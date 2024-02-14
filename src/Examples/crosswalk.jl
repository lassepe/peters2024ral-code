function setup_crosswalk_game(;
    # TODO: the car could also have some other (potentially underactuated, nonlinear) dynamics
    car_dynamics = planar_double_integrator(;
        state_bounds = (; lb = [-Inf, -Inf, -5, -5], ub = [Inf, Inf, 5, 5]),
        #control_bounds = (; lb = [-0.5, -1.0], ub = [0.5, 0.5]),
        control_bounds = (; lb = [-1.0, -2.0], ub = [1.0, 2.0]),
        dt = 0.2,
    ),
    pedestrian_dynamics = planar_double_integrator(;
        state_bounds = (; lb = [-Inf, -Inf, -0.1, -0.1], ub = [Inf, Inf, 0.1, 0.1]),
        control_bounds = (; lb = [-0.5, -0.5], ub = [0.5, 0.5]),
        dt = 0.2,
    ),
    d_collision_constraint = 0.75,
    enable_hard_collision_constraints = true,
    enable_collision_cost = false,
)

    # TODO: also add collision avoidance cost and/or constraint (potentially one-sided (not shared))
    cost = TimeSeparableTrajectoryGameCost(
        sum,
        GeneralSumCostStructure(),
        1.0,
    ) do x, u, t, context_state
        goal_x_p2, goal_y_p2, params... = context_state
        goal_p2 = [goal_x_p2, goal_y_p2]

        d_collision_penalty,
        collision_avoidance_shape_x,
        collision_avoidance_shape_y,
        reference_speed_p1,
        collision_cost_sharpness,
        w_lane_p1,
        w_control_p1,
        w_speed_p1,
        w_collision_p1,
        w_control_p2,
        w_goal_p2, = params

        if enable_collision_cost
            collision_cost = let
                x1, x2 = blocks(x)
                p1, p2 = x1[1:2], x2[1:2]

                3 * smooth_max(
                    (
                        d_collision_penalty - my_norm(
                            (p1 - p2) .* [collision_avoidance_shape_y, collision_avoidance_shape_x],
                        )
                    ) / d_collision_penalty,
                    0.0;
                    sharpness = collision_cost_sharpness,
                )
            end
        else
            collision_cost = 0.0
        end

        cost_p1 = let
            # need to stay close to the lane center on the x axis
            lane_cost = x[Block(1)][1]^2
            # want to main a certain nominal travel velocity
            nominal_speed_cost = (x[Block(1)][4] - reference_speed_p1)^2
            # doesn't want to spend a lot of control input
            control_cost = let
                ux, uy = u[Block(1)]
                ux^2 + uy^2
            end
            w_lane_p1 * lane_cost +
            w_control_p1 * control_cost +
            w_speed_p1 * nominal_speed_cost +
            w_collision_p1 * collision_cost
        end

        cost_p2 = let
            # want to be close to their goal
            goal_cost = my_norm_sqr(x[Block(2)][1:2] - goal_p2)
            # control_cost: doesn't want to spend a lot of control input
            control_cost = let
                ux, uy = u[Block(2)]
                ux^2 + uy^2
            end
            w_goal_p2 * goal_cost + w_control_p2 * control_cost
        end

        [cost_p1, cost_p2]
    end

    # hard-constraints for collision avoidance
    # TODO: would be nice if these could also be private. Currently, they are always shared
    if enable_hard_collision_constraints
        # we only design the pass on the right case and then flip the sign to get the other case
        c = function (d) # d is the relative position between d = p1 - p2
            front_normal = normalize([1.0, -4.0])
            back_normal = normalize([1.0, 4.0])
            c_front = d' * front_normal - d_collision_constraint
            c_back = d' * back_normal - d_collision_constraint
            c_side = d[1] - d_collision_constraint
            smooth_max(c_front, c_back, c_side; sharpness = 2.5)
        end
        coupling_constraints = function (x, hypothesis_index)
            x1, x2 = blocks(x)
            p1, p2 = x1[1:2], x2[1:2]
            d = p1 - p2
            if hypothesis_index == 1
                # stay to the left
                c([-d[1], d[2]])
            else
                # stay to the right
                c(d)
            end
        end
    else
        coupling_constraints = nothing
    end

    p1_env = PolygonEnvironment([[-1.1, -3.0], [-1.1, 3.0], [1.1, 3.0], [1.1, -3.0]])
    p2_env = PolygonEnvironment([[-1.2, -0.5], [-1.2, 0.5], [1.2, 0.5], [1.2, -0.5]])
    # TODO: naming things: ProductEnvironment
    environment = ComposedMultiPlayerEnvironment([p1_env, p2_env])
    dynamics = ProductDynamics([car_dynamics, pedestrian_dynamics])

    TrajectoryGame(dynamics, cost, environment, coupling_constraints)
end

function setup_crosswalk_demo(;
    solver = nothing,
    initial_state = BlockArrays.mortar([[0.0, -2.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]]),
    initial_belief = zip_to_joint_belief(;
        weights = [0.5, 0.5],
        states = [initial_state, initial_state],
        cost_parameters = [[1.0, 0.0], [-1.0, 0.0]],
    ),
    parameterized_game = setup_crosswalk_game(),
    shared_responsibility = [1.0, 0.1],
    context_state_spec = [
        (; name = "d_collision_penalty", range = 0:0.01:2.0, startvalue = 1.0),
        (; name = "collision_avoidance_shape_x", range = 0.1:0.1:20, startvalue = 1),
        (; name = "collision_avoidance_shape_y", range = 0.1:0.1:20, startvalue = 1),
        (; name = "reference_speed_p1", range = 0:0.01:2.0, startvalue = 1.0),
        (; name = "collision_cost_sharpness", range = 0.01:0.01:10.0, startvalue = 1.0),
        (; name = "w_lane_p1", range = 0:0.1:5.0, startvalue = 0.5),
        (; name = "w_control_p1", range = 0:0.01:2.0, startvalue = 1.0),
        (; name = "w_speed_p1", range = 0:0.01:2.0, startvalue = 1.0),
        (; name = "w_collision_p1", range = 0:0.01:10.0, startvalue = 1.0),
        (; name = "w_control_p2", range = 0:0.01:2.0, startvalue = 2.0),
        (; name = "w_goal_p2", range = 0:0.01:2.0, startvalue = 1.0),
    ],
    kwargs...,
)
    setup_demo(;
        solver,
        initial_state,
        initial_belief,
        parameterized_game,
        shared_responsibility,
        context_state_spec,
        visualize_decorations = visualize_crosswalk_decorations,
        get_player_markers = get_crosswalk_player_markers,
        player_marker_sizes = get_crosswalk_player_marker_sizes(),
        kwargs...,
    )
end
