function setup_overtaking_game(;
    # TODO: the car could also have some other (potentially underactuated, nonlinear) dynamics
    car_dynamics = planar_double_integrator(;
        state_bounds = (; lb = [-Inf, -Inf, -5, -5], ub = [Inf, Inf, 5, 5]),
        control_bounds = (; lb = [-1.0, -2.0], ub = [1.0, 2.0]),
        dt = 0.2,
    ),
    d_collision_constraint_front = 1.0,
    d_collision_constraint_side = 0.5,
    enable_hard_collision_constraints = true,
)

    # TODO: also add collision avoidance cost and/or constraint (potentially one-sided (not shared))
    cost = TimeSeparableTrajectoryGameCost(
        sum,
        GeneralSumCostStructure(),
        1.0,
    ) do x, u, t, context_state
        reference_speed_p2, reference_lane_p2, params... = context_state
        reference_speed_p1,
        reference_lane_p1,
        w_lane_p1,
        w_control_p1,
        w_speed_p1,
        w_lane_p2,
        w_control_p2,
        w_speed_p2 = params

        # ego (player 1) is in the right lane
        cost_p1 = let
            # need to stay close to the lane center on the x axis
            lane_cost = (x[Block(1)][1] - reference_lane_p1)^2
            # want to main a certain nominal travel velocity
            nominal_speed_cost = (x[Block(1)][4] - reference_speed_p1)^2
            # doesn't want to spend a lot of control input
            control_cost = let
                ux, uy = u[Block(1)]
                ux^2 + uy^2
            end
            w_lane_p1 * lane_cost + w_control_p1 * control_cost + w_speed_p1 * nominal_speed_cost
        end

        # opponent (player 2) is in front of us with an unknown lateral velocity and unknown target lane
        cost_p2 = let
            # need to stay close to the lane center on the x axis
            lane_cost = (x[Block(2)][1] - reference_lane_p2)^2
            # want to be close to their reference speed
            nominal_speed_cost = (x[Block(2)][4] - reference_speed_p2)^2
            # control_cost: doesn't want to spend a lot of control input
            control_cost = let
                ux, uy = u[Block(2)]
                ux^2 + uy^2
            end
            w_lane_p2 * lane_cost + w_control_p2 * control_cost + w_speed_p2 * nominal_speed_cost
        end

        # opponent (player 3) is in front of player 2 with no uncetainty for now
        cost_p3 = let
            # TODO: maybe introduce separate parameters for this player
            w_lane_p3, w_control_p3, w_speed_p3 = w_lane_p2, w_control_p2, w_speed_p2
            reference_lane_p3, reference_speed_p3 = reference_lane_p2, reference_speed_p2 * 0.8

            # need to stay close to the lane center on the x axis
            lane_cost = (x[Block(3)][1] - reference_lane_p3)^2
            # want to be close to their reference speed
            nominal_speed_cost = (x[Block(3)][4] - reference_speed_p3)^2
            # control_cost: doesn't want to spend a lot of control input
            control_cost = let
                ux, uy = u[Block(3)]
                ux^2 + uy^2
            end
            w_lane_p3 * lane_cost + w_control_p3 * control_cost + w_speed_p3 * nominal_speed_cost
        end

        [cost_p1, cost_p2, cost_p3]
    end

    # hard-constraints for collision avoidance
    # TODO: would be nice if these could also be private. Currently, they are always shared
    if enable_hard_collision_constraints
        # we only design the pass on the right case and then flip the sign to get the other case
        c = function (d, side) # d is the relative position between d = p1 - p2
            if side === :left
                d = [-d[1], d[2]]
            end
            front_normal = normalize([1.0, 2.0])
            back_normal = normalize([1.0, -10.0])
            c_front = d' * front_normal - d_collision_constraint_front
            c_back = d' * back_normal - d_collision_constraint_front
            c_side = d[1] - d_collision_constraint_side
            smooth_max(c_front, c_back, c_side; sharpness = 2.5)
        end
        coupling_constraints = function (x, hypothesis_index)
            x1, x2, x3 = blocks(x)
            p1, p2, p3 = x1[1:2], x2[1:2], x3[1:2]
            d12 = p1 - p2
            d13 = p1 - p3
            d23 = p2 - p3

            if hypothesis_index == 1
                # p2 wants to merge into the left lane, so we have to stay on their right
                c12 = c(d12, :right)
                c23 = c(d23, :right)
                c13 = c(d13, :right)
            else
                # p2 wants to merge into the right lane, so we have to stay on their left
                c12 = c(d12, :left)
                c23 = c(d23, :left)
                c13 = c(d13, :left)
            end

            [c12; c13; c23]
        end
    else
        coupling_constraints = nothing
    end

    environment = PolygonEnvironment([[-0.2, -5.0], [-0.2, 5.0], [0.7, 5.0], [0.7, -5.0]])
    dynamics = ProductDynamics([car_dynamics, car_dynamics, car_dynamics])

    TrajectoryGame(dynamics, cost, environment, coupling_constraints)
end

function setup_overtaking_demo(;
    solver = nothing,
    initial_state = BlockArrays.mortar([
        [0.5, -4.0, 0.0, 1.0],
        [0.6, -2.9, 0.0, 0.75],
        [0.5, -1.0, 0.0, 0.75],
    ]),
    initial_belief = let
        [
            (;
                weight = 0.5,
                state = BlockArrays.mortar([
                    [0.5, -4.0, 0.0, 1.0],
                    [0.6, -2.9, -0.1, 0.75],
                    [0.5, -1.0, 0.0, 0.75],
                ]),
                cost_parameters = [
                    0.5, # reference_speed_p2
                    0.0, # reference_lane_p2
                ],
                dynamics_parameters = nothing,
            ),
            (;
                weight = 0.5,
                state = BlockArrays.mortar([
                    [0.5, -4.0, 0.0, 1.0],
                    [0.6, -2.9, 0.0, 0.75],
                    [0.5, -1.0, 0.0, 0.75],
                ]),
                cost_parameters = [
                    0.5, # reference_speed_p2
                    0.5, # reference_lane_p2
                ],
                dynamics_parameters = nothing,
            ),
        ]
    end,
    parameterized_game = setup_overtaking_game(),
    shared_responsibility = [1.0, 0.001, 0.1],
    context_state_spec = [
        (; name = "reference_speed_p1", range = 0:0.01:2.0, startvalue = 1.0),
        (; name = "reference_lane_p1", range = -1:0.1:1.0, startvalue = 0.5),
        (; name = "w_lane_p1", range = 0:0.1:5.0, startvalue = 0.5),
        (; name = "w_control_p1", range = 0:0.01:2.0, startvalue = 0.2),
        (; name = "w_speed_p1", range = 0:0.01:2.0, startvalue = 1.0),
        (; name = "w_lane_p2", range = 0:0.01:2.0, startvalue = 0.5),
        (; name = "w_control_p2", range = 0:0.01:5.0, startvalue = 0.2),
        (; name = "w_speed_p2", range = 0:0.01:2.0, startvalue = 1.0),
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
        visualize_decorations = visualize_overtaking_decorations,
        get_player_markers = get_overtaking_player_markers,
        player_marker_sizes = get_overtaking_player_marker_sizes(),
        kwargs...,
    )
end
