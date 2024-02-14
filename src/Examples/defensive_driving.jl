function setup_defensive_driving_game(;
    car_dynamics = planar_double_integrator(;
        state_bounds = (; lb = [-Inf, -Inf, -5, -5], ub = [Inf, Inf, 5, 5]),
        control_bounds = (; lb = [-0.25, -0.25], ub = [0.25, 0.25]),
    ),
    reference_velocity_p1 = 1.0,
    reference_velocity_p2 = -1.0,
    enable_hard_collision_constraints = true,
    d_collision_constraint = 0.5,
)
    println("Defining costs.")

    cost = TimeSeparableTrajectoryGameCost(
        sum,
        GeneralSumCostStructure(),
        1.0,
    ) do x, u, t, context_state
        opponent_type, params... = context_state
        w_lane_cost,
        w_control_cost,
        w_nominal_velocity_cost,
        w_control_cost_adversary,
        w_collision_cost,
        d_collision_penality,
        collision_penalty_smoothness = params

        #collision_cost = let
        #    x1, x2 = blocks(x)
        #    p1, p2 = x1[1:2], x2[1:2]
        #    sqrt(
        #        smooth_max(
        #            d_collision_penality - my_norm(p1 - p2),
        #            0.0;
        #            sharpness = collision_penalty_smoothness,
        #        ) + 1e-4,
        #    )
        #end
        collision_cost = let
            x1, x2 = blocks(x)
            p1, p2 = x1[1:2], x2[1:2]
            v1, v2 = x1[3:4], x2[3:4]
            # scale by impact energy
            -my_norm(p1 - p2) * my_norm_sqr(v1 - v2)
        end

        cost_p1 = let
            # collision avoidance.
            # need to stay close to the lane center on the x axis
            lane_x_p1 = 0.5
            lane_cost_p1 = (x[Block(1)][1] - lane_x_p1)^2
            # want to main a certain nominal travel velocity
            nominal_velocity_cost_p1 = (x[Block(1)][4] - reference_velocity_p1)^2
            # doesn't want to spend a lot of control input
            control_cost_p1 = let
                ux, uy = u[Block(1)]
                ux^2 + uy^2
            end

            w_lane_cost * lane_cost_p1 +
            w_control_cost * control_cost_p1 +
            w_nominal_velocity_cost * nominal_velocity_cost_p1 +
            w_collision_cost * collision_cost
        end

        cost_p2 = let
            # collision avoidance.
            # need to stay close to the lane center on the x axis
            lane_x_p2 = -0.5
            lane_cost_p2 = (x[Block(2)][1] - lane_x_p2)^2
            # want to main a certain nominal travel velocity
            nominal_velocity_cost_p2 = (x[Block(2)][4] - reference_velocity_p2)^2
            # doesn't want to spend a lot of control input
            control_cost_p2 = let
                ux, uy = u[Block(2)]
                ux^2 + uy^2
            end

            if opponent_type === :adversarial
                -cost_p1 + w_control_cost_adversary * control_cost_p2
            elseif opponent_type === :nonadversarial
                w_lane_cost * lane_cost_p2 +
                w_control_cost * control_cost_p2 +
                w_nominal_velocity_cost * nominal_velocity_cost_p2 +
                w_collision_cost * collision_cost
            else
                error("Invalid context state")
            end
        end

        [cost_p1, cost_p2]
    end

    println("Defining dynamics, environment, coupling constraints.")

    dynamics = ProductDynamics([car_dynamics, car_dynamics])
    environment = PolygonEnvironment([[-1.5, -3.0], [-1.5, 3.0], [1.5, 3.0], [1.5, -3.0]])
    if enable_hard_collision_constraints
        coupling_constraints = function (xs, us, _)
            # we only design the pass on the right case and then flip the sign to get the other case
            c = function (d) # d is the relative position between d = p1 - p2
                front_normal = normalize([1.0, -4.0])
                c_front = d' * front_normal - d_collision_constraint
                c_side = d[1] - d_collision_constraint
                ifelse(
                    d[2] <= d_collision_constraint,
                    smooth_max(c_front, c_side; sharpness = 2.5),
                    0.0,
                )
            end
            mapreduce(vcat, xs) do x
                x1, x2 = blocks(x)
                p1, p2 = x1[1:2], x2[1:2]
                d = p1 - p2
                # stay to the right
                c(d)
            end
        end
    else
        coupling_constraints = nothing
    end
    TrajectoryGame(dynamics, cost, environment, coupling_constraints)
end

function setup_defensive_driving_demo(;
    solver = nothing,
    initial_state = BlockArrays.mortar([[0.5, -2.5, 0.0, 1.0], [-0.5, 2.5, 0.0, -1.0]]),
    initial_belief = zip_to_joint_belief(;
        weights = [0.5, 0.5],
        cost_parameters = [:adversarial, :nonadversarial],
        states = [initial_state, initial_state],
        # Later: dynamics = [dyn_1, dyn_2],
    ),
    context_state_spec = [
        (; name = "w_lane", range = 0:0.01:2, startvalue = 1.0),
        (; name = "w_control", range = 0.01:0.01:1.0, startvalue = 0.27),
        (; name = "w_nominal_velocity_cost", range = 0.0:0.01:1.0, startvalue = 0.27),
        (; name = "w_control_adversary", range = 0.0:0.01:1.0, startvalue = 0.3),
        (; name = "w_collision_cost", range = 0.0:0.01:1.0, startvalue = 0.3),
        (; name = "d_collision_penality", range = 0.0:0.01:2.0, startvalue = 0.5),
        (; name = "collision_penalty_smoothness", range = 1.0:0.1:10.0, startvalue = 6.0),
    ],
    parameterized_game = setup_defensive_driving_game(),
    kwargs...,
)
    setup_demo(;
        solver,
        initial_state,
        initial_belief,
        parameterized_game,
        context_state_spec,
        kwargs...,
    )
end
