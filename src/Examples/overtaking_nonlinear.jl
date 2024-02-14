function setup_overtaking_nonlinear_game(;
    # TODO: the car could also have some other (potentially underactuated, nonlinear) dynamics
    car_dynamics = UnicycleDynamics(;
        state_bounds = (; lb = [-Inf, -Inf, 0.0, -Inf], ub = [Inf, Inf, 5, Inf]),
        control_bounds = (; lb = [-2, -2], ub = [2, 2]),
        dt = 0.2,
        integration_scheme = :reverse_euler,
    ),
    d_collision_constraint_front = 1.0,
    d_collision_constraint_side = 0.5,
    enable_hard_collision_constraints = true,
    enable_constant_velocity_prediction = false,
)
    if enable_constant_velocity_prediction
        # disable all constraints for the const-vel prediction of the pedestrian
        opponent_dynamics = UnicycleDynamics(;
            dt = car_dynamics.dt,
            state_bounds = (; lb = fill(-Inf, 4), ub = fill(Inf, 4)),
            control_bounds = car_dynamics.control_bounds,
            integration_scheme = car_dynamics.integration_scheme,
        )
    else
        opponent_dynamics = car_dynamics
    end

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
        w_speed_p1,
        w_progress_p1,
        w_orientation_p1,
        w_acceleration_control_p1,
        w_steering_control_p1,
        w_lane_p2,
        w_speed_p2,
        w_progress_p2,
        w_orientation_p2,
        w_acceleration_control_p2,
        w_steering_control_p2,
        emergency_mode = params

        # ego (player 1) is in the right lane
        cost_p1 = let
            lane_cost = (x[Block(1)][1] - reference_lane_p1)^2
            speed_cost = x[Block(1)][3]^2
            progress_cost =
                (x[Block(1)][3] * cos(x[Block(1)][4] - π / 2) - reference_speed_p1)^2
            orientation_cost = (x[Block(1)][4] - π / 2)^2
            acceleration_control_cost = sum(u[Block(1)][1] .^ 2)
            steering_control_cost = sum(u[Block(1)][2] .^ 2)

            nominal_cost =
                w_lane_p1 * lane_cost +
                w_speed_p1 * speed_cost +
                w_progress_p1 * progress_cost +
                w_orientation_p1 * orientation_cost +
                w_acceleration_control_p1 * acceleration_control_cost +
                w_steering_control_p1 * steering_control_cost

            emergency_cost = speed_cost

            (1 - emergency_mode) * nominal_cost + emergency_mode * emergency_cost
        end

        # opponent (player 2) is in front of us with an unknown lateral orientation and unknown target lane
        cost_p2 = let
            lane_cost = (x[Block(2)][1] - reference_lane_p2)^2
            speed_cost = x[Block(2)][3]^2
            progress_cost =
                (x[Block(2)][3] * cos(x[Block(2)][4] - π / 2) - reference_speed_p2)^2
            orientation_cost = (x[Block(2)][4] - π / 2)^2
            acceleration_control_cost = sum(u[Block(2)][1] .^ 2)
            steering_control_cost = sum(u[Block(2)][2] .^ 2)

            if enable_constant_velocity_prediction
                100 * (acceleration_control_cost + steering_control_cost)
            else
                w_lane_p2 * lane_cost +
                w_speed_p2 * speed_cost +
                w_progress_p2 * progress_cost +
                w_orientation_p2 * orientation_cost +
                w_acceleration_control_p2 * acceleration_control_cost +
                w_steering_control_p2 * steering_control_cost
            end
        end

        # opponent (player 3) is in front of player 2 with no uncertainty for now
        cost_p3 = let
            reference_lane_p3 = 0.5 # fixed to the right lane
            reference_speed_p3 = 0.5 # slow vehicle
            w_lane_p3 = w_lane_p2
            w_speed_p3 = w_speed_p2
            w_progress_p3 = w_progress_p2
            w_orientation_p3 = w_orientation_p2
            w_acceleration_control_p3 = w_acceleration_control_p2
            w_steering_control_p3 = w_steering_control_p2

            lane_cost = (x[Block(3)][1] - reference_lane_p3)^2
            speed_cost = x[Block(3)][3]^2
            progress_cost =
                (x[Block(3)][3] * cos(x[Block(3)][4] - π / 2) - reference_speed_p3)^2
            orientation_cost = (x[Block(3)][4] - π / 2)^2
            acceleration_control_cost = sum(u[Block(3)][1] .^ 2)
            steering_control_cost = sum(u[Block(3)][2] .^ 2)

            if enable_constant_velocity_prediction
                100 * (acceleration_control_cost + steering_control_cost)
            else
                w_lane_p3 * lane_cost +
                w_speed_p3 * speed_cost +
                w_progress_p3 * progress_cost +
                w_orientation_p3 * orientation_cost +
                w_acceleration_control_p3 * acceleration_control_cost +
                w_steering_control_p3 * steering_control_cost
            end
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
            else
                # p2 wants to merge into the right lane, so we have to stay on their left
                c12 = c(d12, :left)
            end
            # p2 has to stay on the left of p3 who is always in the right lane
            c23 = c(d23, :left)
            # we have to stay on the left of p3
            c13 = c(d13, :left)

            if enable_constant_velocity_prediction
                # no collision avoidance between constant velocity obstacles c23
                [c12; c13]
            else
                [c12; c13; c23]
            end
        end
    else
        coupling_constraints = nothing
    end

    environment = PolygonEnvironment([[-0.1, -5.0], [-0.1, 5.0], [0.6, 5.0], [0.6, -5.0]])
    if enable_constant_velocity_prediction
        environment = ComposedMultiPlayerEnvironment([
            environment,
            TrivialEnvironment(environment),
            TrivialEnvironment(environment),
        ])
    end

    dynamics = ProductDynamics([car_dynamics, opponent_dynamics, opponent_dynamics])

    TrajectoryGame(dynamics, cost, environment, coupling_constraints)
end

function get_overtaking_player_markers(
    branch_index;
    base_marker_per_player = [
        FileIO.load("$(@__DIR__)/../../media/car1.png"),
        FileIO.load("$(@__DIR__)/../../media/car2.png"),
        FileIO.load("$(@__DIR__)/../../media/car3.png"),
    ],
)
    base_marker_per_player
end

function get_overtaking_player_marker_sizes()
    [0.6, 0.6, 0.6]
end

function visualize_overtaking_decorations(axis; swap_axes = false)
    transformed = swap_axes ? x -> [x[2], -x[1]] : identity

    # lane divider
    Makie.lines!(
        axis,
        [Makie.Point2f(transformed([0.25, -5])), Makie.Point2f(transformed([0.25, 5]))],
        color = :lightgray,
        linestyle = :dash,
    )

    # left lane boundary
    Makie.lines!(
        axis,
        [Makie.Point2f(transformed([-0.2, -5])), Makie.Point2f(transformed([-0.2, 5]))],
        color = :lightgray,
        linewidth = 5,
    )

    # right lane boundary
    Makie.lines!(
        axis,
        [Makie.Point2f(transformed([0.7, -5])), Makie.Point2f(transformed([0.7, 5]))],
        color = :lightgray,
        linewidth = 5,
    )
end

function setup_overtaking_nonlinear_demo(;
    solver = nothing,
    initial_state = BlockArrays.mortar([
        [0.5, -4.0, 1.0, π / 2],
        [0.5, -2.9, 0.75, π / 2],
        [0.5, -1.0, 0.75, π / 2],
    ]),
    initial_belief = let
        [
            (;
                weight = 0.5,
                state = initial_state,
                cost_parameters = [
                    0.5, # reference_speed_p2
                    0.0, # reference_lane_p2
                ],
                dynamics_parameters = nothing,
            ),
            (;
                weight = 0.5,
                state = initial_state,
                cost_parameters = [
                    0.5, # match the speed of the slow vehicle in the right lane
                    0.5, # reference_lane_p2
                ],
                dynamics_parameters = nothing,
            ),
        ]
    end,
    parameterized_game = setup_overtaking_nonlinear_game(),
    shared_responsibility = [1.0, 0.001, 0.001],
    context_state_spec = [
        (; name = "reference_speed_p1", range = 0:0.01:2.0, startvalue = 1.0),
        (; name = "reference_lane_p1", range = -1:0.1:1.0, startvalue = 0.5),
        (; name = "w_lane_p1", range = 0:0.1:1.0, startvalue = 0.5),
        (; name = "w_speed_p1", range = 0:0.01:1, startvalue = 0.0),
        (; name = "w_progress_p1", range = 0:0.01:1.0, startvalue = 1.0),
        (; name = "w_orientation_p1", range = 0:0.01:1.0, startvalue = 0.25),
        (; name = "w_acceleration_control_p1", range = 0:0.01:1.0, startvalue = 1.0),
        (; name = "w_steering_control_p1", range = 0.0:0.01:1.0, startvalue = 0.5),
        (; name = "w_lane_p2", range = 0:0.01:1.0, startvalue = 0.5),
        (; name = "w_speed_p2", range = 0:0.01:1.0, startvalue = 0.0),
        (; name = "w_progress_p2", range = 0:0.01:1.0, startvalue = 1.0),
        (; name = "w_orientation_p2", range = 0:0.01:1.0, startvalue = 0.25),
        (; name = "w_acceleration_control_p2", range = 0:0.01:1.0, startvalue = 1.0),
        (; name = "w_steering_control_p2", range = 0:0.01:1.0, startvalue = 0.5),
    ],
    solver_configuration_kwargs = (;),
    compile_constant_velocity_model = false,
    kwargs...,
)
    if compile_constant_velocity_model
        constant_velocity_game_model =
            setup_overtaking_nonlinear_game(; enable_constant_velocity_prediction = true)
    else
        constant_velocity_game_model = nothing
    end

    default_solver_configuration_kwargs = (;
        option_overrides = (; proximal_perturbation = 1e-3),
        reperturbation_schedule = [2e-2, 3e-2],
    )
    setup_demo(;
        solver,
        initial_state,
        initial_belief,
        parameterized_game,
        shared_responsibility,
        context_state_spec,
        solver_configuration_kwargs = (;
            default_solver_configuration_kwargs...,
            solver_configuration_kwargs...,
        ),
        visualize_decorations = visualize_overtaking_decorations,
        get_player_markers = get_overtaking_player_markers,
        player_marker_sizes = get_overtaking_player_marker_sizes(),
        constant_velocity_game_model,
        kwargs...,
    )
end
