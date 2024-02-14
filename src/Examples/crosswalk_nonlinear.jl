function constant_velocity_game(game)
    n_players = TrajectoryGamesBase.num_players(game)
    dynamics = let
        ego_dynamics = game.dynamics.subsystems[1]
        opponent_dynamics = planar_double_integrator(;
            state_bounds = (; lb = [-Inf, -Inf, -Inf, -Inf], ub = [Inf, Inf, Inf, Inf]),
            control_bounds = (; lb = [0.0, 0.0], ub = [0.0, 0.0]),
            dt = ego_dynamics.dt,
        )

        ProductDynamics([ego_dynamics [opponent_dynamics for _ in 2:n_players]])
    end

    cost = TimeSeparableTrajectoryGameCost(
        sum,
        GeneralSumCostStructure(),
        1.0,
    ) do x, u, t, context_state
        ego_cost = game.cost.state_cost(x, u, t, context_state)[1]
        opponent_costs = map(2:n_players) do ii
            control_cost = let
                ux, uy = u[Block(ii)]
                ux^2 + uy^2
            end
            100 * control_cost
        end
        [ego_cost; opponent_costs]
    end

    environment = ComposedMultiPlayerEnvironment([
        ii == 1 ? game.env : TrivialEnvironment(game.env) for ii in 1:n_players
    ])

    @assert false "not implemented"
    coupling_constraints = game.coupling_constraints

    TrajectoryGame(dynamics, cost, environment, coupling_constraints)
end

function setup_crosswalk_nonlinear_game(;
    # TODO: the car could also have some other (potentially underactuated, nonlinear) dynamics
    car_dynamics = UnicycleDynamics(;
        state_bounds = (; lb = [-Inf, -Inf, 0.0, -Inf], ub = [Inf, Inf, 5, Inf]),
        control_bounds = (; lb = [-2, -2], ub = [2, 2]),
        dt = 0.2,
        integration_scheme = :reverse_euler,
    ),
    pedestrian_dynamics = planar_double_integrator(;
        state_bounds = (; lb = [-Inf, -Inf, -0.1, -0.1], ub = [Inf, Inf, 0.1, 0.1]),
        control_bounds = (; lb = [-0.5, -0.5], ub = [0.5, 0.5]),
        dt = 0.2,
    ),
    d_collision_constraint = 0.75,
    enable_hard_collision_constraints = true,
    enable_collision_cost = false,
    enable_constant_velocity_prediction = false,
)
    if enable_constant_velocity_prediction
        # disable all constraints for the const-vel prediction of the pedestrian
        pedestrian_dynamics.state_bounds.lb .= -Inf
        pedestrian_dynamics.state_bounds.ub .= Inf
    end

    cost = TimeSeparableTrajectoryGameCost(
        sum,
        GeneralSumCostStructure(),
        1.0,
    ) do x, u, t, context_state
        goal_x_p2, goal_y_p2, params... = context_state
        goal_p2 = [goal_x_p2, goal_y_p2]

        w_progress_p1,
        w_lane_p1,
        w_acceleration_control_p1,
        w_steering_control_p1,
        w_speed_p1,
        w_orientation_p1,
        w_control_p2,
        w_goal_p2,
        emergency_mode = params

        if enable_collision_cost
            @assert false "not implemented"
        end

        cost_p1 = let
            # need to stay close to the lane center on the x axis
            lane_cost = x[Block(1)][1]^2
            speed_cost = x[Block(1)][3]^2
            progress_cost = (x[Block(1)][3] * cos(x[Block(1)][4] - π / 2) - 1)^2
            orientation_cost = (x[Block(1)][4] - π / 2)^2
            acceleration_control_cost = sum(u[Block(1)][1] .^ 2)
            steering_control_cost = sum(u[Block(1)][2] .^ 2)

            (1 - emergency_mode) * (
                w_progress_p1 * progress_cost +
                w_lane_p1 * lane_cost +
                w_acceleration_control_p1 * acceleration_control_cost +
                w_steering_control_p1 * steering_control_cost +
                w_orientation_p1 * orientation_cost
            ) + w_speed_p1 * speed_cost
        end

        cost_p2 = let
            # want to be close to their goal
            goal_cost = my_norm_sqr(x[Block(2)][1:2] - goal_p2)
            # control_cost: doesn't want to spend a lot of control input
            control_cost = let
                ux, uy = u[Block(2)]
                ux^2 + uy^2
            end

            if enable_constant_velocity_prediction
                100 * control_cost
            else
                w_goal_p2 * goal_cost + w_control_p2 * control_cost
            end
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
    if enable_constant_velocity_prediction
        p2_env = TrivialEnvironment(p2_env)
    end
    # TODO: naming things: ProductEnvironment
    environment = ComposedMultiPlayerEnvironment([p1_env, p2_env])
    dynamics = ProductDynamics([car_dynamics, pedestrian_dynamics])

    TrajectoryGame(dynamics, cost, environment, coupling_constraints)
end

function get_crosswalk_player_markers(
    branch_index;
    base_marker_per_player = [
        FileIO.load("$(@__DIR__)/../../media/car1.png"),
        FileIO.load("$(@__DIR__)/../../media/ped.png"),
    ],
)
    if branch_index === 1
        base_marker_per_player
    else
        ped_marker = reverse(base_marker_per_player[2]; dims = 2)
        [base_marker_per_player[1], ped_marker]
    end
end

function get_crosswalk_player_marker_sizes()
    [0.6, 0.35]
end

function visualize_crosswalk_decorations(axis)
    # left sidewalk
    Makie.poly!(axis, Makie.Rect(-1.5, -3, 0.5, 6), color = :lightgray)
    # right sidewalk
    Makie.poly!(axis, Makie.Rect(1.0, -3, 0.5, 6), color = :lightgray)
    axis
end

function setup_crosswalk_nonlinear_demo(;
    solver = nothing,
    initial_state = BlockArrays.mortar([[0.0, -2.0, 1.0, π / 2], [0.0, 0.0, 0.0, 0.0]]),
    initial_belief = zip_to_joint_belief(;
        weights = [0.5, 0.5],
        states = [initial_state, initial_state],
        cost_parameters = [[1.0, 0.0], [-1.0, 0.0]],
    ),
    parameterized_game = setup_crosswalk_nonlinear_game(),
    shared_responsibility = [1.0, 0.1],
    context_state_spec = [
        (; name = "w_progress_p1", range = 0:0.01:2.0, startvalue = 1.0),
        (; name = "w_lane_p1", range = 0:0.1:5.0, startvalue = 0.5),
        (; name = "w_acceleration_control_p1", range = 0:0.01:2.0, startvalue = 1.0),
        (; name = "w_steering_control_p1", range = 0:0.01:5.0, startvalue = 0.5),
        (; name = "w_speed_p1", range = 0:0.01:1.0, startvalue = 0.25),
        (; name = "w_orientation_p1", range = 0:0.01:5.0, startvalue = 0.0),
        (; name = "w_control_p2", range = 0:0.01:2.0, startvalue = 2.0),
        (; name = "w_goal_p2", range = 0:0.01:2.0, startvalue = 1.0),
    ],
    solver_configuration_kwargs = (;),
    compile_constant_velocity_model = false,
    kwargs...,
)
    if compile_constant_velocity_model
        constant_velocity_game_model =
            setup_crosswalk_nonlinear_game(; enable_constant_velocity_prediction = true)
    else
        constant_velocity_game_model = nothing
    end

    default_solver_configuration_kwargs = (;
        option_overrides = (; proximal_perturbation = 1e-4),
        reperturbation_schedule = [2e-2, 3e-2, 4e-2],
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
        visualize_decorations = visualize_crosswalk_decorations,
        get_player_markers = get_crosswalk_player_markers,
        player_marker_sizes = get_crosswalk_player_marker_sizes(),
        constant_velocity_game_model,
        kwargs...,
    )
end
