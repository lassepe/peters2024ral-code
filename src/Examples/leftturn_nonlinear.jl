struct LevelSetEnvironment{T}
    """
    level_sets = [level_set(xs) -> v::Real for i in 1:number_of_constraints]
    visualization_limits = ((xmin, xmax), (ymin, ymax))
    """
    fields::T
end

function Base.getproperty(env::LevelSetEnvironment, field::Symbol)
    getproperty(getfield(env, :fields), field)
end

function TrajectoryGamesBase.get_constraints(
    environment::LevelSetEnvironment,
    player_index = nothing,
)
    function (state)
        positions = (substate[1:2] for substate in blocks(state))
        mapreduce(
            vcat,
            Iterators.product(environment.level_sets, positions),
        ) do (level_set, position)
            level_set(position)
        end
    end
end

function TrajectoryGamesBase.visualize!(
    axis,
    environment::LevelSetEnvironment;
    color = :lightgray,
    kwargs...,
)
    xgrid = range(environment.visualization_limits[1]...; length = 200)
    ygrid = range(environment.visualization_limits[2]...; length = 200)
    zgrid = map(Iterators.product(xgrid, ygrid)) do (x, y)
        all(v >= 0 for v in TrajectoryGamesBase.get_constraints(environment, nothing)([x, y]))
    end

    Makie.contourf!(axis, xgrid, ygrid, zgrid; kwargs...)
    #Makie.Colorbar(axis[1, 2], plot)
    axis
end

function intersection_environment(; sharpness = 2, visualization_limits = ((-5, 5), (-5, 5)))
    sharpness = 2
    visualization_limits = ((-5, 5), (-5, 5))
    # carve out each cordner of the square with the smooth_max trick
    level_sets = [
        # cannot be in the top-left corner
        (x) -> smooth_max(x[1] + 1, -x[2] + 1; sharpness),
        # cannot be in the top-right corner
        (x) -> smooth_max(-x[1] + 1, -x[2] + 1; sharpness),
        # cannot be in the bottom-right corner
        (x) -> smooth_max(-x[1] + 1, x[2] + 1; sharpness),
        # cannot be in the bottom-left corner
        (x) -> smooth_max(x[1] + 1, x[2] + 1; sharpness),
    ]
    LevelSetEnvironment((; level_sets, visualization_limits))
end

function test_levelset_environment()
    environment = intersection_environment()
    figure = Makie.Figure()
    TrajectoryGamesBase.visualize!(figure, environment)
    figure
end

function setup_leftturn_nonlinear_game(
    # TODO: the car could also have some other (potentially underactuated, nonlinear) dynamics
    #car_dynamics = UnicycleDynamics(;
    #    state_bounds = (; lb = [-Inf, -Inf, 0.0, -Inf], ub = [Inf, Inf, 5, Inf]),
    #    control_bounds = (; lb = [-2, -2], ub = [2, 2]),
    #    dt = 0.2,
    #    integration_scheme = :reverse_euler,
    #),
    car_dynamics = planar_double_integrator(;
        state_bounds = (; lb = [-Inf, -Inf, -5, -5], ub = [Inf, Inf, 5, 5]),
        control_bounds = (; lb = [-2, -2], ub = [2, 2]),
        dt = 0.2,
    ),
    d_collision_constraint_lon = 1.0,
    d_collision_constraint_lat = 0.5,
    enable_hard_collision_constraints = true,
)

    # TODO: also add collision avoidance cost and/or constraint (potentially one-sided (not shared))
    cost = TimeSeparableTrajectoryGameCost(
        sum,
        GeneralSumCostStructure(),
        1.0,
    ) do x, u, t, context_state
        goal_x_p2, goal_y_p2, params... = context_state

        w_goal_p1,
        w_speed_p1,
        w_acceleration_control_p1,
        w_steering_control_p1,
        w_goal_p2,
        w_speed_p2,
        w_acceleration_control_p2,
        w_steering_control_p2 = params

        # ego (player 1) is in the right lane
        cost_p1 = let
            goal_p1 = [-5, 0.5] # we want to turn left
            goal_cost = my_norm_sqr(x[Block(1)][1:2] - goal_p1)
            speed_cost = x[Block(1)][3]^2
            acceleration_control_cost = sum(u[Block(1)][1] .^ 2)
            steering_control_cost = sum(u[Block(1)][2] .^ 2)

            w_goal_p1 * goal_cost +
            w_speed_p1 * speed_cost +
            w_acceleration_control_p1 * acceleration_control_cost +
            w_steering_control_p1 * steering_control_cost
        end

        # opponent (player 2) is in front of us with an unknown lateral orientation and unknown target lane
        cost_p2 = let
            goal_p2 = [goal_x_p2, goal_y_p2]
            goal_cost = my_norm_sqr(x[Block(2)][1:2] - goal_p2)
            speed_cost = x[Block(2)][3]^2
            acceleration_control_cost = sum(u[Block(2)][1] .^ 2)
            steering_control_cost = sum(u[Block(2)][2] .^ 2)

            w_goal_p2 * goal_cost +
            w_speed_p2 * speed_cost +
            w_acceleration_control_p2 * acceleration_control_cost +
            w_steering_control_p2 * steering_control_cost
        end

        [cost_p1, cost_p2]
    end

    # hard-constraints for collision avoidance
    # TODO: would be nice if these could also be private. Currently, they are always shared
    if enable_hard_collision_constraints
        coupling_constraints = function (xs, us, hypothesis_index)
            # we only design the pass on the right case and then flip the sign to get the other case
            # TODO: Fix to also support "behind"
            c = function (d, lat_side, lon_side) # d is the relative position between d = p1 - p2
                if lat_side === :left
                    lat_normal = [-1, 0]
                elseif lat_side === :right
                    lat_normal = [1, 0]
                else
                    error("Invalid lateral side")
                end
                if lon_side === :below
                    lon_normal = [0, -1]
                elseif lon_side === :above
                    lon_normal = [0, 1]
                else
                    error("Invalid longitudinal side")
                end

                c_lat = d' * lat_normal - d_collision_constraint_lat
                c_lon = d' * lon_normal - d_collision_constraint_lon
                smooth_max(c_lat, c_lon; sharpness = 2.5)
            end

            mapreduce(vcat, xs) do x
                x1, x2 = blocks(x)
                p1, p2 = x1[1:2], x2[1:2]
                d12 = p1 - p2

                if hypothesis_index == 1 || hypothesis_index == 2
                    # p2 wants to turn right (from their perspective) or go straight. Thus, we have to stay to
                    # their right and can only merge behind them
                    c12 = c(d12, :right, :above)
                else
                    # p2 wants to turn left (from their perspective). Thus, we have to stay to their
                    # left and have to turn in front of them
                    c12 = c(d12, :left, :below)
                end

                c12
            end
        end
    else
        coupling_constraints = nothing
    end

    environment = intersection_environment()
    dynamics = ProductDynamics([car_dynamics, car_dynamics])

    TrajectoryGame(dynamics, cost, environment, coupling_constraints)
end

function setup_leftturn_nonlinear_demo(;
    solver = nothing,
    #initial_state = BlockArrays.mortar([[0.5, -4.0, 1.0, π / 2], [-0.5, 4.0, 1.0, -π / 2]]),
    initial_state = BlockArrays.mortar([[0.5, -4.0, 0.0, 0.0], [-0.5, 4.0, 0.0, 0.0]]),
    initial_belief = let
        [
            (;
                weight = 1 / 3,
                state = initial_state,
                # p2 wants to turn right
                cost_parameters = [
                    -5,  # goal_x_p2
                    0.5, # goal_y_p2
                ],
                dynamics_parameters = nothing,
            ),
            (;
                weight = 1 / 3,
                state = initial_state,
                # p2 wanto to go straight
                cost_parameters = [
                    -0.5, # goal_x_p2
                    -5.0,  # goal_y_p2
                ],
                dynamics_parameters = nothing,
            ),
            (;
                weight = 1 / 3,
                state = initial_state,
                # p2 wanto to go straight
                cost_parameters = [
                    5,    # goal_x_p2
                    -0.5, # goal_y_p2
                ],
                dynamics_parameters = nothing,
            ),
        ]
    end,
    parameterized_game = setup_leftturn_nonlinear_game(),
    shared_responsibility = [1.0, 0.01],
    context_state_spec = [
        (; name = "w_goal_p1", range = 0:0.1:1.0, startvalue = 0.1),
        (; name = "w_speed_p1", range = 0:0.1:1.0, startvalue = 0.5),
        (; name = "w_acceleration_control_p1", range = 0:0.1:1.0, startvalue = 0.5),
        (; name = "w_steering_control_p1", range = 0:0.1:1.0, startvalue = 0.5),
        (; name = "w_goal_p2", range = 0:0.1:1.0, startvalue = 0.1),
        (; name = "w_speed_p2", range = 0:0.1:1.0, startvalue = 0.5),
        (; name = "w_acceleration_control_p2", range = 0:0.1:1.0, startvalue = 0.5),
        (; name = "w_steering_control_p2", range = 0:0.1:1.0, startvalue = 0.5),
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
        kwargs...,
    )
end
