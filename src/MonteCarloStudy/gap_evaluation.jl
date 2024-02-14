"""
The main function for the gap evaluation:

Evaluate all solvers across a grid of positions for p2 and visualize the gap. Allow the user to
specify some parameters (branching time etc.) via sliders.
"""
function visualize_gap(;
    contingency_game,
    initial_belief,
    solver, # TODO: we could derive a default solver from the game
    limits_per_opponent = nothing,
    velocity_range1 = -1:0.01:1,
    xgrid_points = 4,
    ygrid_points = 4,
    environment_margin = (0.1, 0.05),
    clip_to_zero = false,
    context_state = Float64[],
    shared_responsibility = nothing,
    grid_opponent_index = 2,
    branching_time = 5,
    warmstart_solver_values_on_grid = nothing,
    min_hypothesis_weight = 0.01,
)
    @assert grid_opponent_index > 1
    if shared_responsibility === nothing
        @warn "No shared responsibility provided. Assuming equal responsibility."
        shared_responsibility = ones(TrajectoryGamesBase.num_players(contingency_game))
    end

    p1_initial = initial_belief[1].state[Block(1)][1:2]
    v1_initial = initial_belief[1].state[Block(1)][3:4]

    # setup a basic canvas with environment rendering in the background
    figure = Makie.Figure(resolution = (800, 800))
    axis = TrajectoryGamesExamples.create_environment_axis(
        figure[1, 1],
        contingency_game.env;
        viz_kwargs = (; strokecolor = :black, strokewidth = 1),
        xrectzoom = false,
        yrectzoom = false,
    )

    bottom_config_area = figure[2, 1]
    right_config_area = figure[1, 2]

    # sliders for horizontal and vertical velocity below the environment rendering
    vx1_slider, vy1_slider, branching_time_slider, belief_sliders... =
        Makie.SliderGrid(
            bottom_config_area[1, 1],
            (; label = "vx1 offset", range = velocity_range1, startvalue = 0),
            (; label = "vy1 offset", range = velocity_range1, startvalue = 0),
            (;
                label = "Branching time",
                range = 0:1:(contingency_game.horizon),
                startvalue = branching_time,
            ),
            [
                (;
                    label = "Hypothesis $i",
                    range = (min_hypothesis_weight):0.01:(1 - min_hypothesis_weight),
                    startvalue = initial_belief[i].weight,
                ) for i in 1:length(initial_belief)
            ]...,
        ).sliders

    menu_area = bottom_config_area[2, 1]
    # select the gap type
    Makie.Label(menu_area[1, 1], "Gap\nType"; rotation = pi / 2)
    gap_type_menu = Makie.Menu(
        menu_area[1, 2];
        tellwidth = false,
        options = ["total gap", "lower gap", "upper gap"],
        default = "upper gap",
    )
    # select which of opponent to grid over (2, 3, ..., num(players(contingency_game)))
    Makie.Label(menu_area[2, 1], "Grid\nOpponent"; rotation = pi / 2)
    grid_opponent_index_menu = Makie.Menu(
        menu_area[2, 2];
        tellwidth = false,
        options = [i for i in 2:TrajectoryGamesBase.num_players(contingency_game)],
        default = grid_opponent_index - 1, # this is the index, not the option
    )

    p1 = Makie.Observable(Makie.Point2f(p1_initial))
    v1 = Makie.@lift Makie.Vec2f(
        v1_initial[1] + $(vx1_slider.value),
        v1_initial[2] + $(vy1_slider.value),
    )

    # for the first player, the user can bick the position, for the second player, we generate a
    # grid of initial possitons. Hence, we must first figure out the limits_per_opponent of the environment for
    # the second player.
    grid_range = Makie.@lift let
        if isnothing(limits_per_opponent)
            xlimits, ylimits = get_position_limits(
                contingency_game.env;
                player = $(grid_opponent_index_menu.selection),
                margin = environment_margin,
            )
        else
            xlimits, ylimits = limits_per_opponent[$(grid_opponent_index_menu.selection) - 1]
        end

        (;
            px = range(xlimits..., length = xgrid_points),
            py = range(ylimits..., length = ygrid_points),
        )
    end

    px_range = Makie.@lift $grid_range.px
    py_range = Makie.@lift $grid_range.py

    grid_positions = Makie.@lift let
        map(Iterators.product($px_range, $py_range)) do (x, y)
            Makie.Point2f(x, y)
        end
    end

    belief = Makie.lift([b.value for b in belief_sliders]...) do unnormed_weights...
        weights = normalize(collect(unnormed_weights), 1)
        [(; b..., weight) for (b, weight) in zip(initial_belief, weights)]
    end

    # update button
    update_button =
        Makie.Button(bottom_config_area[3, 1]; label = "Update Heatmap", tellwidth = false)

    # just for testing, display a dummy heatmap over the grid
    solver_values_on_grid = Makie.@lift let
        $(update_button.clicks)[] # trigger on button click
        grid_evaluate_solvers(
            p1[],
            v1[],
            grid_positions[],
            grid_opponent_index_menu.selection[];
            initial_belief = belief[],
            contingency_game,
            solver,
            branching_time = branching_time_slider.value[],
            context_state,
            shared_responsibility,
            warmstart_solver_values_on_grid,
        )
    end

    total_gaps = Makie.@lift map($solver_values_on_grid) do v
        gap = v.upper_bound_cost - v.lower_bound_cost
        if gap <= -1e-2
            @warn "Total gap was negative, $gap, for state $(v.belief[begin].state)"
            clip_to_zero && return 0.0
        end
        gap
    end

    lower_gaps = Makie.@lift map($solver_values_on_grid) do v
        gap = v.contingency_cost - v.lower_bound_cost
        if gap <= -1e-2
            @warn "Lower gap was negative, $gap, for state $(v.belief[begin].state)"
            clip_to_zero && return 0.0
        end
        gap
    end

    upper_gaps = Makie.@lift map($solver_values_on_grid) do v
        gap = v.upper_bound_cost - v.contingency_cost
        if gap <= -1e-2
            @warn "Upper gap was negative, $gap, for state $(v.belief[begin].state)"
            @assert false
            clip_to_zero && return 0.0
        end
        gap
    end

    gaps = Makie.@lift let
        if $(gap_type_menu.selection) == "total gap"
            $total_gaps
        elseif $(gap_type_menu.selection) == "lower gap"
            $lower_gaps
        elseif $(gap_type_menu.selection) == "upper gap"
            $upper_gaps
        else
            error("Unknown gap type $(gap_type_menu.selection)")
        end
    end

    colorrange = Makie.@lift let
        gap_min, gap_max = extrema($gaps)
        (gap_min, gap_max + 0.01)
    end

    total_gap_heatmap = Makie.heatmap!(
        axis,
        px_range,
        py_range,
        gaps;
        colorrange,
        interpolate = true,
        colormap = Makie.cgrad(:starrynight, 10, categorical = true),
    )

    # color bar legend for the heatmap
    Makie.Colorbar(right_config_area[1, 1], total_gap_heatmap)

    # let the user select p1's position with the mouse
    is_position_locked = Makie.Observable(true)
    Makie.on(Makie.events(figure).mouseposition, priority = 0) do _
        if !is_position_locked[]
            p1[] = clip_to_env(
                contingency_game.env,
                Makie.mouseposition(axis.scene);
                player = 1,
                margin = environment_margin,
            )
        end
        Makie.Consume(false)
    end
    Makie.on(Makie.events(figure).mousebutton, priority = 0) do event
        if event.button == Makie.Mouse.left
            button_was_pressed = event.action == Makie.Mouse.press
            click_in_env =
                is_in_environment(Makie.mouseposition(axis.scene), contingency_game.env; player = 1)
            if button_was_pressed && click_in_env
                is_position_locked[] = !is_position_locked[]
            end
        end
        Makie.Consume(false)
    end
    # render the position of p1
    Makie.scatter!(axis, p1; color = :red, markersize = 10)
    #Makie.arrows!(axis, Makie.@lift([$p1]), Makie.@lift([$v1]); color = :red)

    display(figure)

    solver_values_on_grid
end

"""
Evaluate all solvers across a grid of positions for p2.
"""
function grid_evaluate_solvers(
    p1,
    v1,
    grid_positions,
    grid_opponent_index;
    initial_belief,
    contingency_game,
    solver,
    branching_time,
    context_state,
    shared_responsibility,
    warmstart_solver_values_on_grid,
)
    values_on_grid = ProgressMeter.@showprogress "Updating solver values..." map(
        CartesianIndices(grid_positions),
        grid_positions,
    ) do ci, p2
        # override the position and velocity for player 1 and position for player 2
        belief = map(initial_belief) do b
            new_state = copy(b.state)
            new_state[Block(1)][1:2] = p1
            new_state[Block(1)][3:4] = v1
            new_state[Block(grid_opponent_index)][1:2] = p2
            (; b..., state = new_state)
        end

        warmstart =
            isnothing(warmstart_solver_values_on_grid) ? nothing :
            warmstart_solver_values_on_grid[ci]

        evaluate_solvers(;
            solver,
            contingency_game,
            belief,
            context_state,
            shared_responsibility,
            branching_time,
            warmstart,
        )
    end

    values_on_grid
end

"""
Compute the expect cost for each player for a given joint strategy.
"""
function compute_expected_cost(strategy, contingency_game, belief, context_state)
    sum(enumerate(belief)) do (ground_truth_index, b)
        (; weight, state, cost_parameters, dynamics_parameters) = b
        dynamics = contingency_game.dynamics(dynamics_parameters)
        joint_strategy = TrajectoryGamesBase.JointStrategy([
            s.branch_strategies[ground_truth_index] for s in strategy.substrategies
        ])
        (; xs, us) =
            TrajectoryGamesBase.rollout(dynamics, joint_strategy, state, contingency_game.horizon)
        ego_cost = first(contingency_game.cost(xs, us, vcat(cost_parameters, context_state)))

        weight * ego_cost
    end / length(belief)
end

"""
Evaluate a *specific* solver on a fixed configuration (grid point and belief).
"""
function evaluate_solver(;
    solver,
    contingency_game,
    belief,
    branching_time,
    initial_guess,
    context_state,
    shared_responsibility,
    opponent_strategy = nothing,
    replanning_time = nothing,
    warmstart_tail = true,
    resolve_trunk_on_failure = true,
)
    trunk_solution = solve_contingency_game(
        solver,
        contingency_game,
        belief;
        branching_time,
        initial_guess,
        context_state,
        shared_constraint_premultipliers = shared_responsibility,
    )
    is_trunk_converged =
        trunk_solution.info.raw_solution.status == ParametricMCPs.PATHSolver.MCP_Solved

    if !is_trunk_converged && resolve_trunk_on_failure
        @warn "Trunk did not converge, trying with a different initialization strategy"
        # TODO: think about which order of initialization strategies is faster. This is fine for
        # now.
        new_initialization_strategy =
            solver.initialization_strategy isa Solver.CopyInitialStateInitialization ?
            Solver.RolloutInitialization() : Solver.CopyInitialStateInitialization()
        new_initial_guess = Solver.generate_initial_guess(
            solver,
            contingency_game,
            belief,
            new_initialization_strategy,
        )
        trunk_solution = solve_contingency_game(
            solver,
            contingency_game,
            belief;
            branching_time,
            initial_guess = new_initial_guess,
            context_state,
            shared_constraint_premultipliers = shared_responsibility,
        )
        is_trunk_converged =
            trunk_solution.info.raw_solution.status == ParametricMCPs.PATHSolver.MCP_Solved
    end

    if isnothing(opponent_strategy)
        trunk_strategy = trunk_solution.strategy
    else
        # override the opponent's strategy with the given one
        # evaluate the strategy against the opponent
        trunk_strategy = TrajectoryGamesBase.JointStrategy(
            [
                trunk_solution.strategy.substrategies[begin]
                opponent_strategy.substrategies[(begin + 1):end]
            ],
        )
    end

    is_trunk_converged ||
        @warn "Trunk plan not converged for belief $belief,  branching time $branching_time"
    if !is_trunk_converged
        reproducer = (;
            branching_time,
            belief,
            planning_horizon = contingency_game.horizon,
            initial_guess,
            context_state,
            shared_constraint_premultipliers = shared_responsibility,
        )
    end
    @assert is_trunk_converged
    is_replanning_required =
        !isnothing(replanning_time) &&
        0 < branching_time &&
        replanning_time < contingency_game.horizon

    if is_replanning_required
        # re-plan from the state that we ended up at the `replanning_time`
        new_belief = let
            branch_states = map(eachindex(belief)) do ii
                map(trunk_strategy.substrategies) do strategy
                    strategy.branch_strategies[ii].xs[replanning_time]
                end |> BlockArrays.mortar
            end
            [(; b..., state = branch_states[ii]) for (ii, b) in enumerate(belief)]
        end

        if warmstart_tail
            trunk_initial_guess = let
                initialization_strategy = function (contingency_game, solver, belief)
                    rollout_strategy_per_hypothesis =
                        map(1:length(belief)) do hypothesis_index
                            function (x, t)
                                new_t = t + replanning_time - 1
                                if new_t <= contingency_game.horizon
                                    return BlockArrays.mortar([
                                        substrategy.branch_strategies[hypothesis_index](x, new_t) for substrategy in trunk_strategy.substrategies
                                    ])
                                end
                                # if we are past the horizon of the original plan, just apply zero input (will
                                # not be used anyway)
                                BlockArrays.mortar([
                                    zeros(du) for du in solver.dimensions[begin].control_blocks
                                ])
                            end
                        end
                    Solver.RolloutInitialization(nothing)(
                        contingency_game,
                        solver,
                        belief;
                        rollout_strategy_per_hypothesis,
                    )
                end
                Solver.generate_initial_guess(
                    solver,
                    contingency_game,
                    new_belief,
                    initialization_strategy,
                )
            end
        else
            trunk_initial_guess = nothing
        end
        tail_solution = solve_contingency_game(
            solver,
            contingency_game,
            new_belief;
            branching_time = 0,
            planning_horizon = contingency_game.horizon - replanning_time + 1,
            initial_guess = trunk_initial_guess,
            context_state,
            shared_constraint_premultipliers = shared_responsibility,
        )
        is_tail_converged =
            tail_solution.info.raw_solution.status == ParametricMCPs.PATHSolver.MCP_Solved

        if !is_tail_converged
            @warn "Replanning not converged for belief $new_belief and branching time $branching_time"

            reproducer = (;
                branching_time = 0,
                belief = new_belief,
                planning_horizon = contingency_game.horizon - replanning_time + 1,
                initial_guess = trunk_initial_guess,
                context_state,
                shared_constraint_premultipliers = shared_responsibility,
            )
        end
        @assert is_tail_converged

        # combine the two solutions
        strategy =
            let
                map(
                    Iterators.countfrom(),
                    trunk_strategy.substrategies,
                    tail_solution.strategy.substrategies,
                ) do player_index, strategy, new_strategy
                    # TODO: use a re-planner that is aware of the truncated horizon for replanning
                    branch_strategies = map(eachindex(belief)) do jj
                        # stitch the two solutions together along the time-axis
                        xs = [
                            strategy.branch_strategies[jj].xs[1:(replanning_time - 1)]
                            new_strategy.branch_strategies[jj].xs
                        ]
                        us = [
                            strategy.branch_strategies[jj].us[1:(replanning_time - 1)]
                            new_strategy.branch_strategies[jj].us
                        ]
                        # sanity check that we don't have a weird off-by-one error
                        @assert length(xs) == length(us) == contingency_game.horizon
                        if is_trunk_converged && is_tail_converged
                            @assert isapprox(
                                strategy.branch_strategies[jj].xs[replanning_time],
                                new_strategy.branch_strategies[jj].xs[1],
                                atol = 1e-2,
                            )
                        end
                        OpenLoopStrategy(xs, us)
                    end
                    weights = [b.weight for b in belief]
                    ContingencyStrategy((;
                        player_index,
                        branching_time,
                        branch_strategies,
                        weights,
                    ))
                end
            end |> TrajectoryGamesBase.JointStrategy
    else
        strategy = trunk_strategy
    end

    for substrategy in strategy.substrategies
        for branch_strategy in substrategy.branch_strategies
            @assert length(branch_strategy.xs) == contingency_game.horizon
            @assert length(branch_strategy.us) == contingency_game.horizon
        end
    end

    cost = compute_expected_cost(strategy, contingency_game, belief, context_state)

    cost, strategy, trunk_solution
end

"""
Evaluate all solvers on a fixed configuration (grid point and belief).
"""
function evaluate_solvers(;
    solver,
    contingency_game,
    belief,
    context_state,
    shared_responsibility,
    branching_time,
    enable_replanning = true,
    warmstart = nothing,
    opponent_behavior = :contingency,
)
    replanning_time = enable_replanning ? branching_time + 1 : nothing

    # hacky way of toggline between "opponent knows that we are contingency planning" and "opponent
    if opponent_behavior === :contingency
        lower_bound_cost, lower_bound_strategy, lower_bound_trunk = 0.0, nothing, nothing
        opponent_strategy = nothing
    elseif opponent_behavior === :fully_observed
        lower_bound_cost, lower_bound_strategy, lower_bound_trunk = evaluate_solver(;
            solver,
            contingency_game,
            belief,
            context_state,
            shared_responsibility,
            branching_time = 0,
            initial_guess = isnothing(warmstart) ? nothing :
                            warmstart.lower_bound_trunk.info.raw_solution.z,
            replanning_time,
        )
        opponent_strategy = lower_bound_strategy
    else
        error("Unknown opponent behavior $opponent_behavior")
    end

    upper_bound_cost, upper_bound_strategy, upper_bound_trunk = evaluate_solver(;
        solver,
        contingency_game,
        belief,
        context_state,
        shared_responsibility,
        branching_time = contingency_game.horizon,
        initial_guess = isnothing(warmstart) ? nothing :
                        warmstart.upper_bound_trunk.info.raw_solution.z,
        replanning_time,
        opponent_strategy,
    )

    # catch some special cases where we can reuse earlier solutions
    if branching_time == 0 && !isnothing(lower_bound_stratgy)
        contingency_cost = lower_bound_cost
        contingency_strategy = lower_bound_strategy
        contingency_trunk = lower_bound_trunk
    elseif branching_time == contingency_game.horizon
        contingency_cost = upper_bound_cost
        contingency_strategy = upper_bound_strategy
        contingency_trunk = upper_bound_trunk
    else
        contingency_cost, contingency_strategy, contingency_trunk = evaluate_solver(;
            solver,
            contingency_game,
            belief,
            context_state,
            shared_responsibility,
            branching_time,
            initial_guess = isnothing(warmstart) ? nothing :
                            warmstart.contingency_trunk.info.raw_solution.z,
            #nothing, #upper_bound_trunk.info.raw_solution.z,
            replanning_time,
            opponent_strategy,
        )
    end

    return (;
        belief,
        upper_bound_cost,
        contingency_cost,
        lower_bound_cost,
        upper_bound_strategy,
        contingency_strategy,
        lower_bound_strategy,
        upper_bound_trunk,
        contingency_trunk,
        lower_bound_trunk,
        branching_time,
        context_state,
    )
end

"""
Debug util to visualize the trajectory for the grid position with the largest gap.

If the trajectories are not very different there, contingency planning is likely not very relevant
in this setting at all.
"""
function visualize_trajectory_for_maximum_gap(
    values_on_grid,
    solver,
    game;
    show_closed_loop = true,
    f = argmax,
)
    gap_maximizer = f(values_on_grid) do v
        v.upper_bound_cost - v.contingency_cost
    end

    println("Gap: ", gap_maximizer.upper_bound_cost - gap_maximizer.contingency_cost)

    figure = Makie.Figure(; resolution = (800, 800))

    axis = TrajectoryGamesExamples.create_environment_axis(
        figure[1, 1],
        game.env;
        viz_kwargs = (; strokecolor = :black, strokewidth = 1),
    )

    function plot_contingency_strategy!(axis, strategy, branch_colors; marker = :circle)
        for substrategy in strategy.substrategies
            for (branch_strategy, color) in zip(substrategy.branch_strategies, branch_colors)
                for (x, u) in zip(branch_strategy.xs, branch_strategy.us)
                    Makie.scatter!(
                        axis,
                        x[1],
                        x[2];
                        strokecolor = color,
                        strokewidth = 1,
                        color = (color, 0.2),
                        marker,
                    )
                end
            end
        end
    end

    if show_closed_loop
        println("closed-loop")
        plot_contingency_strategy!(
            axis,
            gap_maximizer.contingency_strategy,
            [:red, :green];
            marker = :circle,
        )
        plot_contingency_strategy!(
            axis,
            gap_maximizer.upper_bound_strategy,
            [:orange, :darkgreen];
            marker = '◀',
        )
    else
        println("open-loop")
        plot_contingency_strategy!(
            axis,
            gap_maximizer.contingency_trunk.strategy,
            [:red, :green];
            marker = :circle,
        )
        plot_contingency_strategy!(
            axis,
            gap_maximizer.upper_bound_trunk.strategy,
            [:orange, :darkgreen];
            marker = '◀',
        )
    end

    display(figure)
end

# Some utils for the environment ===================================================================#

function get_position_limits(
    composed_environment::Examples.ComposedMultiPlayerEnvironment;
    player = nothing,
    margin = (0.0, 0.0),
)
    environment = composed_environment.environment_per_player[player]
    get_position_limits(environment; player, margin)
end

function get_position_limits(environment; player = nothing, margin = (0.0, 0.0))
    vertex_points = environment.set.vertices
    xlimits = extrema(p -> p[1], vertex_points)
    ylimits = extrema(p -> p[2], vertex_points)

    # hack; should be done via dispatch
    x_margin = margin[1]
    if x_margin isa Tuple
        tuple_margin_x = (x_margin[1], -x_margin[2])
    else
        tuple_margin_x = (x_margin, -x_margin)
    end
    y_margin = margin[2]
    if y_margin isa Tuple
        tuple_margin_y = (y_margin[1], -y_margin[2])
    else
        tuple_margin_y = (y_margin, -y_margin)
    end

    xlimits .+ tuple_margin_x, ylimits .+ tuple_margin_y
end

function is_in_environment(position, environment; player = nothing, margin = 0.0)
    xlimits, ylimits = get_position_limits(environment; player, margin)
    xlimits[1] <= position[1] <= xlimits[2] && ylimits[1] <= position[2] <= ylimits[2]
end

function clip_to_env(environment, point; player = nothing, margin = (0.0, 0.0))
    xlimits, ylimits = get_position_limits(environment; player, margin)
    px = clamp(point[1], xlimits...)
    py = clamp(point[2], ylimits...)
    typeof(point)(px, py)
end
