function demo(demo_setup; use_constant_velocity_ego_model = false, sim_kwargs = (;))
    (;
        contingency_game,
        solver,
        constant_velocity_ego_model,
        initial_state,
        initial_belief,
        context_state_spec,
        shared_responsibility,
        visualize_decorations,
        get_player_markers,
        player_marker_sizes,
    ) = demo_setup

    if use_constant_velocity_ego_model
        ego_model = constant_velocity_ego_model
    else
        ego_model = (; solver, game = contingency_game, shared_responsibility)
    end

    run_interactive_simulation(;
        initial_state,
        initial_belief,
        solver,
        game = contingency_game,
        ego_model,
        framerate = 10,
        context_state_spec,
        shared_responsibility,
        visualize_decorations,
        get_player_markers,
        player_marker_sizes,
        sim_kwargs...,
    )
end

function setup_demo(;
    solver,
    initial_state,
    initial_belief,
    parameterized_game,
    visualize_decorations,
    get_player_markers,
    player_marker_sizes,
    solver_configuration_kwargs = (;),
    horizon = 25,
    verbose = true,
    shared_responsibility = ones(num_players(parameterized_game)),
    context_state_spec = [],
    warm_up_solver = true,
    constant_velocity_game_model = nothing,
)
    contingency_game = ContingencyGame((; horizon, parameterized_game))
    solver = @something(
        solver,
        MCPContingencySolver(
            contingency_game,
            initial_belief;
            context_state_dimension = length(context_state_spec),
            solver_configuration_kwargs...,
        )
    )

    # conditionally overwrite the ego model
    if isnothing(constant_velocity_game_model)
        constant_velocity_ego_model = (; solver, game = contingency_game, shared_responsibility)
    else
        ego_game_model =
            ContingencyGame((; horizon, parameterized_game = constant_velocity_game_model))
        ego_solver = MCPContingencySolver(
            ego_game_model,
            initial_belief;
            context_state_dimension = length(context_state_spec),
            solver_configuration_kwargs...,
        )
        constant_velocity_ego_model = (;
            solver = ego_solver,
            game = ego_game_model,
            shared_responsibility = [1.0; zeros(num_players(ego_game_model) - 1)],
        )
    end

    if warm_up_solver
        verbose && @info "triggering initial solve for precompilation..."
        initial_solution = solve_contingency_game(
            solver,
            contingency_game,
            initial_belief;
            context_state = isempty(context_state_spec) ? Float64[] :
                            [ctx.startvalue for ctx in context_state_spec],
            shared_constraint_premultipliers = shared_responsibility,
        )
    else
        initial_solution = nothing
    end

    (;
        contingency_game,
        solver,
        constant_velocity_ego_model,
        initial_state,
        initial_belief,
        initial_solution,
        context_state_spec,
        shared_responsibility,
        visualize_decorations,
        get_player_markers,
        player_marker_sizes,
    )
end
