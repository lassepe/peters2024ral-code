function run_interactive_simulation(;
    initial_state,
    initial_belief,
    solver,
    game,
    ego_model = (; solver, game),
    belief_updater = let
        observation_from_state = function (state)
            mortar(blocks(state)[(begin + 1):(end)])
        end
        observation_dimension = length(observation_from_state(initial_state))
        observation_distribution = Distributions.MvNormal(0.02I(observation_dimension))
        BayesianBeliefUpdater(; observation_from_state, observation_distribution)
    end,
    simulate_noise = true,
    branching_time_estimator = BeliefPropagationBranchingTimeEstimator(),
    branch_colors = get_branch_colors(),
    player_colors = get_player_colors(),
    offscreen_rendering = false,
    framerate = offscreen_rendering ? Inf : 20,
    initial_guess = nothing,
    enable_warmstarting = true,
    ground_truth_hypothesis = 1,
    check_termination = function (state, step)
        step == 30
    end,
    title = "",
    belief_marker_style = (; markersize = 0.5, markerspace = :data, strokewidth = 1),
    is_initially_paused = false,
    verbose = true,
    context_state_spec = NamedTuple{(:name, :range, :startvalue)}[],
    shared_responsibility = ones(num_players(game)),
    planning_horizon = game.horizon,
    turn_length = 2, # set this to a higher value if you don't want to re-plan at every time step
    probability_pruning_threshold = 0.01, # hypotheses below this probability are ignored by the ego
    terminate_on_infeasible_solve = true,
    enable_replanning_in_belief_propagation = true,
    record_open_loop_strategies = false,
    video_save_path = nothing,
    get_player_markers = function (ground_truth_index)
        [:circle, :circle, :cricle]
    end,
    player_marker_sizes = [0.5, 0.5, 0.5],
    visualize_decorations = nothing,
    show_hypothesis_markers = false,
    show_observations = false,
    enable_developer_mode = false,
    show_buttons = true,
    axis_kwargs = (;),
    seed = 1,
    ground_truth_max_change_time = 0,
    intent_change_probability = 0.0,
    hypothesis_labels = ["Hypothesis $i" for i in 1:length(initial_belief)],
    resolution = (400, 800),
    show_obsdistribution = false,
    show_infeasible_step = false,
)
    if ismissing(probability_pruning_threshold)
        error("probability_pruning_threshold is missing; please provide manually")
    end
    has_separate_ego_model = ego_model.game !== game
    rng = Random.MersenneTwister(seed)

    function opponent_hypothesis_selector(belief, step, previous_ground_truth_index)
        if step > ground_truth_max_change_time
            return previous_ground_truth_index
        end

        change_intent = rand(rng) >= 1 - intent_change_probability
        if change_intent
            # sample from the set of hypotheses that are not the previous one
            return rand(rng, setdiff(1:length(belief), previous_ground_truth_index))
        end

        previous_ground_truth_index
    end

    # somehow, we cannot hand in the `canvas` directly since this breaks offscreen rendering...
    canvas = Makie.Figure(; resolution)
    branching_time = get_initial_branching_time(branching_time_estimator, game)
    branching_time_set_by_user = false
    is_reset = true
    current_initial_guess_gt = Ref{Any}(initial_guess)
    current_initial_guess_ego = Ref{Any}(initial_guess)
    pressed_color = Makie.RGBf(0.8, 0.8, 0.8)
    default_color = Makie.RGBf(0.94, 0.94, 0.94)
    step = 1
    feasible = true
    was_stopped_by_user = false
    initial_solution = nothing
    trace = (;
        states = [],
        controls = [],
        beliefs = [],
        stage_costs = [],
        branching_time_estimates = [],
        open_loop_strategies = [],
        ground_truth_indices = [],
        observations = [],
    )
    # the strategy used for the Gaussian mixture observation model
    belief_propagation_strategy = nothing

    function get_rotation(state, player_index)
        player_state = state[Block(player_index)]
        player_dynamics = game.dynamics.subsystems[player_index]

        get_rotation(player_state, player_dynamics)
    end

    function get_rotation(state, dynamics::UnicycleDynamics)
        state[4]
    end

    function get_rotation(state, dynamics::LinearDynamics)
        0.0
    end

    next_state = initial_state

    # close to support optional recording
    function _run(io = nothing)
        scene_canvas = canvas[1, 1] = Makie.GridLayout()
        axis = create_environment_axis(
            scene_canvas[1, 1],
            game.env;
            xlabel = L"p_\text{lat}",
            ylabel = L"p_\text{lon}",
            aspect = Makie.DataAspect(),
            ylabelpadding = 5, # hack to work around makie bug
            viz_kwargs = (; color = :white, strokewidth = 0),
            axis_kwargs...,
        )
        Makie.hidedecorations!(axis; label = false)
        if !isnothing(visualize_decorations)
            visualize_decorations(axis)
        end

        axis.title = title

        continue_playing = Makie.Observable(true)
        is_paused = Makie.Observable(is_initially_paused)
        belief = Makie.Observable(initial_belief)
        ground_truth_state = Makie.Observable(initial_state)
        observation = Makie.Observable(belief_updater.observation_from_state(initial_state))
        auxiliary_state = nothing
        support_options = [
            "$(hypothesis_labels[i]): \
            cost parameters: $(b.cost_parameters), dynamics parameters: $(b.dynamics_parameters)"
            for (i, b) in pairs(initial_belief)
        ]

        Makie.barplot(
            scene_canvas[2, 1],
            Makie.@lift([b.weight for b in $belief]);
            direction = :x,
            color = branch_colors[eachindex(initial_belief)],
            #colormap = :RdYlBu_4,
            axis = (;
                limits = ((0, 1), nothing),
                yticks = (eachindex(initial_belief), hypothesis_labels),
                xlabel = "belief b(θ)",
                ylabel = "hypothesis θ",
            ),
        )
        Makie.rowsize!(scene_canvas, 2, Makie.Fixed(100))
        Makie.rowgap!(scene_canvas, 10)

        if enable_developer_mode
            config_area = canvas[1, 2]
            button_area = canvas[2, 1:2]
        else
            config_area = Makie.Figure() # render elements to a non-displayed figure
            button_area = canvas[2, 1]
        end

        if !show_buttons
            button_area = Makie.Figure() # render elements to a non-displayed figure
        end

        belief_config_area = config_area[1, 1]
        Makie.Label(
            belief_config_area[1, 1],
            "Belief\n(unnormalized)";
            rotation = pi / 2,
            font = :bold,
        )
        belief_sliders =
            Makie.SliderGrid(
                belief_config_area[1, 2],
                [
                    (;
                        label = hypothesis_labels[i],
                        range = 0.01:0.01:0.99,
                        startvalue = initial_belief[i].weight,
                    ) for i in 1:length(initial_belief)
                ]...,
            ).sliders

        menu_area = config_area[2, 1]

        Makie.Label(menu_area[1, 1], "Ground\nTruth"; rotation = pi / 2, font = :bold)
        ground_truth_menu = Makie.Menu(
            menu_area[1, 2];
            options = support_options,
            default = support_options[ground_truth_hypothesis],
        )

        reset_button = Makie.Button(button_area[1, 1]; label = "reset state")
        playpause_button = Makie.Button(
            button_area[1, 2];
            label = "play/pause",
            buttoncolor = Makie.@lift($is_paused ? pressed_color : default_color)
        )
        stop_button = Makie.Button(button_area[1, 3]; label = "stop")

        if solver.branching_time_configuration isa DynamicBranchingTime
            branching_time_range = 0:(game.horizon)
            branching_time_slider_subtitle = "(dynamic)"
        elseif solver.branching_time_configuration isa StaticBranchingTime
            @assert turn_length <= solver.branching_time_configuration.value
            tb = solver.branching_time_configuration.value
            branching_time_range = tb:tb
            branching_time_slider_subtitle = "(static)"
        else
            error("Unknown branching time configuration")
        end

        if solver.planning_horizon_configuration isa DynamicPlanningHorizon
            planning_horizon_range = turn_length:(game.horizon)
            planning_horizon_slider_subtitle = "(dynamic)"
        elseif solver.planning_horizon_configuration isa StaticPlanningHorizon
            @assert turn_length <= solver.planning_horizon_configuration.value
            ph = solver.planning_horizon_configuration.value
            planning_horizon_range = ph:ph
            planning_horizon_slider_subtitle = "(static)"
        else
            error("Unknown planning horizon configuration")
        end

        framerate_slider,
        branching_time_slider,
        planning_horizon_slider,
        preview_time_slider,
        remaining_sliders..., =
            Makie.SliderGrid(
                menu_area[2, 1:2],
                (; label = "Framerate", range = 1:20, startvalue = min(20, framerate)),
                (;
                    label = "Branching time\n$(branching_time_slider_subtitle)",
                    range = branching_time_range,
                    startvalue = branching_time,
                ),
                (;
                    label = "Planning horizon\n$(planning_horizon_slider_subtitle)",
                    range = planning_horizon_range,
                    startvalue = planning_horizon,
                ),
                (; label = "Preview time", range = 1:(game.horizon), startvalue = 1),
                [
                    (;
                        label = "Shared responsibility P$ii",
                        range = 0:0.001:1,
                        startvalue = shared_responsibility[ii],
                    ) for ii in 1:num_players(game)
                ]...,
                [
                    (;
                        label = "ctx: $(context_state_spec[ii].name)",
                        range = context_state_spec[ii].range,
                        startvalue = context_state_spec[ii].startvalue,
                    ) for ii in 1:length(context_state_spec)
                ]...,
            ).sliders

        shared_responsibility_sliders = remaining_sliders[1:num_players(game)]
        context_state_sliders = remaining_sliders[(num_players(game) + 1):end]

        slider_weights =
            Makie.lift([b.value for b in belief_sliders]...) do unnormalized_slider_weights...
                normalize(collect(unnormalized_slider_weights), 1)
            end

        shared_constraint_premultipliers =
            Makie.lift([s.value for s in shared_responsibility_sliders]...) do premultipliers...
                collect(premultipliers)
            end

        context_state_values =
            Makie.lift([s.value for s in context_state_sliders]...) do context_state_values...
                collect(context_state_values)
            end

        Makie.on(slider_weights) do weights
            belief[] = set_belief_weights(belief[], weights)
        end

        Makie.on(branching_time_slider.value) do slider_tb
            branching_time = slider_tb
            branching_time_set_by_user = true
            # trigger re-solve
            belief[] = belief[]
            branching_time_set_by_user = false
        end

        previous_ground_truth_index = ground_truth_hypothesis
        ground_truth_index = Makie.@lift let
            # findfirst(==($(ground_truth_menu.selection)), support_options)
            previous_ground_truth_index =
                opponent_hypothesis_selector($belief, step, previous_ground_truth_index)
        end

        Makie.on(stop_button.clicks) do _
            continue_playing[] = false
            was_stopped_by_user = true
        end

        Makie.on(reset_button.clicks) do _
            is_reset = true
            current_initial_guess_gt[] = nothing
            current_initial_guess_ego[] = nothing
            belief_propagation_strategy = initial_solution.strategy
            is_paused[] = true
            sleep(turn_length / framerate_slider.value[])
            next_state = ground_truth_state[] = initial_state
            # trigger re-solve
            step = 1
            branching_time = get_initial_branching_time(branching_time_estimator, game)
            belief[] = initial_belief
        end

        Makie.on(playpause_button.clicks) do _
            is_paused[] = !is_paused[]
        end

        function update_belief_sliders!(values)
            for (slider, value) in zip(belief_sliders, values)
                Makie.set_close_to!(slider, value)
            end
        end

        verbose && @info "Starting simulation loop..."

        solution = Makie.@lift let
            if branching_time_set_by_user
                branching_time = branching_time_slider.value[]
            else
                branching_time = estimate_branching_time(
                    branching_time_estimator,
                    game,
                    belief_updater,
                    turn_length,
                    belief_propagation_strategy,
                    $belief,
                    step,
                )
            end

            is_point_estimate = iszero(branching_time)
            if is_point_estimate
                pruning_allowed_indicators = [true for _ in initial_belief]
                probability_pruning_threshold = 1 / length(initial_belief) - 1e-3
            else
                pruning_allowed_indicators =
                    [step > ground_truth_max_change_time for _ in initial_belief]
            end

            # the gt solver has no state uncertainty:
            gt_belief = [(; b..., state = next_state) for b in $belief]

            gt_sol = solve_contingency_game(
                solver,
                game,
                gt_belief;
                initial_guess = current_initial_guess_gt[],
                branching_time,
                planning_horizon = $(planning_horizon_slider.value),
                shared_constraint_premultipliers = $shared_constraint_premultipliers,
                context_state = $context_state_values,
                probability_pruning_threshold,
                verbose,
                pruning_allowed_indicators,
            )

            if !has_separate_ego_model
                ego_sol = gt_sol
            else
                ego_sol = solve_contingency_game(
                    ego_model.solver,
                    ego_model.game,
                    $belief;
                    initial_guess = current_initial_guess_ego[],
                    branching_time,
                    planning_horizon = $(planning_horizon_slider.value),
                    shared_constraint_premultipliers = ego_model.shared_responsibility,
                    context_state = $context_state_values,
                    probability_pruning_threshold,
                    verbose,
                    pruning_allowed_indicators,
                )
            end

            is_reset = false

            if enable_replanning_in_belief_propagation
                # for the const-vel baseline to still be able to update beliefs, we need
                # it to access the ground-truth solve for the belief updates
                belief_propagation_strategy = gt_sol.strategy
            end

            (; gt_sol, ego_sol)
        end

        initial_solution = solution[].ego_sol
        belief_propagation_strategy = solution[].ego_sol.strategy

        # re-trigger solve so that that we have a consistent tb estimate
        belief[] = belief[]

        info_per_player = map(1:num_players(game)) do player_index
            γ = Makie.@lift $solution.ego_sol.strategy.substrategies[player_index]
            preview_positions = Makie.@lift map(1:length(initial_belief)) do hypothesis_index
                Makie.Point2f($γ.branch_strategies[hypothesis_index].xs[$(preview_time_slider.value)])
            end
            ground_truth_position =
                Makie.@lift Makie.Point2f($ground_truth_state[Block(player_index)][1:2])

            if player_index === 1
                # ego observes their own state perfectly
                observed_position =
                    Makie.@lift Makie.Point2f($ground_truth_state[Block(player_index)][1:2])
            else
                observed_position =
                    Makie.@lift Makie.Point2f($observation[Block(player_index - 1)][1:2])
            end

            (; γ, preview_positions, ground_truth_position, observed_position)
        end

        for (player_index, player_info) in enumerate(info_per_player)
            if show_hypothesis_markers
                for hypothesis_index in eachindex(initial_belief)
                    color = branch_colors[hypothesis_index]
                    weight = Makie.@lift $belief[hypothesis_index].weight
                    player_position = Makie.@lift $(player_info.preview_positions)[hypothesis_index]
                    Makie.scatter!(
                        axis,
                        player_position;
                        color = (color, weight),
                        belief_marker_style...,
                    )
                    Makie.scatter!(axis, player_position; color = (:black, weight), markersize = 5)
                end
            end

            if player_index > 1
                if show_obsdistribution
                    pxs = Makie.@lift (-1:0.01:1) .+ $(player_info.ground_truth_position)[1]
                    pys = Makie.@lift (-1:0.01:1) .+ $(player_info.ground_truth_position)[2]
                    d = Makie.@lift let
                        Distributions.MvNormal(
                            collect($(player_info.ground_truth_position)),
                            belief_updater.observation_distribution.Σ[1:2, 1:2],
                        )
                    end
                    zs = Makie.@lift map(Iterators.product($pxs, $pys)) do (xo, yo)
                        z = Distributions.pdf($d, [xo, yo])
                    end
                    Makie.contour!(axis, pxs, pys, zs)
                end

                # observation visualization
                if show_observations
                    Makie.scatter!(
                        axis,
                        player_info.observed_position;
                        color = player_colors[player_index],
                    )
                end
            end

            # ground truth visualiation
            Makie.scatter!(
                axis,
                player_info.ground_truth_position;
                color = player_colors[player_index],
                marker = Makie.@lift(get_player_markers($ground_truth_index)[player_index]),
                markersize = player_marker_sizes[player_index],
                rotation = Makie.@lift(get_rotation($ground_truth_state, player_index)),
                markerspace = :data,
            )
            Makie.plot!(axis, player_info.γ; show_only_one_trunk_branch = true)
        end

        if !offscreen_rendering
            display(canvas)
            verbose && @info """
                GUI should be visible now. If not, look for a minimized window ;)

                Instructions:

                - Use the `reset` button to rest all agents to their initial state.
                - Use the `stop` button to gracefully terminate the simulation.
                """
        end

        while continue_playing[]
            ground_truth_state[] = next_state

            if !is_paused[]
                if check_termination(ground_truth_state[], step)
                    break
                end

                joint_strategy = let
                    ego_strategy = solution[].ego_sol.strategy.substrategies[1]
                    opponent_strategies = map(2:num_players(game)) do player_index
                        opponent_strategy = solution[].gt_sol.strategy.substrategies[player_index]
                        deepcopy(opponent_strategy.branch_strategies[ground_truth_index[]])
                    end
                    JointStrategy([[ego_strategy]; opponent_strategies])
                end

                branching_time_estimate = joint_strategy.substrategies[1].branching_time
                ground_truth_dynamics_parameters =
                    belief[][ground_truth_index[]].dynamics_parameters

                immediate_states, immediate_controls = let
                    r = rollout(
                        game.dynamics(ground_truth_dynamics_parameters),
                        joint_strategy,
                        ground_truth_state[],
                        turn_length;
                        skip_last_strategy_call = true,
                    )
                    r.xs, r.us
                end

                has_gt_solver_converged =
                    solution[].gt_sol.info.raw_solution.status ==
                    ParametricMCPs.PATHSolver.MCP_Solved
                has_ego_solver_converged =
                    solution[].ego_sol.info.raw_solution.status ==
                    ParametricMCPs.PATHSolver.MCP_Solved
                has_solver_converged = has_gt_solver_converged && has_ego_solver_converged

                is_step_feasible = is_feasible(
                    game,
                    immediate_states,
                    immediate_controls,
                    ground_truth_index[],
                    ground_truth_dynamics_parameters;
                    verbose,
                )

                if (!has_solver_converged || !is_step_feasible) && terminate_on_infeasible_solve
                    verbose && @info "Solver failed to find a feasible solution; terminating."
                    feasible = false
                    # delay breaking the sim loop so that we also record the final (colliding) state
                    show_infeasible_step || break
                end

                immediate_stage_costs =
                    map(immediate_states[begin:(end - 1)], immediate_controls) do x, u
                        # time varying costs are not supported for now because it's not clear which
                        # "time coordinate system" we would use.
                        t = nothing
                        game.cost.stage_cost(
                            x,
                            u,
                            t,
                            vcat(
                                belief[][ground_truth_index[]].cost_parameters,
                                context_state_values[],
                                false,
                            ),
                        )
                    end
                # update the trace
                # Note: We don't append the last state/control because that's where we'll start next time
                append!(trace.stage_costs, immediate_stage_costs)
                append!(trace.states, immediate_states[begin:(end - 1)])
                append!(trace.controls, immediate_controls)
                append!(trace.beliefs, repeat([belief[]], turn_length - 1))
                append!(
                    trace.branching_time_estimates,
                    repeat([branching_time_estimate], turn_length - 1),
                )
                append!(trace.ground_truth_indices, repeat([ground_truth_index[]], turn_length - 1))
                append!(trace.observations, repeat([observation[]], turn_length - 1))

                @assert length(trace.states) ==
                        length(trace.controls) ==
                        length(trace.beliefs) ==
                        length(trace.stage_costs) ==
                        length(trace.branching_time_estimates) ==
                        length(trace.ground_truth_indices)

                if record_open_loop_strategies
                    append!(
                        trace.open_loop_strategies,
                        repeat([solution[].ego_sol.strategy], turn_length - 1),
                    )
                    @assert length(trace.open_loop_strategies) == length(trace.states)
                end

                for t in 1:(turn_length - 1)
                    step_start_time = time()
                    ground_truth_state[] = immediate_states[t]
                    if !isinf(framerate)
                        step_duration = time() - step_start_time
                        sleep_duration = max(1 / framerate_slider.value[] - step_duration, 0.0)
                        sleep(sleep_duration)
                    end
                end

                @assert length(immediate_states) == turn_length
                next_state = immediate_states[end]

                if !feasible
                    # already decided to terminate the simulation above but still need to record the final state
                    ground_truth_state[] = next_state
                    if !isnothing(io)
                        Makie.recordframe!(io)
                    end
                    break
                end

                # basic warmstarting logic
                if enable_warmstarting && !is_reset
                    if has_gt_solver_converged
                        current_initial_guess_gt[] = derive_warmstart_initial_guess(
                            solution[].gt_sol,
                            game,
                            solver,
                            belief[],
                            turn_length,
                        )
                    else
                        step != 1 &&
                            verbose &&
                            @info "no useful initial guess for ground_truth; generating new one"
                        current_initial_guess_gt[] = nothing
                    end

                    if has_separate_ego_model
                        if has_ego_solver_converged
                            current_initial_guess_ego[] = derive_warmstart_initial_guess(
                                solution[].ego_sol,
                                ego_model.game,
                                ego_model.game,
                                belief[],
                                turn_length,
                            )
                        else
                            step != 1 &&
                                verbose &&
                                @info "no useful initial guess for ego; generating new one"
                            current_initial_guess_ego[] = nothing
                        end
                    end
                else
                    current_initial_guess_gt[] = current_initial_guess_ego[] = nothing
                end
                is_reset = false

                # Construct the updated belief
                new_belief, observation[], auxiliary_state = update_belief(
                    belief_updater,
                    turn_length,
                    belief_propagation_strategy,
                    belief[],
                    next_state,
                    step <= ground_truth_max_change_time ? intent_change_probability : 0.0;
                    rng,
                    simulate_noise,
                    auxiliary_state,
                )

                if !isnothing(io)
                    Makie.recordframe!(io)
                end
                step += 1
                # re-assign to trigger a re-solve via the reactive logic of Observables
                belief[] = new_belief
            else
                sleep(0.1) # hack to avoid aggressive busy waiting
            end
            yield()
        end

        (; trace, initial_solution, feasible, was_stopped_by_user)
    end

    # only generate the video if the user has provided a path
    if isnothing(video_save_path)
        return _run()
    end

    Makie.record(
        _run,
        canvas,
        video_save_path;
        framerate = round(Int, 1 / (game.dynamics.subsystems[begin].dt)),
    )

    (; trace, initial_solution, feasible, was_stopped_by_user)
end
