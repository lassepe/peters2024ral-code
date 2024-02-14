function setup_receding_horizon_crosswalk_demo(;
    warm_up_solver = true,
    planning_horizon = 25,
    solver_configuration_kwargs = (;
        constraint_tightening_factors = (;
            environment_constraints = 1e-3,
            state_box_constraints = 1e-3,
            control_box_constraints = 1e-3,
            coupling_constraints = 1e-3,
        ),
        initial_state_slack = (; maximum_value = 0.0, penalty = 1e3),
        planning_horizon_configuration = Solver.StaticPlanningHorizon(planning_horizon),
    ),
    compile_constant_velocity_model = true,
    kwargs...,
)
    Examples.setup_crosswalk_nonlinear_demo(;
        warm_up_solver,
        horizon = planning_horizon,
        solver_configuration_kwargs,
        compile_constant_velocity_model,
        kwargs...,
    )
end

function entropy(x::AbstractVector)
    -sum(xi * log(length(x), xi) for xi in x)
end

function entropy(p::Real)
    entropy([p, 1 - p])
end

function get_probability_threshold_from_entropy(entropy_threshold)
    Roots.find_zero((1e-10, 0.5)) do p
        entropy(p) - entropy_threshold
    end
end

function setup_receding_horizon_overtaking_demo(;
    warm_up_solver = true,
    planning_horizon = 25,
    solver_configuration_kwargs = (;
        constraint_tightening_factors = (;
            environment_constraints = 1e-3,
            state_box_constraints = 1e-3,
            control_box_constraints = 1e-3,
            coupling_constraints = 1e-3,
        ),
        initial_state_slack = (; maximum_value = 0.0, penalty = 1e3),
        planning_horizon_configuration = Solver.StaticPlanningHorizon(planning_horizon),
    ),
    compile_constant_velocity_model = true,
)
    Examples.setup_overtaking_nonlinear_demo(;
        warm_up_solver,
        horizon = planning_horizon,
        solver_configuration_kwargs,
        compile_constant_velocity_model,
    )
end

function get_method_color(method_name)
    vega_lite_color_scheme = ColorSchemes.glasbey_category10_n256

    if method_name ∈ (:ours, "pessimistic_beliefpropagation_contingency")
        vega_lite_color_scheme[2]
    elseif method_name ∈ (:map, "tb_0")
        colorant"black"
    elseif method_name ∈ (:qmdp, "tb_1")
        colorant"red"
    elseif method_name ∈ (:baseline, "plan_in_expectation")
        :gray
    elseif method_name ∈ (:oracle, "hindsight_contingency")
        :teal
    elseif method_name ∈ (:mpc, "constvel")
        :blue
    else
        error("Unknown method_name: $method_name")
    end
end

function get_streamlined_name(method_name; multiline = false)
    if method_name ∈ (:map, "tb_0")
        multiline ? "Baseline 1\n(tb = 1)" : "Baseline 1 (tb = 1)"
    elseif method_name ∈ (:baseline, "plan_in_expectation")
        multiline ? "Baseline 2\n(tb = T)" : "Baseline 2 (tb = T)"
    elseif method_name ∈ (:ours, "pessimistic_beliefpropagation_contingency")
        multiline ? "Ours\n(heuristic)" : "Ours (heuristic)"
    elseif method_name ∈ (:qmdp, "tb_1")
        multiline ? "Ours \n(tb = 2)" : "Ours (tb = 2)"
    elseif method_name ∈ (:oracle, "hindsight_contingency")
        multiline ? "Ours\n(oracle)" : "Ours (oracle)"
    elseif method_name ∈ (:mpc, "constvel")
        multiline ? "Baseline 3\n(MPC)" : "Baseline 3 (MPC)"
    else
        error("Unknown method_name: $method_name")
    end
end

METHOD_NAMES = [:map, :baseline, :mpc, :ours, :qmdp, :oracle]

function get_grid_range(
    contingency_game,
    player_index;
    xgrid_points = 10,
    ygrid_points = 7,
    environment_margin = (0.1, 0.05),
    limits = nothing,
)
    if isnothing(limits)
        xlimits, ylimits = get_position_limits(
            contingency_game.env;
            player = player_index,
            margin = environment_margin,
        )
    else
        xlimits, ylimits = limits
    end

    @assert xlimits[1] <= xlimits[2]
    @assert ylimits[1] <= ylimits[2]

    if xgrid_points === 1
        px = [Statistics.mean(xlimits)]
    else
        px = range(xlimits..., length = xgrid_points)
    end

    if ygrid_points === 1
        py = [Statistics.mean(ylimits)]
    else
        py = range(ylimits..., length = ygrid_points)
    end

    (; px, py)
end

function get_initial_state_grid(contingency_game, initial_state, player_index; kwarg...)
    grid_range = get_grid_range(contingency_game, player_index; kwarg...)
    map(Iterators.product(grid_range.px, grid_range.py)) do (px, py)
        initial_state = deepcopy(initial_state)
        initial_state[Block(player_index)][1:2] .= [px, py]
        initial_state
    end
end

function get_initial_state_grid(demo_setup, player_index = 2; kwarg...)
    get_initial_state_grid(
        demo_setup.contingency_game,
        demo_setup.initial_state,
        player_index;
        kwarg...,
    )
end

function modified_initial_belief(demo_setup; initial_state, initial_weights)
    @reset demo_setup.initial_state = initial_state
    @reset demo_setup.initial_belief = map(eachindex(demo_setup.initial_belief)) do ii
        b = demo_setup.initial_belief[ii]
        weight = initial_weights[ii]
        @reset b.state = initial_state
        @reset b.weight = weight
        b
    end
end

function set_up_workers(worker_pool_size = :default)
    has_no_workers = Distributed.nprocs() == 1
    if has_no_workers
        worker_pool_size === :default ? Distributed.addprocs() :
        Distributed.addprocs(worker_pool_size)
    end

    @eval Main using Distributed: @everywhere
    @eval Main @everywhere(using ContingencyGames)
end

function receding_horizon_crosswalk_benchmark(demo_setup; kwargs...)
    @assert TrajectoryGamesBase.num_players(demo_setup.contingency_game) == 2
    initial_states = get_initial_state_grid(demo_setup, 2)
    observation_noise_σs = range(start = 0.002, step = 0.004, length = 5)
    probability_pruning_threshold = 0.01
    receding_horizon_benchmark(;
        demo_setup,
        initial_states,
        observation_noise_σs,
        probability_pruning_threshold,
        kwargs...,
    )
end

function receding_horizon_overtaking_benchmark(demo_setup; kwargs...)
    @assert TrajectoryGamesBase.num_players(demo_setup.contingency_game) == 3
    initial_states_p2 = get_initial_state_grid(demo_setup, 2; limits = ((0.0, 0.5), (-2.8, -2.3)))
    initial_states_p3 = get_initial_state_grid(demo_setup, 3; limits = ((0.0, 0.5), (-1.5, -1.0)))
    initial_states = stack([initial_states_p2, initial_states_p3])
    # slightly larger step between noise levels so that we cover also larger tbs
    observation_noise_σs = range(start = 0.002, step = 0.008, length = 5)
    probability_pruning_threshold = 0.04
    receding_horizon_benchmark(;
        demo_setup,
        initial_states,
        probability_pruning_threshold,
        observation_noise_σs,
        kwargs...,
    )
end

function receding_horizon_benchmark(;
    demo_setup,
    initial_states,
    observation_noise_σs,
    worker_pool_size = :default,
    kwargs...,
)
    set_up_workers(worker_pool_size)

    @invokelatest _receding_horizon_benchmark(
        demo_setup,
        initial_states;
        observation_noise_σs,
        kwargs...,
    )
end

function _receding_horizon_benchmark(
    demo_setup,
    initial_states;
    observation_noise_σs = 0.002:0.004:0.02,
    ground_truth_hypotheses = [1, 2],
    meta_seeds = 1:4,
    ground_truth_max_change_times = [0],
    intent_change_probabilities = [0.0],
    epsilon_p = 0.04,
    initial_weights = [0.5, 0.5],
    simulate_noise = true,
    probability_pruning_threshold = 0.01,
    stepable = false,
    kwargs...,
)
    @everywhere Main.__cached_demo_setup = $demo_setup

    # for each initial state we generate a different random seed for noise generation and intent changes
    # this avoids having to run all states with all seeds
    seeded_initial_states = mapreduce(vcat, meta_seeds) do meta_seed
        rng = Random.MersenneTwister(meta_seed)
        seeds = rand(rng, UInt, length(initial_states))
        map(initial_states, seeds) do initial_state, seed
            (; initial_state, seed)
        end
    end

    ProgressMeter.@showprogress pmap(
        Iterators.product(
            seeded_initial_states,
            observation_noise_σs,
            ground_truth_hypotheses,
            ground_truth_max_change_times,
            intent_change_probabilities,
        ),
    ) do (
        (; initial_state, seed),
        observation_noise_σ,
        ground_truth_hypothesis,
        ground_truth_max_change_time,
        intent_change_probability,
    )
        # generate a modified demo_setup:
        this_demo_setup =
            modified_initial_belief(Main.__cached_demo_setup; initial_state, initial_weights)
        run = benchmark_tb_estimators(
            this_demo_setup;
            initial_state,
            observation_noise_σ,
            ground_truth_hypothesis,
            ground_truth_max_change_time,
            intent_change_probability,
            seed,
            simulate_noise,
            probability_pruning_threshold,
            epsilon_p,
            kwargs...,
        )
        if stepable
            @info "Press enter to continue; any other key + enter to abort"
            isempty(readline()) || @assert false
        end
        (;
            run,
            initial_state,
            observation_noise_σ,
            ground_truth_hypothesis,
            ground_truth_max_change_time,
            intent_change_probability,
            epsilon_p,
            seed,
            simulate_noise,
            probability_pruning_threshold,
        )
    end
end

# Note: here, `epsilon_p` is the probability scale, not entropy scale
function epsilon_ablation(
    demo_setup,
    benchmark_function;
    epsilon_p_range = [get_probability_threshold_from_entropy(2.0^x) for x in -10:0],
    benchmark_kwargs = (;),
    save_path = "results/epsilon_ablation",
)
    save_path = "$save_path-$(now())"
    method_list = ["pessimistic_beliefpropagation_contingency"]

    if isdir(save_path)
        @warn "$save_path already exists and is not empty; aborting."
        return nothing
    end

    mkpath(save_path)

    for epsilon_p in epsilon_p_range
        println("==========================================")
        println("epsilon_p = $epsilon_p")
        benchmark = benchmark_function(demo_setup; epsilon_p, method_list, benchmark_kwargs...)
        JLD2.save_object("$save_path/$(epsilon_p).jld2", (; epsilon_p, benchmark))
    end

    @info "All done; results are in $save_path"

    nothing
end

function load_epsilon_ablation(save_path)
    ablation_study = Dictionary()

    for file in readdir(save_path)
        endswith(file, ".jld2") || error("Unexpected file $file in $save_path")
        (; epsilon_p, benchmark) = JLD2.load_object("$save_path/$file")
        insert!(ablation_study, epsilon_p, benchmark)
    end

    Dictionaries.sortkeys!(ablation_study)
end

function visualize_empirical_tb_distribution(
    benchmark;#
    filter_predicate = function (run)
        (run.method_name == "pessimistic_beliefpropagation_contingency")
    end,
)
    flattened_benchmark = flatten_receding_horizon_benchmark(benchmark)
    filtered_runs = filter(filter_predicate, flattened_benchmark)

    Makie.hist([run.empirical_tb for run in filtered_runs])
end

function visualize_epsilon_ablation(
    epsilon_ablation_results,
    baseline_benchmark;
    visualize_failure_counts = true,
    split_by_ground_truth = false,
)
    ground_truth_hypotheses = unique(run.ground_truth_hypothesis for run in baseline_benchmark)

    function _key(run)
        (; run.initial_state, run.ground_truth_hypothesis, run.observation_noise_σ, run.seed)
    end

    flattened_baseline_benchmark = flatten_receding_horizon_benchmark(baseline_benchmark)
    oracle_runs = filter(flattened_baseline_benchmark) do run
        run.method_name == "hindsight_contingency"
    end
    oracle_run_dict = map(only, SplitApplyCombine.group(_key, oracle_runs))

    groups = split_by_ground_truth ? ground_truth_hypotheses : [ground_truth_hypotheses]

    evaluated_benchmarks_per_ground_truth_hypothesis = map(Dictionaries.Indices(groups)) do group
        map(pairs(epsilon_ablation_results)) do (epsilon_p, benchmark)
            flattened_benchmark = flatten_receding_horizon_benchmark(benchmark)
            our_runs = filter(flattened_benchmark) do run
                run.method_name == "pessimistic_beliefpropagation_contingency" &&
                    run.ground_truth_hypothesis ∈ group
            end

            successful_raw_runs = filter(our_runs) do run
                oracle_run = oracle_run_dict[_key(run)]
                run.is_feasible && oracle_run.is_feasible
            end

            number_of_failures = length(our_runs) - length(successful_raw_runs)

            raw_relative_performance = map(successful_raw_runs) do run
                oracle_run = oracle_run_dict[_key(run)]
                run.cost - oracle_run.cost
            end
            mean_relative_performance = Statistics.mean(raw_relative_performance)
            stderr_relative_performance =
                Statistics.std(raw_relative_performance) / sqrt(length(raw_relative_performance))

            raw_tb_error = map(successful_raw_runs) do run
                run.mean_tb_error
            end
            mean_tb_error = Statistics.mean(raw_tb_error)
            stderr_tb_error = Statistics.std(raw_tb_error) / sqrt(length(raw_tb_error))

            raw_empirical_tb = map(our_runs) do run
                run.empirical_tb
            end
            mean_empirical_tb = Statistics.mean(raw_empirical_tb)
            std_empirical_tb = Statistics.std(raw_empirical_tb)

            (;
                epsilon_p,
                number_of_failures,
                mean_relative_performance,
                mean_tb_error,
                mean_empirical_tb,
                stderr_relative_performance,
                stderr_tb_error,
                std_empirical_tb,
                raw_relative_performance,
                raw_tb_error,
                raw_empirical_tb,
            )
        end
    end

    figure = Makie.Figure()

    ax_cost = Makie.Axis(
        figure[1, 1],
        ylabel = "Mean Relative Cost",
        xlabel = "entropy threshold, ϵ",
        xscale = log2,
        xticks = Makie.LogTicks(-10:0),
    )

    ax_tb = Makie.Axis(
        figure[2, 1],
        ylabel = "Mean branching time error",
        xlabel = "entropy threshold, ϵ",
        xscale = log2,
        xticks = Makie.LogTicks(-10:0),
    )

    if visualize_failure_counts
        ax_failures = Makie.Axis(
            figure[3, 1],
            ylabel = "Number of failures",
            xlabel = "entropy threshold, ϵ",
            xscale = log2,
            xticks = Makie.LogTicks(-10:0),
        )
    end

    for group in groups
        evaluated_benchmarks = evaluated_benchmarks_per_ground_truth_hypothesis[group]
        epsilon_hs = collect(keys(evaluated_benchmarks)) .|> entropy
        label = group isa Int ? "ground truth = $group" : ""

        Makie.lines!(
            ax_cost,
            epsilon_hs,
            [v.mean_relative_performance for v in evaluated_benchmarks];
            label,
        )
        Makie.scatter!(
            ax_cost,
            epsilon_hs,
            [v.mean_relative_performance for v in evaluated_benchmarks];
            label,
        )
        Makie.band!(
            ax_cost,
            epsilon_hs,
            [
                v.mean_relative_performance - v.stderr_relative_performance for
                v in evaluated_benchmarks
            ],
            [
                v.mean_relative_performance + v.stderr_relative_performance for
                v in evaluated_benchmarks
            ];
            alpha = 0.5,
            label,
        )

        Makie.lines!(ax_tb, epsilon_hs, [v.mean_tb_error for v in evaluated_benchmarks])
        Makie.scatter!(ax_tb, epsilon_hs, [v.mean_tb_error for v in evaluated_benchmarks])
        Makie.band!(
            ax_tb,
            epsilon_hs,
            [v.mean_tb_error - v.stderr_tb_error for v in evaluated_benchmarks],
            [v.mean_tb_error + v.stderr_tb_error for v in evaluated_benchmarks];
            alpha = 0.5,
        )

        if visualize_failure_counts
            Makie.lines!(
                ax_failures,
                epsilon_hs,
                [v.number_of_failures for v in evaluated_benchmarks],
            )
            Makie.scatter!(
                ax_failures,
                epsilon_hs,
                [v.number_of_failures for v in evaluated_benchmarks],
            )
        end
    end

    if split_by_ground_truth
        Makie.axislegend(ax_cost; merge = true)
    end

    figure
end

function merge_benchmark_results(bench1, bench2)
    map(bench1, bench2) do b1, b2
        @assert b1.observation_noise_σ == b2.observation_noise_σ
        @assert b1.seed == b2.seed
        @assert b1.initial_state == b2.initial_state

        (; b1..., run = merge(b1.run, b2.run))
    end
end

function flatten_receding_horizon_benchmark(receding_horizon_benchmark)
    mapreduce(vcat, receding_horizon_benchmark) do benchmark
        observation_noise_σ = benchmark.observation_noise_σ
        ground_truth_hypothesis = benchmark.ground_truth_hypothesis #get(benchmark, :ground_truth_hypothesis, 1)
        initial_state = benchmark.initial_state
        runs = benchmark.run
        map(pairs(runs)) do (method_name, run)
            is_feasible = run.run.feasible
            empirical_tb = run.tb
            cost = is_feasible ? cost = run.cost : cost = NaN
            if isempty(run.run.trace.branching_time_estimates)
                mean_tb_error = NaN
            else
                mean_tb_error = Statistics.mean(
                    enumerate(run.run.trace.branching_time_estimates[1:empirical_tb]),
                ) do (step, tb)
                    step <= empirical_tb ? abs(step + tb - empirical_tb - 1) : 0.0
                end
            end
            trace = run.run.trace

            # for backwards compatibility:
            seed = get(benchmark, :seed, 1)
            ground_truth_max_change_time = get(benchmark, :ground_truth_max_change_time, 0)
            intent_change_probability = get(benchmark, :intent_change_probability, 0.0)
            epsilon_p = get(benchmark, :epsilon_p, 0.04)
            simulate_noise = get(benchmark, :simulate_noise, false)
            probability_pruning_threshold =
                get(benchmark, :probability_pruning_threshold, missing)

            (;
                method_name,
                observation_noise_σ,
                ground_truth_hypothesis,
                ground_truth_max_change_time,
                intent_change_probability,
                epsilon_p,
                initial_state,
                cost,
                is_feasible,
                empirical_tb,
                mean_tb_error,
                trace,
                seed,
                simulate_noise,
                probability_pruning_threshold,
            )
        end |> collect
    end
end

function reproduce_failures(
    args...;
    filter_run_predicate = r -> !r.is_feasible,
    filter_group_predicate = g -> true,
    runs_priority = runs -> -length(runs),
    offscreen_rendering = false,
    verbose = true,
    stepable = true,
    kwargs...,
)
    reproduce(
        args...;
        filter_run_predicate,
        filter_group_predicate,
        runs_priority,
        offscreen_rendering,
        verbose,
        stepable,
        kwargs...,
    )
end

function reproduce(
    demo_setup,
    benchmark;
    filter_run_predicate,
    filter_group_predicate,
    runs_priority,
    offscreen_rendering = true,
    verbose = false,
    startindex = 1,
    overwrite_probability_pruning_threshold = nothing,
    stepable = false,
    kwargs...,
)
    flattened_results = flatten_receding_horizon_benchmark(benchmark)
    filtered_results = filter(filter_run_predicate, flattened_results)
    results_per_setting = SplitApplyCombine.group(filtered_results) do r
        initial_weights = [b.weight for b in r.trace.beliefs[begin]]
        probability_pruning_threshold =
            @something(overwrite_probability_pruning_threshold, r.probability_pruning_threshold)
        (;
            r.initial_state,
            initial_weights,
            r.ground_truth_hypothesis,
            r.ground_truth_max_change_time,
            r.intent_change_probability,
            r.epsilon_p,
            r.seed,
            r.observation_noise_σ,
            r.simulate_noise,
            probability_pruning_threshold,
        )
    end

    results_per_setting = filter(results_per_setting) do runs
        filter_group_predicate(runs)
    end

    # runs with the most failed methods first
    sorted_failure_cases = sort(results_per_setting; by = runs_priority)

    results = Dictionary()

    for (run_index, (setting, runs)) in
        zip(Iterators.countfrom(startindex), collect(pairs(sorted_failure_cases))[startindex:end])
        println("====================================")
        @show run_index

        # generate a modified demo_setup:
        this_demo_setup =
            modified_initial_belief(demo_setup; setting.initial_state, setting.initial_weights)
        run = benchmark_tb_estimators(
            this_demo_setup;
            setting.initial_state,
            setting.observation_noise_σ,
            setting.ground_truth_hypothesis,
            setting.ground_truth_max_change_time,
            setting.intent_change_probability,
            setting.epsilon_p,
            setting.seed,
            setting.simulate_noise,
            setting.probability_pruning_threshold,
            offscreen_rendering,
            method_list = [r.method_name for r in runs],
            verbose,
            kwargs...,
        )
        if stepable
            # hacky user interrupt
            @info "Press enter to continue; any other key + enter to abort"
            isempty(readline()) || break
        end
        insert!(results, setting, run)
    end

    results
end

function compare_benchmarks(benchmarks...; titles = Iterators.countfrom(), canvas = Makie.Figure())
    for (ii, benchmark, title) in zip(Iterators.countfrom(), benchmarks, titles)
        visualize_benchmark_overview(benchmark; title = title, canvas = canvas[1, ii])
    end
    canvas
end

function visualize_benchmark_overview(
    benchmark;
    title = "",
    canvas = Makie.Figure(),
    filter_state_predicate = (state) -> true,
)
    visualize_receding_horizon_benchmark(
        benchmark;
        title,
        canvas = canvas[1, :],
        filter_state_predicate,
    )
    visualize_solver_failure_stats(benchmark, canvas = canvas[2, :]; filter_state_predicate)
    canvas
end

function generate_all_crosswalk_receding_horizon_figures(;
    benchmark = JLD2.load_object(
        "results/ral2023/other/crosswalk_nonlinear_receding_horizon_benchmark-p01.jld2",
    ),
    qualitative_results = nothing,
)
    set_rss_theme!()
    CairoMakie.activate!()

    if !isnothing(benchmark)
        fig = visualize_quantitative_eval_banner(
            benchmark,
            canvas = Makie.Figure(; resolution = (700, 325), figure_padding = (0, 0, 0, 0)),
        )
        Makie.save("results/figures/crosswalk_receding_horizon_quantitative_eval_banner.pdf", fig)
    end

    if !isnothing(qualitative_results)
        fig_traj, fig_belief_snapshot = visualize_qualitative_crosswalk_receding_horizon_results(
            qualitative_results;
            canvas = Makie.Figure(; resolution = (700, 350), figure_padding = (2, 2, 2, 5)),
            run_index = 1,
            filter_method_predicate = method_name ->
                method_name ∈
                ("tb_0", "pessimistic_beliefpropagation_contingency", "plan_in_expectation"),
            return_belief_snapshot = true,
        )
        Makie.save(
            "results/figures/crosswalk_receding_horizon_qualitative_eval_banner.pdf",
            fig_traj,
        )
        Makie.save(
            "results/figures/crosswalk_receding_horizon_qualitative_eval_banner_belief_snapshot.pdf",
            fig_belief_snapshot,
        )
    end

    GLMakie.activate!()
end

function generate_all_overtaking_receding_horizon_figures(;
    benchmark = JLD2.load_object(
        "results/ral2023/overtaking_nonlinear_receding_horizon_results.jld2",
    ),
    qualitative_results = nothing,
)
    set_rss_theme!()
    CairoMakie.activate!()

    if !isnothing(benchmark)
        fig = visualize_quantitative_eval_banner(
            benchmark,
            canvas = Makie.Figure(; resolution = (700, 325), figure_padding = (0, 0, 0, 0)),
        )
        Makie.save("results/figures/overtaking_receding_horizon_quantitative_eval_banner.pdf", fig)
    end

    if !isnothing(qualitative_results)
        fig_trajs, fig_belief_snapshot = visualize_qualitative_overtaking_receding_horizon_results(
            qualitative_results,
            canvas = Makie.Figure(; resolution = (700, 300), figure_padding = (2, 10, 10, 10)),
            run_index = 12,
            filter_method_predicate = method_name ->
                method_name ∈
                ("tb_0", "pessimistic_beliefpropagation_contingency", "plan_in_expectation"),
            return_belief_snapshot = true,
        )
        Makie.save(
            "results/figures/overtaking_receding_horizon_qualitative_eval_banner.pdf",
            fig_trajs,
        )
        Makie.save(
            "results/figures/overtaking_receding_horizon_qualitative_eval_banner_belief_snapshot.pdf",
            fig_belief_snapshot,
        )
    end

    GLMakie.activate!()
end

function visualize_quantitative_eval_banner(
    benchmark;
    canvas = Makie.Figure(),
    show_branch_time_error = true,
)
    subplot_grid = canvas[1, 1] = Makie.GridLayout()

    failure_canvas = subplot_grid[1, 1]
    visualize_solver_failure_stats(benchmark; canvas = failure_canvas)
    failure_axis = Makie.current_axis(canvas)
    Makie.tightlimits!(failure_axis, Makie.Left(), Makie.Right())
    Makie.Label(failure_canvas[1, 1, Makie.Bottom()], "(a) Failure rate"; padding = (0, 0, 0, 55))

    performance_canvas = subplot_grid[1, 2]
    visualize_receding_horizon_benchmark(
        benchmark;
        canvas = performance_canvas,
        show_legend = false,
    )
    performance_axis = Makie.current_axis(canvas)
    Makie.tightlimits!(performance_axis, Makie.Left(), Makie.Right())
    Makie.Label(
        performance_canvas[1, 1, Makie.Bottom()],
        "(b) Closed-loop performance\nfor successful runs";
        padding = (0, 0, 0, 55),
    )

    if show_branch_time_error
        tb_canvas = subplot_grid[1, 3]
        visualize_tb_error_over_rationality(benchmark; canvas = tb_canvas)
        tb_axis = Makie.current_axis(canvas)
        Makie.tightlimits!(tb_axis, Makie.Left(), Makie.Right())
        Makie.Label(
            tb_canvas[1, 1, Makie.Bottom()],
            "(c) Branching time error";
            padding = (0, 0, 0, 55),
        )
    end

    Makie.colgap!(subplot_grid, 10)

    # generate custom shared legend
    canvas[1, 1, Makie.Top()] = Makie.axislegend(
        performance_axis,
        merge = true,
        orientation = :horizontal,
        tellwidth = false,
        framevisible = false,
        halign = :center,
    )

    canvas
end

function visualize_overtaking_state_region(
    benchmark,
    contingency_game;
    canvas = Makie.Figure(),
    nominal_initial_state = BlockArrays.mortar([
        [0.5, -4.0, 1.0, π / 2],
        [0.5, -2.9, 0.75, π / 2],
        [0.5, -1.0, 0.75, π / 2],
    ]),
    player_colors = get_player_colors(),
    player_markers = get_overtaking_player_markers(1),
    player_marker_sizes = get_overtaking_player_marker_sizes(),
)
    axis = TrajectoryGamesExamples.create_environment_axis(
        canvas,
        contingency_game.env;
        viz_kwargs = (; color = :white, strokewidth = 0),
        aspect = Makie.DataAspect(),
        xlabel = L"p_\text{lat}",
        ylabel = L"p_\text{lon}",
        ylabelpadding = 5, # hack to work around makie bug
        limits = ((-5, 0.0), (-0.75, 0.25)),
        yticks = [-0.75, 0.25],
        xaxisposition = :top,
    )
    visualize_overtaking_decorations(axis; swap_axes = true)

    initial_states = unique([r.initial_state for r in benchmark])

    # box for player 2
    let
        px_min, px_max = extrema([s[Block(2)][1] for s in initial_states])
        py_min, py_max = extrema([s[Block(2)][2] for s in initial_states])
        # swap to "rotate" plot as above
        px_min, py_min = py_min, -px_min
        px_max, py_max = py_max, -px_max
        Makie.poly!(
            axis,
            Makie.Rect(px_min, py_min, px_max - px_min, py_max - py_min),
            color = (:white, 0),
            strokewidth = 1,
            linestyle = :dash,
        )
    end
    # box for player 3
    let
        px_min, px_max = extrema([s[Block(3)][1] for s in initial_states])
        py_min, py_max = extrema([s[Block(3)][2] for s in initial_states])
        # swap to "rotate" plot as above
        px_min, py_min = py_min, -px_min
        px_max, py_max = py_max, -px_max

        Makie.poly!(
            axis,
            Makie.Rect(px_min, py_min, px_max - px_min, py_max - py_min),
            color = (:white, 0),
            strokewidth = 1,
            linestyle = :dash,
        )
    end
    # Plot nominal initial state for all players
    for (position, player_marker, player_color, player_marker_size) in zip(
        blocks(swapped_axes(nominal_initial_state)),
        player_markers,
        player_colors,
        player_marker_sizes,
    )
        Makie.scatter!(
            axis,
            [position[1]],
            [position[2]],
            marker = player_marker,
            markersize = player_marker_size,
            rotations = [position[4]],
            markerspace = :data,
            color = player_color,
        )
    end

    canvas
end

function visualize_overtaking_quantitative_eval_banner(
    contingency_game,
    benchmark;
    canvas = Makie.Figure(),
    show_branch_time_error = false,
)
    subplot_grid = canvas[1, 1] = Makie.GridLayout()
    let
        state_area_canvas = subplot_grid[1, 1:2]
        visualize_overtaking_state_region(benchmark, contingency_game; canvas = state_area_canvas)
        Makie.Label(
            state_area_canvas[1, 1, Makie.Bottom()],
            "(a) Initial state region";
            padding = (0, 0, 0, 10),
        )
    end

    let
        performance_canvas = subplot_grid[2, 1]
        visualize_receding_horizon_benchmark(benchmark; canvas = performance_canvas)
        Makie.Label(
            performance_canvas[1, 1, Makie.Bottom()],
            "(b) Closed-loop performance";
            padding = (0, 0, 0, 55),
        )
        tb_canvas = subplot_grid[2, 2]
        visualize_tb_over_rationality(benchmark; canvas = tb_canvas)
        Makie.Label(
            tb_canvas[1, 1, Makie.Bottom()],
            "(c) Hindsight branching time";
            padding = (0, 0, 0, 55),
        )
    end
    Makie.rowsize!(subplot_grid, 1, Makie.Auto(0.6))
    Makie.colsize!(subplot_grid, 1, Makie.Auto(1.9))
    Makie.rowgap!(subplot_grid, 2)
    Makie.colgap!(subplot_grid, 10)

    canvas
end

function visualize_crosswalk_quantitative_eval_banner(
    contingency_game,
    benchmark;
    canvas = Makie.Figure(),
    zoom_state_predicate = (state) -> abs(state[Block(2)][1]) <= 0.8,
    subcaption_padding = 50,
    player_colors = get_player_colors(),
    nominal_initial_states = [[0.0, -2.0, 0.0, π / 2], [0.0, 0.0, 0.0, 0.0]],
    zoom_color = get_player_colors()[2],
)
    # tb over rationality
    left_column_canvas = canvas[1:2, 1] = Makie.GridLayout()
    state_area_canvas = left_column_canvas[1, 1]
    let
        # visualize regions:
        axis = TrajectoryGamesExamples.create_environment_axis(
            state_area_canvas,
            contingency_game.env;
            viz_kwargs = (; color = :white, strokewidth = 0),
            aspect = Makie.DataAspect(),
            xlabel = L"p_\text{lat}",
            ylabel = L"p_\text{lon}",
            ylabelpadding = 5, # hack to work around makie bug
            limits = ((-1.2, 1.2), (-2.2, 0.6)),
            xticks = [-1, 1],
            xaxisposition = :top,
        )
        visualize_crosswalk_decorations(axis)
        Makie.Label(
            state_area_canvas[1, 1, Makie.Bottom()],
            "(a) Initial state regions";
            padding = (0, 0, 0, 10),
        )
        # player makers at nominal initial state
        player_markers = get_crosswalk_player_markers(1)
        marker_size_per_player = get_crosswalk_player_marker_sizes()
        for (position, player_marker, player_color, player_marker_size) in
            zip(nominal_initial_states, player_markers, player_colors, marker_size_per_player)
            Makie.scatter!(
                axis,
                [position[1]],
                [position[2]],
                marker = player_marker,
                markersize = player_marker_size,
                rotations = [position[4]],
                markerspace = :data,
                color = player_color,
            )
        end

        # show boxes of initial state regions
        initial_states = unique([r.initial_state for r in benchmark])
        let
            px_min, px_max = extrema([s[Block(2)][1] for s in initial_states])
            py_min, py_max = extrema([s[Block(2)][2] for s in initial_states])
            Makie.poly!(
                axis,
                Makie.Rect(px_min, py_min, px_max - px_min, py_max - py_min),
                color = (:white, 0),
                strokewidth = 1,
                linestyle = :dash,
            )
            Makie.text!(
                axis,
                px_min,
                py_max;
                text = "all",
                font = :italic,
                align = (:left, :top),
                offset = (10, -2),
            )
        end
        # zoomed
        let
            zoomed_initial_states = filter(zoom_state_predicate, initial_states)
            px_min, px_max = extrema([s[Block(2)][1] for s in zoomed_initial_states])
            py_min, py_max = extrema([s[Block(2)][2] for s in zoomed_initial_states])
            Makie.poly!(
                axis,
                Makie.Rect(px_min, py_min, px_max - px_min, py_max - py_min),
                color = (:white, 0),
                strokewidth = 1,
                strokecolor = zoom_color,
                linestyle = :dash,
            )
            Makie.text!(
                axis,
                px_min,
                py_max;
                text = "center",
                font = :italic,
                align = (:left, :top),
                offset = (5, -2),
                color = zoom_color,
            )
        end
    end

    tb_canvas = left_column_canvas[2, 1] = Makie.GridLayout()
    let
        visualize_tb_over_rationality(
            benchmark;
            canvas = tb_canvas[1, 1],
            #filter_state_predicate = !zoom_state_predicate,
            axis_kwargs = (; xlabelvisible = false),
        )
        tb_full_axis = Makie.current_axis(canvas)
        Makie.Label(tb_canvas[1, 1, Makie.Top()][1, 1, Makie.Bottom()], "all"; font = :italic)
        visualize_tb_over_rationality(
            benchmark;
            filter_state_predicate = zoom_state_predicate,
            canvas = tb_canvas[1, 2],
            axis_kwargs = (; xlabelvisible = false, ylabelvisible = false, titlecolor = zoom_color),
        )

        tb_zommed_axis = Makie.current_axis(canvas)
        Makie.Label(
            tb_canvas[1, 2, Makie.Top()][1, 1, Makie.Bottom()],
            "center";
            font = :italic,
            color = zoom_color,
        )
        Makie.hideydecorations!(tb_zommed_axis; grid = false)
        Makie.Label(
            tb_canvas[1, 1:2, Makie.Bottom()],
            "(c) Hindsight branching time";
            padding = (0, 0, 0, subcaption_padding),
        )
        Makie.Label(
            tb_canvas[1, 1:2, Makie.Bottom()],
            L"\text{human irrationality, } \sigma^2";
            padding = (0, 0, 0, 0.2 * subcaption_padding),
        )
        Makie.colgap!(tb_canvas, 5)
        Makie.rowsize!(left_column_canvas, 1, Makie.Auto(2))
        Makie.linkyaxes!(tb_full_axis, tb_zommed_axis)
    end

    # full evaluation
    visualize_receding_horizon_benchmark(
        benchmark;
        canvas = canvas[1, 2],
        show_legend = false,
        #filter_state_predicate = !zoom_state_predicate,
    )
    Makie.Label(canvas[1, 2, Makie.Top()][1, 1, Makie.Bottom()], "all"; font = :italic)
    axis_full = Makie.current_axis(canvas)
    Makie.tightlimits!(axis_full, Makie.Left(), Makie.Right())
    Makie.hidexdecorations!(axis_full; grid = false)
    visualize_receding_horizon_benchmark(
        benchmark;
        canvas = canvas[2, 2],
        filter_state_predicate = zoom_state_predicate,
        show_legend = false,
    )
    Makie.Label(
        canvas[2, 2, Makie.Top()][1, 1, Makie.Bottom()],
        "center";
        font = :italic,
        color = zoom_color,
    )
    Makie.Label(
        canvas[2, 2, Makie.Bottom()],
        "(b) Closed-loop performance";
        padding = (0, 0, 0, subcaption_padding),
    )
    axis_zoomed = Makie.current_axis(canvas)
    Makie.tightlimits!(axis_zoomed, Makie.Left(), Makie.Right())
    Makie.linkxaxes!(axis_full, axis_zoomed)

    # generate custom shared legend
    canvas[1, 2, Makie.Top()] = Makie.axislegend(
        axis_zoomed;
        merge = true,
        orientation = :horizontal,
        tellwidth = false,
        framevisible = false,
    )

    Makie.colsize!(canvas.layout, 1, Makie.Auto(0.65))

    canvas
end

function visualize_solver_failure_stats(
    benchmark;
    canvas = Makie.Figure(),
    filter_state_predicate = (state) -> true,
    use_relative_scale = true,
    style = :barplot,
)
    flattened = flatten_receding_horizon_benchmark(benchmark)

    grouped = SplitApplyCombine.group(flattened) do run
        (; method_name = run.method_name, run.observation_noise_σ)
    end

    original_counts = SplitApplyCombine.map(grouped) do group
        length(group)
    end

    filtered_counts = map(grouped) do group
        filter(group) do run
            !run.is_feasible && filter_state_predicate(run.initial_state)
        end |> length
    end

    group_keys = keys(filtered_counts)
    xs = [k.observation_noise_σ for k in group_keys]

    absolute_failure_counts = map(group_keys) do k
        get(filtered_counts, k, 0)::Int
    end

    relative_failure_counts = map(group_keys) do k
        (100absolute_failure_counts[k] / original_counts[k])::Float64
    end

    ys = use_relative_scale ? relative_failure_counts : absolute_failure_counts

    color = map(group_keys) do k
        get_method_color(k.method_name)
    end

    axis = Makie.Axis(
        canvas[1, 1],
        xticks = unique(sort(xs)),
        xlabel = L"\text{human irrationality, } \sigma^2",
        ylabel = use_relative_scale ? "failure rate [%]" : "failure count",
        xtickformat = (xs) -> map(xs) do x
            string(round(x; digits = 3))
        end,
    )

    if style === :barplot
        dodge = [
            findfirst(==(get_streamlined_name(k.method_name)), get_streamlined_name.(METHOD_NAMES)) for k in group_keys
        ]
        Makie.barplot!(axis, collect(xs), collect(ys); dodge, color = collect(color))
    elseif style === :lines
        # plot one lines for each method
        keys_per_method = SplitApplyCombine.group(group_keys) do key
            key.method_name
        end

        for method_keys in keys_per_method
            sorted_method_keys = sort(method_keys; by = k -> k.observation_noise_σ)
            xs = [k.observation_noise_σ for k in sorted_method_keys]
            ys = [relative_failure_counts[k] for k in sorted_method_keys]
            Makie.lines!(axis, xs, ys; color = get_method_color(method_keys[begin].method_name))
        end
    else
        error("unknown plot style")
    end

    #Makie.DataInspector(axis)

    canvas
end

# plot empirical branching time as a function of initial state
function visualize_tb_distribution(receding_horizon_benchmark)
    flattened_results_table = flatten_receding_horizon_benchmark(receding_horizon_benchmark)

    reference_data_per_group = let
        reference_method_name = "hindsight_contingency"
        filtered =
            filter(run -> run.method_name == reference_method_name, flattened_results_table)
        SplitApplyCombine.group(filtered) do r
            (; r.observation_noise_σ, r.ground_truth_hypothesis)
        end
    end

    figure = Makie.Figure()
    levels = [1:30;]
    local last_plot
    group_keys = keys(reference_data_per_group)
    for (ii, group_key) in enumerate(group_keys)
        data = reference_data_per_group[group_key]
        axis, plot = Makie.contourf(
            figure[ii, 1],
            [d.initial_state[5] for d in data],
            [d.initial_state[6] for d in data],
            [d.empirical_tb for d in data];
            levels,
            axis = (; aspect = Makie.DataAspect(), title = "$group_key"),
        )
        last_plot = plot
    end
    Makie.Colorbar(figure[1:length(group_keys), 2], last_plot)

    figure
end

function visualize_tb_error_over_rationality(
    benchmark;
    canvas = Makie.Figure(),
    axis_kwargs = (;),
    method_names = [
        "tb_0",
        "plan_in_expectation",
        "pessimistic_beliefpropagation_contingency",
        "tb_1",
        "hindsight_contingency",
    ],
)
    successful_runs = filter(flatten_receding_horizon_benchmark(benchmark)) do run
        run.is_feasible
    end

    grouped_data = SplitApplyCombine.group(successful_runs) do run
        run.method_name
    end

    results_per_group = map(grouped_data) do group
        result_per_rationality = SplitApplyCombine.group(group) do run
            run.observation_noise_σ
        end |> Dictionaries.sortkeys

        map(result_per_rationality) do runs
            [run.mean_tb_error for run in runs]
        end
    end

    axis = Makie.Axis(
        canvas[1, 1];
        xlabel = L"\text{human irrationality, } \sigma^2",
        ylabel = L"\text{branching time error}",
        xtickformat = (xs) -> map(xs) do x
            string(round(x; digits = 3))
        end,
        axis_kwargs...,
    )

    for method_name in method_names
        xs = collect(keys(results_per_group[method_name]))
        ys = map(Statistics.mean, collect(results_per_group[method_name]))
        errs = map(collect(results_per_group[method_name])) do ys
            Statistics.std(ys) / sqrt(length(ys))
        end

        hi = ys .+ errs
        lo = ys .- errs

        Makie.lines!(axis, xs, ys; color = get_method_color(method_name))
        Makie.band!(axis, xs, lo, hi; color = get_method_color(method_name), alpha = 0.5)
    end

    canvas
end

function visualize_tb_over_rationality(
    benchmark;
    filter_state_predicate = state -> true,
    style = :quantiles,
    reference_method_name = "hindsight_contingency",
    color = get_method_color(:oracle),
    canvas = Makie.Figure(),
    axis_kwargs = (;),
)
    reference_data = filter(flatten_receding_horizon_benchmark(benchmark)) do run
        run.method_name == reference_method_name && filter_state_predicate(run.initial_state)
    end

    axis = Makie.Axis(
        canvas[1, 1];
        xlabel = L"\text{human irrationality, } \sigma^2",
        ylabel = L"\text{Branching time, }t_b",
        xtickformat = (xs) -> map(xs) do x
            string(round(x; digits = 3))
        end,
        axis_kwargs...,
    )

    if style === :mean_stderr
        reference_data_per_noise_level = let
            SplitApplyCombine.group(r -> r.observation_noise_σ, reference_data)
        end
        noise_levels = collect(keys(reference_data_per_noise_level))
        branching_time_stats = map(noise_levels) do σ
            data = reference_data_per_noise_level[σ]
            mean = Statistics.mean([d.empirical_tb for d in data])
            std_err = Statistics.std([d.empirical_tb for d in data]) / sqrt(length(data))
            (; σ, mean, std_err)
        end

        Makie.errorbars!(
            axis,
            [d.σ for d in branching_time_stats],
            [d.mean for d in branching_time_stats],
            [d.std_err for d in branching_time_stats];
            whiskerwidth = 10,
        )
        Makie.lines!(
            axis,
            [d.σ for d in branching_time_stats],
            [d.mean for d in branching_time_stats];
            color,
        )
        Makie.scatter!(
            axis,
            [d.σ for d in branching_time_stats],
            [d.mean for d in branching_time_stats];
            color,
        )
    elseif style === :quantiles
        noise_levels = [d.observation_noise_σ for d in reference_data]
        branching_times = [d.empirical_tb for d in reference_data]
        Makie.boxplot!(
            axis,
            noise_levels,
            branching_times;
            width = 0.001,
            color = (color, 0.5),
            whiskerwidth = :match,
            show_outliers = false,
        )
    end

    canvas
end

function cost_gap(runs)
    ours_run = only(filter(runs) do run
        #run.method_name === "hindsight_contingency"
        run.method_name === "pessimistic_beliefpropagation_contingency"
    end)
    baseline_run = only(filter(runs) do run
        run.method_name === "plan_in_expectation"
    end)

    baseline_run.cost - ours_run.cost
end

"""
Save jld2 with qualitative results and simultaneously generate corresponding videos
"""
function save_crosswalk_qualitative_receding_horizon_results(
    demo_setup,
    benchmark;
    filename = "results/ral2023/other/crosswalk_qualitative/crosswalk_nonlinear_qualitative_receding_horizon_results.jld2",
    kwargs...,
)
    save_qualitative_receding_horizon_results(
        filename,
        demo_setup,
        benchmark;
        video_save_prefix = "crosswalk",
        reproducer_kwargs = (;
            axis_kwargs = (; limits = ((-1.3, 1.3), (-2.5, 2.5))),
            hypothesis_labels = ["right", "left"],
            simulation_horizon = 40,
        ),
        kwargs...,
    )
end

"""
Save jld2 with qualitative results and simultaneously generate corresponding videos
"""
function save_overtaking_qualitative_receding_horizon_results(
    demo_setup,
    benchmark;
    filename = "results/ral2023/overtaking_qualitative/overtaking_nonlinear_qualitative_receding_horizon_results.jld2",
)
    qualitative_results = save_qualitative_receding_horizon_results(
        filename,
        demo_setup,
        benchmark;
        video_save_prefix = "overtaking",
        reproducer_kwargs = (;
            axis_kwargs = (; limits = ((-0.25, 0.75), (-4.15, 3.0))),
            hypothesis_labels = ["merge", "stay"],
            resolution = (400, 1200),
            simulation_horizon = 40,
        ),
    )

    # generate cost videos
    for (ii_rank, reproduced_trajectories) in
        pairs(qualitative_results.top_k_reproduced_trajectories)
        for reproduced_trajectory in reproduced_trajectories
            file_name = "$(dirname(filename))/overtaking_cost_$(ii_rank)-$(get_streamlined_name(reproduced_trajectory.method_name)).gif"
            animate_stage_costs(
                file_name,
                reproduced_trajectory.closed_loop_trajectory.stage_costs;
                visualization_kwargs = (;
                    axis_kwargs = (; limits = ((1, 41), (0, 0.51))),
                    empirical_tb = reproduced_trajectory.empirical_tb,
                    color = get_method_color(reproduced_trajectory.method_name),
                ),
            )
        end
    end
end

function animate_stage_costs(
    file_name,
    stage_costs;
    framerate = 5,
    resolution = (500, 200),
    visualization_kwargs = (;),
)
    figure = Makie.Figure(; resolution)

    Makie.record(figure, file_name; framerate) do io
        stage_costs_up_to_now = Makie.Observable([c[begin] for c in stage_costs])
        visualize_stage_costs(stage_costs_up_to_now; canvas = figure, visualization_kwargs...)

        for time in eachindex(stage_costs)
            stage_costs_up_to_now[] = [c[begin] for c in stage_costs[begin:time]]
            Makie.recordframe!(io)
        end
    end
end

"""
Set the `video_save_prefix` to a path to save an animation of the trajectory
"""
function extract_trajectory_examples(
    demo_setup,
    benchmark;
    reproduce = true,
    reproducer_kwargs = (;),
    k_top = 1:1,
    filter_group_predicate = function (group)
        # focus on safety gap between the approaches: find runs where tb_0 fails while the `pessimistic_beliefpropagation_contingency` and `plan_in_expectation` succeed
        all(group) do run
            if run.method_name == "tb_0"
                !run.is_feasible
            elseif run.method_name ∈
                   ["pessimistic_beliefpropagation_contingency", "plan_in_expectation"]
                run.is_feasible
            else
                true
            end
        end
    end,
    filter_hypothesis_predicate = Returns(true), # ==(2),
    video_save_prefix = nothing,
    video_save_directory = nothing,
    overwrite_probability_pruning_threshold = nothing,
)
    flattened_results = flatten_receding_horizon_benchmark(benchmark)
    filtered = filter(flattened_results) do run
        filter_hypothesis_predicate(run.ground_truth_hypothesis)
    end

    settings = SplitApplyCombine.group(filtered) do r
        initial_weights = [b.weight for b in r.trace.beliefs[begin]]
        probability_pruning_threshold =
            @something(overwrite_probability_pruning_threshold, r.probability_pruning_threshold)
        (;
            r.initial_state,
            initial_weights,
            r.ground_truth_hypothesis,
            r.ground_truth_max_change_time,
            r.intent_change_probability,
            r.epsilon_p,
            r.seed,
            r.observation_noise_σ,
            r.simulate_noise,
            probability_pruning_threshold,
        )
    end

    settings = filter(filter_group_predicate, settings)

    # find the runs with the largest gap
    sorted_settings = sort(map(cost_gap, settings); rev = true) |> pairs |> collect

    map(Dictionaries.Indices(k_top)) do setting_rank
        (setting, expected_gap) = sorted_settings[setting_rank]
        if !reproduce
            return map(settings[setting]) do run
                (;
                    closed_loop_trajectory = (;
                        xs = run.trace.states,
                        us = run.trace.controls,
                        run.trace.stage_costs,
                    ),
                    open_loop_strategies = nothing,
                    run.method_name,
                    run.empirical_tb,
                    run.cost,
                    expected_gap,
                )
            end
        end

        # re-run but this time record the open loop strategies (takes too much space to safe them all on the
        # full benchmark)
        reproduced_runs = let
            this_demo_setup = modified_initial_belief(
                demo_setup;
                setting.initial_state,
                setting.initial_weights,
            )
            reproduced_run = benchmark_tb_estimators(
                this_demo_setup;
                setting.initial_state,
                setting.observation_noise_σ,
                setting.ground_truth_hypothesis,
                setting.ground_truth_max_change_time,
                setting.intent_change_probability,
                setting.epsilon_p,
                setting.seed,
                setting.simulate_noise,
                setting.probability_pruning_threshold,
                record_open_loop_strategies = true,
                video_save_path = !isnothing(video_save_directory) &&
                                  !isnothing(video_save_prefix) ?
                                  "$video_save_directory/$video_save_prefix-$setting_rank.gif" :
                                  nothing,
                show_buttons = false,
                reproducer_kwargs...,
            )
            flatten_receding_horizon_benchmark([(; setting..., run = reproduced_run)])
        end

        reproduced_trajectories = map(reproduced_runs) do reproduced_run
            closed_loop_trajectory = (;
                xs = reproduced_run.trace.states,
                us = reproduced_run.trace.controls,
                reproduced_run.trace.stage_costs,
                reproduced_run.trace.observations,
                reproduced_run.trace.beliefs,
            )
            (;
                closed_loop_trajectory,
                reproduced_run.trace.open_loop_strategies,
                reproduced_run.method_name,
                reproduced_run.empirical_tb,
                reproduced_run.cost,
                expected_gap,
            )
        end

        reproduced_trajectories
    end
end

function save_qualitative_receding_horizon_results(file_name, demo_setup, benchmark; kwargs...)
    qualitative_results = (;
        demo_setup.contingency_game,
        top_k_reproduced_trajectories = extract_trajectory_examples(
            demo_setup,
            benchmark;
            video_save_directory = dirname(file_name),
            kwargs...,
        ),
    )
    JLD2.save_object(file_name, qualitative_results)
    qualitative_results
end

function visualize_qualitative_crosswalk_receding_horizon_results(
    qualitative_results = JLD2.load_object(
        "results/crosswalk_nonlinear_qualitative_receding_horizon_results.jld2",
    );
    canvas = Makie.Figure(),
    axis_kwargs = (; limits = ((-1.2, 1.2), (-2.2, 1.8)), xticks = [-1, 1]),
    run_index = 3,
    kwargs...,
)
    visualize_qualitative_receding_horizon_results(
        qualitative_results,
        visualize_crosswalk_decorations,
        get_crosswalk_player_markers(2),
        get_crosswalk_player_marker_sizes();
        canvas,
        run_index,
        axis_kwargs,
        kwargs...,
    )
end

function visualize_qualitative_overtaking_receding_horizon_results(
    qualitative_results = JLD2.load_object(
        "results/overtaking_nonlinear_qualitative_receding_horizon_results.jld2",
    );
    canvas = Makie.Figure(),
    axis_kwargs = (; limits = ((-4.15, 2.2), (-0.75, 0.25)), yticks = [-0.75, 0.25]),
    kwargs...,
)
    # swap axes for more compact plotting
    qualitative_results = (;
        contingency_game = qualitative_results.contingency_game,
        top_k_reproduced_trajectories = map(
            qualitative_results.top_k_reproduced_trajectories,
        ) do reproduced_trajectories
            map(reproduced_trajectories) do reproduced_trajectory
                (;
                    closed_loop_trajectory = swapped_axes(
                        reproduced_trajectory.closed_loop_trajectory,
                    ),
                    open_loop_strategies = swapped_axes.(
                        reproduced_trajectory.open_loop_strategies
                    ),
                    reproduced_trajectory.method_name,
                    reproduced_trajectory.empirical_tb,
                )
            end
        end,
    )

    visualize_qualitative_receding_horizon_results(
        qualitative_results,
        (axis) -> visualize_overtaking_decorations(axis; swap_axes = true),
        get_overtaking_player_markers(1),
        get_overtaking_player_marker_sizes();
        show_control_profiles = true,
        canvas,
        axis_kwargs,
        swap_axes = true,
        kwargs...,
    )
end

function visualize_stage_costs(closed_loop_trajectory::NamedTuple; kwargs...)
    stage_costs = [c[begin] for c in closed_loop_trajectory.stage_costs]
    visualize_stage_costs(Makie.Observable(stage_costs); kwargs...)
end

function visualize_stage_costs(
    stage_costs;
    canvas = Makie.Figure(),
    color = :black,
    empirical_tb = nothing,
    axis_kwargs = (;),
)
    # ensure everything is a Makie.Observable
    if !(stage_costs isa Makie.Observable)
        stage_costs = Makie.Observable(stage_costs)
    end
    if !(empirical_tb isa Makie.Observable)
        empirical_tb = Makie.Observable(empirical_tb)
    end

    axis = Makie.Axis(
        canvas[1, 1];
        ylabel = "current cost",
        xlabel = "time step",
        subtitlevisible = true,
        #yautolimitmargin = (0.1, 0.2),
        axis_kwargs...,
    )
    Makie.lines!(axis, stage_costs; color)

    time_of_certainty =
        Makie.@lift isnothing($empirical_tb) ? length($stage_costs) + 1 : $empirical_tb + 1
    show_tb_marker = Makie.@lift($time_of_certainty <= length($stage_costs))
    Makie.vlines!(
        axis,
        time_of_certainty,
        color = get_method_color(:oracle),
        linestyle = :dash,
        label = "empirical tb",
        visible = show_tb_marker,
    )

    highlighted_cost = Makie.@lift(
        $time_of_certainty <= length($stage_costs) ? $stage_costs[$time_of_certainty] : NaN
    )
    Makie.scatter!(
        axis,
        time_of_certainty,
        highlighted_cost,
        color = :white,
        marker = :star4,
        markersize = 20,
        strokecolor = get_method_color(:oracle),#player_colors[begin],
        strokewidth = 1.5,
        visible = show_tb_marker,
    )
    if !haskey(axis_kwargs, :limits)
        Makie.tightlimits!(axis, Makie.Left(), Makie.Right())
    end
    Makie.hidespines!(axis, :t, :r)

    canvas
end

function visualize_qualitative_receding_horizon_results(
    qualitative_results,
    visualize_decorations,
    player_markers,
    player_marker_sizes;
    open_loop_viz_time = 5,
    canvas = Makie.Figure(),
    axis_kwargs = (;),
    swap_axes = false,
    show_control_profiles = false,
    run_index = 1,
    filter_method_predicate = (method_name) -> true,
    return_belief_snapshot = false,
    horizon = 30,
)
    (; contingency_game, top_k_reproduced_trajectories) = qualitative_results
    reproduced_trajectories = top_k_reproduced_trajectories[run_index]

    # order by nominal tb to match our front figure
    reproduced_trajectories = sort(
        reproduced_trajectories;
        by = function (reproduced_trajectory)
            if reproduced_trajectory.method_name == "tb_0"
                return 1
            elseif reproduced_trajectory.method_name == "tb_1"
                return 2
            elseif reproduced_trajectory.method_name == "pessimistic_beliefpropagation_contingency"
                return 3
            elseif reproduced_trajectory.method_name == "hindsight_contingency"
                return 4
            elseif reproduced_trajectory.method_name == "plan_in_expectation"
                return 5
            end
            error("unknown method name")
        end,
    )

    reproduced_trajectories =
        filter(r -> filter_method_predicate(r.method_name), reproduced_trajectories)

    subplot_grid = canvas[1, 1] = Makie.GridLayout()
    control_axes = []

    for (ii, reproduced_trajectory) in enumerate(reproduced_trajectories)
        closed_loop_trajectory = reproduced_trajectory.closed_loop_trajectory
        if !isnothing(reproduced_trajectory.open_loop_strategies)
            open_loop_strategy = reproduced_trajectory.open_loop_strategies[open_loop_viz_time]
        else
            open_loop_strategy = nothing
        end
        empirical_tb = reproduced_trajectory.empirical_tb
        method_name = reproduced_trajectory.method_name

        # truncate closed-loop trajectory to specified horizon
        if !isnothing(horizon)
            h = min(horizon, length(closed_loop_trajectory.xs))
            closed_loop_trajectory = (;
                xs = closed_loop_trajectory.xs[begin:h],
                us = closed_loop_trajectory.us[begin:h],
                stage_costs = closed_loop_trajectory.stage_costs[begin:h],
            )
        end

        if swap_axes
            subcanvas = subplot_grid[ii, 1] = Makie.GridLayout()
        else
            subcanvas = subplot_grid[1, ii] = Makie.GridLayout()
        end

        # trajectory in position space
        visualize_receding_horizon_trajectory(
            closed_loop_trajectory,
            open_loop_strategy,
            contingency_game,
            visualize_decorations,
            player_markers,
            player_marker_sizes;
            highlight_time = empirical_tb + 1,
            canvas = subcanvas[1, 1],
            axis_kwargs,
            swap_axes,
        )

        # figure label
        let
            if swap_axes
                label_grid_position = subcanvas[1, 1, Makie.Left()]
                label_rotation = π / 2
            else
                label_grid_position = subcanvas[1, 1, Makie.Top()]
                label_rotation = 0.0
            end

            Makie.Label(
                label_grid_position,
                get_streamlined_name(method_name; multiline = swap_axes),
                color = get_method_color(method_name),
                font = :bold,
                lineheight = 0.8,
                rotation = label_rotation,
            )
        end

        trajectory_axis = Makie.current_axis(canvas)

        # inputs over time
        if show_control_profiles
            visualize_stage_costs(
                closed_loop_trajectory;
                canvas = subcanvas[1, 2],
                color = get_method_color(method_name),
                empirical_tb,
            )
            control_axis = Makie.current_axis(canvas)
            push!(control_axes, control_axis)
        end

        Makie.hidedecorations!(trajectory_axis)
        let
            ((xmin, xmax), (ymin, ymax)) = axis_kwargs.limits
            aspect_ratio = (xmax - xmin) / (ymax - ymin)
            # fix the aspect ratio of the first column so that the other only fills the remaining
            # space
            Makie.colsize!(subcanvas, 1, Makie.Aspect(1, aspect_ratio))
        end
    end

    if show_control_profiles
        Makie.linkaxes!(control_axes...)
        # hide all bu but the last x axis decorations
        for control_axis in control_axes[1:(end - 1)]
            Makie.hidexdecorations!(control_axis)
        end
    end

    Makie.rowgap!(subplot_grid, 30)
    Makie.colgap!(subplot_grid, 0)

    if return_belief_snapshot
        belief_snapshot =
            reproduced_trajectories[begin].closed_loop_trajectory.beliefs[open_loop_viz_time]
        belief_snapshot_canvas = visualize_belief(belief_snapshot)

        return (; traj_canvas = canvas, belief_snapshot_canvas)
    end

    canvas
end

function visualize_belief(
    belief;
    canvas = Makie.Figure(; resolution = (150, 150), figure_padding = (0, 0, 0, 5)),
    branch_colors = get_branch_colors(),
)
    Makie.barplot(
        canvas[1, 1],
        [b.weight for b in belief];
        color = branch_colors,
        axis = (; limits = (nothing, (0, 1))),
    )
    canvas
end

# yikes...; hack for more compact plotting: swap axes to have the highway horizontal
function swapped_axes(strategy::JointStrategy)
    JointStrategy([swapped_axes(substrategy) for substrategy in strategy.substrategies])
end

function swapped_axes(strategy::ContingencyStrategy)
    ContingencyStrategy((;
        strategy.fields...,
        branch_strategies = map(swapped_axes, strategy.branch_strategies),
    ))
end

function swapped_axes(strategy::OpenLoopStrategy)
    swapped_xs = map(strategy.xs) do x
        swapped_axes(x)
    end

    OpenLoopStrategy(swapped_xs, strategy.us)
end

function swapped_axes(x::Vector)
    new_x = copy(x)
    new_x[1] = x[2]
    new_x[2] = -x[1]
    new_x[4] = (x[4] - π / 2)
    new_x
end

function swapped_axes(x::BlockVector)
    mortar([swapped_axes(xi) for xi in blocks(x)])
end

function swapped_axes(trajectory::NamedTuple)
    swapped_xs = map(trajectory.xs) do x
        swapped_axes(x)
    end

    (; trajectory..., xs = swapped_xs)
end

# end hack...

function visualize_receding_horizon_trajectory(
    closed_loop_trajectory,
    open_loop_strategy,
    contingency_game,
    visualize_decorations,
    player_markers,
    marker_size_per_player;
    title = "",
    player_colors = get_player_colors(),
    axis_kwargs = (;),
    highlight_time = nothing,
    canvas = Makie.Figure(),
    swap_axes = false,
    show_observations = false,
)
    if swap_axes
        xlabel = L"p_\text{lon}"
        ylabel = L"p_\text{lat}"
    else
        xlabel = L"p_\text{lat}"
        ylabel = L"p_\text{lon}"
    end

    axis = TrajectoryGamesExamples.create_environment_axis(
        canvas[1, 1],
        contingency_game.env;
        viz_kwargs = (; color = :white, strokewidth = 0),
        aspect = Makie.DataAspect(),
        xlabel,
        ylabel,
        ylabelpadding = 5, # hack to work around Makie padding bug
        title = title,
        axis_kwargs...,
    )

    visualize_decorations(axis)

    # visualize closed-loop trace for each player
    visualize_joint_trajectory_closed_loop!(axis, closed_loop_trajectory; player_colors)

    if show_observations
        num_players = TrajectoryGamesBase.num_players(contingency_game)
        for player_index in 2:num_players
            os = [
                Makie.Point2f(o[Block(player_index - 1)][1:2]) for
                o in closed_loop_trajectory.observations
            ]
            Makie.scatter!(axis, os; alpha = 0.1, color = :black)
        end
    end

    # visualize the initial open-loop strategy
    if !isnothing(open_loop_strategy)
        ego_strategy = open_loop_strategy.substrategies[begin]
        Makie.plot!(
            axis,
            ego_strategy;
            trunk_color = :gray,
            branch_colors = [:gray, :gray],
            show_only_one_trunk_branch = true,
        )
    end

    if !isnothing(highlight_time) && length(closed_loop_trajectory.xs) >= highlight_time
        # highlight the branching state
        branching_state = closed_loop_trajectory.xs[highlight_time]
        Makie.scatter!(
            axis,
            Makie.Point2f(branching_state[1:2]),
            color = :white,
            marker = :star4,
            markersize = 20,
            strokecolor = get_method_color(:oracle),#player_colors[begin],
            strokewidth = 1.5,
        )
    end

    states_to_visualize = [(; state = closed_loop_trajectory.xs[end], alpha = 1.0)]
    if !isnothing(open_loop_strategy)
        pushfirst!(
            states_to_visualize,
            (;
                state = mortar([
                    s.branch_strategies[begin].xs[begin] for s in open_loop_strategy.substrategies
                ]),
                alpha = 0.5,
            ),
        )
    end

    for (; state, alpha) in states_to_visualize
        dimmed_player_markers = map(player_markers) do player_marker
            map(player_marker) do pixel
                Makie.RGBA(pixel.r, pixel.g, pixel.b, pixel.alpha * alpha)
            end
        end

        # show final position of each player
        for (substate, player_marker, player_color, player_marker_size) in
            zip(blocks(state), dimmed_player_markers, player_colors, marker_size_per_player)
            Makie.scatter!(
                axis,
                [substate[1]],
                [substate[2]],
                marker = player_marker,
                markersize = player_marker_size,
                rotations = [substate[4]],
                markerspace = :data,
            )
        end
    end

    canvas
end

function visualize_receding_horizon_benchmark(
    benchmark;
    show_raw_data = false,
    subtract_oracle_performance = false,
    filter_state_predicate = state -> true,
    title = "",
    canvas = Makie.Figure(),
    axis_kwargs = (;),
    legend_kwargs = (; orientation = :horizontal, halign = :center, framevisible = false),
    show_legend = true,
    require_all_feasible = false,
)
    flattened_results_table = filter(
        r -> filter_state_predicate(r.initial_state),
        flatten_receding_horizon_benchmark(benchmark),
    )

    grouped_results = SplitApplyCombine.group(
        p -> (; p.initial_state, p.observation_noise_σ, p.ground_truth_hypothesis, p.seed),
        flattened_results_table,
    )

    if require_all_feasible
        # we only compare runs where all methods are feasible
        grouped_results = filter(group -> all(r -> r.is_feasible, group), grouped_results)
    end

    canvas_grid = canvas[1, 1] = Makie.GridLayout()
    axis = Makie.Axis(
        canvas_grid[1, 1];
        xlabel = L"\text{human irrationality, } \sigma^2",
        ylabel = L"\text{cost}",
        xtickformat = (xs) -> map(xs) do x
            string(round(x; digits = 3))
        end,
        title = string(title),
        axis_kwargs...,
    )

    method_names = [
        "tb_0",
        "plan_in_expectation",
        "constvel",
        "pessimistic_beliefpropagation_contingency",
        "tb_1",
        "hindsight_contingency",
    ]

    if subtract_oracle_performance
        reference_method_name = "hindsight_contingency"
        reference_data = map(grouped_results) do runs
            only(filter(run -> run.method_name == reference_method_name, runs))
        end
        methods_to_show = filter(n -> n != reference_method_name, method_names)
    else
        methods_to_show = method_names
    end

    for method_name in methods_to_show
        method_data = map(grouped_results) do runs
            this_run = only(filter(run -> run.method_name == method_name, runs))
            reference_cost =
                subtract_oracle_performance ?
                reference_data[(; this_run.initial_state, this_run.observation_noise_σ)].cost :
                0.0
            # subtract the reference performance
            (; this_run.observation_noise_σ, cost = (this_run.cost - reference_cost))
        end
        # Note: somehow `filter!` does not work here because dictionaries share the key set
        method_data = filter(p -> !isnan(p.cost), method_data)

        summary_statistics = let
            grouped_data = SplitApplyCombine.group(d -> d.observation_noise_σ, method_data)
            map(pairs(grouped_data)) do (σ, data)
                mid = Statistics.mean([d.cost for d in data])
                std = Statistics.std([d.cost for d in data])
                std_err = std / sqrt(length(data))
                lo = mid - std_err
                hi = mid + std_err
                (; σ, lo, mid, hi)
            end
        end

        Makie.band!(
            axis,
            [p.σ for p in summary_statistics],
            [p.lo for p in summary_statistics],
            [p.hi for p in summary_statistics];
            label = get_streamlined_name(method_name; multiline = true),
            color = (get_method_color(method_name), 0.5),
        )
        Makie.lines!(
            axis,
            [p.σ for p in summary_statistics],
            [p.mid for p in summary_statistics];
            label = get_streamlined_name(method_name; multiline = true),
            color = get_method_color(method_name),
        )

        if show_raw_data
            Makie.scatter!(
                axis,
                [Makie.Point2f(d.observation_noise_σ, d.cost) for d in method_data];
                label = string(method_name),
            )
        end
    end

    if show_legend
        canvas_grid[1, 1, Makie.Top()] = Makie.axislegend(axis; merge = true, legend_kwargs...)
    end
    Makie.rowgap!(canvas_grid, 0)

    canvas
end

function benchmark_tb_estimators(
    demo_setup;
    planning_horizon = demo_setup.contingency_game.horizon,
    verbose = false,
    method_list = [
        "plan_in_expectation",
        "tb_0",
        "tb_1",
        "pessimistic_beliefpropagation_contingency",
        "hindsight_contingency",
        "constvel",
    ],
    simulation_horizon = 30,
    offscreen_rendering = true,
    epsilon_p = 0.04,
    video_save_path = nothing,
    skip_remaining_runs_on_user_interrupt = true,
    probability_pruning_threshold = 0.01,
    kwargs...,
)
    function benchmark_tb_estimator(
        tb_estimator;
        annotation = nothing,
        use_constant_velocity_ego_model = false,
    )
        if !isnothing(annotation) && !isnothing(video_save_path)
            file_name, file_extension = splitext(video_save_path)
            this_video_save_path = "$(file_name)-$(annotation)$(file_extension)"
        else
            this_video_save_path = video_save_path
        end

        run = simulate_receding_horizon_cost(
            demo_setup,
            tb_estimator;
            planning_horizon,
            verbose,
            simulation_horizon,
            offscreen_rendering,
            video_save_path = this_video_save_path,
            probability_pruning_threshold,
            use_constant_velocity_ego_model,
            kwargs...,
        )
        cost = compute_closed_loop_cost(run.trace)
        tb = get_empirical_branching_time(run.trace; threshold = 1 - probability_pruning_threshold)
        (; run = run, cost = cost, tb = tb)
    end

    results = Dictionary()

    function should_stop(result)
        _stop = result.run.was_stopped_by_user && skip_remaining_runs_on_user_interrupt
        _stop && verbose && @info "Benchmarking was interrupted by user."
        _stop
    end

    if "constvel" ∈ method_list
        verbose && @info """
        ======================================================
        Baseline: constant velocity
        """
        constvel_results = benchmark_tb_estimator(
            Examples.ConstantBranchingTimeEstimator(0);
            annotation = "baseline-const-vel",
            use_constant_velocity_ego_model = true,
        )
        should_stop(constvel_results) && return results
        insert!(results, "constvel", constvel_results)
    end

    if "plan_in_expectation" ∈ method_list
        verbose && @info """"
        ======================================================
        Baseline: plan in expectation
        """
        plan_in_expectation_results = benchmark_tb_estimator(
            Examples.ConstantBranchingTimeEstimator(planning_horizon - 1);
            annotation = "baseline",
        )
        should_stop(plan_in_expectation_results) && return results
        insert!(results, "plan_in_expectation", plan_in_expectation_results)
    end

    if "tb_0" ∈ method_list
        verbose && @info """
        ======================================================
        Baseline: tb_0
        """
        map_results = benchmark_tb_estimator(
            Examples.ConstantBranchingTimeEstimator(0);
            annotation = "baseline-tb0",
        )
        should_stop(map_results) && return results
        insert!(results, "tb_0", map_results)
    end

    if "tb_1" ∈ method_list
        verbose && @info """
        ======================================================
        Baseline: tb_1
        """
        qmdp_results = benchmark_tb_estimator(
            Examples.ConstantBranchingTimeEstimator(1);
            annotation = "baseline-tb1",
        )
        should_stop(qmdp_results) && return results
        insert!(results, "tb_1", qmdp_results)
    end

    if "pessimistic_beliefpropagation_contingency" ∈ method_list ||
       "hindsight_contingency" ∈ method_list
        verbose && @info """
        ======================================================
        Pessimistic belief propagation contingency
        """
        pessimistic_beliefpropagation_contingency_results = benchmark_tb_estimator(
            Examples.BeliefPropagationBranchingTimeEstimator(;
                reduction_type = :pessimistic,
                epsilon_p,
            );
            annotation = "ours",
        )
        should_stop(pessimistic_beliefpropagation_contingency_results) && return results
        insert!(
            results,
            "pessimistic_beliefpropagation_contingency",
            pessimistic_beliefpropagation_contingency_results,
        )
    end

    if "hindsight_contingency" ∈ method_list
        verbose && @info """
        ======================================================
        Hindsight contingency
        """

        if pessimistic_beliefpropagation_contingency_results.run.feasible
            oracle_tb_guess = pessimistic_beliefpropagation_contingency_results.tb
        elseif "plan_in_expectation" ∈ method_list && plan_in_expectation_results.run.feasible
            oracle_tb_guess = plan_in_expectation_results.tb
        else
            oracle_tb_guess = simulation_horizon
        end

        hindsight_contingency_results = nothing
        is_tb_consistent = false

        # iteratively refining oracle guess of tb
        for oracle_iteration in 1:3
            candidate_results = benchmark_tb_estimator(
                #In some settings I found that over-approximating tb is better but that observation may not be generally true
                Examples.CountDownBranchingTimeEstimator(; intial_branching_time = oracle_tb_guess);
                annotation = "oracle_iter$(oracle_iteration)",
            )
            should_stop(candidate_results) && return results
            is_candidate_tb_consistent =
                candidate_results.run.feasible && (
                    (candidate_results.tb == oracle_tb_guess) || (
                        candidate_results.tb >= simulation_horizon &&
                        oracle_tb_guess >= simulation_horizon
                    )
                )

            # only store feasible solution candidates
            if isnothing(hindsight_contingency_results) || candidate_results.run.feasible
                hindsight_contingency_results = candidate_results
                is_tb_consistent = is_candidate_tb_consistent
            end

            if !candidate_results.run.feasible
                belief_trace = candidate_results.run.trace.beliefs
                if !isempty(belief_trace) && Solver.is_certain(belief_trace[end])
                    verbose &&
                        @info "Hindisght contingency failed after reaching certainty. Giving up."
                    break
                end

                verbose &&
                    @info "Hindsight contingency solver failed probably because tb was too low."
                oracle_tb_guess += 1
            elseif !is_candidate_tb_consistent
                verbose && @info """
                Hindsight tb estimate was off.
                oracle tb: $(oracle_tb_guess)
                observed tb: $(candidate_results.tb)
                """
                oracle_tb_guess = candidate_results.tb
            else
                verbose && @info "Hindsight tb estimate was consistent at $(oracle_tb_guess)"
                break
            end
        end

        if !is_tb_consistent
            verbose && @warn "Final oracle tb was inconsistent with hindsight tb observation."
        end

        insert!(results, "hindsight_contingency", hindsight_contingency_results)
    end

    results
end

"""
Compare both solvers on a fixed configuration (initial state and belief).

All kwargs are passed to `run_interactive_simulation`.
"""
function simulate_receding_horizon_cost(
    demo_setup,
    branching_time_estimator;
    observation_noise_σ,
    ground_truth_hypothesis,
    ground_truth_max_change_time,
    simulation_horizon,
    seed,
    simulate_noise,
    offscreen_rendering,
    use_constant_velocity_ego_model,
    sim_kwargs...,
)
    belief_updater = let
        observation_from_state = state -> mortar(blocks(state)[(begin + 1):(end)])
        observation_dimension = length(observation_from_state(demo_setup.initial_state))
        observation_distribution =
            Distributions.MvNormal(observation_noise_σ * I(observation_dimension))
        Examples.BayesianBeliefUpdater(; observation_from_state, observation_distribution)
    end

    # Simulate our contingency solver
    run = Examples.demo(
        demo_setup;
        use_constant_velocity_ego_model,
        sim_kwargs = (;
            belief_updater,
            check_termination = (stage, step) -> step >= simulation_horizon,
            ground_truth_hypothesis,
            ground_truth_max_change_time,
            seed,
            simulate_noise,
            branching_time_estimator,
            offscreen_rendering,
            sim_kwargs...,
        ),
    )

    (;
        branching_time_estimator,
        run.trace,
        run.initial_solution,
        run.feasible,
        run.was_stopped_by_user,
    )
end

function visualize_empirical_tb_dynamics(tb_sweep; canvas = Makie.Figure())
    tb_desired = [run.branching_time_estimator.branching_time for run in tb_sweep]
    tb_actual = [get_empirical_branching_time(run.trace) for run in tb_sweep]
    mask = findall(!isnothing, tb_actual)

    Makie.scatter(
        canvas,
        tb_desired[mask],
        tb_actual[mask];
        axis = (; ylabel = "Empirical brancing time"),
    )
end

function compute_closed_loop_cost(trace)
    if isempty(trace.stage_costs)
        return 0.0
    end
    # only for the ego player for now
    sum(stage_cost[begin] for stage_cost in trace.stage_costs)
end

function get_empirical_branching_time(trace; threshold, conservative = false)
    if conservative
        return if isempty(trace.beliefs)
            length(trace.beliefs)
        else
            findlast(
                collect(zip(trace.beliefs, trace.ground_truth_indices)),
            ) do (belief, ground_truth_index)
                belief[ground_truth_index].weight <= threshold
            end
        end
    end

    time_of_certainty = findfirst(
        collect(zip(trace.beliefs, trace.ground_truth_indices)),
    ) do (belief, ground_truth_index)
        belief[ground_truth_index].weight > threshold
    end

    if isnothing(time_of_certainty)
        return length(trace.beliefs)
    end
    # NOTE: there is a subtle difference in the use of tb which we are correcting for here:
    #   - tb in the *planner* is the time until which the inputs are constraint to be equal
    #   - `time_of_certainty` above (tb in our write-ups ) is the time at which the ego *is* certain (which is one step later)
    # Hence, we subtract one here.
    time_of_certainty - 1
end
