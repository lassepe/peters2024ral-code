# TODO: just a rapid prototype; clean up: for a fixed branching time (and initial state for now),
# sweep over a different bliefs
function simulate_belief_sweep(;
    demo_setup,
    nominal_belief,
    branching_time,
    warmstart_per_belief,
    weight_distributions = [[1.0, 0.0], [0.9, 0.1], [0.5, 0.5], [0.1, 0.9], [0.0, 1.0]],
)
    runs = map(weight_distributions) do weight_distribution
        warmstart = warmstart_per_belief[weight_distribution]
        # compute the open-loop strategy
        new_belief =
            [(; b..., weight) for (b, weight) in zip(nominal_belief, weight_distribution)]
        solution = solve_contingency_game(
            demo_setup.solver,
            demo_setup.contingency_game,
            new_belief;
            branching_time,
            context_state = [ctx.startvalue for ctx in demo_setup.context_state_spec],
            shared_constraint_premultipliers = demo_setup.shared_responsibility,
            initial_guess = warmstart,
        )

        warmstart_per_belief[weight_distribution] = solution.info.raw_solution.z

        (; solution.strategy)
    end

    (; demo_setup, runs)
end

function simulate_strategy_matrix(;
    demo_setup,
    nominal_belief,
    branching_times = 0:25,
    weight_distributions = begin
        ws = 0.0:0.1:1.0
        [[w, 1.0 - w] for w in ws]
    end,
)
    warmstart_per_belief = Dict{Vector{Float64},Any}([w => nothing for w in weight_distributions])

    map(branching_times) do branching_time
        result = simulate_belief_sweep(;
            demo_setup,
            nominal_belief,
            branching_time,
            weight_distributions,
            warmstart_per_belief,
        )
        (; result..., branching_time)
    end
end

function simulate_study(
    demo_name;
    demo_setup,
    limits_per_opponent,
    xgrid_points,
    ygrid_points,
    branching_times,
)
    @assert isnothing(limits_per_opponent) ||
            length(limits_per_opponent) ==
            TrajectoryGamesBase.num_players(demo_setup.contingency_game) - 1

    opponent_indices = 2:TrajectoryGamesBase.num_players(demo_setup.contingency_game)

    warmstart_values_on_grid_per_player =
        Any[nothing for _ in 1:TrajectoryGamesBase.num_players(demo_setup.contingency_game)]

    runs = mapreduce(
        vcat,
        Iterators.product(opponent_indices, branching_times),
    ) do (opponent_index, branching_time)
        solver_values_on_grid = visualize_gap(;
            contingency_game = demo_setup.contingency_game,
            demo_setup.solver,
            demo_setup.initial_belief,
            demo_setup.shared_responsibility,
            limits_per_opponent,
            xgrid_points,
            ygrid_points,
            context_state = [ctx.startvalue for ctx in demo_setup.context_state_spec],
            grid_opponent_index = opponent_index,
            branching_time,
            warmstart_solver_values_on_grid = warmstart_values_on_grid_per_player[opponent_index],
        )

        warmstart_values_on_grid_per_player[opponent_index] = solver_values_on_grid[]

        results_on_grid = solver_values_on_grid[]
        map(CartesianIndices(results_on_grid), results_on_grid) do position_index, result
            (; result..., position_index, opponent_index)
        end[:]
    end

    (; demo_name, demo_setup, runs)
end

# quick hack to augment results with opponent indices and position indices
function augment_results(; results, opponent_indices, branching_times, xgrid_points, ygrid_points)
    augmentations = mapreduce(
        vcat,
        Iterators.product(opponent_indices, branching_times),
    ) do (opponent_index, branching_time)
        position_indices = CartesianIndices((xgrid_points, ygrid_points))
        # augmentations
        mapreduce(vcat, position_indices[:]) do position_index
            (; position_index, opponent_index)
        end
    end[:]

    @assert length(augmentations) == length(results)
    map(results, augmentations) do result, augmentation
        (; result..., augmentation...)
    end
end

function evaluate_study_results(raw_study_results; demo_setup = nothing)
    if isnothing(demo_setup)
        demo_setup = raw_study_results.demo_setup
    end

    map(raw_study_results.runs) do run
        # cost gap when replanning
        @assert run.upper_bound_cost > 0
        closedloop_cost = let
            contingency = run.contingency_cost
            upper_bound = run.upper_bound_cost
            absolute_gap = upper_bound - contingency
            relative_gap = absolute_gap / upper_bound
            (; contingency, upper_bound, absolute_gap, relative_gap)
        end

        # open-loop cost gap of plan
        openloop_cost = let
            contingency = compute_expected_cost(
                run.contingency_trunk.strategy,
                demo_setup.contingency_game,
                run.belief,
                run.context_state,
            )
            upper_bound = compute_expected_cost(
                run.upper_bound_trunk.strategy,
                demo_setup.contingency_game,
                run.belief,
                run.context_state,
            )
            @assert upper_bound > 0
            absolute_gap = upper_bound - contingency
            relative_gap = absolute_gap / upper_bound
            (; contingency, upper_bound, absolute_gap, relative_gap)
        end

        closedloop_progress = (;
            contingency = get_final_progress(run.contingency_strategy.substrategies[begin]),
            upper_bound = get_final_progress(run.upper_bound_strategy.substrategies[begin]),
        )

        closedloop_distance = (;
            contingency = get_mean_dist_to_opponents(run.contingency_strategy),
            upper_bound = get_mean_dist_to_opponents(run.upper_bound_strategy),
        )

        weight_distribution = [b.weight for b in run.belief]

        (;
            raw_study_results.demo_name,
            closedloop_cost,
            openloop_cost,
            closedloop_progress,
            closedloop_distance,
            run.position_index,
            run.opponent_index,
            run.belief,
            run.branching_time,
            weight_distribution,
        )
    end
end

"""
Replicating the same structure but now varying the belief rather than the branching time.
"""
function simulate_study2(
    demo_name;
    demo_setup,
    limits_per_opponent,
    xgrid_points,
    ygrid_points,
    weight_distributions = [
        #[1.0, 0.0],
        [0.95, 0.05],
        [0.9, 0.1],
        [0.8, 0.2],
        [0.7, 0.3],
        [0.6, 0.4],
        [0.5, 0.5],
    ],
    branching_time = 10,
)
    @assert isnothing(limits_per_opponent) ||
            length(limits_per_opponent) ==
            TrajectoryGamesBase.num_players(demo_setup.contingency_game) - 1

    nominal_belief = demo_setup.initial_belief
    opponent_indices = 2:TrajectoryGamesBase.num_players(demo_setup.contingency_game)

    warmstart_values_on_grid_per_player =
        Any[nothing for _ in 1:TrajectoryGamesBase.num_players(demo_setup.contingency_game)]

    runs = mapreduce(
        vcat,
        Iterators.product(opponent_indices, weight_distributions),
    ) do (opponent_index, weight_distribution)
        new_belief = [(; b..., weight) for (b, weight) in zip(nominal_belief, weight_distribution)]
        solver_values_on_grid = visualize_gap(;
            contingency_game = demo_setup.contingency_game,
            demo_setup.solver,
            initial_belief = new_belief,
            demo_setup.shared_responsibility,
            limits_per_opponent,
            xgrid_points,
            ygrid_points,
            context_state = [ctx.startvalue for ctx in demo_setup.context_state_spec],
            grid_opponent_index = opponent_index,
            branching_time,
            warmstart_solver_values_on_grid = warmstart_values_on_grid_per_player[opponent_index],
            min_hypothesis_weight = 0.0,
        )

        warmstart_values_on_grid_per_player[opponent_index] = solver_values_on_grid[]

        results_on_grid = solver_values_on_grid[]
        map(CartesianIndices(results_on_grid), results_on_grid) do position_index, result
            (; result..., weight_distribution, position_index, opponent_index)
        end[:]
    end

    (; demo_name, demo_setup, runs)
end

# Evaluation =======================================================================================#

function mean_absolute_acceleration(contingency_strategy)
    @assert sum(contingency_strategy.weights) ≈ 1.0
    sum(
        zip(contingency_strategy.branch_strategies, contingency_strategy.weights),
    ) do (branch_strategy, weight)
        weight * sum(StatsBase.mean(abs.(u) for u in branch_strategy.us))
    end
end

function get_final_progress(contingency_strategy)
    @assert sum(contingency_strategy.weights) ≈ 1.0
    sum(
        zip(contingency_strategy.branch_strategies, contingency_strategy.weights),
    ) do (branch_strategy, weight)
        weight * (branch_strategy.xs[end][2] - branch_strategy.xs[begin][2])
    end
end

function get_mean_dist_to_opponents(joint_contingency_strategy)
    ego_strategy = joint_contingency_strategy.substrategies[begin]
    opponent_strategies = joint_contingency_strategy.substrategies[(begin + 1):end]
    horizon = length(ego_strategy.branch_strategies[begin].xs)
    number_of_hypotheses = length(ego_strategy.branch_strategies)
    StatsBase.mean(1:horizon) do t
        weights = ego_strategy.weights
        @assert sum(weights) ≈ 1.0
        sum(1:number_of_hypotheses) do h
            ego_x = ego_strategy.branch_strategies[h].xs[t]
            # find the closest opponent
            weights[h] * minimum(opponent_strategies) do opponent_strategy
                p_opponent = opponent_strategy.branch_strategies[h].xs[t]
                norm(ego_x[1:2] - p_opponent[1:2])
            end
        end
    end
end

function summarize_closedloop_study_evaluation(study_evaluation)
    method_keys = [:contingency, :upper_bound]

    map(method_keys) do method_key
        method_key => let results = study_evaluation
            (;
                cost = compute_stats(r.closedloop_cost[method_key] for r in results),
                progress = compute_stats(r.closedloop_progress[method_key] for r in results),
                distance = compute_stats(r.closedloop_distance[method_key] for r in results),
            )
        end
    end
end

function new_closedloop_eval(raw_study_results)
    study_evaluation = evaluate_closedloop_study_results(raw_study_results)
    summarize_closedloop_study_evaluation(study_evaluation)
end

function compute_stats(vals)
    val_mean = StatsBase.mean(vals)
    val_std = StatsBase.std(vals)
    val_stderr = val_std / sqrt(length(vals))
    Measurements.measurement(val_mean, val_stderr)
end

# hack to condense the results into tables for the paper
function summarize_openloop_table(study_evaluation)
    [
        :contingency => compute_stats([r.openloop_cost.contingency for r in study_evaluation]),
        :upperbound => compute_stats([r.openloop_cost.upper_bound for r in study_evaluation]),
    ]
end

function run_crosswalk_study(;
    demo_setup = Examples.setup_crosswalk_demo(),
    limits_per_opponent = nothing,
    xgrid_points = 10,
    ygrid_points = 5,
    branching_times = 1:25,
)
    demo_name = "crosswalk"
    global crosswalk_raw_study_results = simulate_study(
        demo_name;
        demo_setup,
        limits_per_opponent,
        xgrid_points,
        ygrid_points,
        branching_times,
    )
    global crosswalk_study_evaluation = evaluate_study_results(crosswalk_raw_study_results)
    global crosswalk_openloop_table = summarize_openloop_table(crosswalk_study_evaluation)
    global crosswalk_closedloop_table =
        summarize_closedloop_study_evaluation(crosswalk_study_evaluation)

    (; crosswalk_openloop_table, crosswalk_closedloop_table)
end

function run_crosswalk_nonlinear_study(;
    demo_setup = Examples.setup_crosswalk_nonlinear_demo(),
    limits_per_opponent = nothing,
    xgrid_points = 7,
    ygrid_points = 10,
    branching_times = 25:-1:1, # iterating in reverse to get feasible warmstarts
)
    demo_name = "crosswalk_nonlinear"
    global crosswalk_nonlinear_raw_study_results = simulate_study(
        demo_name;
        demo_setup,
        limits_per_opponent,
        xgrid_points,
        ygrid_points,
        branching_times,
    )
    global crosswalk_nonlinear_study_evaluation =
        evaluate_study_results(crosswalk_nonlinear_raw_study_results)
    global crosswalk_nonlinear_openloop_table =
        summarize_openloop_table(crosswalk_nonlinear_study_evaluation)
    global crosswalk_nonlinear_closedloop_table =
        summarize_closedloop_study_evaluation(crosswalk_nonlinear_study_evaluation)

    (; crosswalk_nonlinear_openloop_table, crosswalk_nonlinear_closedloop_table)
end

# save an object that can be loaded again with "JLD2.load_object"
function save_crosswalk_nonlinear_results(; prefix = "")
    crosswalk_nonlinear_results = Dict([
        "raw_results" => crosswalk_nonlinear_raw_study_results,
        "evaluation" => crosswalk_nonlinear_study_evaluation,
        "openloop_table" => crosswalk_nonlinear_openloop_table,
        "closedloop_table" => crosswalk_nonlinear_closedloop_table,
    ])

    JLD2.save_object(
        "results/$(prefix)crosswalk_nonlinear_results.jld2",
        crosswalk_nonlinear_results,
    )

    crosswalk_nonlinear_results
end

function load_crosswalk_nonlinear_results(; prefix = "")
    JLD2.load_object("results/$(prefix)crosswalk_nonlinear_results.jld2")
end

"""
collect data for a sweep over beliefs instead of branching times
"""
function run_crosswalk_study2(;
    demo_setup = Examples.setup_crosswalk_nonlinear_demo(),
    limits_per_opponent = nothing,
    xgrid_points = 10,
    ygrid_points = 5,
)
    demo_name = "crosswalk"
    global crosswalk_raw_study_results2 =
        simulate_study2(demo_name; demo_setup, limits_per_opponent, xgrid_points, ygrid_points)
    global crosswalk_study_evaluation2 = evaluate_study_results(crosswalk_raw_study_results2)
    global crosswalk_study_summary2 =
        summarize_study_evaluation(crosswalk_study_evaluation2; group_key = :weight_distribution)
    crosswalk_study_summary2
end

function run_overtaking_study(;
    demo_setup = Examples.setup_overtaking_demo(),
    limits_per_opponent = [((-0.2, 0.7), (-2.9, -2.1)), ((-0.2, 0.7), (-1.8, 0.0))],
    xgrid_points = 5,
    ygrid_points = 10,
    branching_times = 1:25,
)
    demo_name = "overtaking"
    global overtaking_raw_study_results = simulate_study(
        demo_name;
        demo_setup,
        limits_per_opponent,
        xgrid_points,
        ygrid_points,
        branching_times,
    )
    global overtaking_study_evaluation = evaluate_study_results(overtaking_raw_study_results)
    global overtaking_openloop_table = summarize_openloop_table(overtaking_study_evaluation)
    global overtaking_closedloop_table =
        summarize_closedloop_study_evaluation(overtaking_study_evaluation)

    (; overtaking_openloop_table, overtaking_closedloop_table)
end

function run_overtaking_nonlinear_study(;
    demo_setup = Examples.setup_overtaking_nonlinear_demo(),
    limits_per_opponent = [((0.0, 0.5), (-2.8, -2.3)), ((0.0, 0.5), (-1.5, -1.0))],
    xgrid_points = 7,
    ygrid_points = 10,
    # successfully solved for 25:-1:3
    branching_times = 25:-1:1, # could also iterate in reverse order to get feasible warmstart
)
    demo_name = "overtaking_nonlinear"
    global overtaking_nonlinear_raw_study_results = simulate_study(
        demo_name;
        demo_setup,
        limits_per_opponent,
        xgrid_points,
        ygrid_points,
        branching_times,
    )
    global overtaking_nonlinear_study_evaluation =
        evaluate_study_results(overtaking_nonlinear_raw_study_results)
    global overtaking_nonlinear_openloop_table =
        summarize_openloop_table(overtaking_nonlinear_study_evaluation)
    global overtaking_nonlinear_closedloop_table =
        summarize_closedloop_study_evaluation(overtaking_nonlinear_study_evaluation)

    (; overtaking_nonlinear_openloop_table, overtaking_nonlinear_closedloop_table)
end

function save_overtaking_nonlinear_results(; prefix = "")
    overtaking_nonlinear_results = Dict([
        "raw_results" => overtaking_nonlinear_raw_study_results,
        "evaluation" => overtaking_nonlinear_study_evaluation,
        "openloop_table" => overtaking_nonlinear_openloop_table,
        "closedloop_table" => overtaking_nonlinear_closedloop_table,
    ])

    JLD2.save_object(
        "results/$(prefix)overtaking_nonlinear_results.jld2",
        overtaking_nonlinear_results,
    )

    overtaking_nonlinear_results
end

function load_overtaking_nonlinear_results(; prefix = "")
    JLD2.load_object("results/$(prefix)overtaking_nonlinear_results.jld2")
end

function generate_crosswalk_strategy_matrix(;
    demo_setup = Examples.setup_crosswalk_nonlinear_demo(;
        solver_configuration_kwargs = (;
            prune_coupling_constraints_for_zero_probability_branches = true
        ),
    ),
)
    global crosswalk_matrix_demo_setup = demo_setup

    global crosswalk_strategy_matrix =
        simulate_strategy_matrix(; demo_setup, nominal_belief = demo_setup.initial_belief)

    crosswalk_strategy_matrix
end

function print_problem_size_stats(demo_setup)
    # number of variables:
    mcp = demo_setup.solver.mcp_problem_representation
    number_of_decision_variables = ParametricMCPs.get_problem_size(mcp)
    number_of_structural_nonzeros = SparseArrays.nnz(mcp.jacobian_z!)

    (; number_of_decision_variables, number_of_structural_nonzeros)
end
