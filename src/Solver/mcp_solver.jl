struct MCPContingencySolver{T}
    """
    (;
        branching_time_configuration::AbstractBranchTimeConfiguration,
        number_of_hypotheses::Int,
        mcp_problem_representation::ParametricMCPs.ParametricMCP;,
        dimensions::NamedTuple,
        initialization_strategies::Vector{AbstractInitializationType},
    )
    """
    fields::T
end

function Base.getproperty(solver::MCPContingencySolver, name::Symbol)
    if name === :fields
        Base.getfield(solver, name)
    else
        Base.getproperty(solver.fields, name)
    end
end

"""
Abstract trait type to configure the solver branching time.
"""
abstract type AbstractBranchTimeConfiguration end

"""
With this configuration, the solver allows to set the branching time at runtime but may be slower.
"""
struct DynamicBranchingTime <: AbstractBranchTimeConfiguration end

"""

With this configuraiton, the solver sets the branching time at compile time and is thus faster.
"""
struct StaticBranchingTime <: AbstractBranchTimeConfiguration
    value::Int
end

"""
Abstrtact trait type to configure the solver planning horizon
"""
abstract type AbstractPlanningHorizonConfiguration end

"""
With this configuration, the solver allows to set the planning horizon at runtime but may be slower.
"""
struct DynamicPlanningHorizon <: AbstractPlanningHorizonConfiguration
    maximum_value::Int
end

function get_maximum_horizon(c::DynamicPlanningHorizon)
    c.maximum_value
end

"""
With this configuration, the solver sets the planning horizon at compile time and is thus faster.
"""
struct StaticPlanningHorizon <: AbstractPlanningHorizonConfiguration
    value::Int
end

function get_maximum_horizon(c::StaticPlanningHorizon)
    return c.value
end

abstract type AbstractInitializationType end

struct RolloutInitialization{T} <: AbstractInitializationType
    get_rollout_strategy_per_hypothesis::T
end

function RolloutInitialization(opponent_strategy_type::Symbol = :cruise)
    RolloutInitialization(
        function (contingency_game, solver, belief)
            dims = solver.dimensions[begin]
            number_of_hypotheses = length(belief)
            map(1:number_of_hypotheses) do hypothesis_index
                JointStrategy(
                    map(1:num_players(contingency_game)) do player_index
                        dynamics_parameters = belief[hypothesis_index].dynamics_parameters
                        dynamics_hypothesis =
                            contingency_game.dynamics(dynamics_parameters).subsystems[player_index]
                        get_emergency_breaking_strategy(
                            dynamics_hypothesis,
                            belief[hypothesis_index].state,
                            x -> contingency_game.coupling_constraints(x, hypothesis_index),
                            solver.constraint_tightening_factors,
                            player_index;
                            strategy_type = player_index == 1 ? :stop : opponent_strategy_type,
                        )
                    end,
                )
            end
        end,
    )
end

# TODO: move this logic somewhere else
function get_emergency_breaking_strategy(
    dynamics::TrajectoryGamesExamples.UnicycleDynamics,
    initial_state,
    coupling_constraints,
    constraint_tightening_factors,
    player_index;
    strategy_type = :stop,
)
    (u_min, u_max) = control_bounds(dynamics)

    function (x, t)
        px, py, v, θ = x[Block(player_index)]
        # TODO: maybe stop more gracefully
        if strategy_type === :stop && (
            v >= 0 || any(
                (
                    coupling_constraints(x) .-
                    constraint_tightening_factors.coupling_constraints * (t - 1)
                ) .< 0.0,
            )
        )
            [u_min[1] .+ constraint_tightening_factors.control_box_constraints * (t - 1), 0.0]
        else
            zeros(control_dim(dynamics))
        end
    end
end

function get_emergency_breaking_strategy(
    dynamics::TrajectoryGamesBase.LinearDynamics,
    initial_state,
    coupling_constraints,
    constraint_tightening_factors,
    player_index;
    strategy_type = :stop,
)
    umin, umax = control_bounds(dynamics)
    function (x, t)
        if strategy_type === :stop
            px, py = x[Block(player_index)]
            px̂, pŷ = initial_state[Block(player_index)]
            clamp.(
                10 * [px̂ - px, pŷ - py],
                umin .+ constraint_tightening_factors.control_box_constraints * (t - 1),
                umax .- constraint_tightening_factors.control_box_constraints * (t - 1),
            )
        elseif strategy_type === :cruise
            zeros(control_dim(dynamics))
        else
            error("Unknown strategy type: $strategy_type")
        end
    end
end

function (initialization_strategy::RolloutInitialization)(
    contingency_game,
    solver,
    belief;
    rollout_strategy_per_hypothesis = initialization_strategy.get_rollout_strategy_per_hypothesis(
        contingency_game,
        solver,
        belief,
    ),
)
    horizon = contingency_game.horizon
    number_of_hypotheses = length(belief)

    map(1:number_of_hypotheses) do hypothesis_index
        dynamics_parameters = belief[hypothesis_index].dynamics_parameters
        dynamics_hypothesis = contingency_game.dynamics(dynamics_parameters)
        initial_state = belief[hypothesis_index].state
        rollout(
            dynamics_hypothesis,
            rollout_strategy_per_hypothesis[hypothesis_index],
            initial_state,
            horizon,
        )
    end
end

struct CopyInitialStateInitialization <: AbstractInitializationType end

function (initialization_strategy::CopyInitialStateInitialization)(
    contingency_game,
    solver,
    belief;
    kwargs...,
)
    horizon = contingency_game.horizon

    map(1:length(belief)) do hypothesis_index
        xs = [belief[hypothesis_index].state for _ in 1:horizon]
        us = [
            BlockArrays.mortar([
                zeros(d) for d in solver.dimensions[hypothesis_index].control_blocks
            ]) for _ in 1:horizon
        ]
        (; xs, us)
    end
end

function generate_initial_guess(
    solver::MCPContingencySolver,
    contingency_game,
    belief,
    initialization_strategy::AbstractInitializationType,
)
    trajectory_hypotheses = initialization_strategy(contingency_game, solver, belief)
    generate_initial_guess(solver, contingency_game, belief, trajectory_hypotheses)
end

function generate_initial_guess(
    solver::MCPContingencySolver,
    contingency_game,
    belief,
    trajectory_hypotheses;
    z_initial = zeros(ParametricMCPs.get_problem_size(solver.mcp_problem_representation)),
)
    (; horizon) = contingency_game
    (; dimensions, number_of_hypotheses) = solver

    zero_input_primals = reduce(vcat, flatten_hypotheses_per_player(trajectory_hypotheses))

    @assert let
        number_of_primals = sum(enumerate(dimensions)) do (hypothesis_index, dim)
            (dim.state + dim.control) * horizon
        end
        length(zero_input_primals) == number_of_primals
    end

    copyto!(z_initial, zero_input_primals)

    z_initial
end

function to_blockvector(block_dimensions)
    function (data)
        BlockArrays.BlockArray(data, block_dimensions)
    end
end

function to_vector_of_vectors(vector_dimension)
    function (z)
        reshape(z, vector_dimension, :) |> eachcol .|> collect
    end
end

function to_vector_of_blockvectors(block_dimensions)
    vector_dimension = sum(block_dimensions)
    function (z)
        z |> to_vector_of_vectors(vector_dimension) .|> to_blockvector(block_dimensions) |> collect
    end
end

function get_symbolic_parameter_representation(
    parameter::AbstractVector,
    variable_base_name,
    hypothesis_index,
)
    variable_name = Symbol("$(variable_base_name)$(hypothesis_index)")
    Symbolics.@variables($variable_name[1:length(parameter)]) |> only |> scalarize
end

function get_symbolic_parameter_representation(
    parameter::Union{Symbol,Nothing},
    variable_base_name,
    hypothesis_index,
)
    parameter
end

function setup_contingency_constraints(
    ::DynamicBranchingTime,
    hypotheses_symbolic;
    dimensions,
    horizon,
)
    number_of_hypotheses = length(hypotheses_symbolic)
    ego_control_dimension = dimensions[begin].control_blocks[begin]

    branching_time_symbolic = only(Symbolics.@variables(tb))
    contingency_constraints = let
        us_ii = hypotheses_symbolic[begin].us
        mapreduce(vcat, hypotheses_symbolic[(begin + 1):end]) do hypothesis_symbolic_jj
            us_jj = hypothesis_symbolic_jj.us
            mapreduce(vcat, 1:horizon, us_ii[1:horizon], us_jj[1:horizon]) do t, u_ii, u_jj
                is_contingency_constraint_active = ifelse(t <= branching_time_symbolic, 1, 0)
                (u_ii[BlockArrays.Block(1)] - u_jj[BlockArrays.Block(1)]) *
                is_contingency_constraint_active
            end
        end
    end

    (; contingency_constraints, branching_time_symbolic)
end

function setup_contingency_constraints(
    branching_time_configuration::StaticBranchingTime,
    hypotheses_symbolic;
    dimensions,
    horizon,
)
    branching_time_symbolic = branching_time_configuration.value
    contingency_constraints = let
        us_ii = hypotheses_symbolic[begin].us
        mapreduce(vcat, hypotheses_symbolic[(begin + 1):end]) do hypothesis_symbolic_jj
            us_jj = hypothesis_symbolic_jj.us
            mapreduce(
                vcat,
                us_ii[1:(branching_time_configuration.value)],
                us_jj[1:(branching_time_configuration.value)],
            ) do u_ii, u_jj
                u_ii[BlockArrays.Block(1)] - u_jj[BlockArrays.Block(1)]
            end
        end
    end

    (; contingency_constraints, branching_time_symbolic)
end

function setup_horizon(planning_horizon_configuration::DynamicPlanningHorizon)
    planning_horizon_symbolic = only(Symbolics.@variables(horizon))
    stage_active_indicators_symbolic = map(1:(planning_horizon_configuration.maximum_value)) do t
        ifelse(t <= planning_horizon_symbolic, 1, 0)
    end
    (; planning_horizon_symbolic, stage_active_indicators_symbolic)
end

function setup_horizon(planning_horizon_configuration::StaticPlanningHorizon)
    planning_horizon_symbolic = planning_horizon_configuration.value
    stage_active_indicators_symbolic = trues(planning_horizon_configuration.value)
    (; planning_horizon_symbolic, stage_active_indicators_symbolic)
end

function MCPContingencySolver(
    contingency_game::ContingencyGame,
    initial_belief;
    branching_time_configuration = DynamicBranchingTime(),
    planning_horizon_configuration = DynamicPlanningHorizon(contingency_game.horizon),
    context_state_dimension = 0,
    initialization_strategies = [
        RolloutInitialization(:cruise),
        RolloutInitialization(:stop),
        CopyInitialStateInitialization(),
    ],
    option_overrides = (;),
    reperturbation_schedule = [2e-2, 3e-2, 4e-2],
    compute_sensitivities = false,
    constraint_tightening_factors = (;
        environment_constraints = 0.0,
        state_box_constraints = 0.0,
        control_box_constraints = 0.0,
        coupling_constraints = 0.0,
    ),
    initial_state_slack = (; maximum_value = 0.0, penalty = 0.0),
)
    number_of_hypotheses = length(initial_belief)
    # TODO: could also handle additional context states to make a differentiable contingency game
    # set up a joint trajectory for each game hypothesis
    context_state_symbolic =
        Symbolics.@variables(context[1:context_state_dimension]) |> only |> scalarize

    maximum_horizon = get_maximum_horizon(planning_horizon_configuration)
    (; planning_horizon_symbolic, stage_active_indicators_symbolic) =
        setup_horizon(planning_horizon_configuration)

    hypothesis_active_indicators_symbolics =
        Symbolics.@variables(hypothesis_active_indicators[1:number_of_hypotheses]) |>
        only |>
        scalarize

    prune_only_coupling_constraints_symbolic =
        only(Symbolics.@variables(prune_only_coupling_constraints))

    emergency_mode_symbolic = only(Symbolics.@variables(emergency_mode))

    hypotheses_symbolic = map(1:number_of_hypotheses) do hypothesis_index
        dynamics_parameters = get_symbolic_parameter_representation(
            initial_belief[hypothesis_index].dynamics_parameters,
            :dynamics_parameters,
            hypothesis_index,
        )

        dimensions = let
            state_blocks = [
                state_dim(contingency_game.dynamics(dynamics_parameters), player_index)
                for player_index in 1:num_players(contingency_game)
            ]
            state = sum(state_blocks)
            control_blocks = [
                control_dim(contingency_game.dynamics(dynamics_parameters), player_index) for player_index in 1:num_players(contingency_game)
            ]
            control = sum(control_blocks)
            (; state_blocks, state, control_blocks, control)
        end

        belief = let
            weight = let variable_name = Symbol("w_$hypothesis_index")
                Symbolics.@variables($variable_name) |> only
            end
            state = let variable_name = Symbol(:x0, hypothesis_index)
                Symbolics.@variables($variable_name[1:(dimensions.state)]) |>
                only |>
                scalarize |>
                to_blockvector(dimensions.state_blocks)
            end
            cost_parameters = get_symbolic_parameter_representation(
                initial_belief[hypothesis_index].cost_parameters,
                :cost_parameters,
                hypothesis_index,
            )
            (; weight, state, cost_parameters, dynamics_parameters)
        end

        # we prune constraints for hypotheses with probability below a threshold
        is_hypothesis_active = hypothesis_active_indicators_symbolics[hypothesis_index]
        is_hypothesis_active_or_prune_only_coupling_constraints =
            1 - (1 - is_hypothesis_active) * (1 - prune_only_coupling_constraints_symbolic)

        dynamics_hypothesis = contingency_game.dynamics(belief.dynamics_parameters)
        # all scenarios share the same initial state for now but in general it would be possible to have
        # a different initial state for each
        us = let variable_name = Symbol(:u, hypothesis_index)
            Symbolics.@variables($variable_name[1:(dimensions.control * maximum_horizon)]) |>
            only |>
            scalarize |>
            to_vector_of_blockvectors(dimensions.control_blocks)
        end

        xs = let variable_name = Symbol(:x, hypothesis_index)
            Symbolics.@variables($variable_name[1:(dimensions.state * maximum_horizon)]) |>
            only |>
            scalarize |>
            to_vector_of_blockvectors(dimensions.state_blocks)
        end

        cost_per_player = mapreduce(
            .+,
            xs,
            us,
            1:maximum_horizon,
            stage_active_indicators_symbolic,
        ) do x, u, t, is_active
            contingency_game.cost.stage_cost(
                x,
                u,
                t,
                vcat(belief.cost_parameters, context_state_symbolic, emergency_mode_symbolic),
            ) .* is_active
        end

        equality_constraints = let
            dynamics_constraints = mapreduce(vcat, 2:maximum_horizon) do t
                (dynamics_hypothesis(xs[t - 1], us[t - 1], t - 1) - xs[t]) *
                stage_active_indicators_symbolic[t]
            end

            if initial_state_slack.maximum_value <= 0
                initial_state_constraints = xs[begin] - belief.state
            else
                # in this case we handle it in the inequality constraints
                initial_state_constraints = Symbolics.Num[]
            end

            [initial_state_constraints; dynamics_constraints] *
            is_hypothesis_active_or_prune_only_coupling_constraints
        end

        inequality_constraints = let
            if initial_state_slack.maximum_value <= 0
                # in this case we handle it in the equality constraints
                initial_state_constraints = Symbolics.Num[]
            else
                # give some slack to the initial state constraint for feasibility
                initial_state_constraints = [
                    xs[begin] - belief.state .+ initial_state_slack.maximum_value
                    belief.state - xs[begin] .+ initial_state_slack.maximum_value
                ]
            end

            environment_constraints =
                mapreduce(vcat, pairs(unstack_trajectory((; xs, us)))) do (ii, trajectory)
                    environment_constraints_ii = get_constraints(contingency_game.env, ii)
                    mapreduce(vcat, 2:maximum_horizon) do t
                        (
                            environment_constraints_ii(trajectory.xs[t]) .-
                            constraint_tightening_factors.environment_constraints * (t - 1)
                        ) * stage_active_indicators_symbolic[t]
                    end
                end
            # TODO: technically, we could handle the box constraints here in a smarter way to
            # avoid dual multipliers by directly adding them as bounds in the MCP. (thus
            # reducing the problem size)
            state_box_constraints = let
                g_state_box = get_constraints_from_box_bounds(state_bounds(dynamics_hypothesis))
                mapreduce(vcat, 2:maximum_horizon) do t
                    (
                        g_state_box(xs[t]) .-
                        constraint_tightening_factors.state_box_constraints * (t - 1)
                    ) * stage_active_indicators_symbolic[t]
                end
            end
            control_box_constraints = let
                g_control_box = get_constraints_from_box_bounds(control_bounds(dynamics_hypothesis))
                mapreduce(vcat, 1:maximum_horizon) do t
                    (
                        g_control_box(us[t]) .-
                        constraint_tightening_factors.control_box_constraints * (t - 1)
                    ) * stage_active_indicators_symbolic[t]
                end
            end
            [
                initial_state_constraints
                environment_constraints
                state_box_constraints
                control_box_constraints
            ] * is_hypothesis_active_or_prune_only_coupling_constraints
        end

        if isnothing(contingency_game.coupling_constraints)
            coupling_constraints = Symbolics.Num[]
        else
            coupling_constraints = mapreduce(vcat, 2:maximum_horizon) do t
                (
                    contingency_game.coupling_constraints(xs[t], hypothesis_index) .-
                    (t - 1) * constraint_tightening_factors.coupling_constraints
                ) *
                stage_active_indicators_symbolic[t] *
                is_hypothesis_active *
                (1 - emergency_mode_symbolic)
            end
        end

        # also nix the ego-cost for the hypotheses with probability below a threshold
        cost_per_player[begin] *= is_hypothesis_active

        (;
            xs,
            us,
            cost_per_player,
            equality_constraints,
            inequality_constraints,
            coupling_constraints,
            belief,
            dimensions,
            is_active = is_hypothesis_active,
        )
    end

    dimensions = [hypothesis.dimensions for hypothesis in hypotheses_symbolic]
    @assert(
        allequal(d.control_blocks[1] for d in dimensions),
        "The ego agent must have the same number of inputs across all hypotheses."
    )

    belief_symbolic = [h.belief for h in hypotheses_symbolic]

    (; contingency_constraints, branching_time_symbolic) = setup_contingency_constraints(
        branching_time_configuration,
        hypotheses_symbolic;
        dimensions,
        horizon = maximum_horizon,
    )

    expected_cost_per_player_symbolic = sum(1:number_of_hypotheses) do hypothesis_index
        map(enumerate(hypotheses_symbolic[hypothesis_index].cost_per_player)) do (player_index, c)
            w = player_index == 1 ? belief_symbolic[hypothesis_index].weight : 1
            w * c
        end
    end

    h_symbolic =
        [contingency_constraints; mapreduce(h -> h.equality_constraints, vcat, hypotheses_symbolic)]
    μ_symbolic = Symbolics.@variables(μ[1:length(h_symbolic)]) |> only |> scalarize
    g_private_symbolic = mapreduce(h -> h.inequality_constraints, vcat, hypotheses_symbolic)
    λ_private_symbolic =
        Symbolics.@variables(λ_private[1:length(g_private_symbolic)]) |> only |> scalarize
    g_shared_symbolic = mapreduce(h -> h.coupling_constraints, vcat, hypotheses_symbolic)
    λ_shared_symbolic =
        Symbolics.@variables(λ_shared[1:length(g_shared_symbolic)]) |> only |> scalarize

    # multiplier scaling per player as a runtime parameter
    # TODO: technically, we could have this scaling for *every* element of the constraint and
    # actually every constraint but for now let's keep it simple
    shared_constraint_premultipliers_symbolic =
        Symbolics.@variables(γ_scaling[1:num_players(contingency_game)]) |> only |> scalarize

    private_variables_per_player_symbolic = flatten_hypotheses_per_player(hypotheses_symbolic)

    ∇lagrangian_per_player_symbolic = map(
        Iterators.countfrom(),
        expected_cost_per_player_symbolic,
        private_variables_per_player_symbolic,
        shared_constraint_premultipliers_symbolic,
    ) do player_index,
    expected_cost_symbolic,
    private_variables_symbolic,
    shared_constraint_premultiplier_symbolic
        lagrangian = (
            expected_cost_symbolic + #
            μ_symbolic' * h_symbolic - #
            λ_private_symbolic' * g_private_symbolic -
            λ_shared_symbolic' * g_shared_symbolic * shared_constraint_premultiplier_symbolic
        )

        # regularization
        lagrangian += 1e-2 * sum(hypotheses_symbolic) do h
            sum(1:maximum_horizon) do t
                x = h.xs[t]
                u = h.us[t]
                sum(u .^ 2) + sum((x - h.belief.state) .^ 2)
            end
        end

        if initial_state_slack.maximum_value > 0.0
            # compensate for initial state slack
            lagrangian += sum(hypotheses_symbolic) do h
                initial_state_slack.penalty * sum((h.xs[begin] - h.belief.state) .^ 2)
            end
        end

        Symbolics.gradient(lagrangian, private_variables_symbolic)
    end

    f_symbolic = [
        ∇lagrangian_per_player_symbolic...
        h_symbolic
        g_private_symbolic
        g_shared_symbolic
    ]
    z_symbolic = [
        private_variables_per_player_symbolic...
        μ_symbolic
        λ_private_symbolic
        λ_shared_symbolic
    ]

    θ_symbolic = compose_parameter_vector(
        branching_time_configuration,
        planning_horizon_configuration,
        belief_symbolic,
        context_state_symbolic,
        branching_time_symbolic,
        planning_horizon_symbolic,
        shared_constraint_premultipliers_symbolic,
        hypothesis_active_indicators_symbolics,
        prune_only_coupling_constraints_symbolic,
        emergency_mode_symbolic,
    )

    number_of_primal_decision_variables =
        sum(length(p) for p in private_variables_per_player_symbolic)
    lower_bounds = [
        fill(-Inf, number_of_primal_decision_variables + length(h_symbolic))
        fill(0.0, length(g_private_symbolic) + length(g_shared_symbolic))
    ]
    upper_bounds = fill(Inf, length(lower_bounds))
    parameter_dimension = length(θ_symbolic)

    # Using the low-level constructor here to avoid duplicate symbolic differentiation (which runs
    # into some problems with `max` et al).
    mcp_problem_representation = ParametricMCPs.ParametricMCP(
        f_symbolic,
        z_symbolic,
        θ_symbolic,
        lower_bounds,
        upper_bounds;
        compute_sensitivities,
    )

    @info "done generating..."
    MCPContingencySolver((;
        branching_time_configuration,
        planning_horizon_configuration,
        number_of_hypotheses,
        mcp_problem_representation,
        dimensions,
        context_state_dimension,
        initialization_strategies,
        option_overrides,
        reperturbation_schedule,
        constraint_tightening_factors,
    ))
end

function flatten_hypotheses_per_player(hypotheses)
    number_of_players = BlockArrays.blocklength(hypotheses[begin].us[begin])
    variables_per_hypothesis = map(hypotheses) do hypothesis
        trajectory_per_player = unstack_trajectory(hypothesis)
        flatten_trajectory.(trajectory_per_player)
    end
    map(1:number_of_players) do player_index
        reduce(vcat, (v[player_index] for v in variables_per_hypothesis))
    end
end

"""
Reshapes the raw solution into a `JointStrategy` over `ContingencyStrategy`s.
"""
function strategy_from_raw_solution(;
    raw_solution,
    contingency_game,
    solver,
    belief,
    branching_time,
    planning_horizon,
    hypothesis_active_indicators,
)
    number_of_players = num_players(contingency_game)
    (; horizon) = contingency_game
    (; dimensions, number_of_hypotheses) = solver
    z_iter = Iterators.Stateful(raw_solution.z)

    map(1:number_of_players) do player_index
        trajectory_hypotheses = map(1:number_of_hypotheses) do hypothesis_index
            private_state_dimension = dimensions[hypothesis_index].state_blocks[player_index]
            private_control_dimension =
                dimensions[hypothesis_index].control_blocks[player_index]

            number_of_primals =
                horizon * (private_state_dimension + private_control_dimension)
            z_private = Iterators.take(z_iter, number_of_primals) |> collect
            unflatten_trajectory(z_private, private_state_dimension, private_control_dimension)
        end

        ContingencyGames.ContingencyStrategy((;
            player_index,
            branching_time = player_index == 1 ? branching_time : 0,
            branch_strategies = [
                OpenLoopStrategy(
                    trajectory.xs[1:planning_horizon],
                    trajectory.us[1:planning_horizon],
                ) for trajectory in trajectory_hypotheses
            ],
            weights = [b.weight for b in belief],
            hypothesis_active_indicators,
        ))
    end |> JointStrategy
end

function compose_parameter_vector(
    branching_time_configuration,
    planning_horizon_configuration,
    belief,
    context_state,
    branching_time,
    planning_horizon,
    shared_constraint_premultipliers,
    hypothesis_active_indicators,
    prune_only_coupling_constraints,
    emergency_mode,
)
    filter_numbers(x) = eltype(x) <: Number ? x : Float64[]

    parameter_vector = reduce(
        vcat,
        (
            [
                b.weight
                b.state
                filter_numbers(b.cost_parameters)
                filter_numbers(b.dynamics_parameters)
            ] for b in belief
        ),
    )

    append!(parameter_vector, context_state)
    append!(parameter_vector, shared_constraint_premultipliers)
    append!(parameter_vector, hypothesis_active_indicators)
    push!(parameter_vector, prune_only_coupling_constraints)

    if branching_time_configuration isa DynamicBranchingTime
        push!(parameter_vector, branching_time)
    elseif branching_time_configuration isa StaticBranchingTime && !isnothing(branching_time)
        # if the branching time is not dynamic, we don't have to concatenate it to the parameter
        # vector: it is not a runtime variable in this case.
        @assert(
            branching_time_configuration.value == branching_time,
            """
            The branching time chosen at runtime is not supported by the solver. The solver is \
            configured with a static branching time of $(branching_time_configuration.value), but \
            the branching time chosen at runtime is $(branching_time).
            """
        )
    end

    if planning_horizon_configuration isa DynamicPlanningHorizon
        push!(parameter_vector, planning_horizon)
    elseif planning_horizon_configuration isa StaticPlanningHorizon && !isnothing(planning_horizon)
        # if the planning horizon is not dynamic, we don't have to concatenate it to the parameter
        # vector: it is not a runtime variable in this case.
        @assert(
            planning_horizon_configuration.value == planning_horizon,
            """
            The planning horizon chosen at runtime is not supported by the solver. The solver is \
            configured with a static planning horizon of $(planning_horizon_configuration.value), \
            but the planning horizon chosen at runtime is $(planning_horizon).
            """
        )
    end

    push!(parameter_vector, emergency_mode)

    parameter_vector
end

function try_reperturbations(;
    solver,
    raw_solution,
    θ,
    reperturbation_schedule,
    initial_guess,
    options,
    verbose = true,
)
    new_options = options
    for (reperturbation_count, proximal_perturbation) in enumerate(reperturbation_schedule)
        new_options = (; new_options..., proximal_perturbation)
        # try with a higher proximal perturbation
        if is_success(raw_solution.status)
            break
        end

        verbose && @show reperturbation_count
        raw_solution = ParametricMCPs.solve(
            solver.mcp_problem_representation,
            θ;
            initial_guess,
            new_options...,
        )
    end

    raw_solution
end

function is_success(status)
    status == ParametricMCPs.PATHSolver.MCP_Solved
end

# TODO: remove hard-coded certainty threshold (if so, needs to complement `probability_pruning_threshold`)
function is_certain(belief; kwargs...)
    weights = [b.weight for b in belief]
    is_certain(weights; kwargs...)
end

function is_certain(weights::Vector{<:Real}; threshold = 0.99)
    any(w > threshold for w in weights)
end

function ContingencyGames.solve_contingency_game(
    solver::MCPContingencySolver,
    contingency_game::ContingencyGame,
    belief;
    initial_guess = nothing,
    branching_time = solver.branching_time_configuration isa StaticBranchingTime ?
                     solver.branching_time_configuration.value : 5,
    planning_horizon = get_maximum_horizon(solver.planning_horizon_configuration),
    shared_constraint_premultipliers = [1.0, 0.1],
    #shared_constraint_premultipliers = ones(num_players(contingency_game)),
    warn_on_convergence_failure = true,
    context_state = Float64[],
    reperturbation_schedule = solver.reperturbation_schedule,
    probability_pruning_threshold = -1, # by setting this to a negative value, we will never prune any hypothesis
    prune_only_coupling_constraints = true,
    force_prune_indicators = falses(length(belief)),
    initialization_strategies = solver.initialization_strategies,
    pruning_allowed_indicators = trues(length(belief)),
    verbose = true,
    deactivate_infeasible_hypotheses = true,
    allow_emergency_mode = true,
    force_emergency_mode = false,
    infeasibility_check_tolerance = 1e-2,
)
    @assert length(shared_constraint_premultipliers) == num_players(contingency_game)
    (; mcp_problem_representation) = solver

    default_options = (;
        cumulative_iteration_limit = 100_000,
        use_basics = true,
        proximal_perturbation = 1e-2,
        #crash_method = "none",
        #crash_nbchange_limit = 0,
        #        convergence_tolerance = 1e-3,
        warn_on_convergence_failure = false,
        enable_presolve = false,
        #verbose = true,
    )

    options = (; default_options..., solver.option_overrides...)

    @assert length(context_state) == solver.context_state_dimension

    # lazy collection of initial guesses (only generated when iterating over them)
    fallback_initial_guesses = (
        Symbol(typeof(initialization_strategy)) =>
            generate_initial_guess(solver, contingency_game, belief, initialization_strategy) for
        initialization_strategy in initialization_strategies
    )
    original_initial_guess = initial_guess
    if !isnothing(initial_guess)
        initial_guesses =
            Iterators.flatten(((:user_provided => initial_guess,), fallback_initial_guesses))
    else
        initial_guesses = fallback_initial_guesses
    end

    hypothesis_feasibility_indicators = map(eachindex(belief)) do hypothesis_index
        b = belief[hypothesis_index]
        if isnothing(contingency_game.coupling_constraints)
            return true
        end

        all(
            contingency_game.coupling_constraints(b.state, hypothesis_index) .+
            infeasibility_check_tolerance .>= 0,
        )
    end

    hypothesis_active_indicators = map(eachindex(belief)) do hypothesis_index
        if force_prune_indicators[hypothesis_index]
            return false
        end
        if !pruning_allowed_indicators[hypothesis_index]
            return true
        end
        b = belief[hypothesis_index]
        if b.weight <= probability_pruning_threshold
            return false
        end
        if !deactivate_infeasible_hypotheses
            return true
        end
        hypothesis_feasibility_indicators[hypothesis_index]
    end

    function try_solve(; emergency_mode = false)
        local raw_solution
        θ = compose_parameter_vector(
            solver.branching_time_configuration,
            solver.planning_horizon_configuration,
            belief,
            context_state,
            branching_time,
            planning_horizon,
            shared_constraint_premultipliers,
            hypothesis_active_indicators,
            prune_only_coupling_constraints,
            emergency_mode,
        )

        for (index, (initial_guess_type, initial_guess)) in enumerate(initial_guesses)
            if index > 1
                verbose && println("""
                        solve_contingency_game: trying with different initialization strategy:
                        $(initial_guess_type)
                        """)
            end

            raw_solution =
                ParametricMCPs.solve(mcp_problem_representation, θ; initial_guess, options...)
            if is_success(raw_solution.status)
                break
            end

            # TODO: `enable_presolve` causes segfaults with some nonlinear systems. Disable for now.
            # # try with pre-solve toggled to opposite value
            # raw_solution = ParametricMCPs.solve(
            #     mcp_problem_representation,
            #     θ;
            #     initial_guess,
            #     (; options..., enable_presolve = !(options.enable_presolve))...,
            # )
            #if is_success(raw_solution.status)
            #    break
            #end

            # try re-perturbations
            raw_solution = try_reperturbations(;
                solver,
                raw_solution,
                θ,
                reperturbation_schedule,
                initial_guess,
                options,
                verbose,
            )
            if is_success(raw_solution.status)
                break
            end
        end

        raw_solution
    end

    # only try to solve if we're not in emergency mode and all hypotheses are either feasible or not active
    should_try_nominal_solve =
        !force_emergency_mode &&
        all(hypothesis_feasibility_indicators .|| .!hypothesis_active_indicators)

    if should_try_nominal_solve
        raw_solution = try_solve()
    elseif verbose
        @info "Directly skipping solve"
    end

    # none of the warm-starting strategies helped to recover a feasible solution so we have to call the emergency strategy
    if !should_try_nominal_solve || !is_success(raw_solution.status) && allow_emergency_mode
        verbose && @info "Using emergency mode"
        raw_solution = try_solve(; emergency_mode = true)
    end

    if warn_on_convergence_failure && !is_success(raw_solution.status) && verbose
        @warn "Solver did not converge. PATH solver status is $(raw_solution.status)."
    end
    @assert all(x -> !isnan(x) && !isinf(x), raw_solution.z)

    strategy = strategy_from_raw_solution(;
        raw_solution,
        contingency_game,
        solver,
        belief,
        branching_time,
        planning_horizon,
        hypothesis_active_indicators,
    )

    (; strategy, info = (; raw_solution))
end

"""
Like Symbolics.scalarize but robustly handle empty arrays.
"""
function scalarize(num)
    if length(num) == 0
        return Symbolics.Num[]
    end
    Symbolics.scalarize(num)
end
