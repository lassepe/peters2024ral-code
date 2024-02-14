#=== BeliefPropagationBranchingTimeEstimator ===#

@kwdef struct BeliefPropagationBranchingTimeEstimator
    reduction_type::Symbol = :pessimistic
    tb_offset::Int = 0
    tb_min::Int = 1
    epsilon_p::Float64 = 0.04
end

"""
Estimate the future time of certainty (the branching time) based on optimistic simulation.

That is, how quickly *could* we reach certainty if the opponent human operated perfectly at one of the modes.

Note: There's potentially a smarter formula for this but, for now, forward simulation is good enough
"""
function estimate_branching_time(
    branching_time_estimator::BeliefPropagationBranchingTimeEstimator,
    contingency_game,
    updater::BayesianBeliefUpdater,
    turn_length,
    joint_strategy,
    belief,
    step;
    initial_time = step == 1 ? 1 : 2,
    use_relative_observations = false,
)
    if isnothing(joint_strategy)
        return get_initial_branching_time(branching_time_estimator, contingency_game)
    end
    planning_horizon = length(joint_strategy.substrategies[begin].branch_strategies[begin].xs)

    if branching_time_estimator.reduction_type == :pessimistic
        branching_time = 1
        reduce_tbs = max
    elseif branching_time_estimator.reduction_type == :optimistic
        branching_time = planning_horizon
        reduce_tbs = min
    else
        error("Unknown reduction type: $(branching_time_estimator.reduction_type)")
    end

    for (b, hypothesis_index) in zip(belief, Iterators.countfrom())
        # assume that the `b` is the true hypothesis and the future belief
        propagated_belief = belief
        # propagate the belief to each branch
        hypothesis_branching_time = planning_horizon
        for t_future in initial_time:(turn_length - 1):planning_horizon
            if any(b.weight > 1 - branching_time_estimator.epsilon_p for b in propagated_belief)
                hypothesis_branching_time = min(hypothesis_branching_time, t_future)
                break
            end

            simulated_state_observation = predict_state(joint_strategy, t_future, hypothesis_index)

            if use_relative_observations
                simulated_state_observation -=
                    predict_state(joint_strategy, t_future - 1, hypothesis_index)
            end

            propagated_belief, _ = update_belief(
                updater,
                t_future,
                joint_strategy,
                propagated_belief,
                simulated_state_observation;
                is_observation_relative = use_relative_observations,
                simulate_noise = false,
                regularization = 1e-2,
            )
        end
        branching_time = reduce_tbs(branching_time, hypothesis_branching_time)
    end

    # we don't drop tb all the way to 0 because we leave ignoring hypotheses to the solver.
    max(
        branching_time_estimator.tb_min,
        branching_time - initial_time + branching_time_estimator.tb_offset,
    )
end

function get_initial_branching_time(
    branching_time_estimator::BeliefPropagationBranchingTimeEstimator,
    contingency_game,
)
    contingency_game.horizon
end

#=== Constant branching time estimator ===#

@kwdef struct ConstantBranchingTimeEstimator
    branching_time::Int
end

function estimate_branching_time(
    branching_time_estimator::ConstantBranchingTimeEstimator,
    contingency_game,
    updater,
    turn_length,
    joint_strategy,
    belief,
    step,
)
    branching_time_estimator.branching_time
end

function get_initial_branching_time(
    branching_time_estimator::ConstantBranchingTimeEstimator,
    contingency_game,
)
    branching_time_estimator.branching_time
end

#=== CountDownBranchingTimeEstimator ===#

@kwdef struct CountDownBranchingTimeEstimator
    intial_branching_time::Int
    tb_offset::Int = 0
    tb_min::Int = 1
end

function estimate_branching_time(
    branching_time_estimator::CountDownBranchingTimeEstimator,
    contingency_game,
    updater::BayesianBeliefUpdater,
    turn_length,
    joint_strategy,
    belief,
    step;
)
    @assert step >= 1
    # we don't drop tb all the way to 0 because we leave ignoring hypotheses to the solver.
    max(
        branching_time_estimator.tb_min,
        branching_time_estimator.intial_branching_time + branching_time_estimator.tb_offset -
        (step - 1), # -1 for 1-based indexing (initial value of step is 1)),
    )
end

function get_initial_branching_time(
    branching_time_estimator::CountDownBranchingTimeEstimator,
    contingency_game,
)
    branching_time_estimator.intial_branching_time + branching_time_estimator.tb_offset
end
