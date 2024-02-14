function time_shift_strategy(
    strategy::TrajectoryGamesBase.OpenLoopStrategy,
    time_shift;
    x_padding_element = strategy.xs[end],
    u_padding_element = zero(strategy.us[end]),
)
    TrajectoryGamesBase.OpenLoopStrategy(
        [strategy.xs[(begin + time_shift):end]; fill(x_padding_element, time_shift)],
        [strategy.us[(begin + time_shift):end]; fill(u_padding_element, time_shift)],
    )
end

"""
Derive a warm-start initial guess from the given `solution` anticipating that we will get to re-plan
at `turn_length` time steps in the future.

Note: shifting actually doesn't do all that much so I'm disabling it for now.
"""
function derive_warmstart_initial_guess(
    solution,
    contingency_game::ContingencyGame,
    solver,
    belief,
    turn_length;
    apply_shift = false,
    copy_duals = true,
)
    if !apply_shift
        return solution.info.raw_solution.z
    end

    number_of_hypotheses = length(solution.strategy.substrategies[begin].weights)
    rollout_strategy_per_hypothesis = map(1:number_of_hypotheses) do hypothesis_index
        # get the joint, time-shifted trajectory only under this hypothesis
        JointStrategy([
            time_shift_strategy(substrategy.branch_strategies[hypothesis_index], turn_length - 1) #
            for substrategy in solution.strategy.substrategies
        ])
    end

    # override the state component of the belief with the shifted initial state
    belief = map(belief, rollout_strategy_per_hypothesis) do b, strategy
        Accessors.@set b.state =
            mortar([substrategy.xs[turn_length] for substrategy in strategy.substrategies])
    end

    trajectory_hypotheses = Solver.RolloutInitialization(nothing)(
        contingency_game,
        solver,
        belief;
        rollout_strategy_per_hypothesis,
    )

    if copy_duals
        z_initial = solution.info.raw_solution.z
    else
        z_initial = zero(solution.info.raw_solution.z)
    end

    Solver.generate_initial_guess(
        solver,
        contingency_game,
        belief,
        trajectory_hypotheses;
        z_initial,
    )
end
