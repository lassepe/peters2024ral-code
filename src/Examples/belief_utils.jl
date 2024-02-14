struct StateOracleUpdater end

function set_belief_weights(belief, weights)
    [(; b..., weight) for (b, weight) in zip(belief, weights)]
end

function update_belief(::StateOracleUpdater, turn_length, joint_strategy, belief, observed_state)
    # ovewrite the state component of all belief hypotheses with the observed state
    [(; b..., state = observed_state) for b in belief]
end

function zip_to_joint_belief(;
    weights,
    states,
    cost_parameters,
    dynamics_parameters = [nothing for _ in weights],
)
    map(
        weights,
        states,
        cost_parameters,
        dynamics_parameters,
    ) do weight, state, cost_parameters, dynamics_parameters
        (; weight, state, cost_parameters, dynamics_parameters)
    end
end

@kwdef struct BayesianBeliefUpdater{T1,T2}
    observation_from_state::T1
    observation_distribution::T2
end

function get_observation_likelihood(updater::BayesianBeliefUpdater, observation, expected_state)
    # Note: this currently implicitly assumes translation invariance of the observation model
    expected_observation = updater.observation_from_state(expected_state)
    Distributions.pdf(updater.observation_distribution, observation - expected_observation)
end

function update_belief(
    updater::BayesianBeliefUpdater,
    turn_length,
    joint_strategy,
    belief,
    observed_state,
    intent_change_probability = 0.0;
    is_observation_relative = false,
    simulate_noise = false,
    rng = nothing,
    regularization = 0.0,
    auxiliary_state = nothing,
    enable_velocity_estimation = false,
)
    observation = updater.observation_from_state(observed_state)
    # TODO: think about if this should be done in `observation_from_state`
    if simulate_noise
        observation += rand(rng, updater.observation_distribution)
    end

    # TODO: could also do some additional filtering here
    filtered_state = observed_state

    unnormalized_updated_belief = map(belief, Iterators.countfrom()) do b, belief_index
        # 1. propagate the weights of the previous belief through the intent dynamics
        # TODO: Add some process noise. For now, we just assume fixed intents
        wp = b.weight
        # 2. extract the nominal observed_state for this intent: where would we expect the opponents to be
        # if they perfectly executed their intent
        expected_state = mortar([
            sub.branch_strategies[belief_index].xs[turn_length] for
            sub in joint_strategy.substrategies
        ])

        if is_observation_relative
            expected_state -= mortar([
                sub.branch_strategies[belief_index].xs[turn_length - 1] for
                sub in joint_strategy.substrategies
            ])
        end

        # 3. update the weight with the evidence
        wpp = (
            wp * (get_observation_likelihood(updater, observation, expected_state)) +
            regularization
        )

        (; b..., weight = wpp, state = filtered_state)
    end

    # 4. normalize the weights
    normalized_weights = normalize([b.weight for b in unnormalized_updated_belief], 1)

    # account for potential intent changes
    normalized_weights = [
        (1 - intent_change_probability) * w +
        intent_change_probability / length(normalized_weights) for w in normalized_weights
    ]

    (
        set_belief_weights(unnormalized_updated_belief, normalized_weights),
        observation,
        auxiliary_state,
    )
end
