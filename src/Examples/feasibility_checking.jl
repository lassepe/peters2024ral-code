function is_feasible(
    game::ContingencyGame,
    states::AbstractVector{<:AbstractVector},
    controls::AbstractVector{<:AbstractVector},
    hypothesis_index,
    dynamics_parameters;
    verbose = true,
    kwargs...,
)
    for tt in 1:(length(states) - 1)
        if !is_feasible(
            game,
            states[tt],
            controls[tt],
            tt,
            states[tt + 1],
            hypothesis_index,
            dynamics_parameters;
            verbose,
            kwargs...,
        )
            verbose && @info "Infeasible at time $tt"
            return false
        end
    end

    true
end

function is_feasible(
    game::ContingencyGame,
    state::AbstractVector,
    control::AbstractVector,
    time,
    next_state,
    hypothesis_index,
    dynamics_parameters;
    tolerance = 1e-2,
    verbose = true,
)
    dynamics_hypothesis = game.dynamics(dynamics_parameters)
    if !all(isapprox.(dynamics_hypothesis(state, control, time), next_state; atol = tolerance))
        verbose && @info """
                   Dynamics constraints violated;
                   violation was $(minimum(abs.(dynamics_hypothesis(state, control, time) .- next_state)))
                   """
        return false
    end

    sb = TrajectoryGamesBase.state_bounds(dynamics_hypothesis)
    if !all(sb.lb .- tolerance .<= next_state .<= sb.ub .+ tolerance)
        verbose && @info """
                   State constraints violated;
                   violation was $(minimum(abs.(next_state .- sb.lb)))
                   """
        return false
    end

    cb = TrajectoryGamesBase.control_bounds(dynamics_hypothesis)
    if !all(cb.lb .- tolerance .<= control .<= cb.ub .+ tolerance)
        verbose && @info """
                   Control constraints violated
                   violation was $(minimum(abs.(control .- cb.lb)))
                   """
        return false
    end

    environment_constraints = TrajectoryGamesBase.get_constraints(game.env)
    if !all(environment_constraints(next_state) .+ tolerance .>= 0)
        verbose && @info """
                   Environment constraints violated;
                   violation was $(minimum(environment_constraints(next_state)))
                   """
        return false
    end

    if !isnothing(game.coupling_constraints) &&
       !all(game.coupling_constraints(next_state, hypothesis_index) .+ tolerance .>= 0)
        verbose && @info """
                   Coupling constraints violated;
                   violation was $(minimum(game.coupling_constraints(next_state, hypothesis_index)))
                   """
        return false
    end

    true
end
