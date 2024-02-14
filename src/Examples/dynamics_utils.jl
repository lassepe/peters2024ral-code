struct SwitchedDynamics{T} <: TrajectoryGamesBase.AbstractDynamics
    "An indexable collection of discrete modes this switched system can take"
    dynamics_modes::T
end

function (dynamics::SwitchedDynamics)(mode)
    dynamics.dynamics_modes[mode]
end

# TODO: Maybe discontinue this to require the user to be very explicit about parameterization
# (especially if we don't pull in the parameterization interface into TrajectoryGamesBase)
function (dynamics::SwitchedDynamics)(state, control, time, mode)
    dynamics(mode)(state, control, time)
end

function TrajectoryGamesBase.num_players(dynamics::SwitchedDynamics)
    num_players(dynamics.dynamics_modes[begin])
end

# fallback to make it wor for non-parameterized dynamics
function (dynamics::TrajectoryGamesBase.AbstractDynamics)(::Nothing)
    dynamics
end
