"""
An environment that does not impose any constraints but forwards the visualization of the wrapped env
"""
struct TrivialEnvironment{T}
    visualized_environment::T
end

function TrajectoryGamesBase.get_constraints(::TrivialEnvironment, player_id = nothing)
    function (x)
        eltype(x)[]
    end
end

function TrajectoryGamesBase.visualize!(canvas, environment::TrivialEnvironment; kwargs...)
    TrajectoryGamesBase.visualize!(canvas, environment.visualized_environment; kwargs...)
end

"""
An environment that is a product of multiple environments; one for each player.
"""
struct ComposedMultiPlayerEnvironment{T}
    "An indexable collection of single-player environments"
    environment_per_player::T
end

function TrajectoryGamesBase.get_constraints(
    environment::ComposedMultiPlayerEnvironment,
    player_index,
)
    TrajectoryGamesBase.get_constraints(
        environment.environment_per_player[player_index],
        player_index,
    )
end

function TrajectoryGamesBase.get_constraints(environment::ComposedMultiPlayerEnvironment)
    player_indices = eachindex(environment.environment_per_player)

    constraint_getter_per_player = [
        TrajectoryGamesBase.get_constraints(environment, player_ii) for player_ii in player_indices
    ]

    function (state::BlockVector)
        mapreduce(vcat, player_indices) do player_index
            state_ii = state[Block(player_index)]
            constraint_getter_per_player[player_index](state_ii)
        end
    end
end

function TrajectoryGamesExamples.create_environment_axis(
    figure,
    environment::ComposedMultiPlayerEnvironment;
    viz_kwargs = (;),
    kwargs...,
)
    axis = create_environment_axis(
        figure,
        first(environment.environment_per_player);
        viz_kwargs,
        kwargs...,
    )
    for env in environment.environment_per_player[(begin + 1):end]
        TrajectoryGamesBase.visualize!(axis, env; viz_kwargs...)
    end

    axis
end
