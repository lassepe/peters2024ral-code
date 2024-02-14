module Solver

using TrajectoryGamesBase:
    TrajectoryGame,
    OpenLoopStrategy,
    TrajectoryGamesBase,
    JointStrategy,
    ProductDynamics,
    control_dim,
    get_constraints,
    num_players,
    rollout,
    solve_trajectory_game!,
    control_bounds,
    state_bounds,
    state_dim,
    unstack_trajectory,
    stack_trajectories,
    flatten_trajectory,
    unflatten_trajectory,
    get_constraints_from_box_bounds

# TODO: move this logic somewhere else
using TrajectoryGamesExamples: TrajectoryGamesExamples

using Symbolics: Symbolics
using BlockArrays: BlockArrays, Block, blocks, mortar
using ParametricMCPs: ParametricMCPs
using SparseArrays: SparseArrays
using ..ContingencyGames: ContingencyGames, ContingencyGame

using Random: Random

include("mcp_solver.jl")

end
