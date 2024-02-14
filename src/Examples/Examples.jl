module Examples
using TrajectoryGamesBase:
    TrajectoryGamesBase,
    TrajectoryGame,
    LinearDynamics,
    GeneralSumCostStructure,
    TimeSeparableTrajectoryGameCost,
    PolygonEnvironment,
    ProductDynamics,
    JointStrategy,
    rollout,
    num_players
using TrajectoryGamesExamples:
    TrajectoryGamesExamples,
    two_player_meta_tag,
    animate_sim_steps,
    planar_double_integrator,
    UnicycleDynamics,
    create_environment_axis
using BlockArrays: BlockArrays, BlockVector, Block, blocks, mortar
using Makie: Makie
using LinearAlgebra: I, norm, norm_sqr, normalize
using ParametricMCPs: ParametricMCPs
using StatsBase: mean
using Dictionaries: Dictionaries
using Distributions: Distributions
using Accessors: Accessors, @set, @reset
using LaTeXStrings: @L_str
using FileIO: FileIO

using ..ContingencyGames:
    ContingencyGames, ContingencyGame, ContingencyStrategy, solve_contingency_game, predict_state
using ..ContingencyGames.Solver:
    Solver,
    MCPContingencySolver,
    DynamicBranchingTime,
    StaticBranchingTime,
    DynamicPlanningHorizon,
    StaticPlanningHorizon
using ColorSchemes: ColorSchemes
using Random: Random

function get_player_colors()
    base_scheme = ColorSchemes.colorschemes[:PRGn_4][range(0, 1; length = 3)]
    p1_color = base_scheme[1]
    p2_color = ColorSchemes.colorant"#cc338bff"
    p3_color = base_scheme[3]

    [p1_color, p2_color, p3_color]
end

function get_branch_colors()
    ColorSchemes.colorschemes[:RdBu_4][range(0, 1; length = 2)]
end

# common infrastructure for setting up example problems
include("cost_utils.jl")
include("environment_utils.jl")
include("dynamics_utils.jl")

# problems
include("crosswalk.jl")
include("crosswalk_nonlinear.jl")
include("overtaking.jl")
include("overtaking_nonlinear.jl")
include("defensive_driving.jl")

include("leftturn_nonlinear.jl")

# belief updaters for receding-horizon experiments
include("belief_utils.jl")
include("branching_time_utils.jl")
# an iteractive simulator for interacting with the example problems
include("warmstarting.jl")
include("feasibility_checking.jl")
include("interactive_simulation.jl")
# the main entrypoint for quickly spinning up a demp
include("demo.jl")
end
