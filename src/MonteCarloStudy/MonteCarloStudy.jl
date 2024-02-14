module MonteCarloStudy

using Dictionaries: Dictionaries, Dictionary
using BlockArrays: BlockArrays, BlockVector, Block, blocks, mortar
using ParametricMCPs: ParametricMCPs
using Distributions: Distributions
#using JSServe: JSServe
using CairoMakie: CairoMakie
#using WGLMakie: WGLMakie

# if we run on the cluster, we may not be able to load GLMakie
try
    using GLMakie: GLMakie
catch
end

using LinearAlgebra: norm, normalize
using Makie: Makie
using ProgressMeter: ProgressMeter
using Random: Random
using TrajectoryGamesBase: JointStrategy, OpenLoopStrategy, TrajectoryGamesBase
using TrajectoryGamesExamples: TrajectoryGamesExamples
using Infiltrator: Infiltrator
using SplitApplyCombine: SplitApplyCombine
using Statistics: Statistics
using StatsBase: StatsBase
using JLD2: JLD2
using LaTeXStrings: @L_str
using ColorSchemes: ColorSchemes, @colorant_str
using Measurements: Measurements
using SparseArrays: SparseArrays, I
using FileIO: FileIO
using Accessors: @set, @reset
using AlgebraOfGraphics: AlgebraOfGraphics as AOG
using Roots: Roots
using Dates: now

using Distributed: Distributed, pmap, @everywhere

using ..Examples:
    Examples,
    visualize_crosswalk_decorations,
    visualize_overtaking_decorations,
    get_crosswalk_player_markers,
    get_overtaking_player_markers,
    get_crosswalk_player_marker_sizes,
    get_overtaking_player_marker_sizes,
    get_player_colors,
    get_branch_colors

using ..Solver: Solver
using ..ContingencyGames: ContingencyGame, ContingencyStrategy, solve_contingency_game

include("gap_evaluation.jl")
# open loop
include("study.jl")
include("visualization.jl")
# receding horizon
include("receding_horizon_gap_evaluation.jl")

end
