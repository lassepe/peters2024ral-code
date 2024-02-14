module ContingencyGames

using TrajectoryGamesBase: TrajectoryGamesBase
using Makie: Makie
using BlockArrays: mortar

include("api.jl")
include("Solver/Solver.jl")
include("Examples/Examples.jl")
include("MonteCarloStudy/MonteCarloStudy.jl")
end
