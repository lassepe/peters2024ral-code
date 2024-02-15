using ContingencyGames
using Test: @test, @testset

@testset "ContingencyGames.jl" begin
    @testset "integration tests" begin
        @testset "crosswalk nonlinear demo" begin
            demo_setup = ContingencyGames.Examples.setup_crosswalk_nonlinear_demo()
            info = ContingencyGames.Examples.demo(
                demo_setup;
                sim_kwargs = (; check_termination = function (state, step)
                    step > 1
                end),
            )
            @test info.feasible
        end

        @testset "overtaking nonlinear demo" begin
            demo_setup = ContingencyGames.Examples.setup_overtaking_nonlinear_demo()
            info = ContingencyGames.Examples.demo(
                demo_setup;
                sim_kwargs = (; check_termination = function (state, step)
                    step > 1
                end),
            )
            @test info.feasible
        end
    end
end
