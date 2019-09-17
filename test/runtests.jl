using Test
using SparseSampling
using POMDPs
using POMDPModels
using Random

Random.seed!(1)

m = BabyPOMDP()

@testset "POMDP" begin
    solver = SparseSamplingSolver(SSOptions(4, 4))
    planner = solve(solver, m)
    @test action(planner, [true]) == true # feed the baby when hungry
    @test action(planner, [false]) == false # don't feed the baby when it is not hungry
    @time action(planner, [true]) == true # feed the baby when hungry
end

@testset "POW" begin
    solver = SparseSamplingSolver(POWSSOptions(4, 4))
    planner = solve(solver, m)
    @test action(planner, [true]) == true # feed the baby when hungry
    @test action(planner, [false]) == false # don't feed the baby when it is not hungry
    @time action(planner, [false]) == false # don't feed the baby when it is not hungry
end
