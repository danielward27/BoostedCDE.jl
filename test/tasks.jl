using BoostedCDE, Test, Distributions, Random
import BoostedCDE: gaussian_simulator

@testset "gaussian simulator" begin
    θv = rand(3)
    θm = rand(100, 3)

    x1 = gaussian_simulator(MersenneTwister(1), θv)
    x2 = gaussian_simulator(MersenneTwister(1), θv)
    @test x1 == x2
    @test size(x1) == size(θv)

    x1 = gaussian_simulator(MersenneTwister(1), θm)
    x2 = gaussian_simulator(MersenneTwister(1), θm)
    @test x1 == x2
    @test size(x1) == size(θm)
end
