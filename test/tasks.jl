using BoostedCDE, Test, Distributions, Random
import BoostedCDE: gaussian_simulator

@testset "gaussian simulator" begin
    xv = rand(3)
    xm = rand(100, 3)

    x1 = gaussian_simulator(MersenneTwister(1), xv)
    x2 = gaussian_simulator(MersenneTwister(1), xv)
    @test x1 == x2
    @test size(x1) == size(xv)

    x1 = gaussian_simulator(MersenneTwister(1), xm)
    x2 = gaussian_simulator(MersenneTwister(1), xm)
    @test x1 == x2
    @test size(x1) == size(xm)
end
