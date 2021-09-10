using BoostedCDE
using Test
using LinearAlgebra
using Distributions

@testset "loss" begin
    μ1 = [0., 0]
    μ2 = [1., 1]
    L = LowerTriangular([1. 0; 0 1])
    ϕ = [MvnCholeskyϕ(μ1, L), MvnCholeskyϕ(μ2, L)]

    l = loss(ϕ, [0. 0; 0 0])
    expected = -mean([logpdf(MvNormal(μ, 1), zeros(2)) for μ in μs])
    @test l ≈ expected
end
