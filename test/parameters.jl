using BoostedCDE
using Test
using LinearAlgebra
using Flux

@testset "MvnCholeskyϕ" begin
    μ = [1, 2.]
    L = LowerTriangular([1. 0; 3 4])
    ϕ = MvnCholeskyϕ(μ, L)
    @test Flux.params(ϕ) == Flux.Params([μ, [1.,3,4]])
end

