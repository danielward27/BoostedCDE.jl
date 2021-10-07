using BoostedCDE
using Test
using LinearAlgebra
using Distributions
using ReverseDiff

@testset "MeanCholeskyMvn" begin
    ϕ = [1.,2,3,4,5]
    parameterisation = MeanCholeskyMvn(2)
    μ, U = get_params(parameterisation, ϕ)
    
    x = rand(2)
    Λ = Symmetric(U'U)
    h = Λ*μ
    d = MvNormalCanon(h, Λ)
    expected = -logpdf(d, x)
    
    @test loss(parameterisation, ϕ, x) ≈ expected
    @test loss(parameterisation, [ϕ ϕ]', [x x]') ≈ expected
end

@testset "MeanCholeskyMvn gradients" begin
    # Test that computed gradients look reasonable
    ϕ = [0. 10  1 0 1; 0. 10  1 0 1]
    x = [0. 10; 5 5]
    parameterisation = MeanCholeskyMvn(2)
    u = ReverseDiff.gradient(ϕ -> loss(parameterisation, ϕ, x), ϕ)
    @test u[1, 1:2] == [0, 0]
    @test u[2, 1:2] != [0, 0]
    @test u[2, 1] == -u[2, 2]
end



