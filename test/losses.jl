using BoostedCDE
using Test
using LinearAlgebra
using Distributions
using Zygote

@testset "MeanCholeskyMvn" begin
    ϕ = MeanCholeskyMvn([1.,2], UpperTriangular([3. 4; 0 5]))
    x = rand(2)
    Λ = Symmetric(ϕ.U'ϕ.U)
    h = Λ*ϕ.μ
    d = MvNormalCanon(h, Λ)
    expected = -logpdf(d, x)
    ϕv = vectorize(ϕ)
    @test loss(ϕ, x) ≈ expected
    @test loss(MeanCholeskyMvn, ϕv, x) ≈ expected
    @test loss(MeanCholeskyMvn, [ϕv ϕv]', [x x]') ≈ expected
end

@testset "MeanCholeskyMvn gradients" begin
    # Test that computed gradients look reasonable
    ϕ = [0. 10  1 0 1; 0. 10  1 0 1]
    x = [0. 10; 5 5]
    u = Zygote.gradient(ϕ -> loss(MeanCholeskyMvn, ϕ, x), ϕ)[1]
    @test u[1, 1:2] == [0, 0]
    @test u[2, 1:2] != [0, 0]
    @test u[2, 1] == -u[2, 2]
end
