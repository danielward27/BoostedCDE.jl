using BoostedCDE
using Test
using LinearAlgebra
using Distributions

@testset "MeanCholeskyMvn" begin
    ϕ = MeanCholeskyMvn([1,2], UpperTriangular([3 4; 0 5]))
    x = rand(2)
    Λ = Symmetric(ϕ.U'ϕ.U)
    h = Λ*ϕ.μ
    d = MvNormalCanon(h, Λ)
    @test loss(ϕ, x) ≈ -logpdf(d, x)
end
