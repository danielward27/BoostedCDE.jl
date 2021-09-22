using BoostedCDE
using Test
using LinearAlgebra
using Distributions

@testset "mvn_loss" begin
    μ1 = [0., 0]
    μ2 = [1., 1]
    U = [1, 0, 1]
    U_tri = vec_to_triangular(U)
    Σ = inv(U_tri'U_tri)
    ϕ = [(μ1, U), (μ2, U)]
    l = mvn_loss(ϕ, [0. 0; 0 0])
    expected = -mean([logpdf(MvNormal(μ, 1), zeros(2)) for μ in (μ1, μ2)])
    @test l ≈ expected
end
