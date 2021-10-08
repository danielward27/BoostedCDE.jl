using BoostedCDE
using Test
using LinearAlgebra
using ReverseDiff


# Test using basic vector api
@testset "Boosting vector api" begin
    ρ(ϕ, x) = loss(MeanCholeskyMvn(2), ϕ, x)
    N = 10
    J = 5
    init_ϕ = rand(J)
    θ = rand(N, 5)
    x = rand(N, 2)
    bl_too_long = fill(PolyBaseLearner(2), 6)
    @test_throws ArgumentError BoostingModel(init_ϕ, bl_too_long)

    bls = fill(PolyBaseLearner(2), J)
    model = BoostingModel(init_ϕ, bls; η = 0.1)

    # Predict with empty model
    ϕₘ = predict(model, θ)
    expected_ϕₘ = zeros(N,J).+init_ϕ'
    @test ϕₘ == expected_ϕₘ
 
    # Check loss decreases
    init_loss = ρ(ϕₘ, x)
    u = ReverseDiff.gradient(ϕₘ -> ρ(ϕₘ, x), ϕₘ)
    step!(model, θ, u)
    ϕₘ = predict(model, θ)
    loss2 = ρ(ϕₘ, x)
    @test loss2 < init_loss

    # Reset and retrain should get to same solution
    reset!(model)
    ϕₘ = predict(model, θ)
    u = ReverseDiff.gradient(ϕₘ -> ρ(ϕₘ, x), ϕₘ)
    step!(model, θ, u)
    ϕₘ = predict(model, θ)
    loss2_reset = ρ(ϕₘ, x)
    @test loss2_reset ≈ loss2

    # Test predict providing previous ϕ works as expected
    prev_ϕ = predict(model, θ)
    u = ReverseDiff.gradient(ϕₘ -> ρ(ϕₘ, x), ϕₘ)
    step!(model, θ, u)
    @test predict(model, θ, prev_ϕ) == predict(model, θ)

    # Test step 5 + step 5 == step 10
    reset!(model)
    boost!(model, θ, x, loss=ρ, steps=5)
    boost!(model, θ, x, loss=ρ, steps=5)
    ϕ1 = predict(model, θ)

    reset!(model)
    boost!(model, θ, x, loss=ρ, steps=10)
    ϕ2 = predict(model, θ)
    @test ϕ1 == ϕ2
end

