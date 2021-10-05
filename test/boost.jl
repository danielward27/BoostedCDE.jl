using BoostedCDE
using Test
using LinearAlgebra


# Test using basic vector api
@testset "Boosting vector api" begin
    ρ(ϕ, x) = loss(MeanCholeskyMvn, ϕ, x)
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
    step!(model, θ, x, ϕₘ; loss=ρ)
    ϕₘ = predict(model, θ)
    loss2 = ρ(ϕₘ, x)
    @test loss2 < init_loss

    # Reset and retrain should get to same solution
    reset!(model)
    ϕₘ = predict(model, θ)
    step!(model, θ, x, ϕₘ; loss=ρ)
    ϕₘ = predict(model, θ)
    loss2_reset = ρ(ϕₘ, x)
    @test loss2_reset ≈ loss2
end

# TODO Step 5 and 5 should equal steps 10









    #  ϕₘ, losses = boost!(model, θ, x; loss=mvn_loss, steps=10)
    #  @test all([losses[i+1] < losses[i] for i in 1:(length(losses) - 1)])
    #  @test ϕₘ ≈ predict(model, θ)  # Accumulated during training vs predict from scratch
    #  # step 5 and 5 should give same result as steps=10
    #  model = BoostingModel(init_ϕ, bl)
    #  _, _ = boost!(model, θ, x; loss=mvn_loss, steps=5)
    #  ϕₘ2, losses2 = boost!(model, θ, x; loss=mvn_loss, steps=5)
    #  @test losses[6:end] ≈ losses2
    #  @test ϕₘ ≈ ϕₘ2


