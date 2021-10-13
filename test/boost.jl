using BoostedCDE
using Test
using LinearAlgebra
using ReverseDiff
using Random

# setup
N = 20
J = 5
init_ϕ = rand(J)
θ = rand(N, 5)
x = rand(N, 2)
parameterisation = MeanCholeskyMvn(2)
loss(ϕ, x) = cost(parameterisation, ϕ, x)
loss2(ϕ, x) = begin # Loss with non-important ϕ (low gradient norms)
    l = cost(parameterisation, ϕ[:, 1:5], x)
    l += 1e-10*sum(ϕ[:, 6:10])
    return l
end
∇loss(ϕ) = ReverseDiff.gradient(ϕ -> loss(ϕ, x), ϕ)
bls = fill(PolyBaseLearner(2), J)


@testset "argchecks" begin
    bl_too_long = fill(PolyBaseLearner(2), 6)
    @test_throws ArgumentError BoostingModel(init_ϕ, bl_too_long)
end


@testset "predict" begin
    model = BoostingModel(init_ϕ, bls; η = 0.1)
    # Predict with empty model
    ϕₘ = predict(model, θ)
    expected_ϕₘ = zeros(N,J).+ init_ϕ'
    @test ϕₘ == expected_ϕₘ

    # Test predict providing previous ϕ works as expected
    prev_ϕ = predict(model, θ)
    u = ∇loss(ϕₘ)
    step!(model, θ, u)
    @test predict(model, θ, prev_ϕ) == predict(model, θ)

    # Reset and retrain should get to same solution
    reset!(model)
    ϕₘ = predict(model, θ)
    u = ∇loss(ϕₘ)
    step!(model, θ, u)
    ϕₘ2 = predict(model, θ)
    @test prev_ϕ ≈ ϕₘ2
end

# Test using basic vector api
@testset "step! and boost!" begin
    model = BoostingModel(init_ϕ, bls; η = 0.1)
    ϕₘ = predict(model, θ)

    # Check loss decreases
    init_loss = loss(ϕₘ, x)
    u = ∇loss(ϕₘ)
    step!(model, θ, u)
    ϕₘ1 = predict(model, θ)
    loss2 = loss(ϕₘ1, x)
    @test loss2 < init_loss

    # Test step 5 + step 5 == step 10
    reset!(model)
    boost!(model, θ, x; loss, ∇loss, steps=5)
    ϕ1 = boost!(model, θ, x; loss, ∇loss, steps=5).ϕₘ

    reset!(model)
    ϕ2 = boost!(model, θ, x; loss, ∇loss, steps=10).ϕₘ
    @test ϕ1 == ϕ2

    # Test provided gradient result == defualt gradient
    reset!(model)
    ϕ1 = boost!(model, θ, x; loss, ∇loss, steps=10).ϕₘ

    reset!(model)
    ϕ1 = boost!(model, θ, x; loss, steps=10).ϕₘ
    @test ϕ1 == ϕ2
end


@testset "step! vs step_naive!" begin
    init_ϕ = rand(10)
    bls = fill(PolyBaseLearner(2), 10)
    model = BoostingModel(init_ϕ, bls; η = 0.1)

    reset!(model)
    ϕ1 = boost!(model, θ, x; loss=loss2, steps = 5, step! = step!).ϕₘ

    reset!(model)
    ϕ2 = boost!(model, θ, x; loss=loss2, steps = 5, step! = BoostedCDE.step_naive!).ϕₘ

    @test ϕ1 == ϕ2
end


@testset "boostcv!" begin
    data = BoostedCDE.train_val_split(θ, x)
    model = BoostingModel(init_ϕ, bls; η = 0.1)
    
    ∇loss(ϕ) = ReverseDiff.gradient(ϕ -> loss(ϕ, data.x_train), ϕ)

    ϕ1 = boost!(model, data.θ_train, data.x_train; loss, ∇loss, steps=10).ϕₘ

    reset!(model)
    ϕ2 = BoostedCDE.boostcv!(model, data; loss, ∇loss, steps = 10).train.ϕₘ

    reset!(model)
    ϕ3 = BoostedCDE.boostcv!(model, data; loss, steps = 10).train.ϕₘ

    @test ϕ1 == ϕ2
    @test ϕ1 == ϕ3
end

@testset "Unimportant ϕ and θ" begin
    # Test unimportant ϕ and θ aren't commonly chosen.
    Random.seed!(1)
    n = 2000
    θ = rand(n, 2)
    ϵ = randn(size(θ)...)
    x = θ .* [1,2]' .+ ϵ

    θ_scaler = StandardScaler(θ)
    x_scaler = StandardScaler(x)
    θ, x = θ_scaler(θ), x_scaler(x)
    θ = [θ rand(size(θ)...)] # 2 extra "noise" θ

    init_ϕ = rand(10)
    bls = fill(PolyBaseLearner(2), 10)
    model = BoostingModel(init_ϕ, bls; η = 0.1)

    model,_,_ = boost!(model, θ, x; loss=loss2, steps = 10, step! = step!)

    ϕ_idx = model.idx[:ϕ]
    θ_idx = model.idx[:θ]

    @test sum(ϕ_idx.>5) == 0
    @test sum(θ_idx.>2) == 0
end

