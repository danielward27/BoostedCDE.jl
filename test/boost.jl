using BoostedCDE
using Test
using LinearAlgebra
using ReverseDiff
using Random

# setup
Random.seed!(1)
N = 20
J = 5
init_ϕ = rand(J)
x = rand(N, 5)
y = rand(N, 2)
parameterisation = MeanCholeskyMvn(2)
loss(ϕ, y) = cost(parameterisation, ϕ, y)
loss2(ϕ, y) = begin # Loss with non-important ϕ (low gradient norms)
    l = cost(parameterisation, ϕ[:, 1:5], y)
    l += 1e-10*sum(ϕ[:, 6:10])
    return l
end
∇loss(ϕ) = ReverseDiff.gradient(ϕ -> loss(ϕ, y), ϕ)
bls = fill(PolyBaseLearner(2), J)


@testset "argchecks" begin
    bl_too_long = fill(PolyBaseLearner(2), 6)
    @test_throws ArgumentError BoostingModel(init_ϕ, bl_too_long)
end


@testset "predict" begin
    model = BoostingModel(init_ϕ, bls; η = 0.1)
    # Predict with empty model
    ϕₘ = predict(model, x)
    expected_ϕₘ = zeros(N,J).+ init_ϕ'
    @test ϕₘ == expected_ϕₘ

    # Test predict providing previous ϕ works as expected
    prev_ϕ = predict(model, x)
    u = ∇loss(ϕₘ)
    step!(model, x, u)
    @test predict(model, x, prev_ϕ) == predict(model, x)

    # Reset and retrain should get to same solution
    reset!(model)
    ϕₘ = predict(model, x)
    u = ∇loss(ϕₘ)
    step!(model, x, u)
    ϕₘ2 = predict(model, x)
    @test prev_ϕ ≈ ϕₘ2
end

# Test using basic vector api
@testset "step! and boost!" begin
    model = BoostingModel(init_ϕ, bls; η = 0.1)
    ϕₘ = predict(model, x)

    # Check loss decreases
    init_loss = loss(ϕₘ, y)
    u = ∇loss(ϕₘ)
    step!(model, x, u)
    ϕₘ1 = predict(model, x)
    loss2 = loss(ϕₘ1, y)
    @test loss2 < init_loss

    # Test step 5 + step 5 == step 10
    reset!(model)
    boost!(model, x, y; loss, ∇loss, steps=5)
    ϕ1 = boost!(model, x, y; loss, ∇loss, steps=5).ϕₘ

    reset!(model)
    ϕ2 = boost!(model, x, y; loss, ∇loss, steps=10).ϕₘ
    @test ϕ1 == ϕ2

    # Test provided gradient result == defualt gradient
    reset!(model)
    ϕ1 = boost!(model, x, y; loss, ∇loss, steps=10).ϕₘ

    reset!(model)
    ϕ1 = boost!(model, x, y; loss, steps=10).ϕₘ
    @test ϕ1 == ϕ2
end


@testset "step! vs step_naive!" begin
    init_ϕ = rand(10)
    bls = fill(PolyBaseLearner(2), 10)
    model = BoostingModel(init_ϕ, bls; η = 0.1)

    reset!(model)
    ϕ1 = boost!(model, x, y; loss=loss2, steps = 5, step! = step!).ϕₘ

    reset!(model)
    ϕ2 = boost!(model, x, y; loss=loss2, steps = 5, step! = BoostedCDE.step_naive!).ϕₘ

    @test ϕ1 == ϕ2
end


@testset "boostcv!" begin
    data = BoostedCDE.train_val_split(x, y)
    model = BoostingModel(init_ϕ, bls; η = 0.1)
    
    ∇loss(ϕ) = ReverseDiff.gradient(ϕ -> loss(ϕ, data.y_train), ϕ)

    ϕ1 = boost!(model, data.x_train, data.y_train; loss, ∇loss, steps=10).ϕₘ

    reset!(model)
    ϕ2 = BoostedCDE.boostcv!(model, data; loss, ∇loss, steps = 10).train.ϕₘ

    reset!(model)
    ϕ3 = BoostedCDE.boostcv!(model, data; loss, steps = 10).train.ϕₘ

    @test ϕ1 == ϕ2
    @test ϕ1 == ϕ3
end

@testset "Unimportant ϕ and x" begin
    # Test unimportant ϕ and x aren't commonly chosen.
    Random.seed!(1)
    n = 2000
    x = rand(n, 2)
    ϵ = randn(size(x)...)
    y = x .* [1,2]' .+ ϵ

    x_scaler = StandardScaler(x)
    x_scaler = StandardScaler(y)
    x, y = x_scaler(x), x_scaler(y)
    x = [x rand(size(x)...)] # 2 extra "noise" x

    init_ϕ = rand(10)
    bls = fill(PolyBaseLearner(2), 10)
    model = BoostingModel(init_ϕ, bls; η = 0.1)

    model,_,_ = boost!(model, x, y; loss=loss2, steps = 10, step! = step!)

    ϕ_idx = model.idx[:ϕ]
    x_idx = model.idx[:x]

    @test sum(ϕ_idx.>5) == 0
    @test sum(x_idx.>2) == 0
end

