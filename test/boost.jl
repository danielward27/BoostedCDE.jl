using BoostedCDE
using Test
using Distributions

init_ϕ = [1., 2, 3, 4, 5]

@testset "BoostingModel argchecks" begin
    bl_too_long = (fill(PolyBaseLearner(2), 6))
    @test_throws ArgumentError BoostingModel(init_ϕ, bl_too_long)
end


@testset "predict, step! and boost!" begin
    bl = fill(PolyBaseLearner(2), length(init_ϕ))  # TODO in practice we need to make sure models are not copys of eachother!
    model = BoostingModel(init_ϕ, bl)
    x = rand(10, 2)
    θ = rand(10, 5)

    ϕₘ = predict(model, θ)
    @test all([all(init_ϕ[j] .== ϕₘ[:, j]) for j in 1:length(init_ϕ)])  # Empty model

    ϕₘ, losses = boost!(model, θ, x; loss=mvn_loss, steps=10)
    @test all([losses[i+1] < losses[i] for i in 1:(length(losses) - 1)])
    @test ϕₘ ≈ predict(model, θ)  # Accumulated during training vs predict from scratch

    # step 5 and 5 should give same result as steps=10
    model = BoostingModel(init_ϕ, bl)
    _, _ = boost!(model, θ, x; loss=mvn_loss, steps=5)
    ϕₘ2, losses2 = boost!(model, θ, x; loss=mvn_loss, steps=5)
    @test losses[6:end] ≈ losses2
    @test ϕₘ ≈ ϕₘ2
end


# function linear_θ_to_ϕ_map(θ::Matrix{Float64})  # Deterministic mapping to distributional parameters
#     ϕ = Matrix{Float64}(undef, size(θ))
#     [ϕ[:, k] = k*θ[:, k] .+ k for k in 1:size(θ, 2)]
#     ϕ
# end

# θ_true = [1. 2 3 4 5]
# @test linear_θ_to_ϕ_map(θ_true) ≈ [2 6 12 20 30]
# ϕ_true = linear_θ_to_ϕ_map(θ_true)
# d_true = BoostedCDE.mvn_d_from_ϕ(vec(ϕ_true))
# μ_true, Σ_true = mean(d_true), cov(d_true)


# θ = rand(100, 5)
# ϕ = linear_θ_to_ϕ_map(θ)
# x = Matrix{Float64}(undef, 100, 2)
# for (i, ϕᵢ) in enumerate(eachrow(ϕ))
#     d = BoostedCDE.mvn_d_from_ϕ(ϕᵢ)
#     x[i, :] = rand(d)
# end

# bl = fill(PolyBaseLearner(1), length(init_ϕ))
# model = BoostingModel(ones(5), bl; sl=20)
# ϕₘ, loss = boost!(model, θ, x, loss=mvn_loss, steps=100)
# loss[end]

# # Why does a higher step size perform better?


# predict(model, [1. 2 3 4 5])


# ϕ_prediction = predict(model, [1. 2 3 4 5])
# d_prediction = BoostedCDE.mvn_d_from_ϕ(vec(ϕ_prediction))
# Σ_prediction = cov(d_prediction)
# μ_prediction = mean(d_prediction)

# d_actual = BoostedCDE.mvn_d_from_ϕ(vec(linear_θ_to_ϕ_map([1. 2 3 4 5])))
# Σ_actual = cov(d_actual)
# μ_actual = mean(d_actual)

# μ_prediction
# Σ_prediction

# μ_actual
# Σ_actual


