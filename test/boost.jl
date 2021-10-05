using BoostedCDE
using Test
using Distributions
using LinearAlgebra

init_ϕ = MeanCholeskyMvn(rand(2), UpperTriangular(rand(2,2)))

N = 10
J = 5

init_ϕ = rand(N, J)
θ = rand(N, 5)
x = rand(N, 2)

bls = fill(PolyBaseLearner(2), J)

BoostingModel(init_ϕ, bls)







@testset "BoostingModel argchecks" begin
    bl_too_long = (fill(PolyBaseLearner(2), 6))
    @test_throws ArgumentError BoostingModel(init_ϕ, bl_too_long)
end


bl = fill(PolyBaseLearner(2), length(init_ϕ.v))  # TODO in practice we may need to make sure models are not copys of eachother!
model = BoostingModel(init_ϕ, bl)
x = rand(10, 2)
θ = rand(10, 5)

# Empty model prediction should return init_ϕ
ϕₘ = predict(model, θ)
@test all([all(init_ϕ.v[j] .== ϕₘ[:, j]) for j in 1:length(init_ϕ.v)])  

ϕₘ, losses = boost!(model, θ, x; steps=10)



#     ϕₘ, losses = boost!(model, θ, x; loss=mvn_loss, steps=10)
#     @test all([losses[i+1] < losses[i] for i in 1:(length(losses) - 1)])
#     @test ϕₘ ≈ predict(model, θ)  # Accumulated during training vs predict from scratch

#     # step 5 and 5 should give same result as steps=10
#     model = BoostingModel(init_ϕ, bl)
#     _, _ = boost!(model, θ, x; loss=mvn_loss, steps=5)
#     ϕₘ2, losses2 = boost!(model, θ, x; loss=mvn_loss, steps=5)
#     @test losses[6:end] ≈ losses2
#     @test ϕₘ ≈ ϕₘ2
# end

