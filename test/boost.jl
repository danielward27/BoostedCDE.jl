using BoostedCDE
using Test

init_ϕ = ([1.,2], [1.,0,1])
base_learners = (fill(PolynomialBaseLearner(2)), fill(PolynomialBaseLearner(2)))
model = BoostingModel(init_ϕ, base_learners)

@testset "Empty model prediction" begin
    x = rand(10, 3)
    ϕ = predict(model, x)

    @test all([ϕᵢ == init_ϕ for ϕᵢ in ϕ])
    @test all([ϕᵢ !== init_ϕ for ϕᵢ in ϕ])
end
