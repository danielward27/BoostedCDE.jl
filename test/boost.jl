using BoostedCDE
using Test

init_ϕ = [1., 2, 3, 4]
bl = fill(PolyBaseLearner(2), length(init_ϕ))  # TODO in practice we need to make sure models are not copys of eachother!
model = BoostingModel(init_ϕ, bl)

@testset "BoostingModel argchecks" begin
    bl_too_long = (fill(PolyBaseLearner(2), 5))
    @test_throws ArgumentError BoostingModel(init_ϕ, bl_too_long)
end

@testset "Empty model prediction" begin
    x = rand(10, 3)
    θ = rand(10, 5)
    ϕ = predict(model, θ, x)
    @test all([all(init_ϕ[j] .== ϕ[:, j]) for j in 1:length(init_ϕ)])
end
