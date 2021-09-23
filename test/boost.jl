using BoostedCDE
using Test

init_ϕ = ([1.,2], [1.,0,1])
bl = (fill(PolyBaseLearner(2), 2), fill(PolyBaseLearner(2), 3))  # TODO in practice we need to make sure models are not copys of eachother!
model = BoostingModel(init_ϕ, bl)

@testset "BoostingModel argchecks" begin
    bl_too_long = (fill(PolyBaseLearner(2)), fill(PolyBaseLearner(2)))
    bl_array_mismatch =  (fill(PolyBaseLearner(2), 1), fill(PolyBaseLearner(2), 3))
    @test_throws ArgumentError BoostingModel(init_ϕ, bl_too_long)
    @test_throws ArgumentError BoostingModel(init_ϕ, bl_array_mismatch)
end

@testset "Empty model prediction" begin
    x = rand(10, 3)
    θ = rand(10, 5)
    ϕ = predict(model, θ, x)
    @test all([ϕᵢ == init_ϕ for ϕᵢ in ϕ])
    @test all([ϕᵢ !== init_ϕ for ϕᵢ in ϕ])
end
