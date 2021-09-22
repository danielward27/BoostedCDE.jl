using BoostedCDE
using Test

init_ϕ = ([1.,2], [1.,0,1])
bl = (fill(PolyBaseLearner(2), 2), fill(PolyBaseLearner(2), 3))
model = BoostingModel(init_ϕ, bl)

@testset "Boosting model argchecks" begin
    bl_too_long = (fill(PolyBaseLearner(2)), fill(PolyBaseLearner(2)))
    bl_array_mismatch =  (fill(PolyBaseLearner(2), 1), fill(PolyBaseLearner(2), 3))
    @test_throws ArgumentError BoostingModel(init_ϕ, bl_too_long)
    @test_throws ArgumentError BoostingModel(init_ϕ, bl_array_mismatch)
end

@testset "Empty model prediction" begin
    x = rand(10, 3)
    ϕ = predict(model, x)
    @test all([ϕᵢ == init_ϕ for ϕᵢ in ϕ])
    @test all([ϕᵢ !== init_ϕ for ϕᵢ in ϕ])
end
