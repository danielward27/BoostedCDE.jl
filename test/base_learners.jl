using Test
using BoostedCDE

@testset "PolyBaseLearner" begin
    x = 1:10
    β_true = [1,2,3]
    x_poly = [x.^0 x.^1 x.^2]
    y = x_poly * β_true
    bl = PolyBaseLearner(2)
    fit!(bl, x, y)
    ŷ = predict(bl, x)
    @test ŷ ≈ y
    @test bl.β ≈ [1,2,3]
end
