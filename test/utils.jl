using Test
using BoostedCDE
using StatsBase: mean, std

@test BoostedCDE.tri_n_el_to_d.([3,6,10,15]) == [2,3,4,5]

@testset "StandardScaler" begin
    y = [2 4; 4 8; 6 12]
    scaler = StandardScaler(y)
    scaler.μ ≈ [4, 8]
    scaler.σ ≈ [2, 4]
    x_scaled = scaler(y)
    x_unscaled = unscale(scaler, x_scaled)
    @test y ≈ x_unscaled
    @test mean.(eachcol(x_scaled)) ≈ [0,0]
    @test std.(eachcol(x_scaled)) ≈ [1,1]
end

@testset "train_val_split" begin
    x = rand(10, 3)
    y = rand(10, 4)
    data = BoostedCDE.train_val_split(x, y, 0.8)
    @test size(data.x_train) == (8, 3)
    @test size(data.x_val) == (2, 3)
    @test size(data.y_train) == (8, 4)
    @test size(data.y_val) == (2, 4)
end
