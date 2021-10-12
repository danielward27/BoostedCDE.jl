using Test
using BoostedCDE
using StatsBase: mean, std

@test BoostedCDE.tri_n_el_to_d.([3,6,10,15]) == [2,3,4,5]

@testset "StandardScaler" begin
    x = [2 4; 4 8; 6 12]
    scaler = StandardScaler(x)
    scaler.μ ≈ [4, 8]
    scaler.σ ≈ [2, 4]
    x_scaled = scaler(x)
    x_unscaled = unscale(scaler, x_scaled)
    @test x ≈ x_unscaled
    @test mean.(eachcol(x_scaled)) ≈ [0,0]
    @test std.(eachcol(x_scaled)) ≈ [1,1]
end

@testset "train_val_split" begin
    θ = rand(10, 3)
    x = rand(10, 4)
    data = BoostedCDE.train_val_split(θ, x, 0.8)
    @test size(data.θ_train) == (8, 3)
    @test size(data.θ_val) == (2, 3)
    @test size(data.x_train) == (8, 4)
    @test size(data.x_val) == (2, 4)
end
