using Test
using BoostedCDE
using StatsBase: mean, std

@test BoostedCDE.tri_n_el_to_d.([3,6,10,15]) == [2,3,4,5]

@testset "StandardScaler" begin
    x = [2 4; 4 8; 6 12]
    scaler = StandardScaler(x)
    scaler.μ ≈ [4, 8]
    scaler.σ ≈ [2, 4]
    x_scaled = scale(scaler, x)
    x_unscaled = unscale(scaler, x_scaled)
    @test x ≈ x_unscaled
    @test mean.(eachcol(x_scaled)) ≈ [0,0]
    @test std.(eachcol(x_scaled)) ≈ [1,1]
end
