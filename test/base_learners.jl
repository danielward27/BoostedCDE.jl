using Test
using BoostedCDE
using BenchmarkTools

function check_cache_speedup(use_cache::Bool)
    bl = PolyBaseLearner(2; use_cache)
    u = rand(100)
    for i in 1:5
        xₖ = rand(100)
        for i in 1:100
            fit!(bl, xₖ, u)
        end
    end
end

@testset "PolyBaseLearner" begin
    x = 1:10
    β_true = [1,2,3]
    x_poly = [x.^0 x.^1 x.^2]
    y = x_poly * β_true
    bl = PolyBaseLearner(2; use_cache=false)
    fit!(bl, x, y)
    ŷ = predict(bl, x)
    @test ŷ ≈ y
    @test bl.β ≈ [1,2,3]

    bl = PolyBaseLearner(2; use_cache=true)
    fit!(bl, x, y)
    ŷ = predict(bl, x)
    @test ŷ ≈ y
    @test bl.β ≈ [1,2,3]

    t1 = @benchmark check_cache_speedup($true) seconds = 0.1
    t2 = @benchmark check_cache_speedup($false) seconds = 0.1
    mean(x) = sum(x)/length(x)
    speedup = mean(t2.times)/mean(t1.times)
    @test speedup > 2.
end

