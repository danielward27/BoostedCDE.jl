using Test
using BoostedCDE

# TODO USew with debugging code from the repl https://www.julia-vscode.org/docs/stable/userguide/debugging/
bl = PolyBaseLearner(2; use_cache=true)
@enter fit!(bl, x, y)
ŷ = predict(bl, x)
@test ŷ ≈ y
@test bl.β ≈ [1,2,3]

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
end


function check_cache_speedup(
    θ::Matrix{Float64},
    u::Vector{Float64};
    use_cache::Bool = true)
    bl = PolyBaseLearner(2; use_cache)
    for i in 1:100
        for θₖ in eachcol(θ)
            fit!(bl, θₖ, u)
        end
    end            
end


using BenchmarkTools

@btime check_cache_speedup($rand(1000, 10), $rand(1000), use_cache=$true)

@btime check_cache_speedup($rand(1000, 10), $rand(1000), use_cache=$true)


@testset "PolyBaseLearner cache speedup" begin
    θ = rand(1000, 3)
    for col in eachcol

    

end






# Test is caching actually speeds up results?
# We can probably use IdDict
