using BoostedCDE
using Test

@testset "base_learners.jl" begin include("base_learners.jl") end
@testset "utils.jl" begin include("utils.jl") end
@testset "parameters.jl" begin include("parameters.jl") end
@testset "losses.jl" begin include("losses.jl") end
@testset "tasks.jl" begin include("tasks.jl") end


