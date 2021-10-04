using Test
using BoostedCDE
using LinearAlgebra

@testset "MeanCholeskyMvn" begin
    μ = [1.,2]
    U = UpperTriangular([3. 4; 0 5])
    ϕ = MeanCholeskyMvn(μ, U)

    # Check view behavior works as needed
    ϕ.v[1] += 10
    ϕ.v[5] += 10
    @test ϕ.μ[1] == 11
    @test ϕ.U[3] == 15
end
