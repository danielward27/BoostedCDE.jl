using Test
using BoostedCDE
using LinearAlgebra

@testset "MeanCholeskyMvn" begin
    μ = [1.,2, 3]
    U = UpperTriangular([3. 4; 0 5])

    @test_throws ArgumentError MeanCholeskyMvn([1.,2,3,4,5,6])
    @test_throws ArgumentError MeanCholeskyMvn(μ, U)

    # Check view behaviour as expected
    ϕ = MeanCholeskyMvn([1.,2,3,4,5])
    ϕ.v[1] += 10
    ϕ.v[5] += 10
    @test ϕ.μ[1] == 11
    @test ϕ.U[end][end] == 15
end
