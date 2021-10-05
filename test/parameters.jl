using Test
using BoostedCDE
using LinearAlgebra

@testset "MeanCholeskyMvn" begin
    U = UpperTriangular([3. 4; 0 5])

    @test_throws ArgumentError MeanCholeskyMvn([1.,2,3,4,5,6])
    @test_throws ArgumentError MeanCholeskyMvn([1.,2, 3], U)

    a = MeanCholeskyMvn([1.,2,3,4,5])
    b = MeanCholeskyMvn([1., 2], U)
    @test a.μ == b.μ
    @test a.U == b.U
    @test a.d == b.d
end
