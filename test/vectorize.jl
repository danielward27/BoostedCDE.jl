using Test
using BoostedCDE
using LinearAlgebra

@testset "vectorize arrays and special matrices" begin
    @test vectorize([1,2]) == [1,2]
    @test vectorize([1 3; 2 4]) == [1,2,3,4]
    @test vectorize(Diagonal([1,2])) == [1,2]
    @test unvectorize(Diagonal, [1,2]) == Diagonal([1,2])
    @test vectorize(UpperTriangular([1 2; 0 3])) == [1,2,3]
    @test unvectorize(UpperTriangular, [1,2, 3]) == UpperTriangular([1 2; 0 3])
    @test_throws ArgumentError unvectorize(UpperTriangular, [1.,2,3,4])
end

@testset "vectorize ϕ parameter structs" begin
    ϕv = [1.,2,3,4,5]
    ϕ = MeanCholeskyMvn(ϕv)
    ϕv_re = vectorize(ϕ)
    @test ϕv_re == ϕv
end


