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
end

@testset "vectorize ϕ parameter structs" begin
    μ = [1,2]
    U = UpperTriangular([3 4; 0 5])
    ϕ = MeanCholeskyMvn(μ, U)
    ϕ_re = unvectorize_like(ϕ, [1,2,3,4,5])
    @test vectorize(ϕ) == [1,2,3,4,5]
    @test ϕ.μ == ϕ_re.μ
    @test ϕ.U == ϕ_re.U
end
