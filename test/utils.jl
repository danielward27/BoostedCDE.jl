using BoostedCDE
using Test
using LinearAlgebra

@testset "Triangular matrices" begin
    upper = UpperTriangular(reshape(1:2^2, (2,2)))
    expected_vec = [1,3,4]

    @test triangular_to_vec(upper) == expected_vec
    @test vec_to_triangular(expected_vec) == upper

    ϕ = [1.,2,3,4,5]
    μ, U = μ_chol_splitter(ϕ)
    
    @test μ == [1.,2]
    @test U == UpperTriangular([3. 4; 0 5])
    @test_throws ArgumentError μ_chol_splitter([1.,2,3,4])
end

