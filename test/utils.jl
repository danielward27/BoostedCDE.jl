using BoostedCDE
using Test
using LinearAlgebra

@testset "Triangular matrices" begin
    lower = LowerTriangular(reshape(1:2^2, (2,2)))
    upper = lower'
    expected_vec = [1,2,4]

    @test triangular_to_vec(lower) == expected_vec
    @test triangular_to_vec(upper) == expected_vec

    @test vec_to_triangular(expected_vec) == upper
    @test vec_to_triangular(expected_vec, :L) == lower
    @test_throws ArgumentError vec_to_triangular(expected_vec, :Z) == lower

    # TODO Remove below?
    # μ = [1,2,3]
    # a = rand(3,3); a = a'a + I
    # C = cholesky(a)

    # ϕ = μ_and_cholesky_to_vec(μ, C)
    # μ2, C2 = vec_to_μ_and_cholesky(ϕ)
    # @test μ ≈ μ2
    # @test C.U ≈ C2.U
    # @test C.L ≈ C2.L

    # @test triangind(3, :U) == [1,4,5,7,8,9]
    # @test triangind(3, :L) == [1,2,3,5,6,9]
end

