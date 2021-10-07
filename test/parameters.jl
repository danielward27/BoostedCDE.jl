using BoostedCDE
using Test
using LinearAlgebra
using Random


@testset "MeanCholeskyMvn" begin
    ϕ = [1.,2,3,4,5]
    μ_expected = [1,2]
    U_expected = UpperTriangular([3. 4; 0 5])
    parameterisation = MeanCholeskyMvn(2)
    μ, U = get_params(parameterisation, ϕ)
    @test μ == μ_expected
    @test U == U_expected
end

