using Test
using BoostedCDE
using LinearAlgebra

μ = [1,2.]
U = UpperTriangular([3 4; 0 5])
ϕ = BoostedCDE.MeanCholeskyMvn2(μ, U)

BoostedCDE.get_μ(ϕ)
BoostedCDE.get_U(ϕ)