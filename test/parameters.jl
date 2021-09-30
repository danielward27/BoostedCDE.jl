using BoostedCDE
using Test
using LinearAlgebra
using Flux

μ = rand(2)
U = UpperTriangular(rand(2,2))

MeanCholeskyMvn(μ, U)