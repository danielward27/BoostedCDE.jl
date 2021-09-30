
abstract type Abstractϕ end

"""
Parameterization of multivariate normal distribution with mean vector μ and the
upper triangular component of the cholesky decomposition of the covariance
matrix.
"""
struct MeanCholeskyMvn <: Abstractϕ
    "Mean vector"
    μ::Vector{Float64}
    "Upper triangular component of cholesky decomposition of covariance matrix."
    U::UpperTriangular{Float64, Matrix{Float64}}
    "Dimension of the multivariate normal distribution."
    dim::Int64
    MeanCholeskyMvn(μ, U) = new(μ, U, length(μ))
end








# TODO use views? Difficult as triangular matrix is different to reshaping the corresponding ϕ elements and any cat opperations seem to copy.
# struct MeanCholeskyMvn2{T1, T2} <: Abstractϕ
#     "Parameters flattened to a vector"
#     ϕ::Vector{Float64}
#     "Mean vector"
#     μ::T1
#     "Upper triangular component of cholesky decomposition of covariance matrix."
#     U::T2
#     "Dimension of the multivariate normal distribution."
#     dim::Int64
#     function MeanCholeskyMvn2(
#         μ::AbstractVector{Float64},
#         U::UpperTriangular{Float64, Matrix{Float64}})
#         @argcheck length(μ) == size(U, 1)
#         dim = length(μ)
#         ϕ = vcat(μ, vectorize(U))
#         μ = @view ϕ[1:dim]
#         U = vec(U)
#         U = UpperTriangular(reshape(U, dim, dim))    
#         new{typeof(μ), typeof(U)}(μ, U, dim)
#     end
# end
