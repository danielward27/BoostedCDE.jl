"""
Abstract type representing the distribution parameters. Should have a vector
field v which can be used for updating the parameters.
"""
abstract type Abstractϕ end
length(ϕ::Abstractϕ) = length(ϕ.v)

"""
Parameterization of multivariate normal distribution with mean vector μ and the
upper triangular component of the cholesky decomposition of the covariance
matrix.
"""
struct MeanCholeskyMvn{T<:SubArray} <: Abstractϕ
    "Parameters flattened to a vector."
    v::Vector{Float64}
    "Mean vector (view of v)."
    μ::T
    """Upper triangular matrix, represented as a vector of vectors, with the vectors being the columns of length 1:d (views of v)."""
    U::Vector{T}
    "Dimension of the multivariate normal distribution."
    d::Int64
end

MeanCholeskyMvn(v::Vector{Float64}) = begin
    a = 9 + 8*length(v)
    b = isqrt(a)
    b^2 == a || throw(ArgumentError("Invalid vector length."))
    d = (-3 + b) ÷ 2
    μ = @view v[1:d]
    U = vecvec_triangular_view(@view v[d+1:end])
    return MeanCholeskyMvn(v, μ, U, d)
end

MeanCholeskyMvn(
    μ::Vector{Float64},
    U::UpperTriangular{Float64}) = begin
    length(μ) == size(U, 1) || throw(ArgumentError("μ and U dimensions do not match."))
    return MeanCholeskyMvn(vcat(μ, vectorize(U)))
end

