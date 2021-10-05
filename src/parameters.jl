
abstract type Abstractϕ end

"""
Parameterization of multivariate normal distribution with mean vector μ and the
upper triangular component of the cholesky decomposition of the covariance
matrix.
"""
struct MeanCholeskyMvn <: Abstractϕ
    "Mean vector."
    μ::Vector{Float64}
    """Upper triangular matrix."""
    U::UpperTriangular{Float64, Matrix{Float64}}
    "Dimension of the multivariate normal distribution."
    d::Int64
    MeanCholeskyMvn(μ, U) = begin
        d = length(μ)
        d==size(U, 1) || throw(ArgumentError("Inconsistent argument dimensions."))
        return new(μ, U, d)
    end
end

MeanCholeskyMvn(v::AbstractVector{Float64}) = begin
    a = 9 + 8*length(v)
    a == isqrt(a)^2 || throw(ArgumentError("Invalid length for v."))
    d = (-3 + isqrt(a)) ÷ 2
    μ = v[1:d]
    U = @view v[d+1:end]
    U = unvectorize(UpperTriangular, U)
    MeanCholeskyMvn(μ, U)
end

