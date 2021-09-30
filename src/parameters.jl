
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








struct MeanCholeskyMvn2{T} <: Abstractϕ
    "Parameters flattened to a vector"
    ϕ::Vector{Float64}
    "Mean vector view of ϕ"
    μ::T
    "Upper triangular vector view of ϕ."
    U::T
    "Dimension of the multivariate normal distribution."
    dim::Int64
    function MeanCholeskyMvn2(
        μ::AbstractVector,
        U::UpperTriangular)
        U = vectorize(U)
        dim = length(μ)
        ϕ = vcat(μ, U)
        μ = @view ϕ[1:dim]
        U = @view ϕ[dim+1:end]
        new{typeof(μ)}(ϕ, μ, U, dim)
    end
end

get_μ(ϕ::MeanCholeskyMvn2) = ϕ.μ
get_U(ϕ::MeanCholeskyMvn2) = unvectorize(UpperTriangular, ϕ.U)

