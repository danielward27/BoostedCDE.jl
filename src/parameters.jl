
abstract type Abstractϕ end

"""
Parameterization of multivariate normal distribution with mean vector μ and the
upper triangular component of the cholesky decomposition of the covariance
matrix.
"""
struct MeanCholeskyMvn <: Abstractϕ
    "Parameters flattened to a vector"
    v::Vector{Float64}
    "Mean vector view of ϕ."
    μ::SubArray{Float64, 1, Vector{Float64}, Tuple{UnitRange{Int64}}, true}
    "Vectorised Upper triangular view of ϕ."
    U::SubArray{Float64, 1, Vector{Float64}, Tuple{UnitRange{Int64}}, true}
    "Dimension of the multivariate normal distribution."
    d::Int64 
    # Constructor from vector
    MeanCholeskyMvn(v::Vector{Float64}, d::Int64) = begin
        μ = @view v[1:d]
        U = @view v[d+1:end]
        new(v, μ, U, d)
    end
end

MeanCholeskyMvn(v::Vector{Float64}) = begin
    a = 9 + 8*length(v)
    @argcheck a == isqrt(a)^2
    d = (-3 + isqrt(a)) ÷ 2
    MeanCholeskyMvn(v, d)
end

MeanCholeskyMvn(
    μ::Vector{Float64},
    U::UpperTriangular{Float64}) = begin
        MeanCholeskyMvn(vcat(μ, vectorize(U)))
end

length(ϕ::MeanCholeskyMvn) = length(ϕ.v)
