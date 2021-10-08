
abstract type Abstractϕ end

"""
Struct denoting we wish to use a mean vector and cholesky decomposition of the
covariance matrix to parameterize a multivariate normal distribution of
dimension d.
"""
struct MeanCholeskyMvn <: Abstractϕ
    d::Int
end

"""
Get the parameters for the distribution from the vectorised form.
"""
function get_params(
    parameterisation::MeanCholeskyMvn,
    ϕ::AbstractVector{<: Real})
    @unpack d = parameterisation
    length(ϕ) == d*(d+1)÷2+d || throw(ArgumentError("length(ϕ) should be $(d*(d+1)÷2+d)."))
    μ = @view ϕ[1:d]
    U = @view ϕ[d+1:end]
    U = unvectorize(UpperTriangular, U)
    return (μ = μ, U = U)
end
