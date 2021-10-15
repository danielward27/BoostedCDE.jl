
abstract type Parameterisation end


"""
Struct denoting we wish to use a mean vector and cholesky decomposition of the
covariance matrix to parameterize a multivariate normal distribution of
dimension d.
"""
struct MeanCholeskyMvn <: Parameterisation
    d::Int
end

"""
Calculate the expected length of the flattened parameter vector.
"""
expt_length(ϕ::MeanCholeskyMvn) = begin
    d = ϕ.d
    return d+(d*d+d) ÷ 2
end

"""
Get the parameters for the distribution from the vectorised form.
"""
function get_params(
    parameterisation::MeanCholeskyMvn,
    ϕ::AbstractVector{<: Real})
    @unpack d = parameterisation
    length(ϕ) == expt_length(parameterisation) || throw(ArgumentError("length(ϕ) should be $(d*(d+1)÷2+d)."))
    μ = @view ϕ[1:d]
    U = @view ϕ[d+1:end]
    U = unvectorize(UpperTriangular, U)
    return (μ = μ, U = U)
end
