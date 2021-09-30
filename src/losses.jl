"""
Loss functions. The loss function is differentiated with ReverseDiff.jl. It
should take a two matrices each with n rows (number of
simulations/observations): ϕ contains the distributional parameters, and x
contains the observed/simulated points. The loss function should return a single
value i.e. the loss should be accumulated over the batch. Note ForwardDiff has
some requirements to be compatible, e.g. too strict types such as enforcing
ϕ::AbstractMatrix{Float64}, will not work. See ReverseDiff documentation for
more detail.
"""

function loss(
    t::Type{MeanCholeskyMvn},
    ϕ::AbstractMatrix{<: Real},
    x::AbstractMatrix{Float64})
    @argcheck size(ϕ, 1) == size(x, 1)
    batch_loss = 0.

    for (ϕᵢ, xᵢ)  in zip(eachrow(ϕ), eachrow(x))  # TODO time with column major opimized version?
        batch_loss += _loss(t, ϕᵢ, xᵢ)
    end
    return batch_loss/size(x, 1)
end

function _loss(
    t::Type{MeanCholeskyMvn},
    ϕᵢ::AbstractVector{<: Real},
    xᵢ::AbstractVector{Float64})
    ϕ = unvectorize(t, ϕᵢ)
    @unpack μ, U = ϕ
    Λ = Symmetric(U'U)
    h = Λ*μ
    d = MvNormalCanon(h, Λ)
    return -logpdf(d, xᵢ)
end



"""
Negative log-probability of x using vector of parameters ϕ to parameterise the
means and cholesky decomposition of the normal distribution.
"""
function mvn_loss(ϕ::AbstractMatrix{<: Real}, x::AbstractMatrix{Float64})
    @argcheck size(ϕ, 1) == size(x, 1)
    N = size(x, 1)
    batch_loss = 0.
    for (ϕᵢ, xᵢ)  in zip(eachrow(ϕ), eachrow(x))
        d = mvn_d_from_ϕ(ϕᵢ)
        batch_loss += -logpdf(d, xᵢ)
    end
    return batch_loss/N
end

"""
Get the multivariate normal distribution from a ϕ vector, where the first
elements correspond to the mean, and the remaining elements correspond to the
upper triangular elements of the cholesky decomposition of the precision matrix,
listed columnwise.
"""
function mvn_d_from_ϕ(ϕᵢ::AbstractVector{<: Real})
    μ, U = μ_chol_splitter(ϕᵢ)
    Λ = Symmetric(U'U)
    h = Λ*μ
    return MvNormalCanon(h, Λ)
end





# """
# Negative log-probability of x using vector of parameters ϕ to parameterise the
# means and cholesky decomposition of the normal distribution.
# """
# function mvn_loss(μ::Vector{Float64}, u::Vector{Float64}, xᵢ::Vector{Float64})
    
    
#         μ, U = ϕ[i]
#         U = vec_to_triangular(U)
#         Λ = Symmetric(U'U)
#         h = Λ*μ
#         d = MvNormalCanon(h, Λ)
#         l[i] = logpdf(d, x[i, :])
    
    
# end

# length(ϕ) == size(x, 1) || throw(ArgumentError("length(ϕ) should match size(x,1)"))
# N = size(x, 1)
# l = Vector{Float64}(undef, N)
# for i in 1:N
# end
# return agg(-l)

