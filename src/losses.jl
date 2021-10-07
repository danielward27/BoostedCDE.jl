"""
Loss functions. The losses should be differentiable with ReverseDiff using a
GradientTape, i.e. types for ϕ must be general (e.g. <: Real), and the loss
should not have branches depending on the input.
"""


"""
Calculate the loss given a parameterisation specified by Abstractϕ. If matrices
are used, reduction is carried out using summation.
"""
function loss(
    parameterisation::Abstractϕ,
    ϕ::AbstractMatrix{<: Real},
    x::AbstractMatrix{<: Real}
    )
    l = map((ϕᵢ, xᵢ) -> loss(parameterisation, ϕᵢ, xᵢ), eachrow(ϕ), eachrow(x))
    return sum(l)
end

function loss(
    parameterisation::MeanCholeskyMvn,
    ϕ::AbstractVector{<: Real},
    x::AbstractVector{<: Real}
    )
    @unpack d = parameterisation
    μ, U = get_params(parameterisation, ϕ)
    Λ = U'U
    log_det_Σ = -2*sum(log.(diag(U)))
    return (1/2)*(d*log(2π) + log_det_Σ + (x-μ)'Λ*(x-μ))
end







# function five_dim_loss(ϕ::AbstractMatrix{<: Real}, x::AbstractMatrix{<: Real})
#     μs = @view ϕ[:, 1:2]
#     Us = @view ϕ[:, 3:end]
#     d=2
#     ll = 0
#     for (xᵢ, μ, U) in zip(eachrow(x), eachrow(μs), eachrow(Us))
#         U = unvectorize(UpperTriangular, U)
#         Λ = U'U
#         log_det_Σ = -2*sum(log.(diag(U)))
#         ll += (1/2)*(d*log(2π) + log_det_Σ + (xᵢ-μ)'Λ*(xᵢ-μ))
#     end
#     return -ll
# end

