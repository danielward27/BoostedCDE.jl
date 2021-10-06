# TODO Update this documentation
# """ 
# Loss functions. The loss function is differentiated with Zygote.jl. It
# should take a two matrices each with n rows (number of
# simulations/observations): ϕ contains the distributional parameters, and x
# contains the observed/simulated points. The loss function should return a single
# value i.e. the loss should be accumulated over the batch. Note ForwardDiff has
# some requirements to be compatible, e.g. too strict types such as enforcing
# ϕ::AbstractMatrix{Float64}, will not work. See ReverseDiff documentation for
# more detail.
# """
"""
Loss function, if matrices used reduced using mean.
"""
function loss(
    T::Type{<:Abstractϕ},
    ϕv::AbstractMatrix{Float64},
    x::AbstractMatrix{Float64})
    l = sum(loss.(T, eachrow(ϕv), eachrow(x)))/size(ϕv,1)
    l
end

function loss(
    T::Type{<:Abstractϕ},
    ϕv::AbstractVector{Float64},
    x::AbstractVector{Float64})
    ϕ = T(ϕv)
    loss(ϕ, x)
end

function loss(ϕ::MeanCholeskyMvn, x::AbstractVector{Float64})
    @unpack μ, U, d = ϕ
    Λ = Symmetric(U'U)    
    log_det_Σ = -2*sum(log.(diag(U)))
    return (1/2)*(d*log(2π) + log_det_Σ + (x-μ)'Λ*(x-μ))
end
