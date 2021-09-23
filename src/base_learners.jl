"""
Base learners are used to predict the negative gradient vector. These must all
have the methods fit! and predict.
"""
abstract type BaseLearner end
const BaseLearnerTuple = NamedTuple{Vararg{AbstractArray{<: BaseLearner}}}

mutable struct PolyBaseLearner <: BaseLearner  # TODO Support precalculating transform and matrix decomposition somehow?
    degree::Int
    β::AbstractVector
    PolyBaseLearner(degree) = new(degree)
    PolyBaseLearner(; degree=2) = new(degree)
end

function fit!(base_learner::PolyBaseLearner, θ::AbstractVector, u::AbstractVector)
    poly_θ = [θ.^p for p in 0:base_learner.degree]  # TODO Save decomposition for θ. Then just need to check for equality to previous theta and reuse cached version?
    poly_θ = hcat(poly_θ...)
    base_learner.β = poly_θ \ u
    return base_learner
end

function predict(base_learner::PolyBaseLearner, θ::AbstractVector)
    poly_θ = [θ.^p for p in 0:base_learner.degree]
    poly_θ = hcat(poly_θ...)
    return poly_θ * base_learner.β
end


"""
Base learner used at initialization to predict a constant value. Additional
arguments can be provided to fit! for consistency with other base learners but
these are ignored.
"""
struct ConstBaseLearner{T<:Tuple{Vararg{AbstractVector{Real}}}} <: BaseLearner
    "The constant value to be returned."
    ϕ::T
end

function fit!(base_learner::ConstBaseLearner,  _::AbstractVector{Real}, _::AbstractVector{Real})
    return base_learner
end

function predict(base_learner::ConstBaseLearner, θ::AbstractMatrix{Real})
    return fill(base_learner.ϕ, size(θ, 1))
end
