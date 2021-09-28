"""
Base learners are used to predict the negative gradient vector. These must all
have the methods fit! and predict. Note that the same model is used multiple,
so it must be safe to call the fit! method multiple times with different data.
"""
abstract type BaseLearner end

"""
Base learner using polynomial transform. By defualt, with `use_cache = true`,
the model will cache the feature vectors along with QR decompositions of the
polynomial transforms to avoid recalculation. This will speed up training at the
cost of increased memory usage. An IdDict is used as the cache.
$(SIGNATURES)
"""
struct PolyBaseLearner <: BaseLearner
    degree::Int64
    β::Vector{Float64}
    use_cache::Bool
    _cache::IdDict{Any, LinearAlgebra.QRCompactWY{Float64, Matrix{Float64}}}
end

function PolyBaseLearner(degree; use_cache=true)
    PolyBaseLearner(degree, fill(NaN, degree+1), use_cache, IdDict())
end


"""
Fit the Base learner model to the negative gradient vector uⱼ using feature vector θ.
$(SIGNATURES)
"""
function fit!(
    base_learner::PolyBaseLearner,
    θ::AbstractVector,
    u::AbstractVector)
    @unpack degree, β, use_cache, _cache = base_learner
    local qr_poly_θ
    if use_cache & haskey(_cache, θ)
        qr_poly_θ = _cache[θ]
    else
        poly_θ = [θ.^p for p in 0:base_learner.degree]
        poly_θ = reduce(hcat, poly_θ)
        qr_poly_θ = qr(poly_θ)
        if use_cache
            _cache[θ] = qr_poly_θ
        end
    end
    
    β .= qr_poly_θ \ u
    return base_learner
end

"""
Predict the negative gradient vector using θ.
$(SIGNATURES)
"""
function predict(base_learner::PolyBaseLearner, θ::AbstractVector)
    poly_θ = [θ.^p for p in 0:base_learner.degree]
    poly_θ = reduce(hcat, poly_θ)
    return poly_θ * base_learner.β
end
