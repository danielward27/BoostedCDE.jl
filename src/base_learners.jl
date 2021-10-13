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
cost of increased memory usage. Note, an IdDict is used as the cache, which
requires === between objects to be recognised as keys.
$(SIGNATURES)
"""
struct PolyBaseLearner <: BaseLearner
    degree::Int
    β::Vector{Float64}
    use_cache::Bool
    _cache::IdDict{Any,
        NamedTuple{(:poly, :qr), Tuple{Matrix{Float64},
        LinearAlgebra.QRCompactWY{Float64, Matrix{Float64}}}}}
end


function PolyBaseLearner(degree; use_cache=false)
    PolyBaseLearner(degree, fill(NaN, degree+1), use_cache, IdDict())
end


"""
Fit the Base learner model to the negative gradient vector uⱼ using feature vector x.
$(SIGNATURES)
"""
function fit!(
    base_learner::PolyBaseLearner,
    x::AbstractVector,
    u::AbstractVector)
    @unpack degree, β, use_cache, _cache = base_learner
    if use_cache & haskey(_cache, x)
        qr_poly_x = _cache[x].qr
    else
        poly_x = [x.^p for p in 0:base_learner.degree]
        poly_x = reduce(hcat, poly_x)
        qr_poly_x = qr(poly_x)
        if use_cache
            _cache[x] = (poly=poly_x, qr=qr_poly_x)
        end
    end
    β .= qr_poly_x \ u
    return base_learner
end


"""
Predict the negative gradient vector using x.
$(SIGNATURES)
"""
function predict(base_learner::PolyBaseLearner, x::AbstractVector)
    @unpack degree, _cache, use_cache = base_learner
    if use_cache & haskey(_cache, x)
        poly_x = _cache[x].poly
    else
        poly_x = reduce(hcat, [x.^p for p in 0:degree])
    end            
    return poly_x * base_learner.β
end


# This might be the way to fix it. Then base learners still can stay basic.
# wrap the boosting model to take x and k, maybe m too?
# Can we dispatch on m?
# function predict(base_learner::AbstractBaseLearner, x::AbstractMatrix, k::Int)
#     # hash based on x and k
#     # if seen before use results.
#     # if not don't use saved results.
# end



