"""
Base learners are used to predict the negative gradient vector. These must all
have the methods preprocess, fit! and predict. Note that the same model is used
multiple times, so it must be safe to call the fit! method multiple times with
different data.
"""
abstract type BaseLearner end

"""
Base learner using polynomial regression. The polynomial transform and its QR
decomposition are cached for each x in an IdDict to avoid recalculating during
repreated fitting of the model. Note as an IdDict is used caching will
only work if object identity/egality holds.
$(SIGNATURES)
"""
struct PolyBaseLearner <: BaseLearner
    degree::Int
    β::Vector{Float64}
    _cache::IdDict{Any, Tuple{Matrix{Float64},
        LinearAlgebra.QRCompactWY{Float64, Matrix{Float64}}}}
end

function PolyBaseLearner(degree)
    PolyBaseLearner(degree, fill(NaN, degree+1), IdDict())
end

"""
Calculate the polynomial transform and the corresponding QR decomposition of x,
results are cached in an IdDict to avoid recalculation where possible.
"""
function preprocess!(
    base_learner::PolyBaseLearner,
    x::AbstractVector{Float64})
    @unpack degree, _cache = base_learner
    result = get!(_cache, x) do 
        x_poly = reduce(hcat, [x.^p for p in 0:base_learner.degree])
        return (x_poly, qr(x_poly))
    end
    return result
end


"""
Fit the Base learner model to the negative gradient vector u using x.
 $(SIGNATURES)
"""
function fit!(
    base_learner::PolyBaseLearner,
    x::AbstractVector{Float64},
    u::AbstractVector{Float64})
    _, x_qr = preprocess!(base_learner, x)
    base_learner.β .= x_qr \ u
    return base_learner
end

"""
Predict the negative gradient vector using x.
$(SIGNATURES)
"""
function predict(
    base_learner::PolyBaseLearner,
    x::AbstractVector{Float64})
    x_poly, _ = preprocess!(base_learner, x)
    return x_poly * base_learner.β
end
