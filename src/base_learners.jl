"""
Base learners are used to predict the negative gradient vector. These must all
have the methods fit! and predict.
"""
abstract type BaseLearner end

"""
Base learners are used to predict the negative gradient vector. These must all
have the methods fit! and predict. $(SIGNATURES)

By defualt, with `use_cache = true`, the model will cache the views of the
feature vectors along with QR decompositions of the polynomial transforms
to avoid recalculation. This will speed up training at the cost of increased
memory usage. The cache is an IdDict with views of θ as the keys.
"""
struct PolyBaseLearner <: BaseLearner
    degree::Int64
    β::Vector{Float64}
    use_cache::Bool
    _cache::IdDict{
        SubArray{Float64, 1, Matrix{Float64}, Tuple{Base.Slice{Base.OneTo{Int64}}, Int64}, true},
        LinearAlgebra.QRCompactWY{Float64, Matrix{Float64}}}
end


function PolyBaseLearner(degree; use_cache=true)
    _cache = IdDict{
        SubArray{Float64, 1, Matrix{Float64}, Tuple{Base.Slice{Base.OneTo{Int64}}, Int64}, true},
        Cholesky{Float64, Matrix{Float64}}}()
    PolyBaseLearner(degree, fill(NaN, degree+1), use_cache, _cache)
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
        print("cacheing")
    else
        poly_θ = [θ.^p for p in 0:base_learner.degree]
        poly_θ = reduce(hcat, poly_θ)
        qr_poly_θ = qr(poly_θ)
        # use_cache ? _cache[θ] = qr_poly_θ : nothing
        if use_cache
            if θ isa keytype(typeof(_cache))
                _cache[θ] = qr_poly_θ
            else
                @warn "Invalid keytype"
            
            
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
