"""
The boosting algorithm, and related functions.
"""

"""
Constructs a BoostingModel that can be trained using [`boost`]@ref
"""
struct BoostingModel{T <: Vector{<: BaseLearner}}   # TODO do we really need a class for this? We could 
    "Initial ouput parameter predictions. Same for all datapoints."
    init_ϕ::Vector{Float64}
    "Base_learners matching the length of ϕ."
    base_learners::T
    "Step length."
    sl::Float64
    "Base learners selected during training."
    base_learners_selected::T
    "The indices corresponding to the selected base learners (j=ϕ tuple idx, k=element of ϕ[j], l=θ idx))."
    jk::Vector{Tuple{Int64, Int64}}

    function BoostingModel(init_ϕ, base_learners; sl=0.1)
        @argcheck length(init_ϕ) == length(base_learners)
        new{typeof(base_learners)}(
            init_ϕ, base_learners, sl, BaseLearner[],
            Tuple{Int64, Int64, Int64}[])
    end
end

"""
Predict using the boosting model to get the conditional distributional parameters.
"""
function predict(model::BoostingModel, θ::AbstractMatrix{Float64})
    @unpack base_learners_selected, sl, init_ϕ, jk = model
    N = size(θ, 1)
    ϕ = zeros(N, length(init_ϕ)) .+ init_ϕ'

    for (bl, (j, k)) in zip(base_learners_selected, jk)
        ûⱼₖ = predict(bl, θ[:, k])
        ϕ[:, j] .+= sl*ûⱼₖ
    end
    return ϕ
end


"""
Step the boosting model, by adding on a single new base model that minimizes the
loss, returning the corresponding predictions ϕₘ, and loss in a tuple.

$(SIGNATURES)
"""
function step!(
    model::BoostingModel,
    θ::Matrix{Float64},
    x::Matrix{Float64},
    ϕₘ::Matrix{Float64},
    loss::Function)
    @unpack base_learners, base_learners_selected, sl, jk = model
    u = -ForwardDiff.gradient(ϕₘ -> mvn_loss(ϕₘ, x), ϕₘ)::Matrix{Float64}  # N×j

    local best_bl, best_jk, best_ϕ
    best_loss = Inf

    for j in 1:length(base_learners)
        for k in 1:size(θ, 2)  # TODO how to cache poly_θ decomposition? Tricky as base learners may have different transformations, and each is fitted for all θ? Deepcopy could also complicate things. Restricting to Basis functions would provide a cleaner interface? How to handle regularisation still need to be careful. Maybe just an optional transformation or something? Maybe the vector of vectors aproach would allow more optimization? as you can assign the same base learner to a large number of parameters.  We need to cache k results such that each parameter transformation will be remembered after m=1, this would require k to be passed as an argument maybe?. Alteratively, we could try Memoize.jl?
            blⱼₖ = deepcopy(base_learners[j])
            fit!(blⱼₖ, θ[:, k], u[:, j])
            ûⱼ = predict(blⱼₖ, θ[:, k])
            ϕ_proposed = copy(ϕₘ)
            ϕ_proposed[:, j] = ϕ_proposed[:, j] + sl*ûⱼ
            lossⱼₖ = loss(ϕ_proposed, x)
            if lossⱼₖ < best_loss
                best_bl = deepcopy(blⱼₖ)
                best_jk = (j, k)
                best_ϕ = copy(ϕ_proposed)
                best_loss = lossⱼₖ
            end
        end
    end
    push!(base_learners_selected, best_bl)
    push!(jk, best_jk)
    return (ϕₘ = best_ϕ, loss = best_loss)
end


"""
Fit a boosting model by performing M steps. Returns the 
$(SIGNATURES)
"""
function boost!(
    model::BoostingModel,
    θ::Matrix{Float64},
    x::Matrix{Float64};
    loss::Function,
    steps::Int)
    ϕₘ = predict(model, θ)  # ϕ₀ if untrained
    losses = zeros(steps)
    for m in 1:steps
        ϕₘ, lossₘ = step!(model, θ, x, ϕₘ, loss)
        losses[m] = lossₘ
    end
    (ϕₘ = ϕₘ, loss=losses)
end

