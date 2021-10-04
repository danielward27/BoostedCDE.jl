"""
The boosting algorithm, and related functions.
"""

"""
Constructs a BoostingModel that can be trained using [`boost!`](@ref)
"""
struct BoostingModel{T1 <: Abstractϕ, T2 <: Vector{<: BaseLearner}}
    "Initial ouput parameter predictions. Same for all datapoints."
    init_ϕ::T1
    "Base_learners matching the length of ϕ."
    base_learners::T2
    "Step length."
    sl::Float64
    "Base learners selected during training."
    base_learners_selected::T2
    "The indices corresponding to the selected base learners (j=ϕ tuple idx, k=element of ϕ[j], l=θ idx))."
    jk::Vector{Tuple{Int64, Int64}}

    function BoostingModel(init_ϕ, base_learners; sl=0.1)
        @argcheck length(vectorize(init_ϕ)) == length(base_learners)
        new{typeof(init_ϕ), typeof(base_learners)}(
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
    init_ϕ_v = vectorize(init_ϕ)
    ϕ = zeros(N, length(init_ϕ_v)) .+ init_ϕ_v'

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
    N = size(θ, 1)
    J = length(base_learners)
    u = Matrix{Float64}(undef, N, J)

    ϕₘ 
    for i in 1:N

    end
    u = Flux.gradient(Flux.params(ϕₘ)) do 
        batch_loss = 0.
        for (ϕᵢv, xᵢ)  in zip(eachrow(ϕₘ), eachrow(x))  # TODO time with column major opimized version?
            ϕᵢ = unvectorize_like(model.init_ϕ, ϕᵢvᵢ)
            batch_loss += loss(t, ϕᵢ, xᵢ)
        end
    end  # N×j
    # u = -ReverseDiff.gradient(ϕₘ -> mvn_loss(ϕₘ, x), ϕₘ)::Matrix{Float64}  # N×j

    local best_bl, best_jk, best_ϕ
    best_loss = Inf
    θ_cols = eachcol(θ)
    for j in 1:length(base_learners)
        blⱼₖ = base_learners[j]
        for (k, θₖ) in enumerate(θ_cols)
            fit!(blⱼₖ, θₖ, u[:, j])
            ûⱼ = predict(blⱼₖ, θₖ)
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
    steps::Int)
    ϕₘ = predict(model, θ)  # ϕ₀ if untrained
    losses = zeros(steps)
    for m in 1:steps
        ϕₘ, lossₘ = step!(model, θ, x, ϕₘ, loss)
        losses[m] = lossₘ
    end
    (ϕₘ = ϕₘ, loss=losses)
end

