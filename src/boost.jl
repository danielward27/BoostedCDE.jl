"""
The boosting algorithm, and related functions.
"""

"""
Constructs a BoostingModel that can be trained using [`boost!`](@ref)
"""
struct BoostingModel{T <: Vector{<: BaseLearner}}
    "Initial ouput parameter predictions, same for all predictions."
    init_ϕ::Vector{Float64}
    "Base_learners matching the length of ϕ."
    base_learners::T
    "Step length."
    η::Float64
    "Base learners selected during training."
    base_learners_selected::T
    "The indices corresponding to the selected base learners (j=ϕ tuple idx, k=element of ϕ[j], l=θ idx))."
    jk::Vector{Tuple{Int64, Int64}}

    BoostingModel(init_ϕ, base_learners; η=0.1) = begin
        length(init_ϕ) == length(base_learners) || throw(ArgumentError("Mismatch between ϕ length and number of base learners."))
        new{typeof(base_learners)}(
            init_ϕ, base_learners, η, BaseLearner[],
            Tuple{Int64, Int64}[])
    end
end

"""
"Reset" the boosting model, removing all the selected base learners and corresponding indices.
"""
function reset!(model::BoostingModel)
    @unpack base_learners_selected, jk = model
    [deleteat!(x, 1:length(x)) for x in [base_learners_selected, jk]]
end

"""
Predict using the boosting model to get the conditional distributional
parameters. During training, the ϕ matrix from previous iteration can be
provided to avoid recalculating.
"""
function predict(model::BoostingModel, θ::AbstractMatrix{Float64})
    @unpack base_learners_selected, η, init_ϕ, jk = model
    N, J = size(θ, 1), length(init_ϕ)
    ϕ = zeros(N, J) .+ init_ϕ'
    for (bl, (j, k)) in zip(base_learners_selected, jk)
        ûⱼₖ = predict(bl, θ[:, k])
        ϕ[:, j] .-= η*ûⱼₖ
    end
    return ϕ
end


function predict(
    model::BoostingModel,
    θ::AbstractMatrix{Float64},
    last_ϕ::AbstractMatrix{Float64})
    @unpack base_learners_selected, η, jk = model
    ϕ = last_ϕ
    j, k = jk[end]
    bl = base_learners_selected[end]
    ûⱼₖ = predict(bl, θ[:, k])
    ϕ[:, j] .-= η*ûⱼₖ
    return ϕ
end


"""
Step the boosting model, by adding on a single new base model that minimizes the
inner loss. The gradient! function should take ϕ and x and return the gradient
matrix matching the shape of ϕ. Note gradient! may (or may not) be mutating
depending on the definition (e.g. if using tapes with ReverseDiff it would
mutate the tape).

$(SIGNATURES)
"""
function step!(
    model::BoostingModel,
    θ::AbstractMatrix{Float64},
    x::AbstractMatrix{Float64},
    ϕₘ::AbstractMatrix{Float64};
    loss::Function)
    @unpack base_learners, base_learners_selected, jk = model
    J = length(base_learners)

    K = size(θ, 2)
    u = ReverseDiff.gradient(ϕₘ -> loss(ϕₘ, x), ϕₘ)

    local best_bl, best_jk, best_update
    best_inner_loss = Inf
    for j in 1:J
        blⱼₖ = base_learners[j]
        uⱼ = @view u[:, j]
        for k in 1:K
            θₖ = @view θ[:, k]  # TODO Change from ID Dict? This creates a new view each step messing up IDDict. Could make step take in cols views as arguments but that seems a bit dumb?
            fit!(blⱼₖ, θₖ, uⱼ)
            ûⱼ = predict(blⱼₖ, θₖ)
            inner_lossⱼₖ = var(ûⱼ - uⱼ) - var(uⱼ)
            if inner_lossⱼₖ < best_inner_loss
                best_bl = deepcopy(blⱼₖ)
                best_jk = (j, k)
            end
        end
    end
    push!(base_learners_selected, best_bl)
    push!(jk, best_jk)
    return model
end


"""
Fit a boosting model by performing M steps. Returns the model,
predictions and losses from each training iteration.
$(SIGNATURES)
"""
function boost!(
    model::BoostingModel,
    θ::AbstractMatrix{Float64},
    x::AbstractMatrix{Float64};
    loss::Function,
    steps::Int)
    ϕₘ = predict(model, θ)  # ϕ₀ if untrained
    losses = zeros(steps)
    for m in 1:steps
        step!(model, θ, x, ϕₘ, loss = loss)
        ϕₘ = predict(model, θ, ϕₘ)
        losses[m] = loss(ϕₘ, x)
    end
    (model = model, ϕₘ = ϕₘ, loss=losses)
end

