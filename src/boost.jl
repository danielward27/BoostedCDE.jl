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
    "The indices corresponding to the selected base learners."
    idx::NamedTuple{(:ϕ, :x), Tuple{Vector{Int64}, Vector{Int64}}}

    BoostingModel(init_ϕ, base_learners; η=0.1) = begin
        length(init_ϕ) == length(base_learners) || throw(ArgumentError("Mismatch between ϕ length and number of base learners."))
        new{typeof(base_learners)}(
            init_ϕ, base_learners, η, BaseLearner[],
            (ϕ=Int[], x=Int[]))
    end
end

BoostingModel(;init_ϕ, base_learners, η=0.1) = BoostingModel(init_ϕ, base_learners; η)

"""
"Reset" the boosting model, removing all the selected base learners and corresponding indices.
"""
function reset!(model::BoostingModel)
    @unpack base_learners_selected, idx = model
    [empty!(a) for a in [base_learners_selected, idx[:ϕ], idx[:x]]]
    return model
end

"""
Predict using the boosting model to get the conditional distributional
parameters.
$(SIGNATURES)
"""
function predict(model::BoostingModel, x::AbstractMatrix{Float64})
    @unpack base_learners_selected, η, init_ϕ, idx = model
    ϕ = zeros(size(x, 1), length(init_ϕ)) .+ init_ϕ'

    for m in 1:length(base_learners_selected)
        ûⱼₖ = predict(base_learners_selected[m], x[:, idx[:x][m]])
        ϕ[:, idx[:ϕ][m]] .-= η*ûⱼₖ
    end
    return ϕ
end

"""
Predict using the boosting model to get the conditional distributional
parameters, using the ϕ matrix from previous training iteration to avoid
recalculating.
$(SIGNATURES)
"""
function predict(
    model::BoostingModel,
    x::AbstractMatrix{Float64},
    last_ϕ::AbstractMatrix{Float64})
    @unpack base_learners_selected, η, idx = model
    ϕ = last_ϕ
    bl = base_learners_selected[end]
    ûⱼₖ = predict(bl, x[:, idx[:x][end]])
    ϕ[:, idx[:ϕ][end]] .-= η*ûⱼₖ
    return ϕ
end

"""
Step the boosting model, by adding on a single new base model that minimizes the
gradient norm explained. u is the gradient matrix with shape matching ϕ, i.e.
N×J where N is the number observations/simulations, and J is the number of
distributional parameters. $(SIGNATURES)
"""
function step!(
    model::BoostingModel,
    x::AbstractMatrix{Float64},
    u::AbstractMatrix{Float64})
    @unpack base_learners, base_learners_selected, idx = model
    local best_bl, best_idx
    best_norm_explained = -Inf
    u_norms = norm.(eachcol(u))
    for j in sortperm(u_norms, rev=true)  # Biggest norms first
        # No need to fit base learner if perfect predictor can't do better
        u_norms[j] < best_norm_explained && continue
        bl = base_learners[j]
        uⱼ = @view u[:, j]

        for k in 1:size(x, 2)
            xₖ = @view x[:, k]
            fit!(bl, xₖ, uⱼ)
            ûⱼₖ = predict(bl, xₖ)
            norm_explained = u_norms[j] - norm(ûⱼₖ - uⱼ)  # (total-unexplained)
            if norm_explained > best_norm_explained
                best_bl = deepcopy(bl)
                best_idx = (ϕ=j, x=k)
                best_norm_explained = norm_explained
            end
        end
    end
    push!(base_learners_selected, best_bl)
    push!(idx[:x], best_idx[:x])
    push!(idx[:ϕ], best_idx[:ϕ])
    return model
end


# TODO Update docs below
"""
As for [`step!`](@ref), but without skipping training base models where the norm
explained cannot be improved. x is an iterator over the columns of the x matrix
i.e. `eachcol(x)` (using the iterator allows easier of caching of results).
"""
function step_naive!(
    model::BoostingModel,
    x::AbstractMatrix{Float64},
    u::AbstractMatrix{Float64})
    @unpack base_learners, base_learners_selected, idx = model
    local best_bl, best_idx
    best_norm_explained = -Inf
    for j in 1:length(base_learners)
        bl = base_learners[j]
        uⱼ = @view u[:, j]
        for k in 1:size(x, 2)
            xₖ = @view x[:, k]  # TODO Change from ID Dict? This creates a new view each step messing up IDDict. Could make step take in cols views as arguments but that seems a bit dumb?
            fit!(bl, xₖ, uⱼ)
            ûⱼₖ = predict(bl, xₖ)
            norm_explained = norm(ûⱼₖ) - norm(ûⱼₖ - uⱼ)
            if norm_explained > best_norm_explained
                best_bl = deepcopy(bl)
                best_idx = (ϕ=j, x=k)
                best_norm_explained = norm_explained
            end
        end
    end
    push!(base_learners_selected, best_bl)
    push!(idx[:x], best_idx[:x])
    push!(idx[:ϕ], best_idx[:ϕ])
    return model
end


"""
Boosting with cross validation and patience. `loss` should take `ϕ` and `y`,
returning a scalar. `∇loss` should take `ϕ` (i.e. `y_train` should be
abstracted away), returning matrix with size matching `ϕ`. `data` should be an
object that can be unpacked/destructured to (x_train, x_val, y_train, y_val),
e.g. NamedTuple resulting from [`train_val_split`](@ref). Returns a NamedTuple
of results.

If ∇loss is not provided, by default, the gradient is found using ReverseDiff,
using ReverseDiff.GradientTape.
$(SIGNATURES)
"""
function boostcv!(
    model::BoostingModel,
    data::Any;
    steps::Int,
    loss::Function,
    ∇loss::Function = get_tape_∇(loss, predict(model, data.x_train), data.y_train),
    step!::Function = step!,
    max_patience::Int=5)
    @unpack x_train, x_val, y_train, y_val = data
    train = (ϕₘ=predict(model, x_train), loss=zeros(steps), x=x_train, y=y_train)
    val = (ϕₘ=predict(model, x_val), loss=zeros(steps), x=x_val, y=y_val)
    patience = 0
    for m in 1:steps
        u = ∇loss(train.ϕₘ)    
        step!(model, train.x, u)
        for tv in (train, val)
            tv.ϕₘ .= predict(model, tv.x, tv.ϕₘ)
            tv.loss[m] = loss(tv.ϕₘ, tv.y)
        end
        patience = m > 1 && val.loss[m] > val.loss[m-1] ? patience + 1 : 0
        if patience == max_patience
            print("Max patience ($(max_patience)) reached on iteration $(m).")
            if m != steps
                [deleteat!(l, (m+1):length(l)) for l in (train.loss, val.loss)]
            end
            break
        end 
    end
    return (model = model, train = train, val = val)
end

"""
Gradient function with tape, specialised on x and y (usually training data). 
"""
function get_tape_∇(
    loss::Function,
    ϕₘ::AbstractMatrix{Float64},
    y::AbstractMatrix{Float64})
    tape = GradientTape(ϕₘ -> loss(ϕₘ, y), ϕₘ)
    tape = ReverseDiff.compile(tape)
    return ϕₘ -> gradient!(tape, ϕₘ)
end

"""
Minimal function to fit a boosting model.
`loss` should take ϕ and y as arguments, and ∇loss should just take ϕ.
$(SIGNATURES)
"""
function boost!(
    model::BoostingModel,
    x::AbstractMatrix{Float64},
    y::AbstractMatrix{Float64};
    steps::Int,
    loss::Function,
    ∇loss::Function = get_tape_∇(loss, predict(model, x), y),
    step!::Function = step!)
    ϕₘ = predict(model, x)  # ϕ₀ if untrained
    losses = zeros(steps)
    for m in 1:steps
        u = ∇loss(ϕₘ)
        step!(model, x, u)
        ϕₘ = predict(model, x, ϕₘ)
        losses[m] = loss(ϕₘ, y)
    end
    (model = model, ϕₘ = ϕₘ, loss=losses)
end
