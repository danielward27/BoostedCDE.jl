"""
Constructs a BoostingModel that can be trained using [`boost`]@ref
"""
@kwdef mutable struct BoostingModel
    "The base_learners defining a model for each ϕ group."
    base_learners::Vector{BaseLearner}
    "Base learners selected during training. Leave to defualt if untrained."
    base_learners_selected::Vector{BaseLearner} = BaseLearner[]
    "Initial ouput parameter predictions."
    init_ϕ::Abstractϕ
    "Step length."
    sl::AbstractFloat = 0.1
    "The indices corresponding to the selected base learners."
    kj::Vector{Tuple{Int64, Int64}} = Tuple{Int64, Int64}[]  # TODO What is index and if using in predict then either need to flatten, or need to have another index for param group? 
    "The loss after each iteration."
    lossₘ::Vector{AbstractFloat} = Vector{AbstractFloat}[]
end

# TODO Test that this works t least in the untrained case.
function predict(model::BoostingModel, x::AbstractMatrix{Float64})
    @unpack selected_base_learners, sl, init_ϕ, kj = model
    N = size(x, 1)
    ϕ = [deepcopy(ϕᵢ) for _ in 1:N]  # TODO this could be allocated on model construction?

    for (bl, (j, k, l)) in zip(selected_base_learners, jkl)
        ûₖ = predict(bl, θ[:, l])
        [ϕ[i][j][k] .+= sl*ûₖ[j][k] for i in 1:N]
    end
    return ϕ
end



"""
Original guess should be sample covariance matrix.
"""
function boost!(
    model::BoostingModel,
    θ::Matrix{Float64},
    x::Matrix{Float64},
    loss::Function,
    steps::Int)
    @unpack base_learners, base_learners_selected, sl, kj, lossₘ, init_ϕ = model
    base_learners_tuple = Tuple(getfield(base_learners, f) for f in fieldnames(typeof(base_learners)))
    ϕ = deepcopy(init_ϕ)  # TODO Check if copy is sufficient.
    ϕ_tuple = Tuple(getfield(ϕ, f) for f in fieldnames(typeof(ϕ)))
    @argcheck size(θ, 1) == size(x, 1)
    @argcheck base_learners_tuple isa Tuple{Vararg{AbstractArray}}
    @argcheck ϕ_tuple isa Tuple{Vararg{AbstractArray}}
    @argcheck all(size(b) == size(ϕᵢ) for (b, ϕᵢ) in zip(base_learners_tuple, ϕ_tuple))

    best = (loss = loss(ϕ, x), bl = nothing, kj = nothing)

    for m in 1:steps
        u = -gradient(() -> loss(ϕ, x), params(ϕ))[ϕ]

        ϕ_flat, re = Flux.destructure(ϕ)
        models_flat = Flux.destructure(models)
        ϕ_params = [params(ϕᵢ) for ]  # destructure parameters and models into vector.

            
        for k_outer in 1:length(base_learners)  # Loop over ϕ groups then over ϕ
            for k_inner
                blₖ = base_learner_array[k]
                
                for j in 1:size(θ, 2)  # Loop over simulator parameters
                    fit!(blₖ, θ[:, j], u[:, k])
                    ûₖ = predict(blₖ, θ[:, j])
                    ϕ_proposed = copy(ϕ)  # how to update
                    ϕ_proposed[:, k] = ϕ_proposed[:, k] + sl*ûₖ
                    lossⱼ = loss(x, ϕ_proposed)

                    if lossⱼ < best_loss
                        best = (loss = lossⱼ, bl = deepcopy(blₖ), kj = (k, j))
                        ϕ = ϕ_proposed
                    end
                end

            push!(selected_base_learners, best.bl)  # TODO Check if push! limits performance (probably shouldn't?)
            push!(kj, best.kj)
            push!(lossₘ, best.loss)
        end
    end
    return model
end

