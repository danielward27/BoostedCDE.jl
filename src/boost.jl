"""
Constructs a BoostingModel that can be trained using [`boost`]@ref
"""
@kwdef mutable struct BoostingModel
    "The base_learners defining a model for each ϕ group."
    bl_options::Vector{BaseLearner}
    "Base learners selected during training. Leave to defualt if untrained."
    bl_selected::Vector{BaseLearner} = BaseLearner[]
    "Step length."
    sl::AbstractFloat = 0.1
    "The indices corresponding to the selected base learners."
    kj::Vector{Tuple{Int64, Int64}} = Tuple{Int64, Int64}[]
    "The loss after each iteration."
    lossₘ::Vector{AbstractFloat} = Vector{AbstractFloat}[]
end


# function predict(model::BoostingModel, init_ϕ::Abstractϕ, x::AbstractMatrix{Float64})
#     @unpack selected_base_learners, sl = model
#     N = size(x, 1)

#     ϕ = init_ϕ(model)  # This could very well be a feature of 
#     for k in 1:length(selected_base_learners)
#         û = predict(selected_base_learners, θ)

#     end

# end


# """
# Assume centred variables for now. Then original guess can be standard normal
#     base_learner_array should match size of parameters ϕ×N
# """
# function boost!(
#     model::BoostingModel,
#     θ::Matrix{Float64},
#     x::Matrix{Float64},
#     init_ϕ::Vector{AbstractOuput}
#     loss::Function,
#     steps::Int)
#     @unpack bl_options, bl_selected, sl, selected_base_learners, kj, lossₘ = model
#     @argcheck size(θ, 1) == size(x, 1)
#     @argcheck length(Flux.params(init_ϕ)) == length(bl_options)

#     ϕ = copy(init_ϕ)
#     best = (loss = loss(ϕ, x), bl = nothing, kj = nothing)

#     for m in 1:steps
#         u = -gradient(() -> loss(ϕ, x), params(ϕ))[ϕ]

#         ϕ_flat, re = Flux.destructure(ϕ)
#         models_flat = Flux.destructure(models)
#         ϕ_params = [params(ϕᵢ) for ]  # destructure parameters and models into vector.

            
#         for k in 1:length(bl_options)  # Loop over ϕ groups then over ϕ
#             blₖ = base_learner_array[k]
            
#             for j in 1:size(θ, 2)  # Loop over simulator parameters
#                 fit!(blₖ, θ[:, j], u[:, k])
#                 ûₖ = predict(blₖ, θ[:, j])
#                 ϕ_proposed = copy(ϕ)  # how to update
#                 ϕ_proposed[:, k] = ϕ_proposed[:, k] + sl*ûₖ
#                 lossⱼ = loss(x, ϕ_proposed)

#                 if lossⱼ < best_loss
#                     best = (loss = lossⱼ, bl = deepcopy(blₖ), kj = (k, j))
#                     ϕ = ϕ_proposed
#                 end
#             end

#             push!(selected_base_learners, best.bl)  # TODO Check if push! limits performance (probably shouldn't?)
#             push!(kj, best.kj)
#             push!(lossₘ, best.loss)
#         end
#     end
#     return model
# end

