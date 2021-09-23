"""
The boosting algorithm, and related functions.
"""

"""
Constructs a BoostingModel that can be trained using [`boost`]@ref
"""
struct BoostingModel{T1 <: ParamTuple, T2 <: BaseLearnerTuple}   # TODO do we really need a class for this? We could 
    "Initial ouput parameter predictions. Same for all datapoints."
    init_ϕ::T1
    "Base_learners matching the sizes of ϕ."
    base_learners::T2
    "Step length."
    sl::Float64
    "Base learners selected during training. Leave to defualt if untrained."
    base_learners_selected::Vector{BaseLearner}  # TODO Vector of abstract type generally not good practice? Could infer union type from base_learners.
    "The indices corresponding to the selected base learners (j=ϕ tuple idx, k=element of ϕ[j], l=θ idx))."
    jkl::Vector{Tuple{Int64, Int64, Int64}}

    function BoostingModel(init_ϕ, base_learners, sl=0.1)
        @argcheck length(init_ϕ) == length(base_learners)
        @argcheck all(size(blⱼ) == size(ϕⱼ) for (blⱼ, ϕⱼ) in zip(base_learners, init_ϕ))
        new{typeof(init_ϕ), typeof(base_learners)}(
            init_ϕ, base_learners, sl, BaseLearner[], Tuple{Int64, Int64, Int64}[])
    end
end

"""
Predict using the boosting model to get a vector of distributional parameters.
Computes using no cached results (e.g. from previous iterations during
training).
"""
function predict(model::BoostingModel, θ::AbstractMatrix{Float64}, x::AbstractMatrix{Float64})
    @argcheck size(θ, 1) == size(x, 1)
    @unpack base_learners_selected, sl, init_ϕ, jkl = model
    N = size(x, 1)
    ϕ = [deepcopy(init_ϕ) for _ in 1:N]

    for (bl, (j, k, l)) in zip(base_learners_selected, jkl)
        ûⱼₖₗ = predict(bl, θ[:, l])
        [ϕ[i][j][k] .+= sl*ûⱼₖₗ[i] for i in 1:N]
    end

    return ϕ
end


# TODO Just shift the gradient inside the loop? Bigger issue ϕ as Vector of tuples is causing issues?
# Loop over the N, splat the tuple and add the gradients seems the neatest solution?



# """
# Step the boosting model. Can we have a cached previous prediction? Or multiple dispatch where in one we provide previous prediction?
# """
# function step!(model::BoostingModel, θ::Matrix{Float64}, x::Matrix{Float64}, loss::Function)
#     @unpack init_ϕ, base_learners, sl, base_learners_selected, jkl = model
#     u = -gradient(() -> loss(ϕ, x), params(ϕ))[ϕ]

#     for j in 1:length(base_learners)
#         for k in 1:length(base_learners[j])
#             blⱼₖ = base_learners[j][k]
            
#             for l in 1:size(θ, 2)  # Loop over simulator parameters
#                 fit!(blⱼₖ, θ[:, l], u[:, k])  # TODO We may need to copy/deepcopy the models? Unless refitting is always fine?
#                 ûₖ = predict(blₖ, θ[:, j])
#                 ϕ_proposed = copy(ϕ)  # how to update
#                 ϕ_proposed[:, k] = ϕ_proposed[:, k] + sl*ûₖ
#                 lossⱼ = loss(x, ϕ_proposed)

#                 if lossⱼ < best_loss
#                     best = (loss = lossⱼ, bl = deepcopy(blₖ), kj = (k, j))
#                     ϕ = ϕ_proposed
#                 end
#             end

#         push!(selected_base_learners, best.bl)  # TODO Check if push! limits performance (probably shouldn't?)
#         push!(kj, best.kj)
#         push!(lossₘ, best.loss)
#     end

# end


# # TODO original guess should be sample mean and covariance matrix
# # TODO best if we can just provide a step! function e.g. for easy early stopping etc?
# function boost!(
#     model::BoostingModel,
#     θ::Matrix{Float64},
#     x::Matrix{Float64},
#     loss::Function,
#     steps::Int)
#     @unpack init_ϕ, base_learners, sl, base_learners_selected, jkl = model
    

#     ϕ = deepcopy(init_ϕ)  # TODO Would copy be sufficient?
    
#     @argcheck size(θ, 1) == size(x, 1)
    

#     best = (loss = loss(ϕ, x), bl = nothing, kj = nothing)

#     for m in 1:steps
#         u = -gradient(() -> loss(ϕ, x), params(ϕ))[ϕ]

#         ϕ_flat, re = Flux.destructure(ϕ)
#         models_flat = Flux.destructure(models)
#         ϕ_params = [params(ϕᵢ) for ]  # destructure parameters and models into vector.

            
#         for k_outer in 1:length(base_learners)  # Loop over ϕ groups then over ϕ
#             for k_inner in dsads
#                 blₖ = base_learner_array[k]
                
#                 for j in 1:size(θ, 2)  # Loop over simulator parameters
#                     fit!(blₖ, θ[:, j], u[:, k])
#                     ûₖ = predict(blₖ, θ[:, j])
#                     ϕ_proposed = copy(ϕ)  # how to update
#                     ϕ_proposed[:, k] = ϕ_proposed[:, k] + sl*ûₖ
#                     lossⱼ = loss(x, ϕ_proposed)

#                     if lossⱼ < best_loss
#                         best = (loss = lossⱼ, bl = deepcopy(blₖ), kj = (k, j))
#                         ϕ = ϕ_proposed
#                     end
#                 end

#             push!(selected_base_learners, best.bl)  # TODO Check if push! limits performance (probably shouldn't?)
#             push!(kj, best.kj)
#             push!(lossₘ, best.loss)
#         end
#     end
#     return model
# end

