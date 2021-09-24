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

    function BoostingModel(init_ϕ, base_learners, sl=0.1)
        @argcheck length(init_ϕ) == length(base_learners)
        new{typeof(base_learners)}(
            init_ϕ, base_learners, sl, BaseLearner[],
            Tuple{Int64, Int64, Int64}[])
    end
end

"""
Predict using the boosting model to get a vector of distributional parameters.
Computes using no cached results (e.g. from previous iterations during
training).
"""
function predict(model::BoostingModel, θ::AbstractMatrix{Float64}, x::AbstractMatrix{Float64})
    @argcheck size(θ, 1) == size(x, 1)
    @unpack base_learners_selected, sl, init_ϕ, jk = model
    N = size(x, 1)
    ϕ = zeros(N, length(init_ϕ)) .+ init_ϕ'

    for (bl, (j, k)) in zip(base_learners_selected, jk)
        ûⱼₖ = predict(bl, θ[:, k])
        ϕ[:, j] .+= sl*ûⱼₖ
    end
    return ϕ
end


"""
Step the boosting model, by adding on a single new base model. Can we have a
cached previous prediction? Or multiple dispatch where in one we provide
previous prediction?

In addition to mutating the model, the updated predictions ϕ, and the associated
loss are returned in a tuple. Or cache them in the model?
"""
# TODO Test this.
function step!(model::BoostingModel, θ::Matrix{Float64}, x::Matrix{Float64}, ϕₘ₋₁::Matrix{Float64}, loss::Function)  # TODO Provide another method, where ϕₘ₋₁ is found through prediction?
    @unpack base_learners, sl, jk = model
    u = ForwardDiff.gradient(ϕ -> mvn_loss(ϕ, x), ϕ)::Matrix{Float64}  # N×j

    for j in 1:length(base_learners)
        for k in 1:size(θ, 2)
            blⱼₖ = deepcopy(base_learners[j])  # TODO I assume deepcopy is unavoidable? Can reduce if we allow refitting??
            fit!(blⱼₖ, θ[:, k], u[:, j])  
            ûⱼ = predict(blⱼₖ, θ[:, k])
            ϕ_proposed = copy(ϕₘ₋₁)
            ϕ_proposed[:, j] = ϕ_proposed[:, j] + sl*ûⱼ  # TODO clearer notation with u and ûⱼ
            lossⱼₖ = loss(ϕ_proposed, x)
            if lossⱼₖ < best_loss
                best = (bl = deepcopy(blⱼₖ), jk = (j, k),
                    ϕ = copy(ϕ_proposed), loss = lossⱼₖ)                
            end
        end

        push!(selected_base_learners, best.bl)  # TODO Check if push! limits performance (probably shouldn't?)
        push!(jk, best.jk)
        return best.ϕ, best.loss
    end
end


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

