
# Parameters must be vector
# Base learner corresponding vector

struct BoostingModel  # These could all be passed to the function 
    init_ϕ::Vector{AbstractFloat}
    base_learners::Vector{BaseLearner}
    sl::AbstractFloat
    selected_base_learners::Vector{BaseLearner} = BaseLearner[]
    kj::Vector{Tuple{Int64, Int64}} = Tuple{Int64, Int64}[]

    function BoostingModel(
        init_ϕ, base_learners, sl,
        selected_base_learners, kj)
        @argcheck length(init_ϕ) == length(base_learners)
        return new(init_ϕ, base_learners, sl, selected_base_learners, kj)
end


BoostingModel

function boost(model::BoostingModel, x, θ, loss, m_max, sl=0.1)
    """
    Assume centred variables for now. Then original guess can be standard normal
        base_learner_array should match size of parameters ϕ×N
    """
    @unpack init_ϕ, base_learners, sl, selected_base_learners, kj = model
    @argcheck length(selected_base_learners) == 0  # TODO We could support further training of pretrained models.
    @argcheck length(kj)  == 0
    ϕ = repeat(ϕ_init', size(x, 1)) 

    @assert size(ϕ) == (n, K)  "ϕ_init should be defined for all data points."
    
    bbl_vec = BestBaseLearner[]  # Best base learner vector
    for m in 1:m_max
        u = -gradient(() -> loss(x, ϕ), params(ϕ))[ϕ]
        best_loss = Inf
        best_bl = nothing
        
        for k in 1:length(base_learners)  # Loop over ϕ
            uₖ = u[:, k]
            bl = base_learner_array[k]
            
            for j in 1:size(θ, 2)
                fit!(bl, θ[:, j], uₖ)
                ûₖ = predict(bl, θ[:, j])
                ϕ_proposed = copy(ϕ)
                ϕ_proposed[:, k] = ϕ_proposed[:, k] + sl*ûₖ
                lossⱼ = sum(loss(x, ϕ_proposed))

                if lossⱼ < best_loss
                    
                    best_bl = BestBaseLearner(deepcopy(bl), lossⱼ, ûₖ, (k, j))     # TODO Can we get away with just copy? Maybe we need deepcopy as mutable β will change?
                end
            end
        end
        push!(bbl_vec, best_bl)
    end
    return bbl_vec
end

function negative_gaussian_likelihood_loss(x, ϕ)
    N = size(x,1)
    idx = size(ϕ, 2) ÷ 2
    μs = @view ϕ[:, idx]
    σs = @view ϕ[idx + 1, end]
    l = [logprob(MvNormal(μ[i, :], σ[i, :]), x[i, :]) for i in 1:N]
    return -l
end




