var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = BoostedCDE","category":"page"},{"location":"#BoostedCDE","page":"Home","title":"BoostedCDE","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for BoostedCDE.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [BoostedCDE]","category":"page"},{"location":"#BoostedCDE.BaseLearner","page":"Home","title":"BoostedCDE.BaseLearner","text":"Base learners are used to predict the negative gradient vector. These must all have the methods fit! and predict. Note that the same model is used multiple, so it must be safe to call the fit! method multiple times with different data.\n\n\n\n\n\n","category":"type"},{"location":"#BoostedCDE.BoostingModel","page":"Home","title":"BoostedCDE.BoostingModel","text":"Constructs a BoostingModel that can be trained using [boost]@ref\n\n\n\n\n\n","category":"type"},{"location":"#BoostedCDE.PolyBaseLearner","page":"Home","title":"BoostedCDE.PolyBaseLearner","text":"Base learner using polynomial transform. By defualt, with use_cache = true, the model will cache the feature vectors along with QR decompositions of the polynomial transforms to avoid recalculation. This will speed up training at the cost of increased memory usage. An IdDict is used as the cache.\n\n\n\n\n\n","category":"type"},{"location":"#BoostedCDE.boost!-Tuple{BoostingModel, Matrix{Float64}, Matrix{Float64}}","page":"Home","title":"BoostedCDE.boost!","text":"Fit a boosting model by performing M steps. Returns the \n\nboost!(model, θ, x; loss, steps)\n\n\n\n\n\n\n","category":"method"},{"location":"#BoostedCDE.fit!-Tuple{PolyBaseLearner, AbstractVector{T} where T, AbstractVector{T} where T}","page":"Home","title":"BoostedCDE.fit!","text":"Fit the Base learner model to the negative gradient vector uⱼ using feature vector θ.\n\nfit!(base_learner, θ, u)\n\n\n\n\n\n\n","category":"method"},{"location":"#BoostedCDE.gaussian_simulator-Tuple{Random.AbstractRNG, AbstractVector{Float64}}","page":"Home","title":"BoostedCDE.gaussian_simulator","text":"Simulate a three dimensional Gaussian mean vector θ. The covariance is diagonal, and fixed to σ=0.1. Parameter vector θ is the mean vector of the Gaussian.\n\n\n\n\n\n","category":"method"},{"location":"#BoostedCDE.mvn_d_from_ϕ-Tuple{AbstractVector{var\"#s17\"} where var\"#s17\"<:Real}","page":"Home","title":"BoostedCDE.mvn_d_from_ϕ","text":"Get the multivariate normal distribution from a ϕ vector, where the first elements correspond to the mean, and the remaining elements correspond to the upper triangular elements of the cholesky decomposition of the precision matrix, listed columnwise.\n\n\n\n\n\n","category":"method"},{"location":"#BoostedCDE.mvn_loss-Tuple{AbstractMatrix{var\"#s2\"} where var\"#s2\"<:Real, AbstractMatrix{Float64}}","page":"Home","title":"BoostedCDE.mvn_loss","text":"Negative log-probability of x using vector of parameters ϕ to parameterise the means and cholesky decomposition of the normal distribution.\n\n\n\n\n\n","category":"method"},{"location":"#BoostedCDE.predict-Tuple{BoostingModel, AbstractMatrix{Float64}}","page":"Home","title":"BoostedCDE.predict","text":"Predict using the boosting model to get the conditional distributional parameters.\n\n\n\n\n\n","category":"method"},{"location":"#BoostedCDE.predict-Tuple{PolyBaseLearner, AbstractVector{T} where T}","page":"Home","title":"BoostedCDE.predict","text":"Predict the negative gradient vector using θ.\n\npredict(base_learner, θ)\n\n\n\n\n\n\n","category":"method"},{"location":"#BoostedCDE.step!-Tuple{BoostingModel, Matrix{Float64}, Matrix{Float64}, Matrix{Float64}, Function}","page":"Home","title":"BoostedCDE.step!","text":"Step the boosting model, by adding on a single new base model that minimizes the loss, returning the corresponding predictions ϕₘ, and loss in a tuple.\n\nstep!(model, θ, x, ϕₘ, loss)\n\n\n\n\n\n\n","category":"method"},{"location":"#BoostedCDE.triangular_to_vec-Union{Tuple{LinearAlgebra.UpperTriangular{T, S} where S<:AbstractMatrix{T}}, Tuple{T}} where T","page":"Home","title":"BoostedCDE.triangular_to_vec","text":"Convert triangular matrix to vector. This uses the convention that the upper triangular elements are listed columnwise (or equivilently, lower triangular elements listed rowwise). \n\ntriangular_to_vec(M)\n\n\n\n\n\n\n","category":"method"},{"location":"#BoostedCDE.vec_to_triangular-Union{Tuple{AbstractVector{T}}, Tuple{T}} where T","page":"Home","title":"BoostedCDE.vec_to_triangular","text":"Constructs an upper triangular matrix from a vector. Vector should correspond to the upper triangular elements listed columnwise. \n\n\n\n\n\n","category":"method"},{"location":"#BoostedCDE.μ_chol_splitter-Tuple{AbstractVector{var\"#s1\"} where var\"#s1\"<:Real}","page":"Home","title":"BoostedCDE.μ_chol_splitter","text":"Take a flattened vector and triangular matrix and reconstruct it returning a tuple. \n\nμ_chol_splitter(ϕᵢ)\n\n\n\n\n\n\n","category":"method"},{"location":"#BoostedCDE.@loopify-Tuple{Any, Any}","page":"Home","title":"BoostedCDE.@loopify","text":"Convenience macro to define a method for a simulator that allows simulating in     batches. i.e. it takes simulator(rng::AbstractRNG, θ::AbstractVector{Float64}) and     wraps it in a for loop to get simulator(rng::AbstractRNG,     θ::Matrix{Float64}).\n\n\n\n\n\n","category":"macro"}]
}
