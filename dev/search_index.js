var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = BoostedCDE","category":"page"},{"location":"#BoostedCDE","page":"Home","title":"BoostedCDE","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for BoostedCDE.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [BoostedCDE]","category":"page"},{"location":"#BoostedCDE.BaseLearner","page":"Home","title":"BoostedCDE.BaseLearner","text":"Base learners are used to predict the negative gradient vector. These must all have the methods fit! and predict. Note that the same model is used multiple, so it must be safe to call the fit! method multiple times with different data.\n\n\n\n\n\n","category":"type"},{"location":"#BoostedCDE.BoostingModel","page":"Home","title":"BoostedCDE.BoostingModel","text":"Constructs a BoostingModel that can be trained using boost!\n\n\n\n\n\n","category":"type"},{"location":"#BoostedCDE.MeanCholeskyMvn","page":"Home","title":"BoostedCDE.MeanCholeskyMvn","text":"Struct denoting we wish to use a mean vector and cholesky decomposition of the covariance matrix to parameterize a multivariate normal distribution of dimension d.\n\n\n\n\n\n","category":"type"},{"location":"#BoostedCDE.PolyBaseLearner","page":"Home","title":"BoostedCDE.PolyBaseLearner","text":"Base learner using polynomial transform. By defualt, with use_cache = true, the model will cache the feature vectors along with QR decompositions of the polynomial transforms to avoid recalculation. This will speed up training at the cost of increased memory usage. An IdDict is used as the cache.\n\n\n\n\n\n","category":"type"},{"location":"#BoostedCDE.StandardScaler","page":"Home","title":"BoostedCDE.StandardScaler","text":"Create a scaler, fitted to matrix x. Can be used to apply (x - μ)/σ to each column of a matrix. Use as callable struct to scale and use unscale to unscale.\n\n\n\n\n\n","category":"type"},{"location":"#BoostedCDE.StandardScaler-Tuple{AbstractMatrix{T} where T}","page":"Home","title":"BoostedCDE.StandardScaler","text":"Transform a matrix using the scaler\n\n\n\n\n\n","category":"method"},{"location":"#BoostedCDE.boost!-Tuple{BoostingModel, AbstractMatrix{Float64}, AbstractMatrix{Float64}}","page":"Home","title":"BoostedCDE.boost!","text":"Minimal function to fit a boosting model. loss should take ϕ and x as arguments, and ∇loss should just take ϕ.\n\nboost!(model, θ, x; steps, loss, ∇loss, step!)\n\n\n\n\n\n\n","category":"method"},{"location":"#BoostedCDE.boostcv!-Tuple{BoostingModel, Any}","page":"Home","title":"BoostedCDE.boostcv!","text":"Boosting with cross validation and patience. loss should take ϕ and x, returning a scalar. ∇loss should take ϕ (i.e. x_train should be abstracted away), returning matrix with size matching ϕ. data should be an object that can be unpacked/destructured to (θtrain, θval, xtrain, xval), e.g. NamedTuple resulting from train_val_split. Returns a NamedTuple of results.\n\nIf ∇loss is not provided, by default, the gradient is found using ReverseDiff, using ReverseDiff.GradientTape.\n\nboostcv!(model, data; steps, loss, ∇loss, step!, max_patience)\n\n\n\n\n\n\n","category":"method"},{"location":"#BoostedCDE.cost-Tuple{BoostedCDE.Abstractϕ, AbstractMatrix{var\"#s19\"} where var\"#s19\"<:Real, AbstractMatrix{var\"#s18\"} where var\"#s18\"<:Real}","page":"Home","title":"BoostedCDE.cost","text":"Calculate the loss given a parameterisation specified by Abstractϕ. If matrices are used, reduction is carried out using summation.\n\n\n\n\n\n","category":"method"},{"location":"#BoostedCDE.fit!-Tuple{PolyBaseLearner, AbstractVector{T} where T, AbstractVector{T} where T}","page":"Home","title":"BoostedCDE.fit!","text":"Fit the Base learner model to the negative gradient vector uⱼ using feature vector θ.\n\nfit!(base_learner, θ, u)\n\n\n\n\n\n\n","category":"method"},{"location":"#BoostedCDE.gaussian_simulator-Tuple{Random.AbstractRNG, AbstractVector{Float64}}","page":"Home","title":"BoostedCDE.gaussian_simulator","text":"Simulate a three dimensional Gaussian mean vector θ. The covariance is diagonal, and fixed to σ=0.1. Parameter vector θ is the mean vector of the Gaussian.\n\n\n\n\n\n","category":"method"},{"location":"#BoostedCDE.get_params-Tuple{MeanCholeskyMvn, AbstractVector{var\"#s5\"} where var\"#s5\"<:Real}","page":"Home","title":"BoostedCDE.get_params","text":"Get the parameters for the distribution from the vectorised form.\n\n\n\n\n\n","category":"method"},{"location":"#BoostedCDE.get_tape_∇-Tuple{Function, AbstractMatrix{Float64}, AbstractMatrix{Float64}}","page":"Home","title":"BoostedCDE.get_tape_∇","text":"Gradient function with tape, specialised on θ and x (usually training data). \n\n\n\n\n\n","category":"method"},{"location":"#BoostedCDE.linear_θ_to_ϕ_mvn_simulator-Tuple{AbstractMatrix{Float64}}","page":"Home","title":"BoostedCDE.linear_θ_to_ϕ_mvn_simulator","text":"Simulator that applies simple linear transformation of θ to get ϕ. Useful for testing.\n\n\n\n\n\n","category":"method"},{"location":"#BoostedCDE.predict-Tuple{BoostingModel, AbstractMatrix{Float64}, AbstractMatrix{Float64}}","page":"Home","title":"BoostedCDE.predict","text":"Predict using the boosting model to get the conditional distributional parameters, using the ϕ matrix from previous training iteration to avoid recalculating.\n\npredict(model, θ, last_ϕ)\n\n\n\n\n\n\n","category":"method"},{"location":"#BoostedCDE.predict-Tuple{BoostingModel, AbstractMatrix{Float64}}","page":"Home","title":"BoostedCDE.predict","text":"Predict using the boosting model to get the conditional distributional parameters.\n\npredict(model, θ)\n\n\n\n\n\n\n","category":"method"},{"location":"#BoostedCDE.predict-Tuple{PolyBaseLearner, AbstractVector{T} where T}","page":"Home","title":"BoostedCDE.predict","text":"Predict the negative gradient vector using θ.\n\npredict(base_learner, θ)\n\n\n\n\n\n\n","category":"method"},{"location":"#BoostedCDE.reset!-Tuple{BoostingModel}","page":"Home","title":"BoostedCDE.reset!","text":"\"Reset\" the boosting model, removing all the selected base learners and corresponding indices.\n\n\n\n\n\n","category":"method"},{"location":"#BoostedCDE.step!-Tuple{BoostingModel, AbstractMatrix{Float64}, AbstractMatrix{Float64}}","page":"Home","title":"BoostedCDE.step!","text":"Step the boosting model, by adding on a single new base model that minimizes the inner loss. u is the gradient matrix with shape matching ϕ, i.e. N×J where N is the number observations/simulations, and J is the number of distributional parameters.\n\nstep!(model, θ, u)\n\n\n\n\n\n\n","category":"method"},{"location":"#BoostedCDE.step_naive!-Tuple{BoostingModel, AbstractMatrix{Float64}, AbstractMatrix{Float64}}","page":"Home","title":"BoostedCDE.step_naive!","text":"As for step!, but without skipping training base models where inner_loss cannot be improved. \n\n\n\n\n\n","category":"method"},{"location":"#BoostedCDE.train_val_split","page":"Home","title":"BoostedCDE.train_val_split","text":"Train test split for two matrices. Returns named tuple with keys [θtrain, θval, xtrain, xval]\n\n\n\n\n\n","category":"function"},{"location":"#BoostedCDE.tri_n_el_to_d-Tuple{Int64}","page":"Home","title":"BoostedCDE.tri_n_el_to_d","text":"Given the number of triangular elements in a matrix, get the dimension. \n\n\n\n\n\n","category":"method"},{"location":"#BoostedCDE.unscale-Tuple{StandardScaler, AbstractMatrix{T} where T}","page":"Home","title":"BoostedCDE.unscale","text":"Unscale the matrix.\n\n\n\n\n\n","category":"method"},{"location":"#BoostedCDE.unvectorize-Union{Tuple{T}, Tuple{Type{LinearAlgebra.UpperTriangular}, AbstractVector{T}}} where T","page":"Home","title":"BoostedCDE.unvectorize","text":"Given the array type and the vector, reform the array. This does the oposite transformation of vectorize.\n\n\n\n\n\n","category":"method"},{"location":"#BoostedCDE.vectorize-Union{Tuple{LinearAlgebra.UpperTriangular{T, S} where S<:AbstractMatrix{T}}, Tuple{T}} where T","page":"Home","title":"BoostedCDE.vectorize","text":"Flatten to vector, with methods for special matrices, that ignore unwanted elements e.g. for diagonal matrices this will ignore off diagonal elements. These methods may return a copy or a view (behaviour is not kept consistent).\n\n\n\n\n\n","category":"method"},{"location":"#BoostedCDE.@loopify-Tuple{Any, Any}","page":"Home","title":"BoostedCDE.@loopify","text":"Convenience macro to define a method for a simulator that allows simulating in     batches. i.e. it takes simulator(rng::AbstractRNG, θ::AbstractVector{Float64}) and     wraps it in a for loop to get simulator(rng::AbstractRNG,     θ::Matrix{Float64}).\n\n\n\n\n\n","category":"macro"}]
}
