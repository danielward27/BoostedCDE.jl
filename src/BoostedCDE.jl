module BoostedCDE

import Base.@kwdef
using Distributions
using DocStringExtensions
using ReverseDiff
using ReverseDiff: AbstractTape, GradientTape, gradient!, gradient
using LinearAlgebra
using Random
using Random: default_rng, AbstractRNG
using StatsBase
using StatsBase: mean, std
using UnPack

include("utils.jl")
export StandardScaler, unscale, train_val_split

include("parameters.jl")
export MeanCholeskyMvn, get_params

include("vectorize.jl")
export vectorize, unvectorize, unvectorize_like

include("tasks.jl")
export gaussian_simulator

include("base_learners.jl")
export BaseLearner, ConstBaseLearner, PolyBaseLearner, fit!, predict

include("boost.jl")
export BoostingModel, boost!, boostcv!, step!, reset!

include("losses.jl")
export cost

end
