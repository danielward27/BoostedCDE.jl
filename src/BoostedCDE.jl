module BoostedCDE

import Base.@kwdef
using Distributions
using DocStringExtensions
using ReverseDiff
using LinearAlgebra
using Random
using Random: default_rng, AbstractRNG
using StatsBase
using StatsBase: mean
using UnPack

include("utils.jl")

include("parameters.jl")
export MeanCholeskyMvn, get_params

include("vectorize.jl")
export vectorize, unvectorize, unvectorize_like

include("tasks.jl")
export gaussian_simulator

include("base_learners.jl")
export BaseLearner, ConstBaseLearner, PolyBaseLearner, fit!, predict

include("boost.jl")
export BoostingModel, boost!, step!, reset!

include("losses.jl")
export loss

end
