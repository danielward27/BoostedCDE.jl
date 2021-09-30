module BoostedCDE

import Base.@kwdef
using Flux
using ReverseDiff
using DocStringExtensions
using Distributions
using LinearAlgebra
using UnPack
using ArgCheck
using StatsBase
using StatsBase: mean
using Random
using Random: default_rng
using UnPack

include("parameters.jl")
export MeanCholeskyMvn

include("utils.jl")
export triangular_to_vec, vec_to_triangular

include("vectorize.jl")
export vectorize, unvectorize, unvectorize_like

include("tasks.jl")
export gaussian_simulator

include("base_learners.jl")
export BaseLearner, ConstBaseLearner, PolyBaseLearner, fit!, predict

include("boost.jl")
export BoostingModel, boost!, step!

include("losses.jl")
export loss

end
