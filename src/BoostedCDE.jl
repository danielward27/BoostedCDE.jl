module BoostedCDE

import Base.@kwdef
import Base: length
using Zygote
using DocStringExtensions
using Distributions
using LinearAlgebra
using UnPack
using StatsBase
using StatsBase: mean
using Random
using Random: default_rng
using UnPack

include("utils.jl")

include("parameters.jl")
export MeanCholeskyMvn

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
