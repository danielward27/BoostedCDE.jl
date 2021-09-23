module BoostedCDE

import Base.@kwdef
using Flux
using DocStringExtensions
using Distributions
using LinearAlgebra
using UnPack
using ArgCheck
using StatsBase
using StatsBase: mean
using Random
using Random: default_rng


include("utils.jl")
include("tasks.jl")
include("parameters.jl")
include("base_learners.jl")
include("boost.jl")
include("losses.jl")

export BaseLearner, ConstBaseLearner, PolyBaseLearner, fit!, predict

export BoostingModel, boost!

export triangular_to_vec, vec_to_triangular #, μ_and_cholesky_to_vec, vec_to_μ_and_cholesky

export mvn_loss

export gaussian_simulator

end
