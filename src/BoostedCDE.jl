module BoostedCDE

import Base.@kwdef
using Flux
using ForwardDiff
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

export triangular_to_vec, vec_to_triangular, μ_chol_splitter

export mvn_loss

export gaussian_simulator

end
