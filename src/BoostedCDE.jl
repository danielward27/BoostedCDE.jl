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


include("utils.jl")
include("vectorize.jl")
include("tasks.jl")
include("parameters.jl")
include("base_learners.jl")
include("boost.jl")
include("losses.jl")

export BaseLearner, ConstBaseLearner, PolyBaseLearner, fit!, predict

export BoostingModel, boost!, step!

export triangular_to_vec, vec_to_triangular, μ_chol_splitter

export mvn_loss, mvn_d_from_ϕ

export gaussian_simulator

end
