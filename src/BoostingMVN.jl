module BoostingMVN

import Base.@kwdef
using Flux
using DocStringExtensions
using Distributions
using LinearAlgebra
using UnPack
using ArgCheck

include("utils.jl")
include("base_learners.jl")
include("boosting.jl")

export BaseLearner, BoostingModel, PolynomialBaseLearner, fit!, predict, 

end
