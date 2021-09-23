const ParamTuple = NamedTuple{Vararg{AbstractArray{<: Real}}}

# abstract type Abstractϕ end

# Note that all parameters should be defined as vectors (so that they can be mutated).

# """
# Struct containing parameters for a multivariate normal distribution
# parameterised by the mean vector and lower triangular matrix.
# """
# Base.@kwdef struct MvnCholeskyϕ{T} <: Abstractϕ
#     "Mean vector"
#     μ::AbstractVector{Float64}
#     "Lower diagonal component of Σ cholesky decomposition"
#     L::LowerTriangular{Float64, Matrix{Float64}}
#     "Vector view of L"
#     L_flattened::T = @view L[triangind(size(L, 1), :L)]
# end

# MvnCholeskyϕ(
#     μ::AbstractVector{Float64},
#     L::LowerTriangular{Float64, Matrix{Float64}}) = MvnCholeskyϕ(μ=μ, L=L)

# Flux.trainable(ϕ::MvnCholeskyϕ) = (ϕ.μ, ϕ.L_flattened)



# # Flatten and concat and see if still reference

# struct Foo <: Abstractϕ
#     x::Vector{Float64}
#     y::Matrix{Float64}
# end

# foo = Foo([1.,2], [1 2; 3 4])

# function flatten_ϕ(ϕ::Abstractϕ)  # This would mess up types... We could flatten to an array of reals/some abstract type at worst? But surely converting would copy?
#     Tuple(getfield(base_learners, f) for f in fieldnames(typeof(base_learners)))
# end