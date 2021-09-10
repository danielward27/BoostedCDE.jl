abstract type Abstractϕ end

# Note that all trainable params should be vectors

"""
Struct containing parameters for a multivariate normal distribution
parameterised by the mean vector and lower triangular matrix.
"""
Base.@kwdef struct MvnCholeskyϕ{T} <: Abstractϕ
    "Mean vector"
    μ::AbstractVector{Float64}
    "Lower diagonal component of Σ cholesky decomposition"
    L::LowerTriangular{Float64, Matrix{Float64}}
    "Vector view of L"
    L_flattened::T = @view L[triangind(size(L, 1), :L)]
end

MvnCholeskyϕ(
    μ::AbstractVector{Float64},
    L::LowerTriangular{Float64, Matrix{Float64}}) = MvnCholeskyϕ(μ=μ, L=L)

Flux.trainable(ϕ::MvnCholeskyϕ) = (ϕ.μ, ϕ.L_flattened)
