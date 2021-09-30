
abstract type Abstractϕ end

"""
Vectorize(ϕ::Abstractϕ) flattens a struct similarly to Flux.destructure, but
with specialised methods for special matrices. Specifically, it ignores off
diagonal elements for Diagonal matrices, and lower triangular elements in upper
triangular matrices.
"""
function vectorize(ϕ::Abstractϕ)
    ϕ_parts = Array{Float64}[]
    Flux.fmap(ϕ) do ϕ_part
        ϕ_part isa AbstractArray{<: Real} && push!(ϕ_parts, ϕ_part)
        return x
    end
    return reduce(vcat, vectorize.(ϕ_parts))
end

using LinearAlgebra
struct MeanCholeskyMvn{D} <: Abstractϕ
    μ::Vector{Float64}
    U::UpperTriangular{Float64, Matrix{Float64}}
    dim::Int64
end



μ = [1., 2]
U = UpperTriangular(rand(2,2))



Flux.@functor MeanCholeskyMvn

MeanCholeskyMvn(μ, U) = MeanCholeskyMvn{length(μ)}(μ, U, length(μ))
MeanCholeskyMvn(μ, U, dim) = MeanCholeskyMvn{dim}(μ, U, dim) # TODO check dims makes sense
# TODO can we construct fine with MeanCholeskyMvn{2}(μ, U)???


function unvectorize(::Type{MeanCholeskyMvn{D}}, ϕ_vec::AbstractVector{Float64}) where D
    μ = ϕ_vec[1:D]
    U = unvectorize(UpperTriangular, ϕ_vec[d+1:end])
    return MeanCholeskyMvn(μ, U)
end

using LinearAlgebra
μ = rand(2)
U = UpperTriangular(rand(4,4))
MeanCholeskyMvn(μ, U)

SpaceGroup(g::Vector{SMatrix{N,N,Int,N2}}) where {N,N2} = SpaceGroup{N,N2}(g)


# function get_parameters(::MeanCholeskyMvn, ϕᵢ::AbstractVector{<: Real})
#     a = 9 + 8*length(ϕᵢ)
#     @argcheck a == isqrt(a)^2
#     idx = (-3 + isqrt(a)) ÷ 2
#     μ = ϕᵢ[1:idx]
#     U = vec_to_triangular(ϕᵢ[idx+1:end])
#     return μ, U
# end

# """
# Take a flattened vector and triangular matrix and reconstruct it returning a
# tuple. $(SIGNATURES)
# """
# function μ_chol_splitter(ϕᵢ::AbstractVector{<: Real})
#     a = 9 + 8*length(ϕᵢ)
#     @argcheck a == isqrt(a)^2
#     idx = (-3 + isqrt(a)) ÷ 2
#     μ = ϕᵢ[1:idx]
#     U = vec_to_triangular(ϕᵢ[idx+1:end])
#     return μ, U
# end

# https://docs.julialang.org/en/v1/manual/types/
# https://discourse.julialang.org/t/should-i-dispatch-on-singleton-types-or-their-instances/28204/4