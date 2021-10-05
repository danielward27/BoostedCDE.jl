## Arrays and special matrices
"""
Flatten to vector, with methods for special matrices, that ignore unwanted
elements e.g. for diagonal matrices this will ignore off diagonal elements.
These methods may return a copy or a view (behaviour is not kept consistent).
"""
function vectorize(a::UpperTriangular{T}) where T
    n = size(a, 1)
    l = n*(n+1) ÷ 2
    v = Vector{T}(undef, l)
    k = 0
    for i in 1:n
        for j in 1:i
            v[k + j] = a[j, i]
        end
        k += i
    end
    v
end

vectorize(a::AbstractArray) = vec(a)
vectorize(a::Diagonal) = diag(a)

"""
Given the array type and the vector, reform the array. This does the oposite
transformation of [`vectorize`](@ref)
"""
function unvectorize(::Type{UpperTriangular}, v::AbstractVector{T}) where T
    d = tri_n_el_to_d(length(v))
    M = Zygote.Buffer(Matrix{T}(undef, d, d))
    k = 0
    for i in 1:d
        for j in 1:i
            M[j, i] = v[k + j]
        end
        k += i
    end
    return UpperTriangular(copy(M))
end

unvectorize(::Type{Diagonal}, v::AbstractVector) = Diagonal(v)

## ϕ structs

function vectorize(ϕ::MeanCholeskyMvn)
    @unpack μ, U, d = ϕ
    vcat(μ, vectorize(U))
end

# function unvectorize_like(example::MeanCholeskyMvn, ϕ_vec::AbstractVector)
#     d = example.dim
#     μ = ϕ_vec[1:d]
#     U = unvectorize(UpperTriangular, ϕ_vec[d+1:end])
#     return MeanCholeskyMvn(μ, U)
# end