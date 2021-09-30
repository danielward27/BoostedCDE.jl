"""
Flatten to vector, with methods for special matrices, that ignore unwanted
elements e.g. for diagonal matrices this will ignore off diagonal elements.
"""
function vectorize(a::UpperTriangular)
    n = size(a, 1)
    l = n*(n+1) ÷ 2
    v = Vector{T}(undef, l)
    k = 0
    for i in 1:n
        for j in 1:i
            v[k + j] = M[j, i]
        end
        k += i
    end
    v
end
vectorize(a::AbstractArray) = vec(a)
vectorize(a::Diagonal) = diag(a)

"""
Given the array type and the vector, reform the array. This does the oposite
tranformation of [`vectorize`](@ref)
"""
function unvectorize(::Type{UpperTriangular}, v::AbstractVector{T}) where T
    l  = length(v)
    n = (-1 + isqrt(1 + 8l)) ÷ 2
    M = Matrix{T}(undef, n, n)
    k = 0
    for i in 1:n
        for j in 1:i
            M[j, i] = v[k + j]
        end
        k += i
    end
    return UpperTriangular(M)
end

unvectorize(::Type{Diagonal}, v::AbstractVector) = Diagonal(v)
unvectorize(::Type{AbstractVector}, v::AbstractVector) = v
