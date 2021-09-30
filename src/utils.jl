
"""
Convert triangular matrix to vector. This uses the convention that the upper
triangular elements are listed columnwise (or equivilently, lower triangular
elements listed rowwise). $(SIGNATURES)
"""
function triangular_to_vec(M::UpperTriangular{T}) where T
    n = size(M, 1)
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


"""
Constructs an upper triangular matrix from a vector. Vector should correspond to
the upper triangular elements listed columnwise. $(SIGNATURES)
"""
function vec_to_triangular(v::AbstractVector{T}) where T
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


