
"""
Convert vector to a vector of vectors of length 1,2,...,d, representing the 
upper triangular columns of a triangular matrix. Uses views of original vector.
v must be the columns of the triangular matrix, stacked into a vector.
"""
function vecvec_triangular_view(v::AbstractVector)
    d = tri_n_el_to_d(length(v))
    @views [v[range] for range in tri_col_ranges(d)]
end


function triangular_from_vecvec(vv::AbstractVector{<: AbstractVector{T}}) where T<:Real
    d = length(vv)
    m = UpperTriangular(zeros(T,d,d))
    [m[1:length(v), i] = v for (i, v) in enumerate(vv)]
    return m
end

"""
Given the number of triangular elements in a matrix, get the dimension. 
"""
function tri_n_el_to_d(n_el::Int64)
    sqrt_term = isqrt(1 + 8*n_el)
    sqrt_term^2 == 1 + 8*n_el || throw(ArgumentError("Invalid number of elements."))
    return (-1 + sqrt_term) ÷ 2
end

"""
Return the ranges for the upper triangular elements of a matrix columnwise.
e.g. `tri_col_ranges(3)` returns `[1:1, 2:3, 4:6]`.
"""
function tri_col_ranges(d::Int64)
    idx1 = 1
    idx2 = 1
    idxs = Vector{UnitRange{Int64}}(undef, d)
    for i in 1:d
        idxs[i] = idx1:idx2
        idx1 += i
        idx2 = idx1 + i
    end
    return idxs
end