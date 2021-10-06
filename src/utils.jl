

"""
Given the number of triangular elements in a matrix, get the dimension. 
"""
function tri_n_el_to_d(n_el::Int64)
    sqrt_term = isqrt(1 + 8*n_el)
    sqrt_term^2 == 1 + 8*n_el || throw(ArgumentError("Invalid number of elements."))
    return (-1 + sqrt_term) ÷ 2
end


