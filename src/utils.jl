"""
Given the number of triangular elements in a matrix, get the dimension. 
"""
function tri_n_el_to_d(n_el::Int64)
    sqrt_term = isqrt(1 + 8*n_el)
    sqrt_term^2 == 1 + 8*n_el || throw(ArgumentError("Invalid number of elements."))
    return (-1 + sqrt_term) ÷ 2
end

"""
Create a scaler, fitted to matrix x. Can be used to apply (x - μ)/σ to each
column of a matrix.
"""
mutable struct StandardScaler
    μ::Vector{Float64}
    σ::Vector{Float64}
    StandardScaler(x::AbstractMatrix) = new(mean.(eachcol(x)), std.(eachcol(x)))
end

"""
Transform a matrix using the scaler
"""
function scale(scaler::StandardScaler, x::AbstractMatrix)
    @unpack μ, σ = scaler
    (x .- μ') ./ σ'
end

"""
Unscale the matrix.
"""
function unscale(scaler::StandardScaler, x::AbstractMatrix)
    @unpack μ, σ = scaler
    x.*σ' .+ μ'
end
