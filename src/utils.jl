"""
Given the number of triangular elements in a matrix, get the dimension. 
"""
function tri_n_el_to_d(n_el::Int)
    sqrt_term = isqrt(1 + 8*n_el)
    sqrt_term^2 == 1 + 8*n_el || throw(ArgumentError("Invalid number of elements."))
    return (-1 + sqrt_term) ÷ 2
end

"""
Create a scaler, fitted to matrix y. Can be used to apply (y - μ)/σ to each
column of a matrix. Use as callable struct to scale and use [`unscale`](@ref)
to unscale.
"""
mutable struct StandardScaler
    μ::Vector{Float64}
    σ::Vector{Float64}
    absdet::Float64
    function StandardScaler(x::AbstractMatrix)
        μ = mean.(eachcol(x))
        σ = std.(eachcol(x))
        absdet = reduce(*, σ)
        new(μ, σ, absdet)
    end
end


"""
Transform a matrix using the scaler
"""
function (scaler::StandardScaler)(x::AbstractMatrix)
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

"""
Train test split for two matrices. Returns named tuple with keys
[x_train, x_val, y_train, y_val]
"""
function train_val_split(
    x::AbstractMatrix,
    y::AbstractMatrix,
    train_prop::Float64 = 0.8)
    size(x, 1) == size(y, 1) || throw(ArgumentError("Mismatch in size of x and y in dimension 1"))
    0 ≤ train_prop ≤ 1
    n = size(x, 1)
    idx = shuffle(1:n)
    split_at = floor(Int, train_prop*n)
    t = idx[1:split_at]
    v = idx[(split_at+1):n]
    return (x_train=x[t,:], x_val=x[v,:], y_train=y[t,:], y_val=y[v,:])
end
