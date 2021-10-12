"""
Given the number of triangular elements in a matrix, get the dimension. 
"""
function tri_n_el_to_d(n_el::Int)
    sqrt_term = isqrt(1 + 8*n_el)
    sqrt_term^2 == 1 + 8*n_el || throw(ArgumentError("Invalid number of elements."))
    return (-1 + sqrt_term) ÷ 2
end

"""
Create a scaler, fitted to matrix x. Can be used to apply (x - μ)/σ to each
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
[θ_train, θ_val, x_train, x_val]
"""
function train_val_split(
    θ::AbstractMatrix,
    x::AbstractMatrix,
    train_prop::Float64 = 0.8)
    size(θ, 1) == size(x, 1) || throw(ArgumentError("Mismatch in size of θ and x in dimension 1"))
    0 ≤ train_prop ≤ 1
    n = size(θ, 1)
    idx = shuffle(1:n)
    split_at = floor(Int, train_prop*n)
    t = idx[1:split_at]
    v = idx[(split_at+1):n]
    return (θ_train=θ[t,:], θ_val=θ[v,:], x_train=x[t,:], x_val=x[v,:])
end
