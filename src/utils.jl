# TODO Perhaps these functions can be sped up with a custom matrix type if needed

function μ_and_cholesky_to_vec(μ::AbstractVector, C::Cholesky)  # Upper or lower Triangular type?
    [μ; triangular_to_vec(C.U)]
end

function vec_to_μ_and_cholesky(flattened_ϕ::AbstractVector)
    k = length(flattened_ϕ)
    l = (-3 + isqrt(3^2 + 8k)) ÷ 2  # Quadratic formula
    μ = flattened_ϕ[1:l]
    U = vec_to_triangular(flattened_ϕ[l+1:end])
    U = Cholesky(U, :U, 0)
    return μ, U
end

function triangular_to_vec(M::UpperTriangular{T}) where T
    m, n = size(M)
    m == n || throw(error("not square"))
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

function triangular_to_vec(M::LowerTriangular)
    triangular_to_vec(M')
end

function vec_to_triangular(v::AbstractVector{T}, uplo::Symbol=:U) where T
    """
    Constructs an upper `(uplo=:U)` or lower `(uplo=:L)` triangular matrix from
    a vector. 
    """
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

    if uplo==:U
        M = UpperTriangular(M)
    elseif uplo==:L
        M = LowerTriangular(M')
    else
        throw(ArgumentError("uplo should be U: or L:")) 
    end
end


"""Get the indices corresponding to the upper or lower triangular elements of a square array of size n×n."""
function triangind(n::Int, uplo::Symbol=:U)
    uplo ∈ [:U, :L] || throw(ArgumentError("uplo should be :U or :L."))
    l = n*(n + 1) ÷ 2
    indices = Vector{Int64}(undef, l)
    k = 1
    for i in 1:n
        for j in 1:i
            indices[k] = uplo == :U ? n*(i-1) + j : n*(j-1) + i
            k +=1
        end
    end
    sort(indices)
end


typeof(Flux.params([[1]]))

function param_size(p::Flux.Params)

end


p = Flux.params(1, [1,2], [1 2; 1 2])
[size(pᵢ) for pᵢ in p]