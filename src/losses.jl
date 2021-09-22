
"""
Negative log-probability of x using vector of parameters ϕ to parameterise the
means and cholesky decomposition of the normal distribution.
"""
function mvn_loss(ϕ::Vector{<: ParamTuple}, x::Matrix{Float64}, agg = mean)
    length(ϕ) == size(x, 1) || throw(ArgumentError("length(ϕ) should match size(x,1)"))
    N = size(x, 1)
    l = Vector{Float64}(undef, N)
    for i in 1:N
        μ, U = ϕ[i]
        U = vec_to_triangular(U)
        Λ = Symmetric(U'U)
        h = Λ*μ
        d = MvNormalCanon(h, Λ)
        l[i] = logpdf(d, x[i, :])
    end
    return agg(-l)
end
