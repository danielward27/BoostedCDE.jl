
"""
Negative log-probability of x using vector of parameters ϕ to parameterise the
means and cholesky decomposition of the normal distribution.
"""
function mvn_loss(ϕ::Vector{ParamTuple}, x::Matrix{Float64}, reduction=mean)  # TODO This is probably pretty slow? 
    length(ϕ) == size(x, 1) || throw(ArgumentError("length(ϕ) should match size(x,1)"))
    N = size(x, 1)
    l = Vector{Float64}(undef, N)
    for i in 1:N
        Σ = Symmetric(ϕ[i].L*ϕ[i].L')
        l[i] = logpdf(MvNormal(ϕ[i].μ, Σ), x[i, :])
    end
    return reduction(-l)
end




