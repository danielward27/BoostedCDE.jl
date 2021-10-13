"""
Convenience macro to define a method for a simulator that allows simulating in
    batches. i.e. it takes `simulator(rng::AbstractRNG, x::AbstractVector{Float64})` and
    wraps it in a for loop to get `simulator(rng::AbstractRNG,
    x::Matrix{Float64})`.
"""
macro loopify(simulator, x_dim)
    fn_name = Symbol(simulator)
    simulator = esc(simulator)
    quote
        function $(esc(fn_name))(rng::AbstractRNG, x::AbstractMatrix{Float64})
            n = size(x, 1)
            y = Matrix{Float64}(undef, n, $x_dim)
            for i in 1:n
                y[i, :] = $simulator(rng, x[i, :])
            end
            return y
        end
    end
end


"""
Simulate a three dimensional Gaussian mean vector x. The covariance is diagonal,
and fixed to σ=0.1. Parameter vector x is the mean vector of the Gaussian.
"""
function gaussian_simulator(
    rng::AbstractRNG,
    x::AbstractVector{Float64}
    )
    sds = fill(√0.1, length(x))
    d = MvNormal(x, Diagonal(sds.^2))
    rand(rng, d)
end

@loopify gaussian_simulator 3
gaussian_simulator(x::AbstractVector{Float64}) = gaussian_simulator(default_rng(), x)
gaussian_simulator(x::AbstractMatrix{Float64}) = gaussian_simulator(default_rng(), x)



"""
Simulator that applies simple linear transformation of x to get ϕ.
Useful for testing.
"""
function linear_x_to_ϕ_mvn_simulator(x::AbstractMatrix{Float64})
    ϕ = [k+k*xₖ for (k, xₖ) in enumerate(eachcol(x))]
    ϕ = reduce(vcat, ϕ)
    ϕ = [get_params(MeanCholeskyMvn(2), ϕᵢ) for ϕᵢ in eachrow(ϕ)]
 end

