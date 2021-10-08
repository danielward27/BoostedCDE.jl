"""
Convenience macro to define a method for a simulator that allows simulating in
    batches. i.e. it takes `simulator(rng::AbstractRNG, θ::AbstractVector{Float64})` and
    wraps it in a for loop to get `simulator(rng::AbstractRNG,
    θ::Matrix{Float64})`.
"""
macro loopify(simulator, x_dim)
    fn_name = Symbol(simulator)
    simulator = esc(simulator)
    quote
        function $(esc(fn_name))(rng::AbstractRNG, θ::AbstractMatrix{Float64})
            n = size(θ, 1)
            x = Matrix{Float64}(undef, n, $x_dim)
            for i in 1:n
                x[i, :] = $simulator(rng, θ[i, :])
            end
            return x
        end
    end
end


"""
Simulate a three dimensional Gaussian mean vector θ. The covariance is diagonal,
and fixed to σ=0.1. Parameter vector θ is the mean vector of the Gaussian.
"""
function gaussian_simulator(
    rng::AbstractRNG,
    θ::AbstractVector{Float64}
    )
    sds = fill(√0.1, length(θ))
    d = MvNormal(θ, Diagonal(sds.^2))
    rand(rng, d)
end

@loopify gaussian_simulator 3
gaussian_simulator(θ::AbstractVector{Float64}) = gaussian_simulator(default_rng(), θ)
gaussian_simulator(θ::AbstractMatrix{Float64}) = gaussian_simulator(default_rng(), θ)



"""
Simulator that applies simple linear transformation of θ to get ϕ.
Useful for testing.
"""
function linear_θ_to_ϕ_mvn_simulator(θ::AbstractMatrix{Float64})
    ϕ = [k+k*θₖ for (k, θₖ) in enumerate(eachcol(θ))]
    ϕ = reduce(vcat, ϕ)
    ϕ = [get_params(MeanCholeskyMvn(2), ϕᵢ) for ϕᵢ in eachrow(ϕ)]
 end

