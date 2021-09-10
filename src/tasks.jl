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
    sds = fill(√0.1, 3)
    d = MvNormal(θ, sds)
    rand(rng, d)
end

@loopify gaussian_simulator 3
gaussian_simulator(θ::AbstractVector{Float64}) = gaussian_simulator(default_rng(), θ)
gaussian_simulator(θ::AbstractMatrix{Float64}) = gaussian_simulator(default_rng(), θ)






# ## For testing:
# """
# Deterministic simulator (quadratic model) useful for testing.
# """
# function deterministic_test_simulator(
#     rng::AbstractRNG,
#     θ::AbstractVector{Float64}
#     )
#     rng  # Exists just to have a consistent signature
#     @assert length(θ) == 2
#     [θ[1], θ[1]*θ[2], θ[2]^2]
# end

# @loopify 3 deterministic_test_simulator

