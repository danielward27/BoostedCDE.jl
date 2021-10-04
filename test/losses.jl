using BoostedCDE
using Test
using LinearAlgebra
using Distributions

@testset "MeanCholeskyMvn" begin
    ϕ = MeanCholeskyMvn([1.,2], UpperTriangular([3. 4; 0 5]))
    x = rand(2)
    U = unvectorize(UpperTriangular, ϕ.U)
    Λ = Symmetric(U'U)
    h = Λ*ϕ.μ
    d = MvNormalCanon(h, Λ)
    @test loss(ϕ, x) ≈ -logpdf(d, x)
end


ϕ1 = MeanCholeskyMvn([1., 2], UpperTriangular([3. 4; 0 5]))
ϕ2 = MeanCholeskyMvn([2., 3], UpperTriangular([3. 4; 0 5]))

ϕₘ = [ϕ1, ϕ2]
x = [1 1 ; 1 1]
using Flux
params = Flux.params([ϕ.v for ϕ in ϕₘ])
u = Flux.gradient(params) do
    sum(loss.(ϕₘ, eachrow(x)))
end

u = Flux.gradient(Flux.params(ϕ1)) do
    sum(loss(ϕ1, x[1, :]))
end
u[ϕ1]

sum(loss.(ϕₘ, eachrow(x)))

u[ϕₘ[2].v]



using ReverseDiff

for i in 1:length(ϕₘ)
    ReverseDiff.gradient((ϕₘ[i].v) -> loss(ϕₘ[i], x[i, :]))
end

function new_loss(ϕv, ϕ, x)
    ϕ.v .= ϕv
    loss(ϕ, x)
end


new_loss(ones(5), ϕₘ[1], x[1, :])

ReverseDiff.gradient((ϕ_vec) -> new_loss(ϕₘ[1].v, ϕₘ[1], x[1, :]), ϕₘ[1].v)



loss(ϕₘ[2], [1.,1])

u = Flux.gradient(Flux.params(ϕ1.v)) do 
    batch_loss = 0.
    for (ϕᵢv, xᵢ)  in zip(eachrow(ϕₘ), eachrow(x))  # TODO time with column major opimized version?
        ϕᵢ = unvectorize_like(model.init_ϕ, ϕᵢvᵢ)
        batch_loss += loss(t, ϕᵢ, xᵢ)
    end
end  # N×j

x = rand(2)
    U = unvectorize(UpperTriangular, ϕ.U)
    Λ = Symmetric(U'U)
    h = Λ*ϕ.μ
    d = MvNormalCanon(h, Λ)
    @test loss(ϕ, x) ≈ -logpdf(d, x)