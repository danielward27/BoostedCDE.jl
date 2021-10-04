using BoostedCDE
using Test
using LinearAlgebra
using Distributions

@testset "MeanCholeskyMvn" begin
    ϕ = MeanCholeskyMvn([1.,2], UpperTriangular([3. 4; 0 5]))
    x = rand(2)
    U = BoostedCDE.triangular_from_vecvec(ϕ.U)
    Λ = Symmetric(U'U)
    h = Λ*ϕ.μ
    d = MvNormalCanon(h, Λ)
    @test loss(ϕ, x) ≈ -logpdf(d, x)
end


# test gradients - ReverseDiff

using ReverseDiff
ϕ = MeanCholeskyMvn([1., 2], UpperTriangular([3. 4; 0 5]))
x = [-1.,1]
loss2(v) = begin
    ϕ.v .= v
    loss(ϕ, x)
end


loss3(v) = begin
    ϕ = MeanCholeskyMvn(v)
    loss(ϕ, x)
end

ReverseDiff.gradient(loss3, rand(5))


##### Using Flux
using Flux


loss(ϕ, x)
v = rand(5)
u = Flux.gradient(Flux.params(v)) do
    ϕ = MeanCholeskyMvn(v)
    loss(ϕ, x)
end


u[ϕ.v]


u[ϕ.μ]

gs = Flux.gradient(() -> loss(ϕ, x), Flux.params(ϕ))

gs[ϕ]

keys(u)

sum(loss.(ϕₘ, eachrow(x)))

u[ϕₘ[2].v]

struct Foo{T}
    a::T
end

a = rand(3,3)
b = Foo(@view a[1:3])
a .+= 10
b.a



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