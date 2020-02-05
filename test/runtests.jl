using AlphaStableDistributions
using Test, Random

@testset "AlphaStableDistributions.jl" begin

d1 = AlphaStable()
s = rand(d1, 100000)

d2 = fit(AlphaStable, s)

@test d1.α ≈ d2.α rtol=0.1
@test d1.β ≈ d2.β rtol=0.1
@test d1.scale ≈ d2.scale rtol=0.1
@test d1.location ≈ d2.location atol=0.03


d = AlphaSubGaussian(n=96000)
x = rand(d)
x2 = copy(x)
rand!(d, x2)
@test x != x2

d3 = fit(AlphaStable, x)
@test d3.α ≈ 1.5 rtol=0.2
@test d3.β == 0
@test d3.scale ≈ 1 rtol=0.2
@test d3.location ≈ 0 atol=0.03

d4 = AlphaSubGaussian(n=96000)
m = size(d4.R, 1)-1
x = rand(d4)
d5 = fit(AlphaSubGaussian, x, m, p=1.0)
@test d4.α ≈ d5.α rtol=0.1
@test d4.R ≈ d5.R rtol=0.1

end
# 362.499 ms (4620903 allocations: 227.64 MiB)
# 346.520 ms (4621052 allocations: 209.62 MiB) # StaticArrays in outer fun
# 345.925 ms (4225524 allocations: 167.66 MiB) # tempind to tuple
# 395.606 ms (3637770 allocations: 164.76 MiB) # x1 to SVector
# 336.877 ms (10125987 allocations: 236.71 MiB) # typeassert on subgprt
# 328.315 ms (3636312 allocations: 164.69 MiB) # revert x1 svector
# 320.845 ms (3440006 allocations: 161.71 MiB)
# 210.449 ms (3438629 allocations: 86.64 MiB) # full typeinfo in x creation
#
#
# @code_warntype rand(Random.GLOBAL_RNG, AlphaSubGaussian(n=96000))
#
#
#
#
# d = AlphaSubGaussian(n=96)
# using SpecialFunctions
# rand(d)
# rng = Random.GLOBAL_RNG
#
# α=d.α; R=d.R; n=d.n
# α ∈ 1.10:0.01:1.98 || throw(DomainError(α, "α must lie within `1.10:0.01:1.98`"))
# m = size(R, 1)-1
# funk1 = x -> (2^α)*sin(π*α/2)*gamma((α+2)/2)*gamma((α+x)/2)/(gamma(x/2)*π*α/2)
# funk2 = x -> 4*gamma(x/α)/((α*2^2)*gamma(x/2)^2)
# funkmarg = x -> gamma(x/2)/(gamma((x-1)/2)*sqrt(π))
# c = 1.2
# k1 = (funk1(m), funk1(m+1))
# k2 = (funk2(m), funk2(m+1))
#
# kmarg = funkmarg(m+1)
# onetoendm1 = StaticArrays.SOneTo(size(R,1)-1)
# kappa = det(R)/det(R[onetoendm1, onetoendm1])
# invR = inv(R)
# invRx1 = inv(R[onetoendm1, onetoendm1])
# sigrootx1 = cholesky(R[onetoendm1, onetoendm1]).L
# modefactor = R[end, onetoendm1]'*inv(R[onetoendm1, onetoendm1])
#
# @code_warntype AlphaStableDistributions.subgausscondprobtabulate(α, SVector(1.,2,3,4), 3., invRx1, invR, randn(2,2), 0.1, 0.1, 0.1, [1,2], 0.1, k1, k2, kmarg)
#
#
# @MVector zeros(n)
#
#
#
# using Cthulhu
# d = AlphaSubGaussian(n=96)
# @descend_code_warntype rand(Random.GLOBAL_RNG, d)
