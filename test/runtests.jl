using AlphaStableDistributions
using Test, Random, Distributions

@testset "Reproducibility" begin
    @test rand(MersenneTwister(0), AlphaStable()         ) == rand(MersenneTwister(0), AlphaStable()         ) 
    @test rand(MersenneTwister(0), SymmetricAlphaStable()) == rand(MersenneTwister(0), SymmetricAlphaStable())
    @test rand(MersenneTwister(0), AlphaSubGaussian(n=10)) == rand(MersenneTwister(0), AlphaSubGaussian(n=10)) 
end

@testset "cf" begin
    rng = MersenneTwister(1)
    for _ in 1:100
        d = AlphaStable(
            α=rand(rng,Uniform(0,2)), 
            β=rand(rng,Uniform(-1,1)), 
            scale=rand(rng,Uniform(0,10)), 
            location=rand(rng,Uniform(0,10)),
        )
        @test cf(d,0) ≈ 1
        x = rand(rng,Uniform(-10,10))
        @test abs(cf(d, x)) <= 1

        d32 = AlphaStable(Float32.(Distributions.params(d))...)
        @test cf(d32, Float32(x)) isa Complex{Float32}
        @test cf(d32, Float32(x)) ≈ cf(d,x) atol=500*eps(Float32)
    end
    for _ in 1:100
        # test stability under convolution
        d = AlphaStable(
            α=rand(rng,Uniform(0.1,2)), 
            scale=1.0,
            location=0.0,
        )
        x = rand(rng,Uniform(-1,1))
        n = rand(rng,1:10^6)
        s = n^inv(d.α)
        @test cf(d, x) ≈ cf(d, x/s)^n
    end
    xs = range(-1,1,length=100)
    d1 = SymmetricAlphaStable(α=2.0, scale=1/sqrt(2), location=0.0)
    d2 = AlphaStable(d1)
    d_ref = Normal(0.0,1.0)
    @test cf.(Ref(d1),xs) ≈ cf.(Ref(d_ref),xs)
    @test cf.(Ref(d2),xs) ≈ cf.(Ref(d_ref),xs)
    
    xs = range(-10,10,length=100)
    d1 = SymmetricAlphaStable(α=1.0, scale=1.0, location=0.0)
    d2 = AlphaStable(d1)
    d_ref = Cauchy(0.0,1.0)
    @test cf.(Ref(d1),xs) ≈ cf.(Ref(d_ref),xs)
    @test cf.(Ref(d2),xs) ≈ cf.(Ref(d_ref),xs)
    
    d1 = SymmetricAlphaStable(α=1.0, scale=17.9, location=42.0)
    d2 = AlphaStable(d1)
    d_ref = Cauchy(42.0,17.9)
    @test cf.(Ref(d1),xs) ≈ cf.(Ref(d_ref),xs)
    @test cf.(Ref(d2),xs) ≈ cf.(Ref(d_ref),xs)
    
    d1 = AlphaStable(α=1/2, β=1.0, scale=12.0, location=-7.2)
    d_ref = Levy(-7.2, 12.0)
    @test cf.(Ref(d1),xs) ≈ cf.(Ref(d_ref),xs)

    @test @inferred(cf(AlphaStable(α = 1.0) , 1.0)) isa Complex{Float64}
    @test @inferred(cf(AlphaStable(α = 1  ) , 1  )) isa Complex{Float64}
    @test @inferred(cf(AlphaStable(α = 1.0) , 1f0)) isa Complex{Float64}
    @test @inferred(cf(AlphaStable(α = 1f0) , 1  )) isa Complex{Float32}
    @test @inferred(cf(AlphaStable(α = 1f0) , 1f0)) isa Complex{Float32}
end

    
@testset "AlphaStableDistributions.jl" begin
    @test AlphaStable(α=1, scale=1.5) === AlphaStable(α=1.0, scale=1.5)
    @test Distributions.params(AlphaStable()) === (1.5, 0.0, 1.0, 0.0)
    @test Distributions.params(SymmetricAlphaStable()) === (1.5, 1.0, 0.0)
    rng = MersenneTwister(0)
    sampletypes = [Float32,Float64]
    stabletypes = [AlphaStable,SymmetricAlphaStable]
    αs = [0.6:0.1:2,1:0.1:2]
    betas = [-1:0.5:1,0.0]
    sc = 2.0
    for sampletype ∈ sampletypes
        for (i, stabletype) in enumerate(stabletypes)
            for α in αs[i]
                for β in betas[i]
                    d1 = if stabletype == AlphaStable 
                        stabletype(α=sampletype(α), β=sampletype(β), scale=sampletype(sc))
                    else
                        stabletype(α=sampletype(α), scale=sampletype(sc))
                    end
                    s = rand(rng, d1, 10^6)
                    @test eltype(s) == sampletype
                    @test any(isinf.(s)) == false

                    d2 = fit(stabletype, s)
                    @test typeof(d2.α) == sampletype 

                    @test d1.α ≈ d2.α rtol=0.1
                    if (stabletype != SymmetricAlphaStable) && (α != 2)
                        @test d1.β ≈ d2.β atol=0.2
                    end
                    # the quantile method is less accurate
                    @test d1.scale ≈ d2.scale rtol=0.2 * sc
                    @test d1.location ≈ d2.location atol=0.9 * sc
                end
            end

            xnormal = rand(rng,Normal(3.0, 4.0), 96000)
            d = fit(stabletype, xnormal)
            @test d.α ≈ 2 rtol=0.2
            stabletype != SymmetricAlphaStable && @test d.β ≈ 0 atol=0.2
            @test d.scale ≈ 4/√2 rtol=0.2
            @test d.location ≈ 3 rtol=0.1

            xcauchy = rand(rng,Cauchy(3.0, 4.0), 96000)
            d = fit(stabletype, xcauchy)
            @test d.α ≈ 1 rtol=0.2
            stabletype != SymmetricAlphaStable && @test d.β ≈ 0 atol=0.2
            @test d.scale ≈ 4 rtol=0.2
            @test d.location ≈ 3 rtol=0.1
        end
    end

    for α in 1.1:0.1:1.9
        d = AlphaSubGaussian(α=α, n=96000)
        x = rand(rng,d)
        x2 = copy(x)
        rand!(rng,d, x2)
        @test x != x2

        d3 = fit(AlphaStable, x)
        @test d3.α ≈ α rtol=0.2
        @test d3.β ≈ 0 atol=0.2
        @test d3.scale ≈ 1 rtol=0.2
        @test d3.location ≈ 0 atol=0.2
    end

    d4 = AlphaSubGaussian(α=1.5, n=96000)
    m = size(d4.R, 1) - 1
    x = rand(rng,d4)
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
# 
# using AlphaStableDistributions
# d1 = AlphaStable(α=1.5)
# s = rand(d1, 100000)
# using ThreadsX
# @btime fit($AlphaStable, $s, $ThreadsX.MergeSort)
