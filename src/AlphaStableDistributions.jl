module AlphaStableDistributions

using LinearAlgebra, Statistics, Random
using StatsBase, Distributions, StaticArrays
using MAT, SpecialFunctions, ToeplitzMatrices


export AlphaStable, AlphaSubGaussian, fit

Base.@kwdef struct AlphaStable{T} <: Distributions.ContinuousUnivariateDistribution
    α::T = 1.5
    β::T = zero(α)
    scale::T = one(α)
    location::T = zero(α)
end


# sampler(d::AlphaStable) = error("Not implemented")
# pdf(d::AlphaStable, x::Real) = error("Not implemented")
# logpdf(d::AlphaStable, x::Real) = error("Not implemented")
# cdf(d::AlphaStable, x::Real) = error("Not implemented")
# quantile(d::AlphaStable, q::Real) = error("Not implemented")
# minimum(d::AlphaStable) = error("Not implemented")
# maximum(d::AlphaStable) = error("Not implemented")
# insupport(d::AlphaStable, x::Real) = error("Not implemented")
Statistics.mean(d::AlphaStable) = d.α > 1 ? d.location : error("Not defined")
Statistics.var(d::AlphaStable) = d.α == 2 ? 2d.scale^2 : Inf
# modes(d::AlphaStable) = error("Not implemented")
# mode(d::AlphaStable) = error("Not implemented")
# skewness(d::AlphaStable) = error("Not implemented")
# kurtosis(d::Distribution, ::Bool) = error("Not implemented")
# entropy(d::AlphaStable, ::Real) = error("Not implemented")
# mgf(d::AlphaStable, ::Any) = error("Not implemented")
# cf(d::AlphaStable, ::Any) = error("Not implemented")

# lookup table from McCulloch (1986)
const _ena = [
2.4388
2.5120
2.6080
2.7369
2.9115
3.1480
3.4635
3.8824
4.4468
5.2172
6.3140
7.9098
10.4480
14.8378
23.4831
44.2813
]

"""
Fit a symmetric α stable distribution to data.

:param x: data
:returns: (α, c, δ)

α, c and δ are the characteristic exponent, scale parameter
(dispersion^1/α) and location parameter respectively.

α is computed based on McCulloch (1986) fractile.
c is computed based on Fama & Roll (1971) fractile.
δ is the 50% trimmed mean of the sample.
"""
function Distributions.fit(d::Type{<:AlphaStable}, x)
    δ = mean(StatsBase.trim(x,prop=0.25))
    p = quantile.(Ref(sort(x)), (0.05, 0.25, 0.28, 0.72, 0.75, 0.95), sorted=true)
    c = (p[4]-p[3])/1.654
    an = (p[6]-p[1])/(p[5]-p[2])
    if an < 2.4388
        α = 2.
    else
        α = 0.
        j = findfirst(>=(an), _ena) # _np.where(an <= _ena[:,0])[0]
        (j === nothing || j == length(_ena)) && (j = length(_ena))
        t = (an-_ena[j-1])/(_ena[j]-_ena[j-1])
        α = (22-j-t)/10

    end
    if α < 0.5
        α = 0.5
    end
    return AlphaStable(α=α, β=zero(α), scale=c, location=oftype(α, δ))
end

"""
Generate independent stable random numbers.

:param α: characteristic exponent (0.1 to 2.0)
:param β: skew (-1 to +1)
:param scale: scale parameter
:param loc: location parameter (mean for α > 1, median/mode when β=0)


This implementation is based on the method in J.M. Chambers, C.L. Mallows
and B.W. Stuck, "A Method for Simulating Stable Random Variables," JASA 71 (1976): 340-4.
McCulloch's MATLAB implementation (1996) served as a reference in developing this code.
"""
function Base.rand(rng::AbstractRNG, d::AlphaStable)
    α=d.α; β=d.β; scale=d.scale; loc=d.location
    (α < 0.1 || α > 2) && throw(DomainError(α, "α must be in the range 0.1 to 2"))
    abs(β) > 1 && throw(DomainError(β, "β must be in the range -1 to 1"))
    ϕ = (rand(rng) - 0.5) * π
    if α == 1 && β == 0
        return loc + scale*tan(ϕ)
    end
    w = -log(rand(rng))
    α == 2 && (return loc + 2*scale*sqrt(w)*sin(ϕ))
    β == 0 && (return loc + scale * ((cos((1-α)*ϕ) / w)^(1.0/α - 1) * sin(α * ϕ) / cos(ϕ)^(1.0/α)))
    cosϕ = cos(ϕ)
    if abs(α-1) > 1e-8
        ζ = β * tan(π*α/2)
        aϕ = α * ϕ
        a1ϕ = (1-α) * ϕ
        return loc + scale * (( (sin(aϕ)+ζ*cos(aϕ))/cosϕ * ((cos(a1ϕ)+ζ*sin(a1ϕ))) / ((w*cosϕ)^((1-α)/α)) ))
    end
    bϕ = π/2 + β*ϕ
    x = 2/π * (bϕ*tan(ϕ) - β*log(π/2*w*cosϕ/bϕ))
    α == 1 || (x += β * tan(π*α/2))

    return loc + scale*x
end


"""

Generate alpha-sub-Gaussian (aSG) random numbers.

The implementation is based on https://github.com/ahmd-mahm/alpha-SGNm/blob/master/asgn.m

Reference:
A. Mahmood and M. Chitre, "Generating random variates for stable sub-Gaussian processes
with memory", Signal Processing, Volume 131, Pages 271-279, 2017.
(https://doi.org/10.1016/j.sigpro.2016.08.016.)


# Arguments
- `α`: characteristic exponent associated with the aSGN(m) process. This is
a scalar input and should lie within `collect(1.1:0.01:1.98)`.
- `R`: covariance matrix of any adjacent `m+1` samples in an aSGN(m) process.
The dimension of `R` is equal to `m+1`. It should be a symmetric toeplitz matrix.
The maximum acceptable size of `R` is `10x10`
- `n`: number of samples required

# Examples
```jldoctest
julia> x = rand(AlphaSubGaussian(n=1000))
```
"""
Base.@kwdef struct AlphaSubGaussian{T,M<:AbstractMatrix} <: Distributions.ContinuousUnivariateDistribution
    α::T = 1.50
    R::M = SMatrix{5,5}(collect(SymmetricToeplitz([1.0000, 0.5804, 0.2140, 0.1444, -0.0135])))
    n::Int
end

"""
Generates the conditional probability f(X2|X1) if [X1, X2] is a sub-Gaussian
stable random vector such that X1(i)~X2~S(alpha,delta) and rho is the correlation
coefficient of the underlying Gaussian vector. We assume the joint-probabiluty is given by f(X1,X2).
"""
function subgausscondprobtabulate(α, x1, x2_ind, invRx1, invR, vjoint, nmin, nmax, step, rind, kappa, k1, k2, kmarg)::Float64
    m = length(x1)
    r1 = sqrt(x1'*invRx1*x1)
    x = SVector{length(x1)+1, Float64}(x1..., x2_ind)
    r = sqrt(x'*invR*x)

    if r1<nmin
        grad = (vjoint[m, 1]-k2[1])/nmin
        cons = k2[1]
        vjointR1 = grad*r1+cons
    elseif r1>nmax
        vjointR1 = α*k1[1]*(r1^(-α-m))
    else
        ti = (log10(r1)-log10(nmin))/step+1
        tempind = (floor(Int, ti), ceil(Int, ti))
        grad = (vjoint[m, tempind[1]]-vjoint[m, tempind[2]])/(rind[tempind[1]]-rind[tempind[2]])
        cons = vjoint[m, tempind[1]]-grad*rind[tempind[1]]
        vjointR1 = grad*r1+cons
    end

    if r<nmin
        grad = (vjoint[m+1, 1]-k2[2])/nmin
        cons = k2[2]
        vjointR = grad*r+cons
    elseif r>nmax
        vjointR = α*k1[2]*(r^(-α-m-1))
    else
        ti = (log10(r)-log10(nmin))/step+1
        tempind = (floor(Int, ti), ceil(Int, ti))
        grad = (vjoint[m+1, tempind[1]]-vjoint[m+1, tempind[2]])/(rind[tempind[1]]-rind[tempind[2]])
        cons = vjoint[m+1, tempind[1]]-grad*rind[tempind[1]]
        vjointR = grad*r+cons
    end
    (1/sqrt(kappa))*kmarg*vjointR/vjointR1
end


function Random.rand!(rng::AbstractRNG, d::AlphaSubGaussian, x::AbstractArray)
    α=d.α; R=d.R; n=d.n
    length(x) >= n || throw(ArgumentError("length of x must be at least n"))
    α ∈ 1.10:0.01:1.98 || throw(DomainError(α, "α must lie within `1.10:0.01:1.98`"))
    m = size(R, 1)-1
    funk1 = x -> (2^α)*sin(π*α/2)*gamma((α+2)/2)*gamma((α+x)/2)/(gamma(x/2)*π*α/2)
    funk2 = x -> 4*gamma(x/α)/((α*2^2)*gamma(x/2)^2)
    funkmarg = x -> gamma(x/2)/(gamma((x-1)/2)*sqrt(π))
    c = 1.2
    k1 = (funk1(m), funk1(m+1))
    k2 = (funk2(m), funk2(m+1))
    kmarg = funkmarg(m+1)
    onetom = StaticArrays.SOneTo(m)
    kappa = det(R)/det(R[onetom, onetom])
    invR = inv(R)
    invRx1 = inv(R[onetom, onetom])
    sigrootx1 = cholesky(R[onetom, onetom]).L
    modefactor = R[end, onetom]'/R[onetom, onetom]
    matdict = matread(joinpath(@__DIR__(),"vr_repo/vr_alpha=$(α).mat"))
    nmax, nmin, res, rind, vjoint = matdict["Nmax"]::Float64, matdict["Nmin"]::Float64, matdict["res"]::Float64, vec(matdict["rind"])::Vector{Float64}, matdict["vJoint"]::Matrix{Float64}
    step = (log10(nmax)-log10(nmin))/res
    m>size(vjoint, 1)-1 && throw(DomainError(R, "The dimensions of `R` exceed the maximum possible 10x10"))
    A = rand(AlphaStable(α/2, 1.0, 2*cos(π*α/4)^(2.0/α), 0.0))
    T = rand(Chisq(m))
    S = randn(m)
    S = S/sqrt(sum(abs2,S))
    xtmp = ((sigrootx1*sqrt(A*T))*S)'
    if n<=m
        copyto!(x, @view(xtmp[1:n]))
    else
        # x = zeros(n)
        x[onetom] = xtmp
        vstud = α+m
        norms = pdf(TDist(vstud), 0.0)
        @inbounds for i = m+1:n
            x1 = SVector{m,Float64}(view(x,i-m:i-1))
            mode = modefactor*x1
            norm1 = subgausscondprobtabulate(α, x1, mode, invRx1, invR, vjoint, nmin, nmax, step, rind, kappa, k1, k2, kmarg)
            notaccept = true
            while notaccept
                u = rand()
                v = (norms/norm1)*rand(TDist(vstud)) + mode
                gv = (norm1/norms)*pdf(TDist(vstud), (v-mode)*(norm1/norms))
                fv = subgausscondprobtabulate(α, x1, v, invRx1, invR, vjoint, nmin, nmax, step, rind, kappa, k1, k2, kmarg)
                if c*u <= fv/gv
                    x[i] = v
                    notaccept = false
                end
            end
        end
    end
    x
end


Base.rand(rng::AbstractRNG, d::AlphaSubGaussian) = rand!(rng, d, zeros(d.n))

"""
    fit(d::Type{<:AlphaSubGaussian}, x, m; p=1.0)

Fit an aSGN(m) model to data via the covariation method.

The covariation method requires an additional parameter `p`. Ideally, 1 < p < α. In most practical impulsive scenarios p=1.0 is sufficient.
`m` is the number of lags in the covariance matrix.

The implementation is based on https://github.com/ahmd-mahm/alpha-SGNm/blob/master/param_est/asgnfit.m

Reference:
A. Mahmood and M. Chitre, "Generating random variates for stable sub-Gaussian processes
with memory", Signal Processing, Volume 131, Pages 271-279, 2017.
(https://doi.org/10.1016/j.sigpro.2016.08.016.)
"""
function Distributions.fit(d::Type{<:AlphaSubGaussian}, x::AbstractVector{T}, m::Integer; p=one(T)) where T
    d1   = fit(AlphaStable, x)
    α    = d1.α; scale=d1.scale
    cov  = zeros(T, m+1, m+1)
    xlen = length(x)
    c    = ((sum(abs.(x).^p)/xlen)^(1/p))/scale
    for i in 1:m
        tempxlen = xlen-mod(xlen, i)
        xtemp = reshape(x[1:end-mod(xlen, i)], i, tempxlen÷i)
        if mod(tempxlen÷i, 2) != 0
            xtemp = xtemp[:, 1:end-1]
            tempxlen = size(xtemp, 1)*size(xtemp, 2)
        end
        xtemp = reshape(xtemp', 2, tempxlen÷2)
        r = (2/(c^p))*(scale^(2-p))*(xtemp[1, :]'*((sign.(xtemp[2, :]).*(abs.(xtemp[2, :]).^(p-1)))))/(tempxlen/2)
        cov[diagind(cov, i)] .+= r
    end
    cov = (cov+cov')+2*(scale^2)*I(m+1)
    cov ./= 2*scale^2
    AlphaSubGaussian(α=α, R=cov, n=length(x))
end

end # module
