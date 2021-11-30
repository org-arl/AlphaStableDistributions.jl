module AlphaStableDistributions

using LinearAlgebra, Statistics, Random
using StatsBase, Distributions, StaticArrays
using MAT, SpecialFunctions, ToeplitzMatrices
using Interpolations

export AlphaStable, SymmetricAlphaStable, AlphaSubGaussian, fit

Base.@kwdef struct AlphaStable{T} <: Distributions.ContinuousUnivariateDistribution
    α::T = 1.5
    β::T = zero(α)
    scale::T = one(α)
    location::T = zero(α)
end

AlphaStable(α::Integer, β::Integer, scale::Integer, location::Integer) = AlphaStable(float(α), float(β), float(scale), float(location))


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

# lookup tables from McCulloch (1986)
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
const _να = [
    2.439
    2.500
    2.600
    2.700
    2.800
    3.000
    3.200
    3.500
    4.000
    5.000
    6.000
    8.000
    10.000
    15.000
    25.000
]  
const _νβ = [
    0.0 
    0.1 
    0.2
    0.3
    0.5
    0.7
    1.0
]      
const ψ₁ = [
    2.000 2.000 2.000 2.000 2.000 2.000 2.000
    1.916 1.924 1.924 1.924 1.924 1.924 1.924
    1.808 1.813 1.829 1.829 1.829 1.829 1.829
    1.729 1.730 1.737 1.745 1.745 1.745 1.745
    1.664 1.663 1.663 1.668 1.676 1.676 1.676
    1.563 1.560 1.553 1.548 1.547 1.547 1.547
    1.484 1.480 1.471 1.460 1.448 1.438 1.438
    1.391 1.386 1.378 1.364 1.337 1.318 1.318
    1.279 1.273 1.266 1.250 1.210 1.184 1.150
    1.128 1.121 1.114 1.101 1.067 1.027 0.973
    1.029 1.021 1.014 1.004 0.974 0.935 0.874
    0.896 0.892 0.887 0.883 0.855 0.823 0.769
    0.818 0.812 0.806 0.801 0.780 0.756 0.691
    0.698 0.695 0.692 0.689 0.676 0.656 0.595
    0.593 0.590 0.588 0.586 0.579 0.563 0.513
]
const ψ₂ = [
    0.0 2.160 1.000 1.000 1.000 1.000 1.000
    0.0 1.592 3.390 1.000 1.000 1.000 1.000
    0.0 0.759 1.800 1.000 1.000 1.000 1.000
    0.0 0.482 1.048 1.694 2.229 1.000 1.000
    0.0 0.360 0.760 1.232 2.229 1.000 1.000
    0.0 0.253 0.518 0.823 1.575 1.000 1.000
    0.0 0.203 0.410 0.632 1.244 1.906 1.000
    0.0 0.165 0.332 0.499 0.943 1.560 1.000
    0.0 0.136 0.271 0.404 0.689 1.230 2.195
    0.0 0.109 0.216 0.323 0.539 0.827 1.917
    0.0 0.096 0.190 0.284 0.472 0.693 1.759
    0.0 0.082 0.163 0.243 0.412 0.601 1.596
    0.0 0.074 0.147 0.220 0.377 0.546 1.482
    0.0 0.064 0.128 0.191 0.330 0.478 1.362
    0.0 0.056 0.112 0.167 0.285 0.428 1.274
]
const _α =[
    0.5
    0.6
    0.7
    0.8
    0.9
    1.0
    1.1
    1.2
    1.3
    1.4
    1.5
    1.6
    1.7
    1.8
    1.9
    2.0
]
const _β = [
    0.0
    0.25
    0.50
    0.75
    1.00
]
const ϕ₃ = [
    2.588  3.073  4.534  6.636  9.144
    2.337  2.635  3.542  4.808  6.247
    2.189  2.392  3.004  3.844  4.775
    2.098  2.244  2.676  3.265  3.912
    2.040  2.149  2.461  2.886  3.356
    2.000  2.085  2.311  2.624  2.973
    1.980  2.040  2.205  2.435  2.696
    1.965  2.007  2.125  2.294  2.491
    1.955  1.984  2.067  2.188  2.333
    1.946  1.967  2.022  2.106  2.211
    1.939  1.952  1.988  2.045  2.116
    1.933  1.940  1.962  1.997  2.043
    1.927  1.930  1.943  1.961  1.987
    1.921  1.922  1.927  1.936  1.947
    1.914  1.915  1.916  1.918  1.921
    1.908  1.908  1.908  1.908  1.908
]
const ϕ₅ = [
    0.0 0.0    0.0    0.0    0.0
    0.0 -0.017 -0.032 -0.049 -0.064
    0.0 -0.030 -0.061 -0.092 -0.123
    0.0 -0.043 -0.088 -0.132 -0.179
    0.0 -0.056 -0.111 -0.170 -0.232
    0.0 -0.066 -0.134 -0.206 -0.283
    0.0 -0.075 -0.154 -0.241 -0.335
    0.0 -0.084 -0.173 -0.276 -0.390
    0.0 -0.090 -0.192 -0.310 -0.447
    0.0 -0.095 -0.208 -0.346 -0.508
    0.0 -0.098 -0.223 -0.383 -0.576
    0.0 -0.099 -0.237 -0.424 -0.652
    0.0 -0.096 -0.250 -0.469 -0.742
    0.0 -0.089 -0.262 -0.520 -0.853
    0.0 -0.078 -0.272 -0.581 -0.997
    0.0 -0.061 -0.279 -0.659 -1.198
]

"""
    fit(d::Type{<:AlphaStable}, x; alg=QuickSort)

Fit an α stable distribution to data.

returns `AlphaStable`

α∈[0.6,2.0], β∈[-1,1] , c∈[0,∞] and δ∈[-∞,∞] are the characteristic exponent, 
skewness parameter, scale parameter (dispersion^1/α) and location parameter respectively.

α, β, c and δ are computed based on McCulloch (1986) fractile.
"""
function Distributions.fit(::Type{<:AlphaStable}, x::AbstractArray{T}, alg=QuickSort) where {T<:AbstractFloat}
    sx = sort(x, alg=alg)
    p = quantile.(Ref(sx), (0.05, 0.25, 0.28, 0.5, 0.72, 0.75, 0.95), sorted=true)
    να = (p[7]-p[1]) / (p[6]-p[2])
    νβ = (p[7]+p[1]-2p[4]) / (p[7]-p[1])
    (να < _να[1]) && (να = _να[1])
    (να > _να[end]) && (να = _να[end])

    itp₁ = interpolate((_να, _νβ), ψ₁, Gridded(Linear()))
    α = itp₁(να, abs(νβ))
    itp₂ = interpolate((_να, _νβ), ψ₂, Gridded(Linear()))
    β = sign(νβ) * itp₂(να, abs(νβ))

    (β > 1.0) && (β = 1.0)
    (β < -1.0) && (β = -1.0)
    itp₃ = interpolate((_α, _β), ϕ₃, Gridded(Linear()))
    c = (p[6]-p[2]) / itp₃(α, abs(β))
    itp₄ = interpolate((_α, _β), ϕ₅, Gridded(Linear()))
    ζ = p[4] + c * sign(β) * itp₄(α, abs(β))
    if abs(α - 1.0) < 0.1
        δ = ζ
    else
        δ = ζ - β * c * tan(π*α/2)
    end
    return AlphaStable(α=T(α), β=T(β), scale=T(c), location=T(δ))
end

Base.@kwdef struct SymmetricAlphaStable{T} <: Distributions.ContinuousUnivariateDistribution
    α::T = 1.5
    scale::T = one(α)
    location::T = zero(α)
end

"""
    fit(d::Type{<:SymmetricAlphaStable}, x; alg=QuickSort)

Fit a symmetric α stable distribution to data.

returns `SymmetricAlphaStable`

α∈[1,2], c∈[0,∞] and δ∈[-∞,∞] are the characteristic exponent, scale parameter
(dispersion^1/α) and location parameter respectively.

α is computed based on McCulloch (1986) fractile.
scale is computed based on Fama & Roll (1971) fractile.
location is the 50% trimmed mean of the sample.
"""
function Distributions.fit(::Type{<:SymmetricAlphaStable}, x::AbstractArray{T}, alg=QuickSort) where {T<:AbstractFloat}
    sx = sort(x, alg=alg)
    δ = mean(@view(sx[end÷4:(3*end)÷4]))
    p = quantile.(Ref(sx), (0.05, 0.25, 0.28, 0.72, 0.75, 0.95), sorted=true)
    c = (p[4]-p[3]) / 1.654
    an = (p[6]-p[1]) / (p[5]-p[2])
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
    return SymmetricAlphaStable(α=T(α), scale=T(c), location=T(δ))
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
function Base.rand(rng::AbstractRNG, d::AlphaStable{T}) where {T<:AbstractFloat}
    α=d.α; β=d.β; scale=d.scale; loc=d.location
    (α < 0.1 || α > 2) && throw(DomainError(α, "α must be in the range 0.1 to 2"))
    abs(β) > 1 && throw(DomainError(β, "β must be in the range -1 to 1"))
    ϕ = (rand(rng, T) - 0.5) * π
    if α == one(T) && β == zero(T)
        return loc + scale * tan(ϕ)
    end
    w = -log(rand(rng, T))
    α == 2 && (return loc + 2*scale*sqrt(w)*sin(ϕ))
    β == zero(T) && (return loc + scale * ((cos((1-α)*ϕ) / w)^(one(T)/α - one(T)) * sin(α * ϕ) / cos(ϕ)^(one(T)/α)))
    cosϕ = cos(ϕ)
    if abs(α - one(T)) > 1e-8
        ζ = β * tan(π * α / 2)
        aϕ = α * ϕ
        a1ϕ = (one(T) - α) * ϕ
        return loc + scale * (( (sin(aϕ) + ζ * cos(aϕ))/cosϕ * ((cos(a1ϕ) + ζ*sin(a1ϕ))) / ((w*cosϕ)^((1-α)/α)) ))
    end
    bϕ = π/2 + β*ϕ
    x = 2/π * (bϕ * tan(ϕ) - β * log(π/2*w*cosϕ/bϕ))
    α == one(T) || (x += β * tan(π*α/2))

    return loc + scale * x
end

Base.eltype(::Type{<:AlphaStable{T}}) where {T<:AbstractFloat} = T


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
Base.@kwdef struct AlphaSubGaussian{T<:AbstractFloat,M<:AbstractMatrix} <: Distributions.ContinuousUnivariateDistribution
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


function Random.rand!(rng::AbstractRNG, d::AlphaSubGaussian{T}, x::AbstractArray{T}) where {T<:AbstractFloat}
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
    A = rand(AlphaStable(T(α/2), one(T), T(2*cos(π*α/4)^(2.0/α)), zero(T)))
    CT = rand(Chisq(m))
    S = randn(m)
    S = S/sqrt(sum(abs2,S))
    xtmp = ((sigrootx1*sqrt(A*CT))*S)'
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


Base.rand(rng::AbstractRNG, d::AlphaSubGaussian) = rand!(rng, d, zeros(eltype(d), d.n))
Base.eltype(::Type{<:AlphaSubGaussian}) = Float64

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
    c    = ((sum(x->abs(x)^p, x)/xlen)^(1/p))/scale
    for i in 1:m
        tempxlen = xlen-mod(xlen, i)
        xtemp = reshape(x[1:end-mod(xlen, i)], i, tempxlen÷i)
        if mod(tempxlen÷i, 2) != 0
            xtemp = xtemp[:, 1:end-1]
            tempxlen = size(xtemp, 1)*size(xtemp, 2)
        end
        xtemp = reshape(xtemp', 2, tempxlen÷2)
        @views r = (2/(c^p))*(scale^(2-p))*(xtemp[1, :]'*((sign.(xtemp[2, :]).*(abs.(xtemp[2, :]).^(p-1)))))/(tempxlen/2)
        cov[diagind(cov, i)] .+= r
    end
    cov = (cov+cov')+2*(scale^2)*I(m+1)
    cov ./= 2*scale^2
    AlphaSubGaussian(α=α, R=cov, n=length(x))
end

end # module