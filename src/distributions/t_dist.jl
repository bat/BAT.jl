# This file is a part of BAT.jl, licensed under the MIT License (MIT).


get_cov(d::Distributions.GenericMvTDist) = d.Σ
set_cov(d::Distributions.GenericMvTDist{T,M}, Σ::M) where {T,M} = Distributions.GenericMvTDist{T,M}(d.df, d.dim, d.zeromean, deepcopy(d.μ), Σ)
set_cov(d::Distributions.GenericMvTDist{T,M}, Σ::AbstractMatrix{<:Real}) where {T,M<:PDMat} = Distributions.GenericMvTDist{T,M}(d.df, d.dim, d.zeromean, deepcopy(d.μ), PDMat(Σ))


_tdist_scaling_factor(rng::AbstractRNG, x::AbstractVector, df::Real) = sqrt(df / rand(rng, bat_sampler(Chisq(df))))

function _tdist_scaling_factor(rng::AbstractRNG, x::AbstractMatrix, df::Real)
    scaling_factor = similar(x, 1, size(x, 2)) # TODO: Avoid memory allocation
    rand!(rng, bat_sampler(Chisq(df)), scaling_factor)
    scaling_factor .= sqrt.(df ./ scaling_factor)
end


export BATTDistSampler

struct BATTDistSampler{T<:AbstractFloat} <: BATSampler{Univariate,Continuous}
    d::TDist{T}
end

Base.eltype(s::BATTDistSampler{T}) where T = T

function Base.rand(rng::AbstractRNG, s::BATTDistSampler{T}) where T
    ndof = dof(s.d)
    x = randn(rng, T)
    if ndof == Inf
        convert(T, x)
    else
        convert(T, x * sqrt(ndof / rand(rng, bat_sampler(Chisq(ndof)))))
    end
end

bat_sampler(d::TDist) = BATTDistSampler(d)



export BATMvTDistSampler

struct BATMvTDistSampler{T<:Distributions.GenericMvTDist} <: BATSampler{Multivariate,Continuous}
    d::T
end


Base.length(s::BATMvTDistSampler) = length(s.d)

Base.eltype(s::BATMvTDistSampler) = eltype(s.d.Σ)


function Random.rand!(rng::AbstractRNG, s::BATMvTDistSampler, x::StridedVecOrMat{T}) where {T<:Real}
    d = s.d

    scaling_factor = _tdist_scaling_factor(rng, x, d.df)
    unwhiten!(d.Σ, randn!(rng, x))  # TODO: Avoid memory allocation (caused by Σ.chol[:UL] in unwhiten! implementation)
    broadcast!(muladd, x, x, scaling_factor, d.μ)
    x
end


bat_sampler(d::Distributions.GenericMvTDist) = BATMvTDistSampler(d)
