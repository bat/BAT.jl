# This file is a part of BAT.jl, licensed under the MIT License (MIT).


# Workaround for Distributions.jl issue #647
_iszero(x) = iszero(x)
_iszero(::Distributions.ZeroVector) = true


function _check_rand_compat(s::Sampleable{Multivariate}, A::Union{AbstractVector,AbstractMatrix})
    size(A, 1) == length(s) || throw(DimensionMismatch("Output size inconsistent with sample length."))
    nothing
end



@doc """
    bat_sampler(d::Distribution)

Tries to return a BAT-compatible sampler for Distribution d. A sampler is
BAT-compatible if it supports random number generation using an arbitrary
`AbstractRNG`:

    rand(rng::AbstractRNG, s::SamplerType)
    rand!(rng::AbstractRNG, s::SamplerType, x::AbstractArray)

If no specific method of `bat_sampler` is defined for the type of `d`, it will
default to `sampler(d)`, which may or may not return a BAT-compatible
sampler.
"""
function bat_sampler end
export bat_sampler

bat_sampler(d::Distribution) = Distributions.sampler(d)



@doc """
    issymmetric_around_origin(d::Distribution)

Returns `true` (resp. `false`) if the Distribution is symmetric (resp.
non-symmetric) around the origin.
"""
function issymmetric_around_origin end
export issymmetric_around_origin


issymmetric_around_origin(d::Normal) = d.μ ≈ 0

issymmetric_around_origin(d::Gamma) = false

issymmetric_around_origin(d::Chisq) = false

issymmetric_around_origin(d::TDist) = true

issymmetric_around_origin(d::MvNormal) = _iszero(d.μ)

issymmetric_around_origin(d::Distributions.GenericMvTDist) = d.zeromean



function get_cov end


function set_cov end
