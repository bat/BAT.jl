# This file is a part of BAT.jl, licensed under the MIT License (MIT).


_variate_form(::Distribution{VF}) where VF = VF


function _direct_valueshape_sum(vs_a::Union{ScalarShape{T},ArrayShape{T,1}}, vs_b::Union{ScalarShape{<:T},ArrayShape{<:T,1}}) where T
    n = totalndof(vs_a) + totalndof(vs_b)
    ArrayShape{T}(n)
end

_ntshape_names(::NamedTupleShape{names}) where names = names

function _direct_valueshape_sum(vs_a::NamedTupleShape{names_a,AT,VT}, vs_b::NamedTupleShape{names_b}) where {names_a,AT,VT,names_b}
    r = NamedTupleShape(VT; vs_a..., vs_b...)
    _ntshape_names(r) == (names_a..., names_b...) || throw(ArgumentError("Can't generate direct sum of NamedTupleShapes that share field names"))
    r
end

_direct_variateform_sum(::Type{<:Union{Univariate,Multivariate}}, ::Type{<:Union{Univariate,Multivariate}}) = Multivariate

function _direct_variateform_sum(::Type{ValueShapes.NamedTupleVariate{names_a}}, ::Type{ValueShapes.NamedTupleVariate{names_b}}) where {names_a,names_b}
    ValueShapes.NamedTupleVariate{(names_a...,names_b...)}
end



"""
    struct HierarchicalDistribution <: ContinuousDistribution

*Experimental feature, not part of stable public API.*

A hierarchical distribution, useful for hierarchical models/priors.

Constructors:

* ```HierarchicalDistribution(f::Function, primary_dist::NamedTupleDist)```

with a functon `f` that returns a `ContinuousDistribution` for any
variate `v` drawn from `primary_dist`.

Example:

```julia
hd = HierarchicalDistribution(
    v -> NamedTupleDist(
        baz = fill(Normal(v.bar, v.foo), 3)
    ),
    NamedTupleDist(
        foo = Exponential(3.5),
        bar = Normal(2.0, 1.0)
    )
)

varshape(hd) == NamedTupleShape(
    foo = ScalarShape{Real}(),
    bar = ScalarShape{Real}(),
    baz = ArrayShape{Real}(3)
)

v = rand(hd)
```

!!! note

    All fields of `HierarchicalDistribution` are considered internal and
    subject to change without deprecation.
"""
struct HierarchicalDistribution{
    VF <: VariateForm,
    T <: Real,
    F <: Function,
    PD <: ContinuousDistribution,
    VS <: AbstractValueShape,
    SVS <: AbstractValueShape
} <: Distribution{VF,Continuous}
    f::F
    pdist::PD
    vs::VS
    secondary_vs::SVS
    effndof::Int
end

export HierarchicalDistribution


function HierarchicalDistribution(f::Function, primary_dist::ContinuousDistribution)
    vs_primary = varshape(primary_dist)
    vf_primary = _variate_form(primary_dist)
    x_primary_us = rand(bat_determ_rng(), unshaped(primary_dist))
    x_primary = vs_primary(x_primary_us)

    secondary_dist = f(x_primary)
    vs_secondary = varshape(secondary_dist)
    vf_secondary = _variate_form(secondary_dist)
    if any(x -> x isa ConstValueShape, values((;vs_secondary...)))
        throw(ArgumentError("Value shape of secondary distributions of HierarchicalDistribution must not contain a ConstValueShape"))
    end

    vs = _direct_valueshape_sum(vs_primary, vs_secondary)
    VF = _direct_variateform_sum(vf_primary, vf_secondary)
    
    @assert totalndof(vs) == length(unshaped(primary_dist)) + length(unshaped(secondary_dist))

    T = promote_type(eltype(unshaped(primary_dist)), eltype(unshaped(secondary_dist)))
    F = typeof(f)
    PD = typeof(primary_dist)
    VS = typeof(vs)
    SVS = typeof(vs_secondary)

    effndof = eff_totalndof(primary_dist) + eff_totalndof(secondary_dist)

    HierarchicalDistribution{VF,T,F,PD,VS,SVS}(f, primary_dist, vs, vs_secondary, effndof)
end


ValueShapes.varshape(d::HierarchicalDistribution) = d.vs

Base.length(d::HierarchicalDistribution) = totalndof(varshape(d))

eff_totalndof(d::HierarchicalDistribution) = d.effndof


function Base.show(io::IO, d::HierarchicalDistribution)
    print(io, Base.typename(typeof(d)).name, "(")
    Base.show(io, d.f)
    print(io, ", ")
    Base.show(io, d.pdist)
    print(io, ")")
end



struct UnshapedHDist{
    VF <: VariateForm,
    T <: Real,
    F <: Function,
    PD <: ContinuousDistribution,
    VS <: AbstractValueShape,
    SVS <: AbstractValueShape
} <: Distribution{Multivariate,Continuous}
    shaped::HierarchicalDistribution{VF,T,F,PD,VS,SVS}
end


ValueShapes.unshaped(d::HierarchicalDistribution) = UnshapedHDist(d)

Base.length(ud::UnshapedHDist) = length(ud.shaped)

Base.eltype(ud::UnshapedHDist{VF,T}) where {VF,T} = T



_hd_pridist(d::HierarchicalDistribution) = d.pdist

function _hd_secdist(d::HierarchicalDistribution, x::Any)
    secondary_dist = d.f(x)
    @assert eff_totalndof(d.pdist) + eff_totalndof(secondary_dist) == d.effndof
    secondary_dist
end

_hd_pridist(ud::UnshapedHDist) = unshaped(_hd_pridist(ud.shaped))
_hd_secdist(ud::UnshapedHDist, x::AbstractVector{<:Real}) = unshaped(_hd_secdist(ud.shaped, varshape(ud.shaped.pdist)(x)))



#=
@inline @generated function _split_nt(nt::NamedTuple, ::Val{names}) where {names}
    all_names = _nt_type_names(nt)
    rest_names = filter(n -> !(n in names), [all_names...])
    expr1 = Expr(:tuple, map(key -> :($key = nt.$key), names)...)
    expr2 = Expr(:tuple, map(key -> :($key = nt.$key), rest_names)...)
    Expr(:tuple, expr1, expr2)
end

@inline _split_nt(nt::NamedTuple, ::NamedTupleShape{names}) where {names} = _split_nt(nt, Val(names))

@inline function _hd_split(d::HierarchicalDistribution{ValueShapes.NamedTupleVariate{names}}, x::NamedTuple{names}) where names
    vsp = varshape(d.pd)
    _split_nt(x, vsp)
end

@inline function _hd_split(d::HierarchicalDistribution{ValueShapes.NamedTupleVariate{names}}, x::ShapedAsNT{names}) where names
    x_primary_us, x_secondary_us = _hd_split(unshaped(d), x)
    varshape(d.pdist)(x_primary_us), d.secondary_vs(x_secondary_us)
end
=#

function _hd_split(ud::UnshapedHDist, x::AbstractVector{<:Real})
    @argcheck length(ud) == length(eachindex(x))
    np = length(_hd_pridist(ud))
    idxs = eachindex(x)
    idxs_primary = first(idxs):(first(idxs) + np - 1)
    idxs_secondary = (first(idxs) + np):last(idxs)
    (view(x, idxs_primary), view(x, idxs_secondary))
end


function Distributions.logpdf(ud::UnshapedHDist, x::AbstractVector{<:Real})
    x_primary, x_secondary = _hd_split(ud, x)
    logval1 = logpdf(_hd_pridist(ud), x_primary)
    T = typeof(logval1)
    R = promote_type(T, eltype(x))
    # ToDo: Use insupport(_hd_pridist(ud), x_primary) in the future? insupport not yet available for NamedTupleDist.
    if logval1 > convert(T, -Inf)
        logval2 = logpdf(_hd_secdist(ud, x_primary), x_secondary)
        convert(R, logval1 + logval2)
    else
        convert(R, logval1)
    end
end


Distributions.pdf(d::HierarchicalDistribution, x::Any) where names = exp(logpdf(d, x))

Distributions.pdf(ud::UnshapedHDist, x::AbstractVector{<:Real}) = exp(logpdf(ud, x))


Random.rand(rng::AbstractRNG, d::HierarchicalDistribution) = varshape(d)(rand(rng, unshaped(d)))
 

function Random.rand(rng::AbstractRNG, d::HierarchicalDistribution, dims::Dims)
    ud = unshaped(d)
    X_flat = Array{eltype(ud)}(undef, length(ud), dims...)
    X = ArrayOfSimilarVectors(X_flat)
    rand!.(Ref(rng), Ref(ud), X)
    varshape(d).(X)
end

function Distributions._rand!(rng::AbstractRNG, ud::UnshapedHDist, x::AbstractVector{<:Real})
    x_primary, x_secondary = _hd_split(ud, x)
    rand!(rng, _hd_pridist(ud), x_primary)
    rand!(rng, _hd_secdist(ud, x_primary), x_secondary)
    x
end


function Distributions.insupport(ud::UnshapedHDist, x::AbstractVector)
    x_primary, x_secondary = _hd_split(ud, x)
    primary_dist = _hd_pridist(ud)
    insupport(primary_dist, x_primary) && insupport(secondary_dist, _hd_secdist(ud, x_primary))
end


function Statistics.mean(ud::UnshapedHDist)
    mean(nestedview(rand(bat_determ_rng(), ud, 10^5)))
end

function Statistics.cov(ud::UnshapedHDist)
    cov(nestedview(rand(bat_determ_rng(), ud, 10^5)))
end
