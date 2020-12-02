# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    HierarchicalDensity <: DistLikeDensity

A hierarchical density, useful for hierarchical models/priors.

Constructors:

```julia
HierarchicalDensity(f::Function, parent_density::DistLikeDensity)
```

with a functon `f` that returns a `DistLikeDensity` for any variate `v` drawn
from `parent_dist`.

`Distribution`s are automatically converted to `DistLikeDensity`s.
`varshape(parent_density)` and `varshape(f(v))` must be a `NamedTupleShape`.

Example:

```julia
hd = HierarchicalDensity(
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

v = rand(sampler(hd))
)
```
"""
struct HierarchicalDensity{
    F<:Function,
    D<:DistLikeDensity,
    S<:AbstractValueShape
} <: DistLikeDensity
    f::F
    pd::D
    vs::S
end

export HierarchicalDensity


function HierarchicalDensity(f::Function, pd::Any)
    pd_conv = convert(DistLikeDensity, pd)
    vs_pd = varshape(pd_conv)
    v_pd = rand(bat_determ_rng(), sampler(pd_conv))
    cd = _hd_cd(f, vs_pd, v_pd)
    vs = NamedTupleShape(;vs_pd..., varshape(cd)...)
    HierarchicalDensity(f, pd_conv, vs)
end


function _hd_cd(f::Function, vs_pd::AbstractValueShape, v_pd::AbstractVector{<:Real})
    shaped_v = stripscalar(vs_pd(v_pd))
    convert(AbstractDensity, f(shaped_v))
end

_hd_cd(f::Function, vs_pd::AbstractValueShape, v_pd::Any) = convert(AbstractDensity, f(v_pd))


function _hd_cd(d::HierarchicalDensity, v_pd::Any)
    vs_pd = varshape(d.pd)
    _hd_cd(d.f, vs_pd, v_pd)
end


_nt_type_names(::Type{<:NamedTuple{names}}) where names = names

@inline @generated function _split_nt(nt::NamedTuple, ::Val{names}) where {names}
    all_names = _nt_type_names(nt)
    rest_names = filter(n -> !(n in names), [all_names...])
    expr1 = Expr(:tuple, map(key -> :($key = nt.$key), names)...)
    expr2 = Expr(:tuple, map(key -> :($key = nt.$key), rest_names)...)
    Expr(:tuple, expr1, expr2)
end

@inline _split_nt(nt::NamedTuple, ::NamedTupleShape{names}) where {names} =
    _split_nt(nt, Val(names))

@inline function _split_v(d::HierarchicalDensity, v::NamedTuple)
    vsp = varshape(d.pd)
    _split_nt(v, vsp)
end


function _split_v(d::HierarchicalDensity, v::AbstractVector{<:Real})
    np = totalndof(d.pd)
    idxs = eachindex(v)
    idxs_a = first(idxs):(first(idxs) + np - 1)
    idxs_b = (first(idxs) + np):last(idxs)
    (view(v, idxs_a), view(v, idxs_b))
end



function eval_logval_unchecked(
    density::HierarchicalDensity,
    v::Any
)
    d = density
    v1, v2 = _split_v(d, v)
    logval1 = eval_logval_unchecked(d.pd, v1)
    cd = _hd_cd(d, v1)
    logval2 = eval_logval_unchecked(cd, v2)
    logval1 + logval2
end

ValueShapes.varshape(density::HierarchicalDensity) = density.vs

ValueShapes.totalndof(density::HierarchicalDensity) = totalndof(density.vs)


function Statistics.cov(density::HierarchicalDensity)
    cov(nestedview(rand(bat_determ_rng(), sampler(density), 10^5)))
end


Distributions.sampler(density::HierarchicalDensity) = HierarchicalDensitySampler(density)


var_bounds(density::HierarchicalDensity) = HierarchicalDensityBounds(density)



struct HierarchicalDensitySampler{D<:HierarchicalDensity} <: Sampleable{Multivariate,Continuous}
    d::D
end

function Distributions._rand!(rng::AbstractRNG, s::HierarchicalDensitySampler, v::AbstractVector{<:Real})
    d = s.d
    v1, v2 = _split_v(d, v)
    rand!(rng, sampler(d.pd),  v1)
    cd = _hd_cd(d, v1)
    rand!(rng, sampler(cd),  v2)
    v
end

Base.length(s::HierarchicalDensitySampler) = totalndof(varshape(s.d))



struct HierarchicalDensityBounds{D<:HierarchicalDensity} <: AbstractVarBounds
    d::D
end


ValueShapes.totalndof(bounds::HierarchicalDensityBounds) = totalndof(bounds.d)


function Base.eltype(bounds::HierarchicalDensityBounds)
    d = bounds.d
    v = rand(bat_determ_rng(), sampler(d))
    bounds1 = var_bounds(d.pd)
    v1, v2 = _split_v(d, v)
    cd = _hd_cd(d, v1)
    bounds2 = var_bounds(cd)
    promote_type(eltype(bounds1), eltype(bounds2))
end


function Base.in(v::AbstractVector{<:Real}, bounds::HierarchicalDensityBounds)
    d = bounds.d
    bounds1 = var_bounds(d.pd)
    v1, v2 = _split_v(d, v)
    if v1 in bounds1
        cd = _hd_cd(d, v1)
        bounds2 = var_bounds(cd)
        v2 in bounds2
    else
        false
    end
end

Base.in(v::Any, bounds::HierarchicalDensityBounds) = unshaped(v) in bounds


function renormalize_variate!(v_renorm::AbstractVector{<:Real}, bounds::HierarchicalDensityBounds, v::AbstractVector{<:Real})
    d = bounds.d
    bounds1 = var_bounds(d.pd)
    v1, v2 = _split_v(d, v)
    v1_renorm, v2_renorm = _split_v(d, v_renorm)
    renormalize_variate!(v1_renorm, bounds1, v1)

    new_v = if v1_renorm in bounds1
        cd = _hd_cd(d, v1)
        bounds2 = var_bounds(cd)
        renormalize_variate!(v2_renorm, bounds2, v2)
    end

    v_renorm
end
