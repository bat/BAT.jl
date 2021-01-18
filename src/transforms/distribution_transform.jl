# This file is a part of BAT.jl, licensed under the MIT License (MIT).


function eff_totalndof end

eff_totalndof(d::Distribution) = length(d)

# NamedTupleDist, e.g., currently doesn't support `length()`:
eff_totalndof(d::Distribution{<:ValueShapes.StructVariate}) = totalndof(varshape(d))


struct DistributionTransform{
    VF <: VariateForm,
    ST <: VariateSpace,
    SF <: VariateSpace,
    DT <: ContinuousDistribution,
    DF <: Distribution{VF,Continuous}
} <: VariateTransform{VF,ST,SF}
    target_dist::DT
    source_dist::DF
    target_space::ST
    source_space::SF
end

# ToDo: Add field to cache dist-specific pre-calculated values? May be useful
# for truncated dists and others.


function _distrafo_ctor_impl(target_dist::Distribution, source_dist::Distribution)
    @argcheck eff_totalndof(target_dist) == eff_totalndof(source_dist)
    target_space = getspace(target_dist)
    source_space = getspace(source_dist)
    DistributionTransform(target_dist, source_dist, target_space, source_space)
end

DistributionTransform(target_dist::Distribution{VF,Continuous}, source_dist::Distribution{VF,Continuous}) where VF =
    _distrafo_ctor_impl(target_dist, source_dist)

DistributionTransform(target_dist::Distribution{Multivariate,Continuous}, source_dist::Distribution{VF,Continuous}) where VF =
    _distrafo_ctor_impl(target_dist, source_dist)

DistributionTransform(target_dist::Distribution{VF,Continuous}, source_dist::Distribution{Multivariate,Continuous}) where VF =
    _distrafo_ctor_impl(target_dist, source_dist)

DistributionTransform(target_dist::Distribution{Multivariate,Continuous}, source_dist::Distribution{Multivariate,Continuous}) =
    _distrafo_ctor_impl(target_dist, source_dist)


# apply_dist_trafo(trg_d, src_d, src_v, prev_ladj)
function apply_dist_trafo end


target_space(trafo::DistributionTransform) = trafo.target_space
source_space(trafo::DistributionTransform) = trafo.source_space

import Base.∘
function ∘(a::DistributionTransform, b::DistributionTransform)
    @argcheck a.source_dist == b.target_dist
    DistributionTransform(a.target_dist, b.source_dist)
end

Base.inv(trafo::DistributionTransform) = DistributionTransform(trafo.source_dist, trafo.target_dist)

ValueShapes.varshape(trafo::DistributionTransform) = varshape(trafo.target_dist)


function apply_vartrafo_impl(trafo::DistributionTransform, v::Any, prev_ladj::Real)
    apply_dist_trafo(trafo.target_dist, trafo.source_dist, v, prev_ladj)
end


function apply_vartrafo_impl(trafo::InvVT{<:DistributionTransform}, v::Any, prev_ladj::Real)
    apply_vartrafo_impl(inv(trafo.orig), v, prev_ladj)
end



const _StdDistType = Union{Uniform, Normal}

_trg_disttype(::Type{Uniform}, ::Type{Univariate}) = StandardUvUniform
_trg_disttype(::Type{Uniform}, ::Type{Multivariate}) = StandardMvUniform
_trg_disttype(::Type{Normal}, ::Type{Univariate}) = StandardUvNormal
_trg_disttype(::Type{Normal}, ::Type{Multivariate}) = StandardMvNormal

function _trg_dist(disttype::Type{<:_StdDistType}, source_dist::Distribution{Univariate,Continuous})
    trg_dt = _trg_disttype(disttype, Univariate)
    trg_dt()
end

function _trg_dist(disttype::Type{<:_StdDistType}, source_dist::Distribution{Multivariate,Continuous})
    trg_dt = _trg_disttype(disttype, Multivariate)
    trg_dt(eff_totalndof(source_dist))
end

function _trg_dist(disttype::Type{<:_StdDistType}, source_dist::ContinuousDistribution)
    trg_dt = _trg_disttype(disttype, Multivariate)
    trg_dt(eff_totalndof(source_dist))
end


function DistributionTransform(disttype::Type{<:_StdDistType}, source_dist::ContinuousDistribution)
    trg_d = _trg_dist(disttype, source_dist)
    DistributionTransform(trg_d, source_dist)
end



function std_dist_from(src_d::Distribution)
    throw(ArgumentError("No standard intermediate distribution defined to tranform into from $(typename(src_d).name)"))
end

function std_dist_to(trg_d::Distribution)
    throw(ArgumentError("No standard intermediate distribution defined to tranform into $(typename(trg_d).name) from"))
end


@inline function _intermediate_std_dist(trg_d::Distribution, src_d::Distribution)
    _select_intermediate_dist(std_dist_to(trg_d), std_dist_from(src_d))
end

@inline _intermediate_std_dist(::Union{StdUvDist,StdMvDist}, src_d::Distribution) = std_dist_from(src_d)

@inline _intermediate_std_dist(trg_d::Distribution, ::Union{StdUvDist,StdMvDist}) = std_dist_to(trg_d)

function _intermediate_std_dist(::Union{StdUvDist,StdMvDist}, ::Union{StdUvDist,StdMvDist})
    throw(ArgumentError("Direct conversions must be used between standard intermediate distributions"))
end

@inline _select_intermediate_dist(a::D, ::D) where D<:Union{StdUvDist,StdMvDist} = a
@inline _select_intermediate_dist(a::D, ::D) where D<:Union{StandardUvUniform,StandardMvUniform} = a
@inline _select_intermediate_dist(a::Union{StandardUvUniform,StandardMvUniform}, ::Union{StdUvDist,StdMvDist}) = a
@inline _select_intermediate_dist(::Union{StdUvDist,StdMvDist}, b::Union{StandardUvUniform,StandardMvUniform}) = b

_check_conv_eff_totalndof(trg_d::Uniform, src_d::Uniform) = nothing

function _check_conv_eff_totalndof(trg_d::Distribution, src_d::Distribution)
    trg_d_n = eff_totalndof(trg_d)
    src_d_n = eff_totalndof(src_d)
    if trg_d_n != src_d_n
        throw(ArgumentError("Can't convert to $(typeof(trg_d).name) with $(trg_d_n) eff. DOF from $(typeof(src_d).name) with $(src_d_n) eff. DOF"))
    end
    nothing
end

function apply_dist_trafo(trg_d::Distribution, src_d::Distribution, src_v::Any, prev_ladj::Real)
    _check_conv_eff_totalndof(trg_d, src_d)
    intermediate_d = _intermediate_std_dist(trg_d, src_d)
    intermediate_v, intermediate_ladj = apply_dist_trafo(intermediate_d, src_d, src_v, prev_ladj)
    apply_dist_trafo(trg_d, intermediate_d, intermediate_v, intermediate_ladj)
end


function apply_dist_trafo(trg_d::DT, src_d::DT, src_v::Real, prev_ladj::Real) where {DT <: StdUvDist}
    (v = src_v, ladj = prev_ladj)
end

function apply_dist_trafo(trg_d::DT, src_d::DT, src_v::AbstractVector{<:Real}, prev_ladj::Real) where {DT <: StdMvDist}
    @argcheck length(trg_d) == length(src_d) == length(eachindex(src_v))
    (v = src_v, ladj = prev_ladj)
end


function apply_dist_trafo(trg_d::Distribution{Univariate}, src_d::StdMvDist, src_v::AbstractVector{<:Real}, prev_ladj::Real)
    @argcheck length(src_d) == length(eachindex(src_v)) == 1
    apply_dist_trafo(trg_d, view(src_d, 1), first(src_v), prev_ladj)
end

function apply_dist_trafo(trg_d::StdMvDist, src_d::Distribution{Univariate}, src_v::Real, prev_ladj::Real)
    @argcheck length(trg_d) == 1
    r = apply_dist_trafo(view(trg_d, 1), src_d, first(src_v), prev_ladj)
    (v = unshaped(r.v), ladj = r.ladj)
end



@inline function _value_and_ladj(::typeof(cdf), d::Distribution{Univariate,Continuous}, x::Real)
    u = cdf(d, x)
    ladj = + logpdf(d, x)
    (u, ladj)
end

@inline function _value_and_ladj(::typeof(quantile), d::Distribution{Univariate,Continuous}, u::Real)
    x = quantile(d, u)
    ladj = - logpdf(d, x)
    (x, ladj)
end

function _eval_distr_trafo_error(f::Function, d::Distribution, x::Any)
    ErrorException("Evaluating transformation via $f for distribution of type $(typeof(d)) failed at $x")
end

function _eval_dist_trafo_func(f::Function, d::Distribution{Univariate,Continuous}, src_v::Real, prev_ladj::Real)
    try
        if isnan(prev_ladj)
            trg_v = f(d, src_v)
            var_trafo_result(trg_v, src_v)
        else
            trg_v, trafo_ladj = _value_and_ladj(f, d, src_v)
            var_trafo_result(trg_v, src_v, trafo_ladj, prev_ladj)
        end
    catch err
        if err isa DomainError
            var_trafo_result(NaN, src_v)
        else
            rethrow(_eval_distr_trafo_error(f, d, src_v))
        end
    end
end

std_dist_from(src_d::Distribution{Univariate,Continuous}) = StandardUvUniform()

function apply_dist_trafo(::StandardUvUniform, src_d::Distribution{Univariate,Continuous}, src_v::Real, prev_ladj::Real)
    _eval_dist_trafo_func(cdf, src_d, src_v, prev_ladj)
end

std_dist_to(trg_d::Distribution{Univariate,Continuous}) = StandardUvUniform()

function apply_dist_trafo(trg_d::Distribution{Univariate,Continuous}, ::StandardUvUniform, src_v::Real, prev_ladj::Real)
    TV = float(typeof(src_v))
    # Avoid src_v ≈ 0 and src_v ≈ 1 to avoid infinite variate values for target distributions with infinite support:
    mod_src_v = ifelse(src_v == 0, zero(TV) + eps(TV), ifelse(src_v == 1, one(TV) - eps(TV), convert(TV, src_v)))
    trg_v, ladj = _eval_dist_trafo_func(quantile, trg_d, mod_src_v, prev_ladj)
    (v = trg_v, ladj = ladj)
end



function _dist_trafo_rescale_impl(trg_d, src_d, src_v::Real, prev_ladj::Real)
    R = float(typeof(src_v))
    trg_offs, trg_scale = location(trg_d), scale(trg_d)
    src_offs, src_scale = location(src_d), scale(src_d)
    rescale_factor = trg_scale / src_scale
    trg_v = (src_v - src_offs) * rescale_factor + trg_offs
    trafo_ladj = log(rescale_factor)
    var_trafo_result(trg_v, src_v, trafo_ladj, prev_ladj)
end

apply_dist_trafo(trg_d::Uniform, src_d::Uniform, src_v::Real, prev_ladj::Real) = _dist_trafo_rescale_impl(trg_d, src_d, src_v, prev_ladj)
apply_dist_trafo(trg_d::StandardUvUniform, src_d::Uniform, src_v::Real, prev_ladj::Real) = _dist_trafo_rescale_impl(trg_d, src_d, src_v, prev_ladj)
apply_dist_trafo(trg_d::Uniform, src_d::StandardUvUniform, src_v::Real, prev_ladj::Real) = _dist_trafo_rescale_impl(trg_d, src_d, src_v, prev_ladj)

# ToDo: Use StandardUvNormal as standard intermediate dist for Normal? Would
# be useful if StandardUvNormal would be a better standard intermediate than
# StandardUvUniform for some other uniform distributions as well.
#
#     std_dist_from(src_d::Normal) = StandardUvNormal()
#     std_dist_to(trg_d::Normal) = StandardUvNormal()

apply_dist_trafo(trg_d::Normal, src_d::Normal, src_v::Real, prev_ladj::Real) = _dist_trafo_rescale_impl(trg_d, src_d, src_v, prev_ladj)
apply_dist_trafo(trg_d::StandardUvNormal, src_d::Normal, src_v::Real, prev_ladj::Real) = _dist_trafo_rescale_impl(trg_d, src_d, src_v, prev_ladj)
apply_dist_trafo(trg_d::Normal, src_d::StandardUvNormal, src_v::Real, prev_ladj::Real) = _dist_trafo_rescale_impl(trg_d, src_d, src_v, prev_ladj)


# ToDo: Optimized implementation for Distributions.Truncated <-> StandardUvUniform



function apply_dist_trafo(trg_d::StandardMvUniform, src_d::StandardMvNormal, src_v::AbstractVector{<:Real}, prev_ladj::Real)
    @argcheck length(trg_d) == length(src_d)
    apply_dist_trafo(convert(Product, trg_d), convert(Product, src_d), src_v, prev_ladj)
end

function apply_dist_trafo(trg_d::StandardMvNormal, src_d::StandardMvUniform, src_v::AbstractVector{<:Real}, prev_ladj::Real)
    @argcheck length(trg_d) == length(src_d)
    apply_dist_trafo(convert(Product, trg_d), convert(Product, src_d), src_v, prev_ladj)
end


std_dist_from(src_d::MvNormal) = StandardMvNormal(length(src_d))

function apply_dist_trafo(trg_d::StandardMvNormal, src_d::MvNormal, src_v::AbstractVector{<:Real}, prev_ladj::Real)
    @argcheck length(trg_d) == length(src_d)
    A = cholesky(src_d.Σ).U
    trg_v = transpose(A) \ (src_v - src_d.μ)
    trafo_ladj = -logabsdet(A)[1]
    var_trafo_result(trg_v, src_v, trafo_ladj, prev_ladj)
end

std_dist_to(trg_d::MvNormal) = StandardMvNormal(length(trg_d))

function apply_dist_trafo(trg_d::MvNormal, src_d::StandardMvNormal, src_v::AbstractVector{<:Real}, prev_ladj::Real)
    @argcheck length(trg_d) == length(src_d)
    A = cholesky(trg_d.Σ).U
    trg_v = transpose(A) * src_v + trg_d.μ
    trafo_ladj = logabsdet(A)[1]
    var_trafo_result(trg_v, src_v, trafo_ladj, prev_ladj)
end


function apply_dist_trafo(trg_d::Distributions.Product, src_d::Distributions.Product, src_v::AbstractVector{<:Real}, prev_ladj::Real)
    rs = apply_dist_trafo.(trg_d.v, src_d.v, src_v, zero(Float32))
    trg_v = broadcast(r -> r.v, rs)
    trafo_ladj = sum(map(r -> r.ladj, rs))
    var_trafo_result(trg_v, src_v, trafo_ladj, prev_ladj)
end

function apply_dist_trafo(trg_d::StdMvDist, src_d::Distributions.Product, src_v::AbstractVector{<:Real}, prev_ladj::Real)
    apply_dist_trafo(convert(Distributions.Product, trg_d), src_d, src_v, prev_ladj)
end

function apply_dist_trafo(trg_d::Distributions.Product, src_d::StdMvDist, src_v::AbstractVector{<:Real}, prev_ladj::Real)
    apply_dist_trafo(trg_d, convert(Distributions.Product, src_d), src_v, prev_ladj)
end


function _ntdistelem_to_stdmv(trg_d::StdMvDist, sd::Distribution, src_v_unshaped::AbstractVector{<:Real}, acc::ValueAccessor)
    td = view(trg_d, ValueShapes.view_range(Base.OneTo(length(trg_d)), acc))
    sv = stripscalar(view(src_v_unshaped, acc))
    apply_dist_trafo(td, sd, sv, 0)
end

function _ntdistelem_to_stdmv(trg_d::StdMvDist, sd::ConstValueDist, src_v_unshaped::AbstractVector{<:Real}, acc::ValueAccessor)
    (v = Bool[], ladj = zero(Float32))
end

function apply_dist_trafo(trg_d::StdMvDist, src_d::ValueShapes.UnshapedNTD, src_v::AbstractVector{<:Real}, prev_ladj::Real)
    src_vs = varshape(src_d.shaped)
    @argcheck length(trg_d) == length(eachindex(src_v))
    rs = map((acc, sd) -> _ntdistelem_to_stdmv(trg_d, sd, src_v, acc), values(src_vs), values(src_d.shaped))
    trg_v = vcat(map(r -> r.v, rs)...)
    trafo_ladj = sum(map(r -> r.ladj, rs))
    var_trafo_result(trg_v, src_v, trafo_ladj, prev_ladj)
end

function apply_dist_trafo(trg_d::StdMvDist, src_d::NamedTupleDist, src_v::Union{NamedTuple,ShapedAsNT}, prev_ladj::Real)
    src_v_unshaped = unshaped(src_v, varshape(src_d))
    apply_dist_trafo(trg_d, unshaped(src_d), src_v_unshaped, prev_ladj)
end

function _stdmv_to_ntdistelem(td::Distribution, src_d::StdMvDist, src_v::AbstractVector{<:Real}, acc::ValueAccessor)
    sd = view(src_d, ValueShapes.view_range(Base.OneTo(length(src_d)), acc))
    sv = view(src_v, ValueShapes.view_range(axes(src_v, 1), acc))
    apply_dist_trafo(td, sd, sv, 0)
end

function _stdmv_to_ntdistelem(td::ConstValueDist, src_d::StdMvDist, src_v::AbstractVector{<:Real}, acc::ValueAccessor)
    (v = Bool[], ladj = zero(Float32))
end

function apply_dist_trafo(trg_d::ValueShapes.UnshapedNTD, src_d::StdMvDist, src_v::AbstractVector{<:Real}, prev_ladj::Real)
    trg_vs = varshape(trg_d.shaped)
    @argcheck totalndof(trg_vs) == length(src_d)
    rs = map((acc, td) -> _stdmv_to_ntdistelem(td, src_d, src_v, acc), values(trg_vs), values(trg_d.shaped))
    trg_v_unshaped = vcat(map(r -> unshaped(r.v), rs)...)
    trafo_ladj = sum(map(r -> r.ladj, rs))
    var_trafo_result(trg_v_unshaped, src_v, trafo_ladj, prev_ladj)
end

function apply_dist_trafo(trg_d::NamedTupleDist, src_d::StdMvDist, src_v::AbstractVector{<:Real}, prev_ladj::Real)
    unshaped_result = apply_dist_trafo(unshaped(trg_d), src_d, src_v, prev_ladj)
    (v = varshape(trg_d)(unshaped_result.v), ladj = unshaped_result.ladj)
end


function apply_dist_trafo(trg_d::Distribution{Multivariate}, src_d::ReshapedDist, src_v::Any, prev_ladj::Real)
    src_vs = varshape(src_d)
    @argcheck length(trg_d) == totalndof(src_vs)
    apply_dist_trafo(trg_d, unshaped(src_d), unshaped(src_v, src_vs), prev_ladj)
end

function apply_dist_trafo(trg_d::ReshapedDist, src_d::Distribution{Multivariate}, src_v::AbstractVector{<:Real}, prev_ladj::Real)
    trg_vs = varshape(trg_d)
    @argcheck totalndof(trg_vs) == length(src_d)
    r = apply_dist_trafo(unshaped(trg_d), src_d, src_v, prev_ladj)
    (v = trg_vs(r.v), ladj = r.ladj)
end

function apply_dist_trafo(trg_d::ReshapedDist, src_d::ReshapedDist, src_v::AbstractVector{<:Real}, prev_ladj::Real)
    trg_vs = varshape(trg_d)
    src_vs = varshape(src_d)
    @argcheck totalndof(trg_vs) == totalndof(src_vs)
    r = apply_dist_trafo(unshaped(trg_d), unshaped(src_d), unshaped(src_v, src_vs), prev_ladj)
    (v = trg_vs(r.v), ladj = r.ladj)
end


function apply_dist_trafo(trg_d::StdMvDist, src_d::UnshapedHDist, src_v::AbstractVector{<:Real}, prev_ladj::Real)
    src_v_primary, src_v_secondary = _hd_split(src_d, src_v)
    trg_d_primary = typeof(trg_d)(length(eachindex(src_v_primary)))
    trg_d_secondary = typeof(trg_d)(length(eachindex(src_v_secondary)))
    trg_v_primary, ladj_primary = apply_dist_trafo(trg_d_primary, _hd_pridist(src_d), src_v_primary, prev_ladj)
    trg_v_secondary, ladj = apply_dist_trafo(trg_d_secondary, _hd_secdist(src_d, src_v_primary), src_v_secondary, ladj_primary)
    trg_v = vcat(trg_v_primary, trg_v_secondary)
    (v = trg_v, ladj = ladj)
end

function apply_dist_trafo(trg_d::StdMvDist, src_d::HierarchicalDistribution, src_v::Any, prev_ladj::Real)
    src_v_unshaped = unshaped(src_v, varshape(src_d))
    apply_dist_trafo(trg_d, unshaped(src_d), src_v_unshaped, prev_ladj)
end

function apply_dist_trafo(trg_d::UnshapedHDist, src_d::StdMvDist, src_v::AbstractVector{<:Real}, prev_ladj::Real)
    src_v_primary, src_v_secondary = _hd_split(trg_d, src_v)
    src_d_primary = typeof(src_d)(length(eachindex(src_v_primary)))
    src_d_secondary = typeof(src_d)(length(eachindex(src_v_secondary)))
    trg_v_primary, ladj_primary = apply_dist_trafo(_hd_pridist(trg_d), src_d_primary, src_v_primary, prev_ladj)
    trg_v_secondary, ladj = apply_dist_trafo(_hd_secdist(trg_d, trg_v_primary), src_d_secondary, src_v_secondary, ladj_primary)
    trg_v = vcat(trg_v_primary, trg_v_secondary)
    (v = trg_v, ladj = ladj)
end

function apply_dist_trafo(trg_d::HierarchicalDistribution, src_d::StdMvDist, src_v::AbstractVector{<:Real}, prev_ladj::Real)
    unshaped_result = apply_dist_trafo(unshaped(trg_d), src_d, src_v, prev_ladj)
    (v = varshape(trg_d)(unshaped_result.v), ladj = unshaped_result.ladj)
end
