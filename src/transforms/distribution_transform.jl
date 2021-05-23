# This file is a part of BAT.jl, licensed under the MIT License (MIT).


const StdUvDist = Union{StandardUvUniform, StandardUvNormal}
const StdMvDist = Union{StandardMvUniform, StandardMvNormal}



_adignore(f) = f()

function ChainRulesCore.rrule(::typeof(_adignore), f)
    result = _adignore(f)
    _nogradient_pullback(ΔΩ) = (ChainRulesCore.NO_FIELDS, ZeroTangent())
    return result, _nogradient_pullback
end

macro _adignore(expr)
    :(_adignore(() -> $(esc(expr))))
end


function _pushfront(v::AbstractVector, x)
    T = promote_type(eltype(v), typeof(x))
    r = similar(v, T, length(eachindex(v)) + 1)
    r[firstindex(r)] = x
    r[firstindex(r)+1:lastindex(r)] = v
    r
end

function ChainRulesCore.rrule(::typeof(_pushfront), v::AbstractVector, x)
    result = _pushfront(v, x)
    function _pushfront_pullback(thunked_ΔΩ)
        ΔΩ = ChainRulesCore.unthunk(thunked_ΔΩ)
        (ChainRulesCore.NO_FIELDS, ΔΩ[firstindex(ΔΩ)+1:lastindex(ΔΩ)], ΔΩ[firstindex(ΔΩ)])
    end
    return result, _pushfront_pullback
end


function _pushback(v::AbstractVector, x)
    T = promote_type(eltype(v), typeof(x))
    r = similar(v, T, length(eachindex(v)) + 1)
    r[lastindex(r)] = x
    r[firstindex(r):lastindex(r)-1] = v
    r
end

function ChainRulesCore.rrule(::typeof(_pushback), v::AbstractVector, x)
    result = _pushback(v, x)
    function _pushback_pullback(thunked_ΔΩ)
        ΔΩ = ChainRulesCore.unthunk(thunked_ΔΩ)
        (ChainRulesCore.NO_FIELDS, ΔΩ[firstindex(ΔΩ):lastindex(ΔΩ)-1], ΔΩ[lastindex(ΔΩ)])
    end
    return result, _pushback_pullback
end


_dropfront(v::AbstractVector) = v[firstindex(v)+1:lastindex(v)]

_dropback(v::AbstractVector) = v[firstindex(v):lastindex(v)-1]


_rev_cumsum(xs::AbstractVector) = reverse(cumsum(reverse(xs)))

function ChainRulesCore.rrule(::typeof(_rev_cumsum), xs::AbstractVector)
    result = _rev_cumsum(xs)
    function _rev_cumsum_pullback(ΔΩ)
        ∂xs = ChainRulesCore.@thunk cumsum(ChainRulesCore.unthunk(ΔΩ))
        (ChainRulesCore.NO_FIELDS, ∂xs)
    end
    return result, _rev_cumsum_pullback
end


# Equivalent to `cumprod(xs)``:
_exp_cumsum_log(xs::AbstractVector) = exp.(cumsum(log.(xs)))

function ChainRulesCore.rrule(::typeof(_exp_cumsum_log), xs::AbstractVector)
    result = _exp_cumsum_log(xs)
    function _exp_cumsum_log_pullback(ΔΩ)
        ∂xs = inv.(xs) .* _rev_cumsum(exp.(cumsum(log.(xs))) .* ChainRulesCore.unthunk(ΔΩ))
        (ChainRulesCore.NO_FIELDS, ∂xs)
    end
    return result, _exp_cumsum_log_pullback
end



struct DistributionTransform{
    VT <: AbstractValueShape,
    VF <: AbstractValueShape,
    DT <: ContinuousDistribution,
    DF <: ContinuousDistribution
} <: VariateTransform{VT,VF}
    target_dist::DT
    source_dist::DF
    target_varshape::VT
    source_varshape::VF
end

# ToDo: Add specialized dist trafo types able to cache relevant quantities, etc.


function _distrafo_ctor_impl(target_dist::Distribution, source_dist::Distribution)
    @argcheck eff_totalndof(target_dist) == eff_totalndof(source_dist)
    target_varshape = varshape(target_dist)
    source_varshape = varshape(source_dist)
    DistributionTransform(target_dist, source_dist, target_varshape, source_varshape)
end

DistributionTransform(target_dist::Distribution{VF,Continuous}, source_dist::Distribution{VF,Continuous}) where VF =
    _distrafo_ctor_impl(target_dist, source_dist)

DistributionTransform(target_dist::Distribution{Multivariate,Continuous}, source_dist::Distribution{VF,Continuous}) where VF =
    _distrafo_ctor_impl(target_dist, source_dist)

DistributionTransform(target_dist::Distribution{VF,Continuous}, source_dist::Distribution{Multivariate,Continuous}) where VF =
    _distrafo_ctor_impl(target_dist, source_dist)

DistributionTransform(target_dist::Distribution{Multivariate,Continuous}, source_dist::Distribution{Multivariate,Continuous}) =
    _distrafo_ctor_impl(target_dist, source_dist)


show_distribution(io::IO, d::Distribution) = show(io, d)
function show_distribution(io::IO, d::NamedTupleDist)
    print(io, Base.typename(typeof(d)).name, "{")
    show(io, propertynames(d))
    print(io, "}(…)")
end
    
function Base.show(io::IO, trafo::DistributionTransform)
    print(io, Base.typename(typeof(trafo)).name, "(")
    show_distribution(io, trafo.target_dist)
    print(io, ", ")
    show_distribution(io, trafo.source_dist)
    print(io, ")")
end

Base.show(io::IO, M::MIME"text/plain", trafo::DistributionTransform) = show(io, trafo)

# apply_dist_trafo(trg_d, src_d, src_v, prev_ladj)
function apply_dist_trafo end

apply_dist_trafo_noladj(trg_d::Distribution, src_d::Distribution, src_v::Any) = apply_dist_trafo(trg_d, src_d, src_v, missing).v


import Base.∘
function ∘(a::DistributionTransform, b::DistributionTransform)
    @argcheck a.source_dist == b.target_dist
    DistributionTransform(a.target_dist, b.source_dist)
end

Base.inv(trafo::DistributionTransform) = DistributionTransform(trafo.source_dist, trafo.target_dist)

ValueShapes.varshape(trafo::DistributionTransform) = trafo.source_varshape


function apply_vartrafo_impl(trafo::DistributionTransform, v::Any, prev_ladj::OptionalLADJ)
    apply_dist_trafo(trafo.target_dist, trafo.source_dist, v, prev_ladj)
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
    throw(ArgumentError("No standard intermediate distribution defined to transform from $(typeof(src_d).name)"))
end

function std_dist_to(trg_d::Distribution)
    throw(ArgumentError("No standard intermediate distribution defined to transform into $(typeof(trg_d).name)"))
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

function apply_dist_trafo(trg_d::Distribution, src_d::Distribution, src_v::Any, prev_ladj::OptionalLADJ)
    _check_conv_eff_totalndof(trg_d, src_d)
    intermediate_d = _intermediate_std_dist(trg_d, src_d)
    intermediate_v, intermediate_ladj = apply_dist_trafo(intermediate_d, src_d, src_v, prev_ladj)
    apply_dist_trafo(trg_d, intermediate_d, intermediate_v, intermediate_ladj)
end


function apply_dist_trafo(trg_d::DT, src_d::DT, src_v::Real, prev_ladj::OptionalLADJ) where {DT <: StdUvDist}
    (v = src_v, ladj = prev_ladj)
end

function apply_dist_trafo(trg_d::DT, src_d::DT, src_v::AbstractVector{<:Real}, prev_ladj::OptionalLADJ) where {DT <: StdMvDist}
    @argcheck length(trg_d) == length(src_d) == length(eachindex(src_v))
    (v = src_v, ladj = prev_ladj)
end


function apply_dist_trafo(trg_d::Distribution{Univariate}, src_d::StdMvDist, src_v::AbstractVector{<:Real}, prev_ladj::OptionalLADJ)
    @_adignore @argcheck length(src_d) == length(eachindex(src_v)) == 1
    apply_dist_trafo(trg_d, view(src_d, 1), first(src_v), prev_ladj)
end

function apply_dist_trafo(trg_d::StdMvDist, src_d::Distribution{Univariate}, src_v::Real, prev_ladj::OptionalLADJ)
    @argcheck length(trg_d) == 1
    r = apply_dist_trafo(view(trg_d, 1), src_d, first(src_v), prev_ladj)
    (v = unshaped(r.v), ladj = r.ladj)
end


@inline _trafo_cdf(d::Distribution{Univariate,Continuous}, x::Real) = _trafo_cdf_impl(params(d), d, x)

@inline _trafo_cdf_impl(::NTuple, d::Distribution{Univariate,Continuous}, x::Real) = cdf(d, x)

@inline function _trafo_cdf_impl(::NTuple{N,Union{Integer,AbstractFloat}}, d::Distribution{Univariate,Continuous}, x::ForwardDiff.Dual{TAG}) where {N,TAG}
    x_v = ForwardDiff.value(x)
    u = cdf(d, x_v)
    dudx = pdf(d, x_v)
    ForwardDiff.Dual{TAG}(u, dudx * ForwardDiff.partials(x))
end


@inline _trafo_quantile(d::Distribution{Univariate,Continuous}, u::Real) = _trafo_quantile_impl(params(d), d, u)

@inline _trafo_quantile_impl(::NTuple, d::Distribution{Univariate,Continuous}, u::Real) = _trafo_quantile_impl_generic(d, u)

@inline function _trafo_quantile_impl(::NTuple{N,Union{Integer,AbstractFloat}}, d::Distribution{Univariate,Continuous}, u::ForwardDiff.Dual{TAG}) where {N,TAG}
    x = _trafo_quantile_impl_generic(d, ForwardDiff.value(u))
    dxdu = inv(pdf(d, x))
    ForwardDiff.Dual{TAG}(x, dxdu * ForwardDiff.partials(u))
end

# Workaround for Beta dist, ForwardDiff doesn't work for parameters:
_trafo_quantile_impl(::NTuple{N,ForwardDiff.Dual}, d::Beta, x::Real) where N = convert(float(typeof(x)), NaN)

@inline _trafo_quantile_impl_generic(d::Distribution{Univariate,Continuous}, u::Real) = quantile(d, u)

# Workaround for rounding errors that can result in quantile values outside of support of Truncated:
@inline function _trafo_quantile_impl_generic(d::Truncated{<:Distribution{Univariate,Continuous}}, u::Real)
    x = quantile(d, u)
    T = typeof(x)
    min_x = T(minimum(d))
    max_x = T(maximum(d))
    if x < min_x && isapprox(x, min_x, atol = 4 * eps(T))
        min_x
    elseif x > max_x && isapprox(x, max_x, atol = 4 * eps(T))
        max_x
    else
        x
    end
end


@inline function _eval_dist_trafo_func(f::typeof(_trafo_cdf), d::Distribution{Univariate,Continuous}, src_v::Real, prev_ladj::OptionalLADJ)
    R_V = float(promote_type(typeof(src_v), eltype(params(d)), ismissing(prev_ladj) ? Bool : typeof(prev_ladj)))
    R_LADJ = !ismissing(prev_ladj) ? R_V : Missing
    if insupport(d, src_v)
        trg_v = f(d, src_v)
        trafo_ladj = !ismissing(prev_ladj) ? + logpdf(d, src_v) : missing
        var_trafo_result(convert(R_V, trg_v), src_v, convert(R_LADJ, trafo_ladj), prev_ladj)
    else
        var_trafo_result(convert(R_V, NaN), src_v, convert(R_LADJ, !ismissing(prev_ladj) ? NaN : missing), prev_ladj)
    end
end

@inline function _eval_dist_trafo_func(f::typeof(_trafo_quantile), d::Distribution{Univariate,Continuous}, src_v::Real, prev_ladj::OptionalLADJ)
    R_V = float(promote_type(typeof(src_v), eltype(params(d)), ismissing(prev_ladj) ? Bool : typeof(prev_ladj)))
    R_LADJ = !ismissing(prev_ladj) ? R_V : Missing
    if 0 <= src_v <= 1
        trg_v = f(d, src_v)
        trafo_ladj = !ismissing(prev_ladj) ? - logpdf(d, trg_v) : missing
        var_trafo_result(convert(R_V, trg_v), src_v, convert(R_LADJ, trafo_ladj), prev_ladj)
    else
        var_trafo_result(convert(R_V, NaN), src_v, convert(R_LADJ, !ismissing(prev_ladj) ? NaN : missing), prev_ladj)
    end
end


std_dist_from(src_d::Distribution{Univariate,Continuous}) = StandardUvUniform()

function apply_dist_trafo(::StandardUvUniform, src_d::Distribution{Univariate,Continuous}, src_v::Real, prev_ladj::OptionalLADJ)
    _eval_dist_trafo_func(_trafo_cdf, src_d, src_v, prev_ladj)
end

std_dist_to(trg_d::Distribution{Univariate,Continuous}) = StandardUvUniform()

function apply_dist_trafo(trg_d::Distribution{Univariate,Continuous}, ::StandardUvUniform, src_v::Real, prev_ladj::OptionalLADJ)
    TV = float(typeof(src_v))
    # Avoid src_v ≈ 0 and src_v ≈ 1 to avoid infinite variate values for target distributions with infinite support:
    mod_src_v = ifelse(src_v == 0, zero(TV) + eps(TV), ifelse(src_v == 1, one(TV) - eps(TV), convert(TV, src_v)))
    trg_v, ladj = _eval_dist_trafo_func(_trafo_quantile, trg_d, mod_src_v, prev_ladj)
    (v = trg_v, ladj = ladj)
end



function _dist_trafo_rescale_impl(trg_d, src_d, src_v::Real, prev_ladj::OptionalLADJ)
    R = float(typeof(src_v))
    trg_offs, trg_scale = location(trg_d), scale(trg_d)
    src_offs, src_scale = location(src_d), scale(src_d)
    rescale_factor = trg_scale / src_scale
    trg_v = (src_v - src_offs) * rescale_factor + trg_offs
    trafo_ladj = !ismissing(prev_ladj) ? log(rescale_factor) : missing
    var_trafo_result(trg_v, src_v, trafo_ladj, prev_ladj)
end

@inline apply_dist_trafo(trg_d::Uniform, src_d::Uniform, src_v::Real, prev_ladj::OptionalLADJ) = _dist_trafo_rescale_impl(trg_d, src_d, src_v, prev_ladj)
@inline apply_dist_trafo(trg_d::StandardUvUniform, src_d::Uniform, src_v::Real, prev_ladj::OptionalLADJ) = _dist_trafo_rescale_impl(trg_d, src_d, src_v, prev_ladj)
@inline apply_dist_trafo(trg_d::Uniform, src_d::StandardUvUniform, src_v::Real, prev_ladj::OptionalLADJ) = _dist_trafo_rescale_impl(trg_d, src_d, src_v, prev_ladj)

# ToDo: Use StandardUvNormal as standard intermediate dist for Normal? Would
# be useful if StandardUvNormal would be a better standard intermediate than
# StandardUvUniform for some other uniform distributions as well.
#
#     std_dist_from(src_d::Normal) = StandardUvNormal()
#     std_dist_to(trg_d::Normal) = StandardUvNormal()

@inline apply_dist_trafo(trg_d::Normal, src_d::Normal, src_v::Real, prev_ladj::OptionalLADJ) = _dist_trafo_rescale_impl(trg_d, src_d, src_v, prev_ladj)
@inline apply_dist_trafo(trg_d::StandardUvNormal, src_d::Normal, src_v::Real, prev_ladj::OptionalLADJ) = _dist_trafo_rescale_impl(trg_d, src_d, src_v, prev_ladj)
@inline apply_dist_trafo(trg_d::Normal, src_d::StandardUvNormal, src_v::Real, prev_ladj::OptionalLADJ) = _dist_trafo_rescale_impl(trg_d, src_d, src_v, prev_ladj)


# ToDo: Optimized implementation for Distributions.Truncated <-> StandardUvUniform


@inline function apply_dist_trafo(trg_d::StandardMvUniform, src_d::StandardMvNormal, src_v::AbstractVector{<:Real}, prev_ladj::OptionalLADJ)
    @_adignore @argcheck eff_totalndof(trg_d) == eff_totalndof(src_d)
    _product_dist_trafo_impl(StandardUvUniform(), StandardUvNormal(), src_v, prev_ladj)
end

@inline function apply_dist_trafo(trg_d::StandardMvNormal, src_d::StandardMvUniform, src_v::AbstractVector{<:Real}, prev_ladj::OptionalLADJ)
    @_adignore @argcheck eff_totalndof(trg_d) == eff_totalndof(src_d)
    _product_dist_trafo_impl(StandardUvNormal(), StandardUvUniform(), src_v, prev_ladj)
end


std_dist_from(src_d::MvNormal) = StandardMvNormal(length(src_d))

function apply_dist_trafo(trg_d::StandardMvNormal, src_d::MvNormal, src_v::AbstractVector{<:Real}, prev_ladj::OptionalLADJ)
    @argcheck length(trg_d) == length(src_d)
    A = cholesky(src_d.Σ).U
    trg_v = transpose(A) \ (src_v - src_d.μ)
    trafo_ladj = !ismissing(prev_ladj) ? - logabsdet(A)[1] : missing
    var_trafo_result(trg_v, src_v, trafo_ladj, prev_ladj)
end

std_dist_to(trg_d::MvNormal) = StandardMvNormal(length(trg_d))

function apply_dist_trafo(trg_d::MvNormal, src_d::StandardMvNormal, src_v::AbstractVector{<:Real}, prev_ladj::OptionalLADJ)
    @argcheck length(trg_d) == length(src_d)
    A = cholesky(trg_d.Σ).U
    trg_v = transpose(A) * src_v + trg_d.μ
    trafo_ladj = !ismissing(prev_ladj) ? + logabsdet(A)[1] : missing
    var_trafo_result(trg_v, src_v, trafo_ladj, prev_ladj)
end


eff_totalndof(d::Dirichlet) = length(d) - 1
eff_totalndof(d::DistributionsAD.TuringDirichlet) = length(d) - 1

std_dist_to(trg_d::Dirichlet) = StandardMvUniform(eff_totalndof(trg_d))
std_dist_to(trg_d::DistributionsAD.TuringDirichlet) = StandardMvUniform(eff_totalndof(trg_d))


function apply_dist_trafo(trg_d::Dirichlet, src_d::StandardMvUniform, src_v::AbstractVector{<:Real}, prev_ladj::OptionalLADJ)
    apply_dist_trafo(DistributionsAD.TuringDirichlet(trg_d.alpha), src_d, src_v, prev_ladj)
end

function _dirichlet_beta_trafo(α::Real, β::Real, src_v::Real)
    R = float(promote_type(typeof(α), typeof(β), typeof(src_v)))
    convert(R, apply_dist_trafo(Beta(α, β), StandardUvUniform(), src_v, missing).v)::R
end

_a_times_one_minus_b(a::Real, b::Real) = a * (1 - b)

function apply_dist_trafo(trg_d::DistributionsAD.TuringDirichlet, src_d::StandardMvUniform, src_v::AbstractVector{<:Real}, prev_ladj::Missing)
    # See M. J. Betancourt, "Cruising The Simplex: Hamiltonian Monte Carlo and the Dirichlet Distribution",
    # https://arxiv.org/abs/1010.3436

    @_adignore @argcheck length(trg_d) == length(src_d) + 1
    αs = _dropfront(_rev_cumsum(trg_d.alpha))
    βs = _dropback(trg_d.alpha)
    beta_v = fwddiff(_dirichlet_beta_trafo).(αs, βs, src_v)
    beta_v_cp = _exp_cumsum_log(_pushfront(beta_v, 1))
    beta_v_ext = _pushback(beta_v, 0)
    trg_v = fwddiff(_a_times_one_minus_b).(beta_v_cp, beta_v_ext)
    var_trafo_result(trg_v, src_v, missing, prev_ladj)
end

function apply_dist_trafo(trg_d::DistributionsAD.TuringDirichlet, src_d::StandardMvUniform, src_v::AbstractVector{<:Real}, prev_ladj::Real)
    trg_v = apply_dist_trafo(trg_d, src_d, src_v, missing).v
    trafo_ladj = - logpdf(trg_d, trg_v)
    var_trafo_result(trg_v, src_v, trafo_ladj, prev_ladj)
end


g_debug = nothing
function _product_dist_trafo_impl(trg_ds, src_ds, src_v::AbstractVector{<:Real}, prev_ladj::Missing)
    global g_debug = (trg_ds=trg_ds, src_ds=src_ds, src_v=src_v, prev_ladj=missing)
    trg_v = fwddiff(apply_dist_trafo_noladj).(trg_ds, src_ds, src_v)
    var_trafo_result(trg_v, src_v, missing, missing)
end

function _product_dist_trafo_impl(trg_ds, src_ds, src_v::AbstractVector{<:Real}, prev_ladj::Real)
    rs = fwddiff(apply_dist_trafo).(trg_ds, src_ds, src_v, zero(prev_ladj))
    trg_v = map(r -> r.v, rs)
    trafo_ladj = sum(map(r -> r.ladj, rs))
    var_trafo_result(trg_v, src_v, trafo_ladj, prev_ladj)
end

function apply_dist_trafo(trg_d::Distributions.Product, src_d::Distributions.Product, src_v::AbstractVector{<:Real}, prev_ladj::OptionalLADJ)
    @_adignore @argcheck eff_totalndof(trg_d) == eff_totalndof(src_d)
    _product_dist_trafo_impl(trg_d.v, src_d.v, src_v, prev_ladj)
end

function apply_dist_trafo(trg_d::StandardMvUniform, src_d::Distributions.Product, src_v::AbstractVector{<:Real}, prev_ladj::OptionalLADJ)
    @_adignore @argcheck eff_totalndof(trg_d) == eff_totalndof(src_d)
    _product_dist_trafo_impl(StandardUvUniform(), src_d.v, src_v, prev_ladj)
end

function apply_dist_trafo(trg_d::StandardMvNormal, src_d::Distributions.Product, src_v::AbstractVector{<:Real}, prev_ladj::OptionalLADJ)
    @_adignore @argcheck eff_totalndof(trg_d) == eff_totalndof(src_d)
    _product_dist_trafo_impl(StandardUvNormal(), src_d.v, src_v, prev_ladj)
end

function apply_dist_trafo(trg_d::Distributions.Product, src_d::StandardMvUniform, src_v::AbstractVector{<:Real}, prev_ladj::OptionalLADJ)
    @_adignore @argcheck eff_totalndof(trg_d) == eff_totalndof(src_d)
    _product_dist_trafo_impl(trg_d.v, StandardUvUniform(), src_v, prev_ladj)
end

function apply_dist_trafo(trg_d::Distributions.Product, src_d::StandardMvNormal, src_v::AbstractVector{<:Real}, prev_ladj::OptionalLADJ)
    @_adignore @argcheck eff_totalndof(trg_d) == eff_totalndof(src_d)
    _product_dist_trafo_impl(trg_d.v, StandardUvNormal(), src_v, prev_ladj)
end


function _ntdistelem_to_stdmv(trg_d::StdMvDist, sd::Distribution, src_v_unshaped::AbstractVector{<:Real}, trg_acc::ValueAccessor, init_ladj::OptionalLADJ)
    td = view(trg_d, ValueShapes.view_range(Base.OneTo(length(trg_d)), trg_acc))
    sv = stripscalar(view(src_v_unshaped, trg_acc))
    apply_dist_trafo(td, sd, sv, init_ladj)
end

function _ntdistelem_to_stdmv(trg_d::StdMvDist, sd::ConstValueDist, src_v_unshaped::AbstractVector{<:Real}, trg_acc::ValueAccessor, init_ladj::OptionalLADJ)
    (v = Bool[], ladj = init_ladj)
end


_transformed_ntd_elshape(d::Distribution{Univariate}) = varshape(d)
_transformed_ntd_elshape(d::Distribution{Multivariate}) = ArrayShape{Real}(eff_totalndof(d))
function _transformed_ntd_elshape(d::Distribution)
    vs = varshape(d)
    @argcheck totalndof(vs) == eff_totalndof(d)
    vs
end

function _transformed_ntd_accessors(d::NamedTupleDist{names}) where names
    shapes = map(_transformed_ntd_elshape, values(d))
    vs = NamedTupleShape(NamedTuple{names}(shapes))
    values(vs)
end

function apply_dist_trafo(trg_d::StdMvDist, src_d::ValueShapes.UnshapedNTD, src_v::AbstractVector{<:Real}, prev_ladj::OptionalLADJ)
    src_vs = varshape(src_d.shaped)
    @argcheck length(src_d) == length(eachindex(src_v))
    trg_accessors = _transformed_ntd_accessors(src_d.shaped)
    init_ladj = ismissing(prev_ladj) ? missing : zero(Float32)
    rs = map((acc, sd) -> _ntdistelem_to_stdmv(trg_d, sd, src_v, acc, init_ladj), trg_accessors, values(src_d.shaped))
    trg_v = vcat(map(r -> r.v, rs)...)
    trafo_ladj = !ismissing(prev_ladj) ? sum(map(r -> r.ladj, rs)) : missing
    var_trafo_result(trg_v, src_v, trafo_ladj, prev_ladj)
end

function apply_dist_trafo(trg_d::StdMvDist, src_d::NamedTupleDist, src_v::Union{NamedTuple,ShapedAsNT}, prev_ladj::OptionalLADJ)
    src_v_unshaped = unshaped(src_v, varshape(src_d))
    apply_dist_trafo(trg_d, unshaped(src_d), src_v_unshaped, prev_ladj)
end

function _stdmv_to_ntdistelem(td::Distribution, src_d::StdMvDist, src_v::AbstractVector{<:Real}, src_acc::ValueAccessor, init_ladj::OptionalLADJ)
    sd = view(src_d, ValueShapes.view_range(Base.OneTo(length(src_d)), src_acc))
    sv = view(src_v, ValueShapes.view_range(axes(src_v, 1), src_acc))
    apply_dist_trafo(td, sd, sv, init_ladj)
end

function _stdmv_to_ntdistelem(td::ConstValueDist, src_d::StdMvDist, src_v::AbstractVector{<:Real}, src_acc::ValueAccessor, init_ladj::OptionalLADJ)
    (v = Bool[], ladj = init_ladj)
end

function apply_dist_trafo(trg_d::ValueShapes.UnshapedNTD, src_d::StdMvDist, src_v::AbstractVector{<:Real}, prev_ladj::OptionalLADJ)
    trg_vs = varshape(trg_d.shaped)
    @argcheck length(src_d) == length(eachindex(src_v))
    src_accessors = _transformed_ntd_accessors(trg_d.shaped)
    init_ladj = ismissing(prev_ladj) ? missing : zero(Float32)
    rs = map((acc, td) -> _stdmv_to_ntdistelem(td, src_d, src_v, acc, init_ladj), src_accessors, values(trg_d.shaped))
    trg_v_unshaped = vcat(map(r -> unshaped(r.v), rs)...)
    trafo_ladj = !ismissing(prev_ladj) ? sum(map(r -> r.ladj, rs)) : missing
    var_trafo_result(trg_v_unshaped, src_v, trafo_ladj, prev_ladj)
end

function apply_dist_trafo(trg_d::NamedTupleDist, src_d::StdMvDist, src_v::AbstractVector{<:Real}, prev_ladj::OptionalLADJ)
    unshaped_result = apply_dist_trafo(unshaped(trg_d), src_d, src_v, prev_ladj)
    (v = varshape(trg_d)(unshaped_result.v), ladj = unshaped_result.ladj)
end


const AnyReshapedDist = Union{ReshapedDist,MatrixReshaped}

function apply_dist_trafo(trg_d::Distribution{Multivariate}, src_d::AnyReshapedDist, src_v::Any, prev_ladj::OptionalLADJ)
    src_vs = varshape(src_d)
    @argcheck length(trg_d) == totalndof(src_vs)
    apply_dist_trafo(trg_d, unshaped(src_d), unshaped(src_v, src_vs), prev_ladj)
end

function apply_dist_trafo(trg_d::AnyReshapedDist, src_d::Distribution{Multivariate}, src_v::AbstractVector{<:Real}, prev_ladj::OptionalLADJ)
    trg_vs = varshape(trg_d)
    @argcheck totalndof(trg_vs) == length(src_d)
    r = apply_dist_trafo(unshaped(trg_d), src_d, src_v, prev_ladj)
    (v = trg_vs(r.v), ladj = r.ladj)
end

function apply_dist_trafo(trg_d::AnyReshapedDist, src_d::AnyReshapedDist, src_v::AbstractVector{<:Real}, prev_ladj::OptionalLADJ)
    trg_vs = varshape(trg_d)
    src_vs = varshape(src_d)
    @argcheck totalndof(trg_vs) == totalndof(src_vs)
    r = apply_dist_trafo(unshaped(trg_d), unshaped(src_d), unshaped(src_v, src_vs), prev_ladj)
    (v = trg_vs(r.v), ladj = r.ladj)
end


function apply_dist_trafo(trg_d::StdMvDist, src_d::UnshapedHDist, src_v::AbstractVector{<:Real}, prev_ladj::OptionalLADJ)
    src_v_primary, src_v_secondary = _hd_split(src_d, src_v)
    trg_d_primary = typeof(trg_d)(length(eachindex(src_v_primary)))
    trg_d_secondary = typeof(trg_d)(length(eachindex(src_v_secondary)))
    trg_v_primary, ladj_primary = apply_dist_trafo(trg_d_primary, _hd_pridist(src_d), src_v_primary, prev_ladj)
    trg_v_secondary, ladj = apply_dist_trafo(trg_d_secondary, _hd_secdist(src_d, src_v_primary), src_v_secondary, ladj_primary)
    trg_v = vcat(trg_v_primary, trg_v_secondary)
    (v = trg_v, ladj = ladj)
end

function apply_dist_trafo(trg_d::StdMvDist, src_d::HierarchicalDistribution, src_v::Any, prev_ladj::OptionalLADJ)
    src_v_unshaped = unshaped(src_v, varshape(src_d))
    apply_dist_trafo(trg_d, unshaped(src_d), src_v_unshaped, prev_ladj)
end

function apply_dist_trafo(trg_d::UnshapedHDist, src_d::StdMvDist, src_v::AbstractVector{<:Real}, prev_ladj::OptionalLADJ)
    src_v_primary, src_v_secondary = _hd_split(trg_d, src_v)
    src_d_primary = typeof(src_d)(length(eachindex(src_v_primary)))
    src_d_secondary = typeof(src_d)(length(eachindex(src_v_secondary)))
    trg_v_primary, ladj_primary = apply_dist_trafo(_hd_pridist(trg_d), src_d_primary, src_v_primary, prev_ladj)
    trg_v_secondary, ladj = apply_dist_trafo(_hd_secdist(trg_d, trg_v_primary), src_d_secondary, src_v_secondary, ladj_primary)
    trg_v = vcat(trg_v_primary, trg_v_secondary)
    (v = trg_v, ladj = ladj)
end

function apply_dist_trafo(trg_d::HierarchicalDistribution, src_d::StdMvDist, src_v::AbstractVector{<:Real}, prev_ladj::OptionalLADJ)
    unshaped_result = apply_dist_trafo(unshaped(trg_d), src_d, src_v, prev_ladj)
    (v = varshape(trg_d)(unshaped_result.v), ladj = unshaped_result.ladj)
end
