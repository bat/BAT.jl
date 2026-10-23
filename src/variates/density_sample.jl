# This file is a part of BAT.jl, licensed under the MIT License (MIT).


const _default_float_WT = Float64 # Default type for float weights
const _default_int_WT = Int # Default type for int weights
const _default_LDT = Float64 # Default type for log-density values


"""
    struct DensitySample

A weighted sample drawn according to an statistical density,
e.g. a [`BAT.MeasureLike`](@ref).

Constructors:

* ```DensitySample(v::Any, logd::Real, weight::Real, info::Any, aux::Any)```

Fields:

$(TYPEDFIELDS)

Use [`DensitySampleVector`](@ref) to store vectors of multiple samples with
an efficient column-based memory layout.
"""
struct DensitySample{
    P,
    T<:Real,
    W<:Real,
    R,
    Q
}
    "variate value"
    v::P

    "log(density) value at `v`"
    logd::T

    "Weight of the sample"
    weight::W

    "Additional info on the provenance of the sample. Content depends
    on the sampling algorithm."
    info::R

    "Custom user-defined information attached to the sample."
    aux::Q
end

export DensitySample


# DensitySample behaves as a scalar type under broadcasting:
@inline Base.Broadcast.broadcastable(s::DensitySample) = Ref(s)

# Necessary to make StructArrays.maybe_convert_elt happy:
Base.convert(::Type{DensitySample{P,T,W,R,Q}}, s::DensitySample) where {P,T,W,R,Q} = DensitySample{P,T,W,R,Q}(
    convert(P, s.v), convert(T, s.logd), convert(W, s.weight), convert(R, s.info), convert(Q, s.aux)
)

import Base.==
function ==(A::DensitySample, B::DensitySample)
    A.v == B.v && A.logd == B.logd &&
        A.weight == B.weight && A.info == B.info && A.aux == B.aux
end


# Required for TypedTables.showtable:
Base.getindex(s::DensitySample, k::Symbol) = getproperty(s, k)


function Base.similar(s::DensitySample{P,T,W,R,Q}) where {P<:AbstractVector{<:Real},T,W,R,Q}
    v = fill!(similar(s.v), eltype(s.v)(NaN))
    logd = convert(T, NaN)
    weight = zero(W)
    info = R()
    aux = Q()
    P_new = typeof(v)
    DensitySample{P_new,T,W,R,Q}(v, logd, weight, info, aux)
end


function _apply_shape(shape::AbstractValueShape, s::DensitySample)
    DensitySample(
        shape(s.v),
        s.logd,
        s.weight,
        s.info,
        s.aux,
    )
end

(shape::AbstractValueShape)(s::DensitySample) = _apply_shape(shape, s)



"""
    struct DensitySampleVector <: AbstractVector{<:DensitySample}

A vector of [`DensitySample`](@ref) elements.

`DensitySampleVector` is currently a type alias for
`StructArrays.StructArray{<:DensitySample,...}`, though this is subject to
change without deprecation.

The variates are stored in the layout that `rand(μ^n)` produces for the
measure `μ` they belong to: a vector of numbers for scalar variates, nested
arrays (ArraysOfArrays) for array variates and a `StructArray` for named
tuple variates, with constant components stored as `Fill` columns.

Constructors:

```julia
DensitySampleVector(;
    v::AbstractVector,
    logd::AbstractVector{<:Real},
    weight::Union{AbstractVector{<:Real}, Symbol},
    info::AbstractVector,
    aux::AbstractVector
)
```

```julia
DensitySampleVector(X, μ::AbstractMeasure; logd, weight, info, aux)
```

```julia
DensitySampleVector(
    (
        v::AbstractVector,
        logd::AbstractVector{<:Real},
        weight::AbstractVector{<:Real},
        info::AbstractVector{<:Any},
        aux::AbstractVector{<:Any}
    )
)
```

With `weight = :multiplicity` repeated samples will be replaced by a
single sample, with a weight equal to the number of repetitions.

`DensitySampleVector(X, μ)` takes the variates `X` of the measure `μ`,
either as a batch of variates (e.g. from `rand(μ^n)`) or as a matrix whose
columns are the free components of the variates in the flat
standard-variate layout of `μ`. Point-mass components of `μ` are stored as
`Fill` columns.
"""
const DensitySampleVector{
    P,T<:AbstractFloat,W<:Real,R,Q,
    PV<:AbstractVector{P},TV<:AbstractVector{T},WV<:AbstractVector{W},RV<:AbstractVector{R},QV<:AbstractVector{Q},
    IDX # Can't be just Int: transported batches come with CartesianIndex{1} indices
} = StructArray{
    DensitySample{P,T,W,R,Q},
    1,
    NamedTuple{(:v, :logd, :weight, :info, :aux), Tuple{PV,TV,WV,RV,QV}},
    IDX
}

export DensitySampleVector


function DensitySampleVector(contents::Tuple{PV,TV,WV,RV,QV}) where {
    P,T<:AbstractFloat,W<:Real,R,Q,
    PV<:AbstractVector{P},TV<:AbstractVector{T},WV<:AbstractVector{W},RV<:AbstractVector{R},QV<:AbstractVector{Q}
}
    StructArray{DensitySample{P,T,W,R,Q}}(contents)
end


# Supertype of the per-sample provenance stored in the `info` field of a
# DensitySample, e.g. BAT.MCMCSampleID:
abstract type SampleID end


_create_undef_vector(::Type{T}, len::Integer) where T = Vector{T}(undef, len)


# Flat, growable storage, e.g. for MCMC output in standard space:
function DensitySampleVector{P,T,W,R,Q}(::UndefInitializer, len::Integer, npar::Integer) where {
    PT<:Real, P<:AbstractVector{PT}, T<:AbstractFloat, W<:Real, R, Q
}
    contents = (
        VectorOfSimilarVectors(ElasticArray{PT}(undef, npar, len)),
        Vector{T}(undef, len),
        Vector{W}(undef, len),
        _create_undef_vector(R, len),
        _create_undef_vector(Q, len)
    )

    DensitySampleVector(contents)
end

DensitySampleVector(::Type{S}, varlen::Integer) where {P<:AbstractVector{<:Real},T<:AbstractFloat,W<:Real,R,Q,S<:DensitySample{P,T,W,R,Q}} =
    DensitySampleVector{P,T,W,R,Q}(undef, 0, varlen)


# Batches of variates in the layout of BAT's sample storage, from batches
# in that layout or from plain vectors of variates:
_canonical_batch(X::AbstractVector) = X
_canonical_batch(X::AbstractVector{<:AbstractArray}) = convert(VectorOfSimilarArrays, X)
_canonical_batch(X::AbstractVectorOfSimilarArrays) = X
_canonical_batch(X::AbstractSlices) = convert(VectorOfSimilarArrays, X)
_canonical_batch(X::AbstractFill{<:AbstractArray,1}) = X
_canonical_batch(X::AbstractFill{<:NamedTuple,1}) = X
_canonical_batch(X::AbstractVector{<:NamedTuple}) = _canonical_batch(StructArray(X))
_canonical_batch(X::StructVector{<:NamedTuple}) = StructArray(map(_canonical_batch, StructArrays.components(X)))
_canonical_batch(X::ShapedAsNTArray{<:NamedTuple,1}) = unflatbatch(elshape(X), flatview(unshaped.(X)))


function DensitySampleVector(;
    v::AbstractVector,
    logd::AbstractVector{<:Real} = fill(NaN, length(v)),
    weight::Union{AbstractVector{<:Real}, Symbol} = fill(1, length(eachindex(v))),
    info::AbstractVector = fill(nothing, length(eachindex(v))),
    aux::AbstractVector = fill(nothing, length(eachindex(v)))
)
    X = _canonical_batch(v)
    if weight == :multiplicity
        idxs, weight = repetition_to_weights(X)
        return DensitySampleVector((X[idxs], logd[idxs], weight, info[idxs], aux[idxs]))
    else
        return DensitySampleVector((X, logd, weight, info, aux))
    end
end

function DensitySampleVector(X::AbstractVector, μ::AbstractMeasure; kwargs...)
    vs = varshape(μ)
    v = _own_batch_impl(_canonical_batch(X), vs)
    batch_varshape(v) <= vs || throw(ArgumentError("Variates with shape $(batch_varshape(v)) don't match the variate shape $vs of the measure"))
    return DensitySampleVector(; v = v, kwargs...)
end

DensitySampleVector(M::AbstractMatrix{<:Real}, μ::AbstractMeasure; kwargs...) =
    DensitySampleVector(; v = unflatbatch(μ, M), kwargs...)

DensitySampleVector(X::AbstractArray, d::Distribution; kwargs...) = DensitySampleVector(X, batmeasure(d); kwargs...)


function Base.similar(s::DensitySampleVector, sz::Tuple)
    DensitySampleVector(map(c -> similar(c, sz), values(StructArrays.components(s))))
end


function Base.show(io::IO, A::DensitySampleVector)
    print(io, "DensitySampleVector(length = ")
    show(io, length(eachindex(A)))
    print(io, ", varshape = ")
    show_value_shape(io, varshape(A))
    print(io, ")")
end

# Required for TypedTables.showtable:
TypedTables.columnnames(A::DensitySampleVector) = propertynames(A)
# Required for TypedTables.showtable:
Base.getindex(A::DensitySampleVector, k::Symbol) = getproperty(A, k)

function Base.show(io::IO, M::MIME"text/plain", A::DensitySampleVector)
    print(io, "DensitySampleVector, ")
    TypedTables.showtable(io, A)
end


# Specialize getindex to properly support ArraysOfArrays and similar, preventing
# conversion to exact element type:

@inline Base.getindex(A::StructArray{<:DensitySample}, I::Int...) =
    DensitySample(A.v[I...], A.logd[I...], A.weight[I...], A.info[I...], A.aux[I...])

@inline Base.getindex(A::StructArray{<:DensitySample}, I::Union{Int,AbstractArray,Colon}...) =
    DensitySampleVector((A.v[I...], A.logd[I...], A.weight[I...], A.info[I...], A.aux[I...]))

# Specialize push! and append! likewise, StructArrays would convert the
# sample to the element type of A first. Only flat storage can grow:
function Base.push!(A::DensitySampleVector, s::DensitySample)
    _check_growable(A.v)
    push!(A.v, s.v)
    push!(A.logd, s.logd)
    push!(A.weight, s.weight)
    push!(A.info, s.info)
    push!(A.aux, s.aux)
    A
end

function Base.append!(A::DensitySampleVector, B::DensitySampleVector)
    _check_growable(A.v)
    append!(A.v, B.v)
    append!(A.logd, B.logd)
    append!(A.weight, B.weight)
    append!(A.info, B.info)
    append!(A.aux, B.aux)
    A
end

# Specialize IndexStyle, current default for StructArray seems to be IndexCartesian()
Base.IndexStyle(::StructArray{<:DensitySample, 1}) = IndexLinear()

# Specialize comparison, currently StructArray seems fall back to `(==)(A::AbstractArray, B::AbstractArray)`
import Base.==
function(==)(A::DensitySampleVector, B::DensitySampleVector)
    _batches_equal(==, A.v, B.v) &&
    A.logd == B.logd &&
    A.weight == B.weight &&
    A.info == B.info &&
    A.aux == B.aux
end


function Base.vcat(X::DensitySampleVector, Xs::DensitySampleVector...)
    DensitySampleVector((
        _vcat_batches(X.v, map(Y -> Y.v, Xs)...),
        vcat(X.logd, map(Y -> Y.logd, Xs)...),
        vcat(X.weight, map(Y -> Y.weight, Xs)...),
        vcat(X.info, map(Y -> Y.info, Xs)...),
        vcat(X.aux, map(Y -> Y.aux, Xs)...),
    ))
end

function Base.merge!(X::DensitySampleVector, Xs::DensitySampleVector...)
    for Y in Xs
        append!(X, Y)
    end
    X
end

Base.merge(X::DensitySampleVector, Xs::DensitySampleVector...) = vcat(X, Xs...)

function checked_push!(samples::DensitySampleVector, new_sample::DensitySample)
    if !isempty(samples) && last(samples).v == new_sample.v
        samples.weight[end] += new_sample.weight
    else
        push!(samples, new_sample)
    end
    samples
end


# Samples with the free components of their variates as flat real vectors:
_flat_samples(A::DensitySampleVector) = DensitySampleVector((nestedview(flatbatch(A.v)), A.logd, A.weight, A.info, A.aux))

# Samples with flat variates reshaped to variates of shape `vs`:
_reshaped_samples(vs::AbstractValueShape, A::DensitySampleVector) =
    DensitySampleVector((unflatbatch(vs, flatbatch(A.v)), A.logd, A.weight, A.info, A.aux))

Base.Broadcast.broadcasted(::typeof(unshaped), A::DensitySampleVector) = _flat_samples(A)

function Base.Broadcast.broadcasted(vs::AbstractValueShape, A::DensitySampleVector)
    Base.depwarn("Broadcasting a value shape over a DensitySampleVector is deprecated, construct samples from their measure via `DensitySampleVector(X, μ; logd, weight, info, aux)` instead.", :broadcasted)
    _reshaped_samples(vs, A)
end


Base.Broadcast.broadcasted(::typeof(identity), s_src::DensitySampleVector) = deepcopy(s_src)

function Base.Broadcast.broadcasted(f::ComposedFunction, s_src::DensitySampleVector)
    broadcast(f.outer, broadcast(f.inner, s_src))
end



ValueShapes.varshape(A::DensitySampleVector) = batch_varshape(A.v)


@inline _ncolons(::Val{N}) where N = ntuple(_ -> Colon(), Val{N}())

_mean(X::AbstractVectorOfSimilarArrays{T,M}, w::AbstractWeights) where {T,M} =
    mean(flatview(X), w, dims = M + 1)[_ncolons(Val{M}())...]

_var(X::AbstractVectorOfSimilarArrays{T,M}, w::AbstractWeights; mean = nothing, corrected::Bool = true) where {T,M} =
    var(flatview(X), w, M + 1; mean = mean, corrected = corrected)[_ncolons(Val{M}())...]

_std(X::AbstractVectorOfSimilarArrays{T,M}, w::AbstractWeights; mean = nothing, corrected::Bool = true) where {T,M} =
    std(flatview(X), w, M + 1; mean = mean, corrected = corrected)[_ncolons(Val{M}())...]

_cov(X::AbstractVectorOfSimilarVectors, w::AbstractWeights; corrected::Bool = true) =
    cov(flatview(X), w, 2; corrected = corrected)

_cor(X::AbstractVectorOfSimilarVectors, w::AbstractWeights) =
    cor(flatview(X), w, 2)


# Weight provenance is deliberately erased at this level (weights may be
# repetition counts, importance weights, etc.), so the statistics are the
# plain moments of the represented weighted empirical measure - without a
# finite-sample bias correction, which would require knowing how the
# samples were generated. These moments are invariant under global weight
# rescaling and under repetition compression/expansion. They are computed
# leaf by leaf and have the structure of the variates:
_stat_weights(samples::DensitySampleVector) = ProbabilityWeights(_canonical_rel_weights(samples.weight))

_map_leafkinds(f, X::AbstractVector) = _map_leaves(L -> f(_leafkind(L), L), X)

_const_zero(x::Number) = zero(float(x))
_const_zero(x) = map(_const_zero, x)

function Statistics.mean(samples::DensitySampleVector)
    w = _stat_weights(samples)
    _map_leafkinds(samples.v) do kind, X
        _leaf_mean(kind, X, w)
    end
end

_leaf_mean(::_ScalarLeaf, X::AbstractVector, w::AbstractWeights) = mean(X, w)
_leaf_mean(::_ArrayLeaf, X::AbstractVector, w::AbstractWeights) = _mean(X, w)
_leaf_mean(::_ConstLeaf, X::AbstractVector, ::AbstractWeights) = getindex_value(X)

function Statistics.var(samples::DensitySampleVector)
    w = _stat_weights(samples)
    _map_leafkinds(samples.v) do kind, X
        _leaf_var(kind, X, w)
    end
end

_leaf_var(::_ScalarLeaf, X::AbstractVector, w::AbstractWeights) = var(X, w, corrected = false)
_leaf_var(::_ArrayLeaf, X::AbstractVector, w::AbstractWeights) = _var(X, w, corrected = false)
_leaf_var(::_ConstLeaf, X::AbstractVector, ::AbstractWeights) = _const_zero(getindex_value(X))

Statistics.std(samples::DensitySampleVector) = _sqrt_leaves(var(samples))

_sqrt_leaves(x::Number) = sqrt(x)
_sqrt_leaves(x) = map(_sqrt_leaves, x)

Statistics.median(samples::DensitySampleVector) = quantile(samples, 0.5)

function _weighted_empirical_quantile(v::AbstractVector, w::AbstractVector, p::Real)
    isempty(v) && throw(ArgumentError("quantile of an empty array is undefined"))
    0 <= p <= 1 || throw(ArgumentError("input probability out of [0,1] range"))

    nan_idx = findfirst(i -> !iszero(w[i]) && isnan(v[i]), eachindex(v, w))
    isnothing(nan_idx) || return v[nan_idx]

    order = sortperm(v)
    first_idx = first(i for i in order if !iszero(w[i]))
    last_idx = first(i for i in Iterators.reverse(order) if !iszero(w[i]))
    iszero(p) && return v[first_idx]
    isone(p) && return v[last_idx]

    T = _weight_accum_type(w)
    target_weight = p * sum(T, w)
    cumulative_weight = zero(T)
    # Stop when cumulative weight reaches target.
    for i in order
        cumulative_weight += w[i]
        cumulative_weight >= target_weight && return v[i]
    end
    return v[last_idx]
end

# Weighted quantiles need scalar indexing, they are host-only:
function Statistics.quantile(samples::DensitySampleVector, p::Real)
    w = _canonical_rel_weights(samples.weight)
    _map_leafkinds(samples.v) do kind, X
        _leaf_quantile(kind, X, w, p)
    end
end

_leaf_quantile(::_ScalarLeaf, X::AbstractVector, w::AbstractVector, p::Real) = _weighted_empirical_quantile(X, w, p)
_leaf_quantile(::_ConstLeaf, X::AbstractVector, ::AbstractVector, ::Real) = getindex_value(X)

function _leaf_quantile(::_ArrayLeaf, X::AbstractVector, w::AbstractVector, p::Real)
    rows = _leaf_rows(X)
    reshape([_weighted_empirical_quantile(view(rows, i, :), w, p) for i in axes(rows, 1)], innersize(X))
end

# Flat storage of a leaf of array variates, one row per array element:
_leaf_rows(X::AbstractVectorOfSimilarArrays) = reshape(flatview(X), :, length(eachindex(X)))

Base.minimum(samples::DensitySampleVector) = _map_leafkinds(_leaf_extremum(minimum), samples.v)
Base.maximum(samples::DensitySampleVector) = _map_leafkinds(_leaf_extremum(maximum), samples.v)

_leaf_extremum(f) = (kind, X) -> _leaf_extremum(f, kind, X)
_leaf_extremum(f, ::_ScalarLeaf, X::AbstractVector) = f(X)
_leaf_extremum(f, ::_ArrayLeaf, X::AbstractVector) = reshape(f(_leaf_rows(X), dims = 2), innersize(X))
_leaf_extremum(::Any, ::_ConstLeaf, X::AbstractVector) = getindex_value(X)

Statistics.cov(samples::DensitySampleVector) = cov(flatbatch(samples.v), _stat_weights(samples), 2; corrected = false)
Statistics.cor(samples::DensitySampleVector) = cor(flatbatch(samples.v), _stat_weights(samples), 2)

function Base.isapprox(a::DensitySampleVector, b::DensitySampleVector; kwargs...)
    axes(a) == axes(b) || return false
    return varshape(a) == varshape(b) &&
        isapprox(flatbatch(a.v), flatbatch(b.v); kwargs...) &&
        isapprox(a.logd, b.logd; kwargs...) &&
        isapprox(a.weight, b.weight; kwargs...)
end

# The variate with the highest log-density, and its index:
function _get_mode(samples::DensitySampleVector)
    i = findmax(samples.logd)[2]
    v = _map_leafkinds(samples.v) do kind, X
        _leaf_variate(kind, X, i)
    end
    (v, i)
end

_leaf_variate(::_ScalarLeaf, X::AbstractVector, i) = X[i]
_leaf_variate(::_ArrayLeaf, X::AbstractVector, i) = collect(X[i])
_leaf_variate(::_ConstLeaf, X::AbstractVector, ::Any) = getindex_value(X)


StatsBase.mode(samples::DensitySampleVector) = _get_mode(samples)[1]


"""
    drop_low_weight_samples(
        samples::DensitySampleVector,
        fraction::Real = 10^-4
    )

*BAT-internal, not part of stable public API.*

Drop `fraction` of the total probability mass from samples to filter out the
samples with the lowest weight.
"""
function drop_low_weight_samples(samples::DensitySampleVector, fraction::Real = 10^-5; threshold::Real=10^-2)
    W = float(samples.weight)
    if isempty(W) || all(iszero, W)
        samples
    elseif minimum(W) / maximum(W) > threshold
        samples
    else
        W_s = sort(W)
        Q = cumsum(W_s)
        Q ./= maximum(Q)
        @assert last(Q) ≈ 1
        ind = searchsortedlast(Q, fraction)
        if ind !== 0
            thresh = W_s[searchsortedlast(Q, fraction)]
            idxs = findall(x -> x >= thresh, samples.weight)
            samples[idxs]
        else
            samples
        end
    end
end


# Return validated weights scaled by their maximum. Log weights normalize
# before conversion. The result is finite and invariant to positive scaling.
function _canonical_rel_weights(W::AbstractVector{<:Real})
    all(w -> isfinite(w) && w >= zero(w), W) || throw(ArgumentError("Sample weights must be finite and non-negative"))
    T = promote_type(Float32, float(eltype(W)))
    isempty(W) && return T.(W)
    max_w = maximum(W)
    max_w > zero(max_w) || throw(ArgumentError("Sample weights must contain at least one strictly positive entry"))
    return _normalize_rel_weights(W, max_w, T)
end

_normalize_rel_weights(W, max_w::AbstractFloat, ::Type{T}) where {T} = T.(W) ./ T(max_w)
_normalize_rel_weights(W, max_w::ULogarithmic, ::Type{T}) where {T} = exp.(T.(log.(W ./ max_w)))
_normalize_rel_weights(W, max_w, ::Type{T}) where {T} = T.(W ./ max_w)

_weight_accum_type(W::AbstractVector{<:Real}) = promote_type(Float64, eltype(W))


"""
    repetition_to_weights(v::AbstractVector)

*BAT-internal, not part of stable public API.*

Drop (subsequently) repeated samples by adding weights.
"""
function repetition_to_weights(v::AbstractVector)
    idxs = Vector{Int}()
    counts = Vector{Int}()
    isempty(v) && return (idxs, counts)
    push!(idxs, 1)
    push!(counts, 1)
    for i in 2:length(v)
        if v[i] == v[i-1]
            counts[end] += 1
        else
            push!(idxs, i)
            push!(counts, 1)
        end
    end
    return (idxs, counts)
end


function LazyReports.pushcontent!(rpt::LazyReport, smplv::DensitySampleVector)
    # ToDo: Forward context somehow instead of creating a new one here?
    context = BATContext()

    usmplv = unshaped.(smplv)
    nsamples = length(eachindex(smplv))
    total_weight = sum(smplv.weight)
    ess = round.(Int, bat_eff_sample_size(usmplv, context).result)

    lazyreport!(rpt, """
    ### Sampling result

    * Total number of samples: $nsamples
    * Total weight of samples: $total_weight
    * Effective sample size: between $(minimum(ess)) and $(maximum(ess))
    """)

    only_one_ci(viv::AbstractVector{<:AbstractInterval}) = length(viv) == 1 ? only(viv) : :multiple

    marg_tbl = _marginal_table(smplv)
    mod_marg_tbl = merge(Tables.columns(marg_tbl), (credible_intervals = map(only_one_ci, marg_tbl.credible_intervals),))
    marg_headermap = Dict(:parameter => "Parameter", :mean => "Mean", :std => "Std. dev.", :global_mode => "Gobal mode", :marginal_mode => "Marg. mode", :credible_intervals => "Cred. interval", :marginal_histogram => "Histogram")
    lazyreport!(
        rpt,
        "#### Marginals",
        lazytable(mod_marg_tbl, headers = marg_headermap)
    )

    fixed_tbl = _rpt_table_of_constparvals(smplv)
    if !isempty(fixed_tbl)
        marg_headermap = Dict(:parameter => "Parameter", :value => "Value")

        lazyreport!(rpt,
            "#### Fixed parameters",
            lazytable(fixed_tbl, headers = marg_headermap)
        )
    end

    return nothing
end


function _marginal_table(smplv::DensitySampleVector)
    parnames = map(string, all_active_names(varshape(smplv)))

    usmplv = unshaped.(smplv)

    credible_intervals = smallest_credible_intervals(usmplv)

    mhists = _marginal_histograms(usmplv)

    mm_alg = bat_default(bat_marginalmode, Val(:algorithm), usmplv)
    marginal_mode = bat_marginalmode_impl(usmplv, mm_alg, _g_dummy_context).result

    TypedTables.Table(
        parameter = parnames,
        mean = mean(usmplv),
        std = std(usmplv),
        global_mode = mode(usmplv),
        marginal_mode = marginal_mode,
        credible_intervals = credible_intervals,
        marginal_histogram = mhists,
    )
end


function _rpt_table_of_constparvals(smplv::DensitySampleVector)
    constpars = _const_parameters(varshape(smplv))
    # Need to convert, otherwise these can become Vector{Union{}} if constpars is empty:
    parkeys = convert(Vector{Symbol}, Symbol.(first.(constpars)))::Vector{Symbol}
    str_parvalues = convert(Vector{String}, string.(last.(constpars)))::Vector{String}
    TypedTables.Table(parameter = parkeys, value = str_parvalues)
end

function _marginal_histograms(smpl::DensitySampleVector{<:AbstractVector{<:Real}}; nbins = 40)
    trimmed_smpl = drop_low_weight_samples(smpl)
    V = flatview(trimmed_smpl.v)
    W = Weights(trimmed_smpl.weight)
    [fit(Histogram, V[i,:], W, nbins = nbins) for i in axes(V,1)]
end
