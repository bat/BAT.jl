# This file is a part of BAT.jl, licensed under the MIT License (MIT).


const _default_float_WT = Float64 # Default type for float weights
const _default_int_WT = Int # Default type for int weights
const _default_LDT = Float64 # Default type for log-density values


"""
    struct DensitySample

A weighted sample drawn according to an statistical density,
e.g. a [`BAT.MeasureLike`](@ref).

Constructors:

* ```DensitySampleVector(v::Any, logd::Real, weight::Real, info::Any, aux::Any)```

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

Constructors:

```julia
function DensitySampleVector(;
    v::AbstractVector;
    logval::AbstractVector{<:Real},
    weight::Union{AbstractVector{<:Real}, Symbol},
    info::AbstractVector,
    aux::AbstractVector
)
```

```julia
DensitySampleVector(
    (
        v::AbstractVector{<:AbstractVector{<:Real}},
        logd::AbstractVector{<:Real},
        weight::AbstractVector{<:Real},
        info::AbstractVector{<:Any},
        aux::AbstractVector{<:Any}
    )
)
```

With `weight = :multiplicity` repeated samples will be replaced by a
single sample, with a weight equal to the number of repetitions.
"""
const DensitySampleVector{
    P,T<:AbstractFloat,W<:Real,R,Q,
    PV<:AbstractVector{P},TV<:AbstractVector{T},WV<:AbstractVector{W},RV<:AbstractVector{R},QV<:AbstractVector{Q},
    IDX # Can't be just Int yet, for compatibility with older versions of ArraysOfArrays and ValueShapes
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


_create_undef_vector(::Type{T}, len::Integer) where T = Vector{T}(undef, len)


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


_canonical_variates(xs::VectorOfSimilarArrays) = xs
_canonical_variates(xs::AbstractSlices) = VectorOfSimilarArrays(xs)
_canonical_variates(xs::AbstractVector{<:AbstractArray}) = VectorOfSimilarArrays(xs)
_canonical_variates(xs::AbstractVector{<:Real}) = xs
_canonical_variates(xs::AbstractVector{<:NamedTuple}) = StructVector(xs)
_canonical_variates(xs::StructVector) = xs
_canonical_variates(xs::ShapedAsNTArray) = xs


# ToDo: Use Fill instead of fill? Will likely require an `as_appendable` function.
function DensitySampleVector(;
    v::AbstractVector,
    logd::AbstractVector{<:Real} = fill(NaN, length(v)),
    weight::Union{AbstractVector{<:Real}, Symbol} = fill(1, length(eachindex(v))),
    info::AbstractVector = fill(nothing, length(eachindex(v))),
    aux::AbstractVector = fill(nothing, length(eachindex(v)))
)
    if weight == :multiplicity
        idxs, weight = repetition_to_weights(v)
        return DensitySampleVector((_canonical_variates(v[idxs]), logd[idxs], weight, info[idxs], aux[idxs]))
    else
        return DensitySampleVector((_canonical_variates(v), logd, weight, info, aux))
    end
end


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

# Specialize IndexStyle, current default for StructArray seems to be IndexCartesian()
Base.IndexStyle(::StructArray{<:DensitySample, 1}) = IndexLinear()

# Specialize comparison, currently StructArray seems fall back to `(==)(A::AbstractArray, B::AbstractArray)`
import Base.==
function(==)(A::DensitySampleVector, B::DensitySampleVector)
    A.v == B.v &&
    A.logd == B.logd &&
    A.weight == B.weight &&
    A.info == B.info &&
    A.aux == B.aux
end


function Base.merge!(X::DensitySampleVector, Xs::DensitySampleVector...)
    for Y in Xs
        append!(X, Y)
    end
    X
end

Base.merge(X::DensitySampleVector, Xs::DensitySampleVector...) = merge!(deepcopy(X), Xs...)

function checked_push!(samples::DensitySampleVector, new_sample::DensitySample)
    if !isempty(samples) && last(samples).v == new_sample.v
        samples.weight[end] += new_sample.weight
    else
        push!(samples, new_sample)
    end
    samples
end


function Base.Broadcast.broadcasted(shaper::Union{AbstractValueShape, typeof(unshaped)}, A::DensitySampleVector)
    DensitySampleVector((
        shaper.(A.v),
        A.logd,
        A.weight,
        A.info,
        A.aux
    ))
end

function foobar(shaper::Union{AbstractValueShape, typeof(unshaped)}, A::DensitySampleVector)
    DensitySampleVector((
        shaper.(A.v),
        A.logd,
        A.weight,
        A.info,
        A.aux
    ))
end


Base.Broadcast.broadcasted(::typeof(identity), s_src::DensitySampleVector) = deepcopy(s_src)

function Base.Broadcast.broadcasted(f::ComposedFunction, s_src::DensitySampleVector)
    broadcast(f.outer, broadcast(f.inner, s_src))
end



ValueShapes.varshape(A::DensitySampleVector) = elshape(A.v)


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


function _get_statw(f::Function, samples::DensitySampleVector, resultshape::AbstractValueShape)
    shape = varshape(samples)
    X = unshaped.(samples.v)
    w = FrequencyWeights(samples.weight)
    r_unshaped = f(X, w)
    resultshape(r_unshaped)
end

Statistics.mean(samples::DensitySampleVector) = _get_statw(_mean, samples, varshape(samples))
Statistics.var(samples::DensitySampleVector) = _get_statw(_var, samples, replace_const_shapes(ValueShapes.const_zero_shape, varshape(samples)))
Statistics.std(samples::DensitySampleVector) = _get_statw(_std, samples, replace_const_shapes(ValueShapes.const_zero_shape, varshape(samples)))

Statistics.median(samples::DensitySampleVector) = quantile(samples, 0.5)

function Statistics.quantile(samples::DensitySampleVector, p::Real)
    shape = varshape(samples)
    flat_samples = flatview(unshaped.(samples.v))
    n_params = size(flat_samples)[1]
    median_params = Vector{Float64}()

    for param in Base.OneTo(n_params)
        median_param = quantile(flat_samples[param,:], FrequencyWeights(samples.weight), p)
        push!(median_params, median_param)
    end
    shape(median_params)
end

function _get_stat(f::Function, samples::DensitySampleVector)
    shape = varshape(samples)
    X = unshaped.(samples.v)
    r_unshaped = f(X)
    shape(r_unshaped)
end

Base.minimum(samples::DensitySampleVector) = _get_stat(minimum, samples)
Base.maximum(samples::DensitySampleVector) = _get_stat(maximum, samples)

Statistics.cov(samples::DensitySampleVector{<:AbstractVector{<:Real}}) = _cov(samples.v, FrequencyWeights(samples.weight))
Statistics.cor(samples::DensitySampleVector{<:AbstractVector{<:Real}}) = _cor(samples.v, FrequencyWeights(samples.weight))

function _get_mode(samples::DensitySampleVector)
    shape = varshape(samples)
    i = findmax(samples.logd)[2]
    v_unshaped = unshaped.(samples.v)[i]
    v = deepcopy(shape(v_unshaped))
    (v, i)
end


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
    if minimum(W) / maximum(W) > threshold
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


"""
    repetition_to_weights(v::AbstractVector)

*BAT-internal, not part of stable public API.*

Drop (subsequently) repeated samples by adding weights.
"""
function repetition_to_weights(v::AbstractVector)
    idxs = Vector{Int}()
    counts = Vector{Int}()
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
    marg_headermap = Dict(:parameter => "Variate", :mean => "Mean", :std => "Std. dev.", :global_mode => "Gobal mode", :marginal_mode => "Marg. mode", :credible_intervals => "Cred. interval", :marginal_histogram => "Histogram")
    lazyreport!(
        rpt,
        "#### Marginals",
        lazytable(mod_marg_tbl, headers = marg_headermap)
    )

    fixed_tbl = _rpt_table_of_constparvals(smplv)
    if !isempty(fixed_tbl)
        marg_headermap = Dict(:parameter => "Variate", :value => "Value")

        lazyreport!(rpt,
            "#### Fixed parameters",
            lazytable(fixed_tbl, headers = marg_headermap)
        )
    end

    return nothing
end


function _marginal_table(smplv::DensitySampleVector)
    parnames = map(string, all_active_names(elshape(smplv.v)))

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
    vs = elshape(smplv.v)
    # Need to convert, otherwise these can become Vector{Union{}} if parkeys is empty:
    parkeys = convert(Vector{Symbol}, Symbol.(get_fixed_names(vs)))::Vector{Symbol}
    parvalues = convert(Vector{Any}, [getproperty(vs, f).shape.value for f in parkeys])
    str_parvalues = convert(Vector{String}, string.(parvalues))::Vector{String}
    TypedTables.Table(parameter = parkeys, value = str_parvalues)
end

function _marginal_histograms(smpl::DensitySampleVector{<:AbstractVector{<:Real}}; nbins = 40)
    trimmed_smpl = drop_low_weight_samples(smpl)
    V = flatview(trimmed_smpl.v)
    W = Weights(trimmed_smpl.weight)
    [fit(Histogram, V[i,:], W, range(minimum(V[i,:]), maximum(V[i,:]), length = 41)) for i in axes(V,1)]
end


function _rand_subsample_idxs(gen::GenContext, smpls::DensitySampleVector, n::Integer)
    # ToDo: Use PSIS (possible, efficiently?).

    orig_idxs = eachindex(smpls)
    weights = FrequencyWeights(float(smpls.weight))
    # Always generate idxs on CPU for now:
    idxs = sample(get_rng(gen), orig_idxs, weights, n, replace=true, ordered=false)
    return idxs
end
