# This file is a part of BAT.jl, licensed under the MIT License (MIT).


@doc """
    AbstractMCMCCallback <: Function

Subtypes (e.g. `X`) must support

    (::X)(level::Integer, chain::MCMCIterator) => nothing
    (::X)(level::Integer, tuner::AbstractMCMCTuner) => nothing

to be compabtible with `mcmc_iterate!`, `mcmc_tune_burnin!`, etc.
"""
abstract type AbstractMCMCCallback <: Function end
export AbstractMCMCCallback

@inline Base.convert(::Type{AbstractMCMCCallback}, x::AbstractMCMCCallback) = x

@inline Base.convert(::Type{Vector{<:AbstractMCMCCallback}}, V::Vector{<:AbstractMCMCCallback}) = V

Base.convert(::Type{Vector{<:AbstractMCMCCallback}}, V::Vector) =
            [convert(AbstractMCMCCallback, x) for x in V]


function mcmc_callback_vector(x, idxs::AbstractVector{<:Integer})
    V = convert(Vector{<:AbstractMCMCCallback}, x)
    if eachindex(V) != idxs
        throw(DimensionMismatch("Indices of callback vector incompatible with reference indices"))
    end
    V
end

mcmc_callback_vector(x::Tuple{}, idxs::AbstractVector{<:Integer}) =
    [MCMCNopCallback() for _ in idxs]



@doc """
    MCMCCallbackWrapper{F} <: AbstractMCMCCallback

Wraps a callable object to turn it into an `AbstractMCMCCallback`.

Constructor:

    MCMCCallbackWrapper(f::Any)

`f` needs to support the call syntax of an `AbstractMCMCCallback`.
"""
struct MCMCCallbackWrapper{F} <: AbstractMCMCCallback
    f::F
end


@inline (wrapper::MCMCCallbackWrapper)(args...) = wrapper.f(args...)

Base.convert(::Type{AbstractMCMCCallback}, f::Function) = MCMCCallbackWrapper(f)



struct MCMCNopCallback <: AbstractMCMCCallback end

(cb::MCMCNopCallback)(level::Integer, obj::Any) = nothing

Base.convert(::Type{AbstractMCMCCallback}, ::Tuple{}) = MCMCNopCallback()



struct MCMCMultiCallback{N,TP<:NTuple{N,AbstractMCMCCallback}} <: AbstractMCMCCallback
    funcs::TP

    MCMCMultiCallback(fs::Vararg{AbstractMCMCCallback,N}) where {N} =
        new{N, typeof(fs)}(fs)

    function fMCMCMultiCallback(fs::Vararg{Any,N}) where {N}
        fs_conv = map(x -> convert(AbstractMCMCCallback, x), fs)
        new{N, typeof(fs_conv)}(fs_conv)
    end
end


function (cb::MCMCMultiCallback)(level::Integer, obj::Any)
    for f in cb.funcs
        f(level, obj)
    end
    nothing
end

Base.convert(::Type{AbstractMCMCCallback}, fs::Tuple) = MCMCMultiCallback(fs...)



struct MCMCAppendCallback{T,F<:Function} <: AbstractMCMCCallback
    appendable::T
    max_level::Int
    get_data_func!::F
    nonzero_weights::Bool
end

export MCMCAppendCallback


function (cb::MCMCAppendCallback)(level::Integer, subject::Any)
    if (level <= cb.max_level)
        cb.get_data_func!(cb.appendable, subject, cb.nonzero_weights)
    end
    nothing
end


Base.convert(::Type{AbstractMCMCCallback}, x::DensitySampleVector) = MCMCAppendCallback(x)

MCMCAppendCallback(x::DensitySampleVector, nonzero_weights::Bool = true) =
    MCMCAppendCallback(x, 1, get_samples!, nonzero_weights)


Base.convert(::Type{AbstractMCMCCallback}, x::MCMCSampleIDVector) = MCMCAppendCallback(x)

MCMCAppendCallback(x::MCMCSampleIDVector, nonzero_weights::Bool = true) =
    MCMCAppendCallback(x, 1, get_sample_ids!, nonzero_weights)
