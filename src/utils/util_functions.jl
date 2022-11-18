# This file is a part of BAT.jl, licensed under the MIT License (MIT).


@inline nop_func(x...) = nothing


struct CombinedCallback{N,Fs<:NTuple{N,Function}} <: Function
    fs::Fs
end


function Base.show(io::IO, f::CombinedCallback)
    print(io, Base.typename(typeof(f)).name, "(")
    show(io, f.fs)
    print(io, ")")
end

function Base.show(io::IO, M::MIME"text/plain", f::CombinedCallback)
    print(io, Base.typename(typeof(f)).name, "(")
    show(io, M, f.fs)
    print(io, ")")
end


@inline _run_callbacks(fs::NTuple{0,Function}, args...; kwargs...) = nothing

@inline function _run_callbacks(fs::NTuple{1,Function}, args...; kwargs...)
    first(fs)(args...; kwargs...)
    nothing
end

@inline function _run_callbacks(fs::NTuple{N,Function}, args...; kwargs...) where N
    first(fs)(args...; kwargs...)
    _run_callbacks(Base.tail(fs), args...; kwargs...)
    nothing
end

@inline (f::CombinedCallback)(args...; kwargs...) = _run_callbacks(f.fs, args...; kwargs...)


combine_callbacks(::typeof(nop_func), f::Function) = f
combine_callbacks(f::Function, ::typeof(nop_func)) = f
combine_callbacks(::typeof(nop_func), ::typeof(nop_func)) = nop_func

combine_callbacks(f::Function, g::Function) = CombinedCallback((f, g))
combine_callbacks(f::CombinedCallback, g::Function) = CombinedCallback((f.fs..., g))
combine_callbacks(f::Function, g::CombinedCallback) = CombinedCallback((f, g.fs...))
combine_callbacks(f::CombinedCallback, g::CombinedCallback) = CombinedCallback((f.fs..., g.fs...))



near_neg_inf(::Type{T}) where T<:Real = T(-1E38) # Still fits into Float32

isneginf(x) = isinf(x) && x < 0 
isposinf(x) = isinf(x) && x > 0 
