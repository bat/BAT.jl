# This file is a part of BAT.jl, licensed under the MIT License (MIT).


forwarddiff_dualized(::Type{TagType}, x::Real) where TagType = ForwardDiff.Dual{TagType}(x, one(x))

function forwarddiff_dualized(::Type{TagType}, x::NTuple{N,T}) where {TagType,N,T<:Real}
    ntuple(j -> ForwardDiff.Dual{TagType}(x[j], ntuple(i -> i == j ? one(x[j]) : zero(x[j]), Val(N))), Val(N))
end

# Equivalent to ForwardDiff internal function dualize(T, x):
forwarddiff_dualized(::Type{TagType}, x::SVector{N,T}) where {TagType,N,T<:Real} = SVector(forwarddiff_dualized(TagType, (x...,)))



# Equivalent to ForwardDiff internal function static_dual_eval(TagType, f, x) (for SVector):
function forwarddiff_eval(f::Base.Callable, x::Union{T, NTuple{N,T}, SVector{N,T}}) where {N,T<:Real}
    TagType = typeof(ForwardDiff.Tag(f, T))
    x_dual = forwarddiff_dualized(T, x)
    y_dual = f(x_dual)
end


forwarddiff_vjp(ΔΩ::Real, y_dual::ForwardDiff.Dual{TagType,T,1}) where {TagType,T<:Real} = ΔΩ * first(ForwardDiff.partials(y_dual))

function forwarddiff_vjp(ΔΩ::Union{NTuple{N,T},SVector{N,T}}, y_dual::NTuple{N,<:ForwardDiff.Dual}) where {N,T<:Real}
    (sum(map((ΔΩ_i, y_dual_i) -> ForwardDiff.partials(y_dual_i) * ΔΩ_i, ΔΩ, y_dual))...,)
end

# BAT.forwarddiff_vjp(ΔΩ, BAT.forwarddiff_eval(f, x)) == ForwardDiff.jacobian(f, x)' * ΔΩ (for SVector):
forwarddiff_vjp(ΔΩ::Union{NTuple{N,T},SVector{N,T}}, y_dual::SVector{N,<:ForwardDiff.Dual}) where {N,T<:Real} = SVector(forwarddiff_vjp((ΔΩ...,), (y_dual...,)))


# Return type of fwddiff_back (`SVector` or `NTuple`) currently depeds on type of `f(x)`, not type of `x`:
function forwarddiff_pullback(f::Base.Callable, x::Union{NTuple{N,T}, SVector{N,T}}) where {N,T<:Real}
    # Seems faster this way, according to benchmarking (benchmark artifact?):
    TagType = typeof(ForwardDiff.Tag(f, T))
    x_dual = forwarddiff_dualized(T, x)
    y_dual = f(x_dual)

    # Seems slower this way, for some reason:
    # y_dual = forwarddiff_eval(f, x)

    y = map(ForwardDiff.value, y_dual)
    fwddiff_back(ΔΩ) = forwarddiff_vjp(ΔΩ, y_dual)
    y, fwddiff_back
end


function forwarddiff_broadcast_pullback(fs, X::AbstractArray{<:Union{NTuple{N,T},SVector{N,T}}}) where {N,T<:Real}
    Y_dual = broadcast(forwarddiff_eval, fs, X)
    Y = broadcast(y_dual -> map(ForwardDiff.value, y_dual), Y_dual)
    fwddiff_back(ΔΩs) = broadcast(forwarddiff_vjp, ΔΩs, Y_dual)
    Y, fwddiff_back
end

#=
# May be faster for cheap `fs`, according to benchmarking, in spite of double evalutation (why?). Limited to SVector elements:
function forwarddiff_broadcast_pullback(fs, X::AbstractArray{<:SVector{N,T}}) where {N,T<:Real}
    Y = broadcast(fs, X)
    fwddiff_back(ΔΩs) = broadcast((f, x, ΔΩ) -> ForwardDiff.jacobian(f, x)' * ΔΩ, fs, X, ΔΩs)
    Y, fwddiff_back
end
=#
