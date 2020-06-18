# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using Random123: Philox4x, Threefry4x
using Random123: random123_r, gen_seed


@doc doc"""
    bat_rng()

Return a new BAT-compatible random number generator, with a random seed drawn
from the system entropy pool.
"""
function bat_rng end
export bat_rng

bat_rng() = Philox4x()::Philox4x{UInt64,10}

bat_determ_rng() = Philox4x((0, 0))::Philox4x{UInt64,10}



struct RNGPartition{R<:AbstractRNG,S,C,I}
    seed::S
    partctrsbase::C
    depth::Int
    idxs::I
end


function RNGPartition(parent_rng::AbstractRNG, partidxs::AbstractUnitRange{<:Integer})
    seed = rngpart_getseed(parent_rng)
    partcounters, depth = rngpart_getpartctrs(parent_rng)

    child_rngpart_partctrsbase = _rngpart_inc_partctrs(partcounters, depth, 1)

    parent_mod_partcounters = _rngpart_inc_partctrs(child_rngpart_partctrsbase, depth, length(partidxs))
    rngpart_setpartctrs!(parent_rng, parent_mod_partcounters, depth)

    R = typeof(parent_rng)
    S = typeof(seed)
    C = typeof(child_rngpart_partctrsbase)
    I = typeof(partidxs)
    RNGPartition{R,S,C,I}(seed, child_rngpart_partctrsbase, depth, partidxs)
end


Base.eachindex(rngpart::RNGPartition) = rngpart.idxs
Base.length(rngpart::RNGPartition) = length(eachindex(rngpart))
Base.size(rngpart::RNGPartition) = (length(rngpart),)


function set_rng!(rng::R, rngpart::RNGPartition{R}, i::Integer) where R <: AbstractRNG
    idxs = eachindex(rngpart)
    Base.checkindex(Bool, idxs, i) || throw(ArgumentError("Index $i not in partition indices $idxs of $rngpart"))

    j = i - minimum(idxs)
    mod_partcounters = _rngpart_inc_partctrs(rngpart.partctrsbase, rngpart.depth, j)
    mod_depth = rngpart.depth + 1

    Random.seed!(rng, rngpart.seed)
    rngpart_setpartctrs!(rng, mod_partcounters, mod_depth)

    rng
end


Random.AbstractRNG(rngpart::RNGPartition{R}, i::Integer) where R =
    set_rng!(rngpart_createrng(R), rngpart, i)


rngpart_createrng(::Type{T}) where {T <: Philox4x} = T(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

rngpart_getseed(rng::Philox4x) = (rng.key1, rng.key2)

# function set_rng!(target::Philox4x, src::Philox4x)
#     target.ctr1 = src.ctr1
#     target.ctr2 = src.ctr2
#     target.ctr3 = src.ctr3
#     target.ctr4 = src.ctr4
#     target.key1 = src.key1
#     target.key2 = src.key2
#     target.p = src.p
#     target.x1 = src.x1
#     target.x2 = src.x2
#     target.x3 = src.x3
#     target.x4 = src.x4
#     target
# end


rngpart_createrng(::Type{T}) where {T <: Threefry4x} = T(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

rngpart_getseed(rng::Threefry4x) = (rng.key1, rng.key2, rng.key3, rng.key4)

# function set_rng!(target::Threefry4x, src::Threefry4x)
#     target.x1 = src.x1
#     target.x2 = src.x2
#     target.x3 = src.x3
#     target.x4 = src.x4
#     target.key1 = src.key1
#     target.key2 = src.key2
#     target.key3 = src.key3
#     target.key4 = src.key4
#     target.ctr1 = src.ctr1
#     target.ctr2 = src.ctr2
#     target.ctr3 = src.ctr3
#     target.ctr4 = src.ctr4
#     target.p = src.p
# end


function rngpart_getpartctrs(rng::Union{Philox4x,Threefry4x})
    partctrinfo = _rngpart_split_uints(rng.ctr4, rng.ctr3, rng.ctr2)
    depth = _rngpart_getdepth(partctrinfo)
    partctrs = map(_rngpart_getpartctr, partctrinfo)
    (partctrs = partctrs, depth = depth)
end


function rngpart_setpartctrs!(rng::Union{Philox4x,Threefry4x}, partctrs::NTuple{6,UInt32}, depth::Int)
    tagged_partctrs = _rngpart_settopbit(partctrs, depth)
    merged_partctrs = _rngpart_merge_uints(tagged_partctrs...)::NTuple{3,UInt64}

    rng.ctr4 = merged_partctrs[1]
    rng.ctr3 = merged_partctrs[2]
    rng.ctr2 = merged_partctrs[3]
    rng.ctr1 = 0

    random123_r(rng)

    rng
end


@inline _rngpart_split_uints() = ()
@inline _rngpart_split_uints(x::UInt64, xs::UInt64...) =
    (UInt32(x >> 32), UInt32(x << 32 >> 32), _rngpart_split_uints(xs...)...)

@inline _rngpart_merge_uints() = ()
@inline _rngpart_merge_uints(x1::UInt32, x2::UInt32, xs::UInt32...) =
    (UInt64(x1) << 32 | UInt64(x2), _rngpart_merge_uints(xs...)...)


_rngpart_topbit_mask(::Type{T}) where {T <: Unsigned} = ((typemax(T) >> 1) + one(T))

_rngpart_lowbits_mask(::Type{T}) where {T <: Unsigned} = typemax(T) >> 1


_rngpart_haspartctrtag(x::T) where {T<:Unsigned} = (x & _rngpart_topbit_mask(T)) > 0 

_rngpart_getpartctr(x::T) where {T<:Unsigned} = (x & _rngpart_lowbits_mask(T))


function _rngpart_getdepth(partctrinfo::NTuple{N,T}) where {N,T<:Unsigned}
    cycle::Int = 1
    for i in eachindex(partctrinfo)
        x = partctrinfo[i]
        if _rngpart_haspartctrtag(x)
            cycle = i
        else
            x == 0 || throw(ArgumentError("Inconsistent partition counter information"))
        end
    end
    return cycle
end


function _rngpart_inc_partctrs(partctrs::NTuple{N,T}, depth::Int, x::Integer) where {N,T<:Unsigned}
    1 <= depth <= length(partctrs) || throw(ArgumentError("Partition depth out of allowed range"))
    m = ntuple(i -> i == depth ? T(x) : zero(T), Val(length(partctrs)))
    partctrs .+ m
end


function _rngpart_settopbit(partctrs::NTuple{N,T}, depth::Int) where {N,T<:Unsigned}
    1 <= depth <= length(partctrs) || throw(ArgumentError("Partition depth out of allowed range"))
    any(_rngpart_haspartctrtag, partctrs) && throw(ArgumentError("Partition counter(s) out of allowed range"))
    m = ntuple(i -> i <= depth ? _rngpart_topbit_mask(T) : zero(T), Val(length(partctrs)))
    partctrs .| m
end
