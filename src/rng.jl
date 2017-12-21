# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using RandomNumbers.Random123: Philox4x, Threefry4x
using RandomNumbers.Random123: random123_r, gen_seed

const Random123_UInt = Union{UInt32, UInt64}
const Random123RNG4x = Union{Philox4x, Threefry4x}


abstract type AbstractRNGSeed end

export AbstractRNGSeed

AbstractRNGSeed() = Philox4xSeed()



struct Philox4xSeed{T<:Random123_UInt} <: AbstractRNGSeed
    seed::NTuple{2,T}

    Philox4xSeed{T}(seed::NTuple{2,T}) where {T<:Random123_UInt} = new{T}(seed)
    Philox4xSeed{T}() where {T<:Random123_UInt} = new{T}(gen_seed(T, 2))
end

export Philox4xSeed

# (::Type{Philox4xSeed{T}})() where {T<:Random123_UInt} = Philox4xSeed(gen_seed(T, 2))

Philox4xSeed() = Philox4xSeed{UInt64}()

function (rngseed::Philox4xSeed{T})() where {T}
    rng = Philox4x{T,10}(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    srand(rng, rngseed.seed)
end



struct Threefry4xSeed{T<:Random123_UInt} <: AbstractRNGSeed
    seed::NTuple{4,T}

    Threefry4xSeed{T}(seed::NTuple{4,T}) where {T<:Random123_UInt} = new{T}(seed)
    Threefry4xSeed{T}() where {T<:Random123_UInt} = new{T}(gen_seed(T, 4))
end

export Threefry4xSeed

Threefry4xSeed() = Threefry4xSeed{UInt64}()

function (rngseed::Threefry4xSeed{T})() where {T}
    rng = Threefry4x{T,20}(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    srand(rng, rngseed.seed)
end



reset_rng_counters!(rng::AbstractRNG, tags::Integer...) = reset_rng_counters!(rng, tags)

# ToDo: More flexible tagging scheme, allow for more/less than 3 tags and
# support counter-based RNGs with more/less than 4 counters.

function reset_rng_counters!(rng::Random123RNG4x, tags::NTuple{3,Integer})
    rng.ctr3 += Base.Threads.threadid()
    rng.ctr4 = tags[1]
    rng.ctr3 = tags[2]
    rng.ctr2 = tags[3]
    rng.ctr1 = zero(rng.ctr1)
    random123_r(rng)
end



struct MersenneTwisterSeed{R<:AbstractRNG} <: AbstractRNGSeed
    rng::R
end

export MersenneTwisterSeed

MersenneTwisterSeed() = MersenneTwisterSeed(RandomDevice())

(rngseed::MersenneTwisterSeed)() =
    MersenneTwister(rand(rngseed.rng, UInt64))

# ToDo (maybe): Implement tagging (resp. multiple streams) for Mersenne
# Twister via skip-ahead
