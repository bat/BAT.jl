# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using MultiThreadingTools, Base.Threads


struct ThreadSafeRNG{RNG<:AbstractRNG} <: AbstractRNG
    rng::ThreadLocal{RNG}
end

export ThreadSafeRNG

function ThreadSafeRNG(rngtype::Type{RNG}) where {RNG<:AbstractRNG}
    systemrng = RandomDevice(false)
    seeds = rand(systemrng, UInt64, nthreads())
    rngs = rngtype.(seeds)
    ThreadSafeRNG(ThreadLocal{eltype(rngs)}(rngs))
end


@inline Base.rand(rng::ThreadSafeRNG, tp::Type, args...) = rand(rng.rng[], tp, args...)

@inline Base.rand!(rng::ThreadSafeRNG, A::Array, args...) = rand!(rng.rng[], A, args...)
