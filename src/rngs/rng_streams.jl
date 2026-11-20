# This file is a part of BAT.jl, licensed under the MIT License (MIT).

@deprecate bat_rng() HeterogeneousComputing.get_rng(BAT.get_batcontext())

_bat_determ_rng() = rootrng(0)

# MCMC derives the random streams of chains, steps and walkers from stream
# keys (see `HeterogeneousComputing.subrng`). A context RNG without a key
# serves as the entropy of a keyed root on its own compute unit:
_keyed_rng(rng::AbstractRNG) = has_rng_streamkey(rng) ? rng : rootrng(rng)
