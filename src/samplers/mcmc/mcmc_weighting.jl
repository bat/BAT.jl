# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    abstract type AbstractMCMCWeightingScheme{T<:Real}

Abstract class for weighting schemes for MCMC samples.

Weight values will have type `T`.
"""
abstract type AbstractMCMCWeightingScheme{T<:Real} end
export AbstractMCMCWeightingScheme


sample_weight_type(::Type{<:AbstractMCMCWeightingScheme{T}}) where {T} = T



"""
    struct RepetitionWeighting{T<:AbstractFloat} <: AbstractMCMCWeightingScheme{T}

Sample weighting scheme suitable for sampling algorithms which may repeated
samples multiple times in direct succession (e.g.
[`RandomWalk`](@ref)). The repeated sample is stored only once,
with a weight equal to the number of times it has been repeated (e.g.
because a Markov chain has not moved during a sampling step).

Constructors:

* ```$(FUNCTIONNAME)()```
"""
struct RepetitionWeighting{T<:Real} <: AbstractMCMCWeightingScheme{T} end
export RepetitionWeighting

RepetitionWeighting() = RepetitionWeighting{Int}()

mcmc_weight_type(::RepetitionWeighting) = Int

function mcmc_weight_values(
    ::RepetitionWeighting,
    p_accept::Real,
    accepted::Bool
) where Q
    if accepted
        (0, 1)
    else
        (1, 0)
    end
end


"""
    ARPWeighting{T<:AbstractFloat} <: AbstractMCMCWeightingScheme{T}

*Experimental feature, not part of stable public API.*

Sample weighting scheme suitable for accept/reject-based sampling algorithms
(e.g. [`RandomWalk`](@ref)). Both accepted and rejected samples
become part of the output, with a weight proportional to their original
acceptance probability.

Constructors:

* ```$(FUNCTIONNAME)()```
"""
struct ARPWeighting{T<:AbstractFloat} <: AbstractMCMCWeightingScheme{T} end
export ARPWeighting

ARPWeighting() = ARPWeighting{Float64}()

mcmc_weight_type(::ARPWeighting) = Float64

function mcmc_weight_values(
    scheme::ARPWeighting,
    p_accept::Real,
    accepted::Bool
)
    T = typeof(p_accept)
    if p_accept ≈ 1
        (zero(T), one(T))
    elseif p_accept ≈ 0
        (one(T), zero(T))
    else
        (T(1 - p_accept), p_accept)
    end
end
