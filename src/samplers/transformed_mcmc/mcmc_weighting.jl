# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    abstract type TransformedAbstractMCMCWeightingScheme{T<:Real}

Abstract class for weighting schemes for MCMC samples.

Weight values will have type `T`.
"""
abstract type TransformedAbstractMCMCWeightingScheme{T<:Real} end
export TransformedAbstractMCMCWeightingScheme


sample_weight_type(::Type{<:AbstractMCMCWeightingScheme{T}}) where {T} = T



"""
    struct TransformedRepetitionWeighting{T<:AbstractFloat} <: TransformedAbstractMCMCWeightingScheme{T}

Sample weighting scheme suitable for sampling algorithms which may repeated
samples multiple times in direct succession (e.g.
[`MetropolisHastings`](@ref)). The repeated sample is stored only once,
with a weight equal to the number of times it has been repeated (e.g.
because a Markov chain has not moved during a sampling step).

Constructors:

* ```$(FUNCTIONNAME)()```
"""
struct TransformedRepetitionWeighting{T<:Real} <: TransformedAbstractMCMCWeightingScheme{T} end
export TransformedRepetitionWeighting

TransformedRepetitionWeighting() = TransformedRepetitionWeighting{Int}()


"""
    TransformedARPWeighting{T<:AbstractFloat} <: TransformedAbstractMCMCWeightingScheme{T}

Sample weighting scheme suitable for accept/reject-based sampling algorithms
(e.g. [`MetropolisHastings`](@ref)). Both accepted and rejected samples
become part of the output, with a weight proportional to their original
acceptance probability.

Constructors:

* ```$(FUNCTIONNAME)()```
"""
struct TransformedARPWeighting{T<:AbstractFloat} <: TransformedAbstractMCMCWeightingScheme{T} end
export TransformedARPWeighting

TransformedARPWeighting() = TransformedARPWeighting{Float64}()
