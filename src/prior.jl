# This file is a part of BAT.jl, licensed under the MIT License (MIT).


const PriorDistribution = Distribution{Multivariate}

abstract type NoPrior end

const OptionalPrior = Union{NoPrior,PriorDistribution}
