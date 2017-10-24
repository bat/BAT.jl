# This file is a part of BAT.jl, licensed under the MIT License (MIT).


const PriorDistribution = Distribution{Multivariate}

abstract type OptionalPrior end

abstract type NoPrior <: OptionalPrior end
