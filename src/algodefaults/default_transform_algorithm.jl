# This file is a part of BAT.jl, licensed under the MIT License (MIT).


bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::NoDensityTransform, ::AnyDensityLike) = DensityIdentityTransform()
bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::NoDensityTransform, ::AbstractPosteriorDensity) = DensityIdentityTransform()

bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::PriorToUniform, ::AbstractPosteriorDensity) = PriorSubstitution()
bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::PriorToGaussian, ::AbstractPosteriorDensity) = PriorSubstitution()


# ToDo: Add ToUnitBounded and ToUnbounded
