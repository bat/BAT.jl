# This file is a part of BAT.jl, licensed under the MIT License (MIT).


bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::DoNotTransform, ::MeasureLike) = IdentityTransformAlgorithm()
bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::DoNotTransform, ::AbstractPosteriorMeasure) = IdentityTransformAlgorithm()

bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::PriorToUniform, ::AbstractPosteriorMeasure) = PriorSubstitution()
bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::PriorToUniform, ::BATDistMeasure) = PriorSubstitution()
bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::PriorToUniform, ::EvaluatedMeasure) = PriorSubstitution()
bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::PriorToUniform, ::BATDistMeasure{<:StandardUniformDist}) = IdentityTransformAlgorithm()

bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::PriorToNormal, ::AbstractPosteriorMeasure) = PriorSubstitution()
bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::PriorToNormal, ::BATDistMeasure) = PriorSubstitution()
bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::PriorToNormal, ::EvaluatedMeasure) = PriorSubstitution()
bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::PriorToNormal, ::BATDistMeasure{<:StandardNormalDist}) = IdentityTransformAlgorithm()

bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::Function, ::DensitySampleVector) = SampleTransformation()
bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::AbstractValueShape, ::DensitySampleVector) = SampleTransformation()

bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::ToRealVector, ::Union{BATMeasure,DensitySampleVector}) = UnshapeTransformation()

bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::Base.Fix2{typeof(unshaped)}, ::BATMeasure) = FullMeasureTransform()
bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::Base.Fix2{typeof(unshaped)}, ::DensitySampleVector) = SampleTransformation()
