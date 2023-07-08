# This file is a part of BAT.jl, licensed under the MIT License (MIT).


bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::DoNotTransform, ::AnyMeasureOrDensity) = IdentityTransformAlgorithm()
bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::DoNotTransform, ::AbstractPosteriorMeasure) = IdentityTransformAlgorithm()

bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::PriorToUniform, ::AbstractPosteriorMeasure) = PriorSubstitution()
bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::PriorToUniform, ::DistMeasure) = PriorSubstitution()
bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::PriorToUniform, ::Renormalized{<:DistMeasure}) = PriorSubstitution()
bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::PriorToUniform, ::EvaluatedMeasure) = PriorSubstitution()
bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::PriorToUniform, ::DistMeasure{<:StandardUniformDist}) = IdentityTransformAlgorithm()

bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::PriorToGaussian, ::AbstractPosteriorMeasure) = PriorSubstitution()
bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::PriorToGaussian, ::DistMeasure) = PriorSubstitution()
bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::PriorToGaussian, ::Renormalized{<:DistMeasure}) = PriorSubstitution()
bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::PriorToGaussian, ::EvaluatedMeasure) = PriorSubstitution()
bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::PriorToGaussian, ::DistMeasure{<:StandardNormalDist}) = IdentityTransformAlgorithm()

bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::Function, ::DensitySampleVector) = SampleTransformation()
bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::AbstractValueShape, ::DensitySampleVector) = SampleTransformation()

bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::ToRealVector, ::Union{BATMeasure,DensitySampleVector}) = UnshapeTransformation()

bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::Base.Fix2{typeof(unshaped)}, ::BATMeasure) = FullMeasureTransform()
bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::Base.Fix2{typeof(unshaped)}, ::DensitySampleVector) = SampleTransformation()
