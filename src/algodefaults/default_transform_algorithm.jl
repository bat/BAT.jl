# This file is a part of BAT.jl, licensed under the MIT License (MIT).


bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::DoNotTransform, ::MeasureLike) = IdentityTransformAlgorithm()
bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::DoNotTransform, ::AbstractPosteriorMeasure) = IdentityTransformAlgorithm()

bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::UniformBased, ::AbstractPosteriorMeasure) = PriorSubstitution()
bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::UniformBased, ::BATDistMeasure) = PriorSubstitution()
bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::UniformBased, ::BATPushFwdMeasure) = PriorSubstitution()
bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::UniformBased, ::EvaluatedMeasure) = PriorSubstitution()
bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::UniformBased, ::BATWeightedMeasure) = PriorSubstitution()
bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::UniformBased, ::BATDistMeasure{<:StandardUniformDist}) = IdentityTransformAlgorithm()

bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::NormalBased, ::AbstractPosteriorMeasure) = PriorSubstitution()
bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::NormalBased, ::BATDistMeasure) = PriorSubstitution()
bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::NormalBased, ::BATPushFwdMeasure) = PriorSubstitution()
bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::NormalBased, ::EvaluatedMeasure) = PriorSubstitution()
bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::NormalBased, ::BATWeightedMeasure) = PriorSubstitution()
bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::NormalBased, ::BATDistMeasure{<:StandardNormalDist}) = IdentityTransformAlgorithm()

bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::Function, ::BATMeasure) = FullMeasureTransform()
bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::Function, ::DensitySampleVector) = SampleTransformation()
bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::Function, ::DensitySampleMeasure) = SampleTransformation()
bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::AbstractValueShape, ::DensitySampleVector) = SampleTransformation()
bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::AbstractValueShape, ::DensitySampleMeasure) = SampleTransformation()

bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::ToRealVector, ::Union{BATMeasure,DensitySampleVector}) = UnshapeTransformation()

bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::Base.Fix2{typeof(unshaped)}, ::BATMeasure) = FullMeasureTransform()
bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::Base.Fix2{typeof(unshaped)}, ::DensitySampleVector) = SampleTransformation()
bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::Base.Fix2{typeof(unshaped)}, ::DensitySampleMeasure) = SampleTransformation()
