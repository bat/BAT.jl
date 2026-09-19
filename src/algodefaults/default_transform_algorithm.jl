# This file is a part of BAT.jl, licensed under the MIT License (MIT).


bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::DoNotTransform, ::MeasureLike) = IdentityTransformAlgorithm()

bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::Union{UniformBased,NormalBased}, ::AbstractMeasure) = PriorSubstitution()
bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::UniformBased, ::_StdUniformMeasure) = IdentityTransformAlgorithm()
bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::NormalBased, ::_StdNormalMeasure) = IdentityTransformAlgorithm()

bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::Function, ::AbstractMeasure) = FullMeasureTransform()
bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::Function, ::DensitySampleVector) = SampleTransformation()
bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::Function, ::DensitySampleMeasure) = SampleTransformation()
bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::AbstractValueShape, ::DensitySampleVector) = SampleTransformation()
bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::AbstractValueShape, ::DensitySampleMeasure) = SampleTransformation()

bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::ToRealVector, ::Union{AbstractMeasure,DensitySampleVector}) = UnshapeTransformation()

bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::Base.Fix2{typeof(unshaped)}, ::AbstractMeasure) = FullMeasureTransform()
bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::Base.Fix2{typeof(unshaped)}, ::DensitySampleVector) = SampleTransformation()
bat_default(::typeof(bat_transform), ::Val{:algorithm}, ::Base.Fix2{typeof(unshaped)}, ::DensitySampleMeasure) = SampleTransformation()
