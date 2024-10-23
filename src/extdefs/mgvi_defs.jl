# This file is a part of BAT.jl, licensed under the MIT License (MIT).

struct MGVISampleInfo <: SampleID
    stepno::Int
    converged::Bool
    mnlp::Float64
end


const MGVISampleInfoVector{
    UV<:AbstractVector{<:Int}, BV<:AbstractVector{<:Bool}, FV<:AbstractVector{<:Float64}
} = StructArray{
    MGVISampleInfo,
    1,
    NamedTuple{
        (:stepno, :converged, :mnlp),
        Tuple{UV,BV,FV}
    },
    Int
}


MGVISampleInfoVector(contents::NTuple{3,Any}) = StructArray{MGVISampleInfo}(contents)

MGVISampleInfoVector(::UndefInitializer, len::Integer) = MGVISampleInfoVector((
    Vector{Int}(undef, len), Vector{Bool}(undef, len), Vector{Float64}(undef, len)
))

MGVISampleInfoVector() = MGVISampleInfoVector(undef, 0)

_create_undef_vector(::Type{MGVISampleInfo}, len::Integer) = MGVISampleInfoVector(undef, len)



# Specialize comparison, currently StructArray seems fall back to `(==)(A::AbstractArray, B::AbstractArray)`
import Base.==
function(==)(A::MGVISampleInfoVector, B::MGVISampleInfoVector)
    A.stepno == B.stepno &&
    A.converged == B.converged &&
    A.mnlp == B.mnlp
end


function Base.merge!(X::MGVISampleInfoVector, Xs::MGVISampleInfoVector...)
    for Y in Xs
        append!(X, Y)
    end
    X
end

Base.merge(X::MGVISampleInfoVector, Xs::MGVISampleInfoVector...) = merge!(deepcopy(X), Xs...)




"""
    abstract type BAT.MGVISchedule

Abstract supertype for MGVI sampling schedules.

See [`MGVISampling`](@ref).
"""
abstract type MGVISchedule end


"""
    abstract type FixedMGVISchedule <: BAT.MGVISchedule

Abstract supertype for MGVI sampling schedules.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)


Constructors:

* `FixedMGVISchedule(nsamples::AbstractVector{<:Real})`: The number of
  samples to draw at each MGVI step. The length of `nsamples` implies the
  total number of steps. The number of samples will be rounded to integer
  values if necessary, to allow for constructions like
  `FixedMGVISchedule(range(12, 1000, length = 10))`.

Fields:

* `nsamples::AbstractVector{<:Real}`: See constructor above.

See [`MGVISampling`](@ref).
"""
@with_kw struct FixedMGVISchedule{TV<:AbstractVector{<:Real}} <: MGVISchedule
    nsamples::TV = range(12, 1000, length = 10)
end
export FixedMGVISchedule


"""
    struct MGVISampling <: AbstractUltraNestAlgorithmReactiv


Samples via
[Metric Gaussian Variational Inference](https://arxiv.org/abs/1901.11033),
using the [MGVI.jl](https://github.com/bat/MGVI.jl) Julia implementation
of the algorithm.


Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

* `pretransform::AbstractTransformTarget`: Pre-transformation to apply to the target measure before sampling.

* `nsamples::Int`: Number is independent samples to draw. MGVI will generate symmetical samples, so it will generate
  `2*nsamples`` samples in total, but only `nsamples` independent samples.

* `schedule::MGVISchedule`: MGVI schedule, by default a [`FixedMGVISchedule`](@ref).

* `config::MGVI.MGVIConfig`: MGVI configuration.

!!! note

    This functionality is only available when the package [MGVI](https://github.com/bat/MGVI.jl) is loaded (e.g. via
    `import MGVI`).
"""
@with_kw struct MGVISampling{
    TR<:AbstractTransformTarget, IA<:InitvalAlgorithm,
    CFG, SD<:MGVISchedule
} <: AbstractSamplingAlgorithm
    pretransform::TR = (pkgext(Val(:MGVI)); PriorToNormal())
    init::IA = InitFromTarget()
    nsamples::Int = 10^4
    schedule::SD = FixedMGVISchedule(range(12, nsamples, length = 10))
    config::CFG = ext_default(pkgext(Val(:MGVI)), Val(:MGVI_CONFIG))
    store_unconverged::Bool = false
end
export MGVISampling
