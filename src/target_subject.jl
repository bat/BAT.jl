# This file is a part of BAT.jl, licensed under the MIT License (MIT).

abstract type AbstractTargetSubject end



mutable struct TargetSubject{
    F<:AbstractTargetDensity,
    B<:AbstractParamBounds
} <: AbstractTargetSubject
    tdensity::F
    bounds::B
end

export TargetSubject

Base.length(subject::TargetSubject) = length(subject.bounds)

target_function(subject::TargetSubject) = subject.tdensity
param_bounds(subject::TargetSubject) = subject.bounds
nparams(subject::TargetSubject) = nparams(subject.bounds)


rand_initial_params(rng::AbstractRNG, target::TargetSubject) =
    rand_initial_params!(rng, target, Vector{float(eltype(target.bounds))}(nparams(target)))

rand_initial_params(rng::AbstractRNG, target::TargetSubject, n::Integer) =
    rand_initial_params!(rng, target, Matrix{float(eltype(target.bounds))}(nparams(target), n))

rand_initial_params!(rng::AbstractRNG, target::TargetSubject, x::StridedVecOrMat{<:Real}) =
    rand!(rng, target.bounds, x)


#=

# ToDo: Something like

Base.rand!(
    rng::AbstractRNG,
    target::TargetSubject,
    S::Type{<:DensitySample},
    nsamples::Integer,
    exec_context::ExecContext = ExecContext();
    max_nsteps::Int = 1000,
    max_time::Float64 = Inf,
    granularity::Int = 1,
    ll::LogLevel = LOG_NONE
)
    ...
end

=#


#=

# ToDo:

mutable struct TransformedTargetSubject{
    SO<:AbstractTargetSubject,
    SN<:TargetSubject
} <: AbstractTargetSubject
   before::SO
   after::SN
   # ... transformation, Jacobi matrix of transformation, etc.
end

export TransformedTargetSubject

...

=#
