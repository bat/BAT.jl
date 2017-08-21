# This file is a part of BAT.jl, licensed under the MIT License (MIT).

abstract type AbstractTargetSubject end



mutable struct TargetSubject <: AbstractTargetSubject {
    F<:AbstractTargetFunction,
    B<:AbstractParamBounds
}
    tfunc::F
    bounds::B
end

export TargetSubject

Base.length(subject::TargetSubject) = length(subject.bounds)


target_function(subject::TargetSubject) = subject.tfunc
param_bounds(subject::TargetSubject) = subject.bounds



#=

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
