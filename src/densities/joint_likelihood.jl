# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    struct BAT.JointLikelihood <: MeasureBase.AbstractLikelihood

*BAT-internal, not part of stable public API.*

Combines several likelihoods that share a common parameter space.

User code should not instantiate `JointLikelihood` directly, but use
[`joint_likelihood`](@ref) instead.
"""
struct JointLikelihood{L<:Tuple} <: MeasureBase.AbstractLikelihood
    likelihoods::L
end


"""
    joint_likelihood(likelihoods...)

Combine several likelihoods over a common parameter space into a joint
likelihood.

All component likelihoods are evaluated at the same (i.e. shared) parameter
point. The log-density of the joint likelihood is the sum of the component
log-densities.

The components may be given in any form that can serve as a likelihood in a
[`PosteriorMeasure`](@ref) and are converted accordingly.

`MeasureBase.insupport` is only defined for the joint likelihood if it is
defined for all of its components.
"""
function joint_likelihood end
export joint_likelihood

function joint_likelihood(likelihood, likelihoods...)
    ls = map(l -> _convert_likelihood(l, DensityKind(l)), (likelihood, likelihoods...))
    return JointLikelihood(_flat_likelihoods(ls))
end

_flat_likelihoods(ls::Tuple) = (_flat_likelihoods(first(ls))..., _flat_likelihoods(Base.tail(ls))...)
_flat_likelihoods(::Tuple{}) = ()
_flat_likelihoods(l::JointLikelihood) = l.likelihoods
_flat_likelihoods(l) = (l,)


function DensityInterface.logdensityof(ℒ::JointLikelihood, p::Any)
    sum(map(l -> logdensityof(l, p), ℒ.likelihoods))
end

(ℒ::JointLikelihood)(p) = exp(ULogarithmic, logdensityof(ℒ, p))

MeasureBase.insupport(ℒ::JointLikelihood, p) = all(map(l -> MeasureBase.insupport(l, p), ℒ.likelihoods))

_precompose_density(ℒ::JointLikelihood, g) = JointLikelihood(map(l -> _precompose_density(l, g), ℒ.likelihoods))
