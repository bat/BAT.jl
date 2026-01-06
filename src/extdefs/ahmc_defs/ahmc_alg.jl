# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# TODO: MD, Adjust docstring to new typestructure
"""
    struct HamiltonianMC <: MCMCAlgorithm

The [Hamiltonian Monte Carlo](https://en.wikipedia.org/wiki/Hamiltonian_Monte_Carlo)
(HMC) sampling algorithm.

Uses the HMC implementation provided by the package
[AdvancedHMC](https://github.com/TuringLang/AdvancedHMC.jl).

HMC uses gradients of the target measure's density, so your [`BATContext`](@ref)
needs to include an `ADSelector` to specify which automatic differentiation
backend should be used.

* Note: The fields of `HamiltonianMC` are still subject to change, and not
yet part of stable public BAT API!*

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)

!!! note

    `HamiltonianMC` is only available if the AdvancedHMC package is loaded
    (e.g. via `import AdvancedHMC`). 
"""
@with_kw struct HamiltonianMC{
    TA<:Real,
    TAI<:Tuple{Vararg{Real}},
    MT<:HMCMetric,
    IT,
    TC
} <: MCMCProposal
    target_acceptance::TA = 0.8
    target_acceptance_int::TAI = (0.9 * target_acceptance, one(Float64))
    metric::MT = UnitEuclideanMetric()
    integrator::IT = ext_default(pkgext(Val(:AdvancedHMC)), Val(:DEFAULT_INTEGRATOR))
    termination::TC = ext_default(pkgext(Val(:AdvancedHMC)), Val(:DEFAULT_TERMINATION_CRITERION))
end

export HamiltonianMC


mutable struct HMCProposalState{
    TA<:Real,
    TAI<:Tuple{Vararg{Real}},
    IT,
    TC,
    HA,   # <:AdvancedHMC.Hamiltonian,
    KRNL, # <:AdvancedHMC.HMCKernel
    TR    # <:AdvancedHMC.Transition
} <: MCMCProposalState
    target_acceptance::TA
    target_acceptance_int::TAI
    integrator::IT
    termination::TC
    hamiltonian::HA
    kernel::KRNL
    transition::TR
end

export HMCProposalState
