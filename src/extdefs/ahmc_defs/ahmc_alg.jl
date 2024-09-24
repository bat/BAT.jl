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
    MT<:HMCMetric,
    IT,
    TC,
    WS<:AbstractMCMCWeightingScheme
} <: MCMCProposal
    metric::MT = DiagEuclideanMetric()
    integrator::IT = ext_default(pkgext(Val(:AdvancedHMC)), Val(:DEFAULT_INTEGRATOR))
    termination::TC = ext_default(pkgext(Val(:AdvancedHMC)), Val(:DEFAULT_TERMINATION_CRITERION))
    weighting::WS = RepetitionWeighting()
end

export HamiltonianMC


mutable struct HMCProposalState{
    IT,
    TC,
    HA,#<:AdvancedHMC.Hamiltonian,
    KRNL,#<:AdvancedHMC.HMCKernel
    TR,# <:AdvancedHMC.Transition
    WS<:AbstractMCMCWeightingScheme
} <: MCMCProposalState
    integrator::IT
    termination::TC
    hamiltonian::HA
    kernel::KRNL
    transition::TR
    weighting::WS 
end

export HMCProposalState

const HMCState = MCMCState{<:BATMeasure,
                          <:RNGPartition,
                          <:Function,
                          <:HMCProposalState,
                          <:DensitySampleVector,
                          <:DensitySampleVector,
                          <:BATContext
}
