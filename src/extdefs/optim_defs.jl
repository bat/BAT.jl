# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    OptimAlg

Selects an optimization algorithm from the
[Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl)
package.

Note that when using first order algorithms like `Optim.LBFGS`, your
[`BATContext`](@ref) needs to include an `ADSelector` that specifies
which automatic differentiation backend should be used.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

`optimalg` must be an `Optim.AbstractOptimizer`.

Fields:

$(TYPEDFIELDS)

!!! note

    This algorithm is only available if the Optim package is loaded (e.g. via
        `import Optim`.
"""
@with_kw struct OptimAlg{
    ALG,
    TR<:AbstractTransformTarget,
    IA<:InitvalAlgorithm
} <: AbstractModeEstimator
    optalg::ALG = ext_default(pkgext(Val(:Optim)), Val(:DEFAULT_OPTALG))
    trafo::TR = PriorToGaussian()
    init::IA = InitFromTarget()
end
export OptimAlg
