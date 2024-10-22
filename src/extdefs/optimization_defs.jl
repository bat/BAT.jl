# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    struct OptimizationAlg

Selects an optimization algorithm from the
[Optimization.jl](https://github.com/SciML/Optimization.jl)
package.
Note that when using first order algorithms like `OptimizationOptimJL.LBFGS`, your
[`BATContext`](@ref) needs to have `ad` set to an automatic differentiation
backend.

Constructors:
* ```$(FUNCTIONNAME)(; fields...)```
`optalg` must be an `Optimization.AbstractOptimizer`.
The field `kwargs` can be used to pass additional keywords to the optimizers
See the [Optimization.jl documentation](https://docs.sciml.ai/Optimization/stable/) for the available keyword arguments.
Fields:
$(TYPEDFIELDS)
!!! note
    This algorithm is only available if the `Optimization` package or any of its submodules, like `OptimizationOptimJL`, is loaded (e.g. via
        `import Optimization`).
"""
@with_kw struct OptimizationAlg{
    ALG,
    TR<:AbstractTransformTarget,
    IA<:InitvalAlgorithm
} <: AbstractModeEstimator
    optalg::ALG = ext_default(pkgext(Val(:Optimization)), Val(:DEFAULT_OPTALG))
    trafo::TR = PriorToGaussian()
    init::IA = InitFromTarget()
    maxiters::Int64 = 1_000
    maxtime::Float64 = NaN
    abstol::Float64 = NaN
    reltol::Float64 = 0.0
    kwargs::NamedTuple = (;)
end
export OptimizationAlg
