# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    struct OptimizationAlg

Selects an optimization algorithm from the
[OptimizationBase.jl](https://github.com/SciML/OptimizationBase.jl)
package as the backend for density maximization.

Used via [`TransformedMaxDensity`](@ref) for mode estimation; a bare
`OptimizationAlg` used as a mode estimator is auto-wrapped in a
`TransformedMaxDensity` with default settings.

Note that when using first order algorithms like `OptimizationOptimJL.LBFGS`, your
[`BATContext`](@ref) needs to have `ad` set to an automatic differentiation
backend.

Constructors:
* ```$(FUNCTIONNAME)(; fields...)```
`optalg` must be an `OptimizationBase.AbstractOptimizer`.
The field `kwargs` can be used to pass additional keywords to the optimizers
See the [OptimizationBase.jl documentation](https://docs.sciml.ai/Optimization/stable/) for the available keyword arguments.
Fields:
$(TYPEDFIELDS)
!!! note
    This algorithm is only available if the `OptimizationBase` package or any of its submodules, like `OptimizationOptimJL`, is loaded (e.g. via
        `import OptimizationOptimJL`).
"""
@with_kw struct OptimizationAlg{ALG}
    optalg::ALG = ext_default(pkgext(Val(:OptimizationBase)), Val(:DEFAULT_OPTALG))
    maxiters::Int64 = 1_000
    maxtime::Float64 = NaN
    abstol::Float64 = NaN
    reltol::Float64 = 0.0
    store_trace::Bool = false
    kwargs::NamedTuple = (;)
end
export OptimizationAlg
