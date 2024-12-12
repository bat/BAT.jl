# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    struct OptimizationOpt

Selects an optimization algorithm from the
[Optimization.jl](https://github.com/SciML/Optimization.jl)
package for minimization tasks.

Note that when using gradient-based algorithms like
`OptimizationOptimJL.LBFGS`, your [`BATContext`](@ref) needs to have `ad` set
to an automatic differentiation backend.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

The field `optargs` can be used to pass additional keyword arguments to
`Optimization.solve`, see the
[Optimization.jl documentation](https://docs.sciml.ai/Optimization/stable/).

Fields:

$(TYPEDFIELDS)

!!! note

    This algorithm is only available if the `Optimization` package and
    (depending on `optalg`) additional required packages (e.g.
    `OptimizationOptimJL`) are loaded (e.g. via `import Optimization, ...`).
"""
@with_kw struct OptimizationOpt{ALG} <: AbstractModeEstimator
    optalg::ALG = ext_default(pkgext(Val(:Optimization)), Val(:DEFAULT_OPTALG))
    maxiters::Int64 = 1_000
    maxtime::Float64 = NaN
    abstol::Float64 = NaN
    reltol::Float64 = 0.0
    optargs::NamedTuple = (;)
end
export OptimizationOpt
