# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    OptimOpt

Selects an optimization algorithm from the
[Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl)
package for minimization tasks.

Note that when using gradient-based algorithms like `Optim.LBFGS`, your
[`BATContext`](@ref) needs to include an `ADSelector` that specifies which
automatic differentiation backend should be used.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

`optimalg` must be an `Optim.AbstractOptimizer`.

The field `optargs` can be used to pass additional keyword arguments to
`Optim.Options`, see the
[Optimization.jl documentation](https://julianlsolvers.github.io/Optim.jl/stable/user/config/#General-Options).

Fields:

$(TYPEDFIELDS)

!!! note

    This algorithm is only available if the Optim package is loaded (e.g. via
    `import Optim`.
"""
@with_kw struct OptimOpt{ALG} <: AbstractMinimizer
    optalg::ALG = ext_default(pkgext(Val(:Optim)), Val(:DEFAULT_OPTALG))
    maxiters::Int = 1_000
    maxtime::Float64 = NaN
    abstol::Float64 = NaN
    reltol::Float64 = 0.0
    optargs::NamedTuple = (;)
end
export OptimOpt
