# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# UltraNest docstrings are reproduced here under MIT License with the kind
# permission of the original author
# Johannes Buchner <johannes.buchner.acad@gmx.com>.

"""
    struct MGVInference <: AbstractUltraNestAlgorithmReactiv

*Experimental feature, not part of stable public API.*

Samples via
[Metric Gaussian Variational Inference](https://arxiv.org/abs/1901.11033),
using the [MGVI.jl](https://github.com/bat/MGVI.jl) Julia implementation
of the algorithm.


Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

* trafo::AbstractTransformTarget
* config::MGVI.MGVIConfig


!!! note

    This functionality is only available when the
    [MGVI](https://github.com/bat/MGVI.jl) package is loaded (e.g. via
    `import MGVI`).
"""
@with_kw struct MGVInference{TR<:AbstractTransformTarget,CFG} <: AbstractSamplingAlgorithm
    trafo::TR = (pkgext(Val(:MGVI)); PriorToNormal())
    config::CFG
end
export MGVInference
