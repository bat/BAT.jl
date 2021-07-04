# BAT.jl Documentation

BAT.jl is a Bayesian Analysis Toolkit in Julia. It is a high-performance tool box for Bayesian inference with statistical models expressed in a general-purpose programming language instead of a domain-specific language.

Typical applications for this package are parameter inference given a model (in the form of a likelihood function and prior), the comparison of different models in the light of a given data set, and the test of the validity of a model to represent the data set at hand. BAT.jl provides access to the full Bayesian posterior distribution to enable parameter estimation, limit setting and uncertainty propagation. BAT.jl also provides supporting functionality like plotting recipes and reporting functions.

BAT.jl is implemented in pure Julia and allows for a flexible definition of mathematical models and applications while enabling the user to code for the performance required for computationally expensive numerical operations. BAT.jl provides implementations (internally and via other Julia packages) of algorithms for sampling, optimization and integration. BAT's main focus is on the analysis of complex custom models. It is designed to enable parallel code execution at various levels (running multiple MCMC chains in parallel is provided out-of-the-box).

It's possible to use BAT.jl with likelihood functions implemented in languages other than Julia: Julia allows for [calling code in C and Fortran](https://docs.julialang.org/en/v1/manual/calling-c-and-fortran-code/index.html), [C++](https://github.com/JuliaInterop/Cxx.jl), [Python](https://github.com/JuliaPy/PyCall.jl) and [several other languages](https://github.com/JuliaInterop) directly. In addition, BAT.jl provides (as an experimental feature) a very lightweight binary RPC protocol that is easy to implement, to call non-Julia likelihood functions written in another language and running in separate processes.

BAT.jl originated as a rewrite/redesign of [BAT](https://github.com/bat/bat), the Bayesian Analysis Toolkit in C++. BAT.jl now offer a different set of functionality and a wider variety of algorithms than it's C++ predecessor.

!!! note

    BAT.jl requires Julia >= v1.3, we recommend to use Julia >= v1.6.


## Table of contents

```@contents
Pages = [
    # "basics.md",
    "installation.md",
    "tutorial.md",
    # "faq.md",
    # "examples.md",
    # "algorithms.md",
    # "benchmarks.md",
    # "publications.md",
    "api.md",
    "developing.md",
    "license.md",
]
Depth = 1
```

## Citing BAT.jl

When using BAT.jl for research, teaching or similar, please cite
*Schulz et al. "BAT.jl: A Julia-Based Tool for Bayesian Inference", [SNCS (2021)](https://doi.org/10.1007/s42979-021-00626-4)*.

```
@article{Schulz:2021BAT,
  author  = {Schulz, Oliver and Beaujean, Frederik and Caldwell, Allen and Grunwald, Cornelius and Hafych, Vasyl and Kr{\"o}ninger, Kevin and Cagnina, Salvatore La and R{\"o}hrig, Lars and Shtembari, Lolian},
  journal = {SN Computer Science},
  title   = {BAT.jl: A Julia-Based Tool for Bayesian Inference},
  year    = {2021},
  issn    = {2661-8907},
  month   = {Apr},
  number  = {3},
  pages   = {210},
  volume  = {2},
  day     = {12},
  doi     = {10.1007/s42979-021-00626-4},
  url     = {https://doi.org/10.1007/s42979-021-00626-4},
}
```

If you use [`BAT.AHMIntegration`](@ref) as an important part of your work, please also cite 
*Caldwell et al. "Integration with an Adaptive Harmonic Mean Algorithm", [IJMPA (2020)](http://doi.org/10.1142/S0217751X20501420).*

```
@article{Caldwell:2020AHMI,
  author    = {Caldwell, Allen and Eller, Philipp and Hafych, Vasyl and Schick, Rafael and Schulz, Oliver and Szalay, Marco},
  journal   = {International Journal of Modern Physics A},
  title     = {Integration with an adaptive harmonic mean algorithm},
  year      = {2020},
  number    = {24},
  pages     = {2050142},
  volume    = {35},
  doi       = {10.1142/S0217751X20501420},
  publisher = {World Scientific},
}
```

If you use [`BAT.PartitionedSampling`](@ref) (experimental feature) as an important part of your work, please also cite 
*Hafych et al. "Parallelizing MCMC Sampling via Space Partitioning", [arXiv:2008.03098 (2020)](https://arxiv.org/abs/2008.03098)*.

```
@article{Hafych:2008.03098,
  author        = {Hafych, Vasyl and Eller, Philipp and Caldwell, Allen and Schulz, Oliver},
  title         = {Parallelizing MCMC Sampling via Space Partitioning},
  year          = {2018},
  month         = {8},
  archiveprefix = {arXiv},
  eprint        = {2008.03098},
  primaryclass  = {stat.CO},
}
```


## Learning (more about) Julia

BAT.jl supersedes [BAT in C++](https://github.com/bat/bat). If you're considering to switch to BAT.jl, but you're new to Julia and want to learn more about the the language, here are a few resources to get started:

The [Julia website](https://julialang.org/) provides many [links to introductory videos and written tutorials](https://julialang.org/learning/), e.g. ["Intro to Julia"](https://www.youtube.com/watch?v=fMa1qSg_LxA),
[Think Julia: How to Think Like a Computer Scientist](https://benlauwens.github.io/ThinkJulia.jl/latest/book.html)
and ["The Fast Track to Julia"](https://juliadocs.github.io/Julia-Cheat-Sheet/). If you are familar with MATLAB or Python, you may also want to take a look at the ["MATLAB–Python–Julia cheatsheet"](https://cheatsheets.quantecon.org/).

The in-depth article [Why Numba and Cython are not substitutes for Julia](http://www.stochasticlifestyle.com/why-numba-and-cython-are-not-substitutes-for-julia/) explains how Julia addresses several fundamental challenges inherent to scientific high-performance computing.


## Acknowledgements

We acknowledge the contributions from all the BAT.jl users, they help us make BAT.jl a better project. Your help is most welcome!

Development of BAT.jl has been supported by funding from

* [Deutsche Forschungsgemeinschaft (DFG, German Research Foundation)](https://www.dfg.de/)

* European Union Framework Programme for Research and Innovation Horizon 2020 (2014-2020) under the Marie Sklodowska-Curie Grant Agreement No.765710
