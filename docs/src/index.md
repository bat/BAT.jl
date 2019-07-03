# BAT.jl Documentation

BAT.jl stands for Bayesian Analysis Toolkit in Julia. It is a high high-performance tool box for Bayesian inference with statistical models expressed in a general-purpose programming language, instead of a domain-specific language.

Typical applications for this package are the extraction of the values of the parameters of a model, the comparison of different models in the light of a given data set and the test of the validity of a model to represent the data set at hand. BAT.jl provides access to the full Bayesian posterior distribution to enable parameter estimation, limit setting and uncertainty propagation. BAT.jl also provides supporting functionality like plotting recipes and reporting functions.

BAT.jl is implemented in pure Julia and allows for a flexible definition of mathematical models and applications while enabling the user to code for the performance required for computationally expensive numerical operations. BAT.jl provides implementations (internally and via other Julia packages) of algorithms for sampling, optimization and integration. A few predefined models will be  provided for standard cases such as histogram fitting and simple counting experiments (work in progress), but BAT's main focus is on the analysis of complex custom models. It is designed to enable multi-threaded and distributed (work in progress) code execution at various levels, running multiple MCMC chains in parallel is provided out-of-the-box.

It's possible to use BAT.jl with likelihood functions implemented in languages other than Julia: Julia allows for [calling code in C and Fortran](https://docs.julialang.org/en/v1/manual/calling-c-and-fortran-code/index.html), [C++](https://github.com/JuliaInterop/Cxx.jl), [Python](https://github.com/JuliaPy/PyCall.jl) and [several other languages](https://github.com/JuliaInterop) directly. In addition, BAT.jl provides a very lightweight binary RPC protocol that is easy to implement, to call non-Julia functions written in any language, running in separate processes.

!!! note

    BAT.jl requires Julia v1.2.


## History and Status

BAT.jl is a complete rewrite of [BAT](https://github.com/bat/bat), the Bayesian Analysis Toolkit in C++. BAT.jl is still a work in progress: It already provides several improvements over it's C++ predecessor, but has not yet reached feature parity.


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

When using BAT.jl for work that will result in a scientific publication, please cite

> Caldwell et al., *BAT.jl - A Bayesian Analysis Toolkit in Julia*, [**doi:10.5281/zenodo.2605312**](https://doi.org/10.5281/zenodo.2587213)

The DOI above is [version-independent](http://help.zenodo.org/#versioning), you may want to use the DOI of the specific BAT.jl version used in your work.


## Learning (more about) Julia

BAT.jl is intended to supersede [BAT in C++](https://github.com/bat/bat), long term. If you're considering to switch to BAT.jl but you're new to Julia and want to learn more about the the language, here are a few resources to get started:

The [Julia website](https://julialang.org/) provides many [links to introductory videos and written tutorials](https://julialang.org/learning/), e.g. ["Intro to Julia"](https://www.youtube.com/watch?v=fMa1qSg_LxA),
["A Deep Introduction to Julia for Data Science and Scientific Computing"](http://ucidatascienceinitiative.github.io/IntroToJulia/)
and ["The Fast Track to Julia 1.0"](https://juliadocs.github.io/Julia-Cheat-Sheet/). If you are familar with MATLAB or Python, you may also want to take a look at the ["MATLAB–Python–Julia cheatsheet"](https://cheatsheets.quantecon.org/).

Note: Try to avoid tutorials and books written for older versions of Julia, as there have been quite a few changes to the language in v1.0.

There are also a lot of interesting talks and tutorials on the [Julia YouTube Channel](https://www.youtube.com/user/JuliaLanguage). Have a look at the [talks at JuliaCon 2018](https://www.youtube.com/playlist?list=PLP8iPy9hna6Qsq5_-zrg0NTwqDSDYtfQB) to get an impression on the kinds of scientific applications Julia is being used for and why, e.g. ["Why Julia is the most suitable language for science"](https://youtu.be/7y-ahkUsIrY).

The in-depth article [Why Numba and Cython are not substitutes for Julia](http://www.stochasticlifestyle.com/why-numba-and-cython-are-not-substitutes-for-julia/) explains how Julia addresses several fundamental challenges inherent to scientific high-performance computing.

If you want to get an impression of the attention to detail so typical for Julia, watch ["0.1 vs 1//10: How numbers are compared"](https://youtu.be/CE1x130lYkA).


## Acknowledgements

We acknowledge the contributions from all the BAT.jl users, they help us make BAT.jl a better project. Your help is most welcome!

Development of BAT.jl has been supported by funding from

* [Deutsche Forschungsgemeinschaft (DFG, German Research Foundation)](https://www.dfg.de/)
