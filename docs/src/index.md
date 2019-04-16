# BAT.jl Documentation

BAT.jl is the Julia version of the Bayesian Analysis Toolkit. It is designed to help solve statistical problems encountered in Bayesian inference. Typical examples are the extraction of the values of the free parameters of a model, the comparison of different models in the light of a given data set, and the test of the validity of a model to represent the data set at hand. BAT.jl aims to provide multiple algorithms that give access to the full Bayesian posterior distribution, to enable parameter estimation, limit setting and uncertainty propagation. BAT.jl also provides supporting functionality like plotting recipes and reporting functions.

This package is a complete rewrite of the previous [C++-BAT](https://github.com/bat/bat) in Julia. BAT.jl provides several improvements over it's C++ predecessor, but has not yet reached feature parity in all areas. There is no backward compatibility, but the spirit is the same: providing a tool for Bayesian computations of complex models that require application-specific code.

BAT.jl is implemented in pure Julia and allows for a flexible definition of mathematical models and applications while enabling the user to code for the performance required for computationally expensive numerical operations. BAT.jl provides implementations (internally of via other Julia packages) of algorithms for sampling, optimization and integration. While predefined models are (resp. will soon be) provided for standard cases, such as simple counting experiments, binomial problems or Gaussian models, BAT's main strength lies in the analysis of complex models. The package is designed to enable multi-threaded and distributed code execution at various levels, multi-threaded MCMC chains are provided out-of-the-box.

In addition to likelihood functions implemented in Julia, BAT.jl provides a lightweight binary protocol to connect to functions written in other languages and running in separate processes (code for likelihoods written in C++ is included).

Table of contents:

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

BAT.jl is intended replace [C++-BAT](https://github.com/bat/bat), long term. If you're new to Julia and want to learn more about the the language, here are a few resources to get started:

The [Julia website](https://julialang.org/) provides many [links to introductory videos and written tutorials](https://julialang.org/learning/), e.g. ["Intro to Julia"](https://www.youtube.com/watch?v=fMa1qSg_LxA),
["A Deep Introduction to Julia for Data Science and Scientific Computing"](http://ucidatascienceinitiative.github.io/IntroToJulia/)
and ["The Fast Track to Julia 1.0"](https://juliadocs.github.io/Julia-Cheat-Sheet/)

Note: Try to avoid tutorials and books written for Julia v0.6 as there have been quite a few changes to the language in v1.0.

There are also a lot of interesting talk and tutorials on the [Julia YouTube Channel](https://www.youtube.com/user/JuliaLanguage). Have a look at the [talks at JuliaCon 2018](https://www.youtube.com/playlist?list=PLP8iPy9hna6Qsq5_-zrg0NTwqDSDYtfQB) to get an impression on the kinds of scientific applications Julia is being used for and why, e.g. ["Why Julia is the most suitable language for science"](https://youtu.be/7y-ahkUsIrY).

The in-depth article [Why Numba and Cython are not substitutes for Julia](http://www.stochasticlifestyle.com/why-numba-and-cython-are-not-substitutes-for-julia/) explains how Julia addresses several fundamental challenges inherent to scientific computing.

If you want to get an impression of the attention to detail so typical for Julia, watch ["0.1 vs 1//10: How numbers are compared"](https://youtu.be/CE1x130lYkA).


## Acknowledgements

We acknowledge the contributions from all the BAT.jl users who help us make BAT.jl a better project. Your help is very welcome!

Development of BAT.jl has been supported by funding from

* [Deutsche Forschungsgemeinschaft (DFG, German Research Foundation)](https://www.dfg.de/)
