# BAT.jl

[![Documentation for stable version](https://img.shields.io/badge/docs-stable-blue.svg)](https://bat.github.io/BAT.jl/stable)
[![Documentation for development version](https://img.shields.io/badge/docs-dev-blue.svg)](https://bat.github.io/BAT.jl/dev)
[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md)
[![Build Status](https://github.com/bat/BAT.jl/workflows/CI/badge.svg?branch=master)](https://github.com/bat/BAT.jl/actions?query=workflow%3ACI)
[![Codecov](https://codecov.io/gh/bat/BAT.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/bat/BAT.jl)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2587213.svg)](https://doi.org/10.5281/zenodo.2587213)

Welcome to BAT, the Bayesian analysis toolkit. This is a rewrite of the
previous [C++-BAT](https://github.com/bat/bat) in Julia. BAT.jl provides
several improvements over it's C++ predecessor, but has not reached feature
parity yet in some areas.

BAT.jl currently includes:

* Metropolis-Hastings MCMC sampling
* Adaptive Harmonic Mean Integration ([AHMI](https://arxiv.org/abs/1808.08051))
* Plotting recipes for MCMC samples and statistics

Additional sampling algorithms and other features are in preparation.


## Installation

To install BAT.jl, start Julia and run

```julia
julia> using Pkg
julia> pkg"add BAT"
```

!!! note

    BAT.jl requires Julia >= v1.3.


## Documentation

* [Documentation for stable version](https://bat.github.io/BAT.jl/stable)
* [Documentation for development version](https://bat.github.io/BAT.jl/dev)


## Citing BAT.jl

When using BAT.jl for research, teaching or similar, please cite our work, see [CITATION.bib](CITATION.bib).
