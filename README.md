<h1> <img style="height:5em;" alt="BAT.jl" src="docs/src/assets/logo.svg"/> </h1> 

[![Documentation for stable version](https://img.shields.io/badge/docs-stable-blue.svg)](https://bat.github.io/BAT.jl/stable)
[![Documentation for development version](https://img.shields.io/badge/docs-dev-blue.svg)](https://bat.github.io/BAT.jl/dev)
[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md)
[![Build Status](https://github.com/bat/BAT.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/bat/BAT.jl/actions?query=workflow%3ACI)
[![Codecov](https://codecov.io/gh/bat/BAT.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/bat/BAT.jl)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2587213.svg)](https://doi.org/10.5281/zenodo.2587213)

Welcome to BAT, a Bayesian analysis toolkit in Julia.

BAT.jl offers a variety of posterior sampling, mode estimation and integration algorithms, supplemented by plotting recipes and I/O functionality.

BAT.jl originated as a rewrite/redesign of [BAT](https://github.com/bat/bat), the Bayesian Analysis Toolkit in C++. BAT.jl now offer a different set of functionality and a wider variety of algorithms than its C++ predecessor.


## Installation

To install BAT.jl, start Julia and run

```julia
julia> using Pkg
julia> pkg"add BAT"
```

Note: BAT.jl requires Julia >= v1.10, we recommend to use
[the latest stable Julia version](https://julialang.org/downloads/)
for optimal performance.


## Documentation

* [Documentation for stable version](https://bat.github.io/BAT.jl/stable)
* [Documentation for development version](https://bat.github.io/BAT.jl/dev)


## Citing BAT.jl

When using BAT.jl for research, teaching or similar, please cite our work, see [CITATION.bib](CITATION.bib).
