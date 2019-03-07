# BAT.jl

[![Documentation for stable version](https://img.shields.io/badge/docs-stable-blue.svg)](https://bat.github.io/BAT.jl/stable)
[![Documentation for development version](https://img.shields.io/badge/docs-dev-blue.svg)](https://bat.github.io/BAT.jl/dev)
[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md)
[![Travis Build Status](https://travis-ci.com/bat/BAT.jl.svg?branch=master)](https://travis-ci.com/bat/BAT.jl)
[![Appveyor Build Status](https://ci.appveyor.com/api/projects/status/github/bat/BAT.jl?branch=master&svg=true)](https://ci.appveyor.com/project/bat/BAT-jl)
[![Codecov](https://codecov.io/gh/bat/BAT.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/bat/BAT.jl)

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


## Documentation

* [Documentation for stable version](https://bat.github.io/BAT.jl/stable)
* [Documentation for development version](https://bat.github.io/BAT.jl/dev)
