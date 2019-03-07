# BAT Documentation

BAT.jl is the Julia version of the Bayesian Analysis Toolkit. It is designed to help solve statistical problems encountered in Bayesian inference. Typical examples are the extraction of the values of the free parameters of a model, the comparison of different models in the light of a given data set, and the test of the validity of a model to represent the data set at hand. BAT.jl aims to provide multiple algorithms that give access to the full Bayesian posterior distribution, to enable parameter estimation, limit setting and uncertainty propagation. BAT.jl also provides supporting functionality like plotting recipes and reporting functions.

This package is a complete rewrite of the previous [C++-BAT](https://github.com/bat/bat) in Julia. BAT.jl provides several improvements over it's C++ predecessor, but has not yet reached feature parity in all areas. There is no backward compatibility, but the spirit is the same: providing a tool for Bayesian computations of complex models that require application-specific code.

BAT.jl is implemented in pure Julia and allows for a flexible definition of mathematical models and applications while enabling the user to code for the performance required for computationally expensive numerical operations. BAT.jl provides implementations (internally of via other Julia packages) of algorithms for sampling, optimization and integration. While predefined models are (resp. will soon be) provided for standard cases, such as simple counting experiments, binomial problems or Gaussian models, BAT's main strength lies in the analysis of complex models. The package is designed to enable multi-threaded and distributed code execution at various levels, multi-threaded MCMC chains are provided out-of-the-box.

In addition to likelihood functions implemented in Julia, BAT.jl provides a lightweight binary protocol to connect to functions written in other languages and running in separate processes (code for likelihoods written in C++ is included).


## Getting started

### Prerequisites

TODO: ...

### How-to-get-started / Tutorial

TODO: ...


### Developer Instructions

To generate and view a local version of the documentation, run

```shell
cd docs
julia make.jl local
```

then open "docs/build/index.html" in your browser.

When changing the code of BAT.jl and testing snippets and examples in the REPL, automatic code reloading comes in very handy. Try out [Revise.jl](https://github.com/timholy/Revise.jl).


## Documentation

### User guide

### FAQ

### Publications and talks

### How to cite BAT.jl

### LICENSE


## Algorithms

### Sampling algorithms (and interfaces)

TODO: List algorithms and short short descriptions.

### Integration algorithms (and interfaces)

TODO: List algorithms and short short descriptions.

### Optimization algorithms (and interfaces)
TODO: List algorithms and short short descriptions.

### Other algorithms (and interfaces)



## Interfaces

TODO: List interfaces



## Examples

### Common models and problems

#### The 1-D Gaussian model

#### The Poisson problem (counting experiments)

#### The binomial case

### Add more models and problems

### Published scientific examples 

#### A multivariate Gaussian combination model (similar to BLUE)

#### The EFTfitter



## Benchmarks and performance tests



### Acknowledgements

We acknowledge the contributions from all the BAT.jl users who help us make BAT.jl a better project. Your help is very welcome!

Development of BAT.jl has been supported by funding from

* [Deutsche Forschungsgemeinschaft (DFG, German Research Foundation)](http://www.dfg.de/)
