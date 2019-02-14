# BAT Documentation

BAT.jl , the julia version of the Bayesian Analysis Toolkit, is a software package which is designed to help solve statistical problems encountered in Bayesian inference. Typical examples are the extraction of the values of the free parameters of a model, the comparison of different models in the light of a given data set, and the test of the validity of a model to represent the data set at hand. BAT.jl is based on Bayes’ Theorem and it is realized with the use of different algorithms. These give access to the full posterior probability distribution, and they enable parameter estimation, limit setting and uncertainty propagation.

BAT.jl is implemented in julia and allows for a flexible definition of mathematical models and applications while keeping in mind the reliability and speed requirements of the numerical operations. It provides implementations (or links to implementations) of algorithms for sampling, optimization and integration. While predefined models exist for standard cases, such as simple counting experiments, binomial problems or Gaussian models, its full strength lies in the analysis of complex and high-dimensional models.

BAT.jl is a completely re-written code based on the original [BAT](https://github.com/bat/bat) code written in C++. There is no backward compatibility whatsoever, but the spirit is the same: providing a tool for Bayesian computations of complex models.



## Code
TODO: Link to the code here...



## Getting started

### Prerequisites

TODO: write up prerequisites.

### Installation

BAT.jl is under development and not a registered Julia package yet. Install via

```julia
using Pkg
pkg"add https://github.com/BAT/BAT.jl.git"
```

### The How-to-get-started instructions

TODO: Write instructions.

### Developer Instructions

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

#### Add more scientific models



## Benchmark and performance tests



## Contact

### The BAT core developer team

### The BAT.jl mailing list



### Acknowledgements
We acknowledge the contributions from the various BAT.jl user who help us make BAT.jl a better project. Your help is very welcome!

The development of BAT.jl is funded by the German Research Foundation (DFG).
