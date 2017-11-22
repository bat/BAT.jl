# BAT Documentation

Welcome to BAT, the Bayesian analysis toolkit. This is a (still incomplete) rewrite of the previous [C++ version](https://github.com/bat/bat) in Julia.


## Installation

BAT.jl is under development and not a registered Julia package yet. Install via

```julia
julia> Pkg.clone("https://github.com/oschulz/MultiThreadingTools.jl.git")
julia> Pkg.clone("https://github.com/bat/BAT.jl.git")
```


## Developer Instructions

When changing the code of BAT.jl and testing snippets and examples in the REPL, automatic code reloading comes in very handy. Try out [Revise.jl](https://github.com/timholy/Revise.jl):

```julia
julia> Pkg.add("Revise")
julia> using Revise
julia> using BAT
```

Note: It's essential to load `Revise` *before* `BAT`. `using Revise` must be done within the REPL (or via ".juliarc.jl" [in a special way](https://github.com/timholy/Revise.jl#using-revise-by-default)). Putting `using Revise` in a Julia script will not work.


## Manual Outline

```@contents
Pages = [
    "man/tutorial.md",
    "man/basics.md",
]
Depth = 1
```
