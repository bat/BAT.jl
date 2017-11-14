# BAT Documentation

Welcome to BAT, the Bayesian analysis toolkit. This is a complete rewrite in julia
of the previous [C++ version](https://github.com/bat/bat).

## Developer Installation instructions

Clone and switch to the dev branch. Fire up julia >=0.6 Then create a link to the repository inside the julia package directory

    julia> cd(Pkg.dir())
    shell> ln -s ../../workspace/BAT.jl BAT

Now install the requirements

    julia> Pkg.clone("https://github.com/oschulz/MultiThreadingTools.jl.git")
    julia> Pkg.checkout("MultiThreadingTools", "dev")

When working on the code and testing snippets in the REPL, it's easiest if BAT is automatically reloaded. The `Revise` package does this but not if structs or their constructor signatures are changed, which requires restarting julia.

    julia> Pkg.add("Revise")
    julia> using Revise

Now start working

    julia> using BAT


## Manual Outline

```@contents
Pages = [
    "man/tutorial.md",
    "man/basics.md",
]
Depth = 1
```

## Library Reference
