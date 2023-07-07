# Installation

BAT.jl is written in the [Julia programming language](https://julialang.org/). To use BAT.jl, you will need to install Julia itself, the BAT.jl Julia package, and some additional Julia packages. Depending on your use case, you may also want to need a Python installation with certain Python packages (optional).

Table of contents:

```@contents
Pages = ["installation.md"]
Depth = 3
```

## Installing Julia

Julia is available for Linux, OS-X and Windows, and easy to install:

* [Download Julia](https://julialang.org/downloads/).

* Extract the archive (Linux), resp. drag Julia into Applications (OS-X) or run the installer (Windows).

* You may want to add the Julia `bin` directory to your `$PATH`. To get the location of the Julia `bin` directory on OS-X or Windows, start a Julia session (via applications menu) and run the Julia command `Sys.BINDIR`.

!!! note

    BAT.jl requires Julia >= v1.6, we strongly recommend to use the latest Julia version for optimal performance.


## Installing BAT.jl and related Julia packages

BAT.jl is provided as a registered Julia package. To install it, simply run

```julia
julia> using Pkg
julia> pkg"add BAT"
```

However, you will likely need other Julia packages too. We recommend that you install certain statistics, plotting, I/O and array packages as well:

```
julia> using Pkg
julia> pkg"add BAT ArraysOfArrays Distributions ElasticArrays IntervalSets Parameters Plots StatsBase Tables TypedTables"
```

In addition, these packages will need to be installed and loaded (`using PackageName` or `import PackageName`) to enable some optional BAT algorithms/functionalities:

```
julia> pkg"add AutoDiffOperators AdvancedHMC Cuba Folds HDF5 NestedSamplers Optim"
```

To install the latest development version of BAT (main branch) instead of the latest stable release, use

```julia
julia> pkg"add BAT#main"
```


## Installing Visual Studio Code and Jupyter (Optional)

Please download and install [the lastest Julia release](https://julialang.org/downloads/).

You may also want to install [Visual Studio Code](https://code.visualstudio.com/download) with the [VS-Code Julia extension](https://code.visualstudio.com/docs/languages/julia) and/or a have a working Jupyter installation. [JupyterLab Desktop](https://github.com/jupyterlab/jupyterlab-desktop/releases) is easy to install (but a full Anaconda or custom Python installation with Jupyter will work too, of course). For details regarding Julia and Jupyter, see the [IJulia.jl](https://github.com/JuliaLang/IJulia.jl#installation) documentation.


#### IJulia (Jupyter Julia kernel)

To use the the Julia Jupyter kernel, you need to add the package ["IJulia.jl"](https://github.com/JuliaLang/IJulia.jl):

On Linux, simply use

```
pkg"add IJulia"
```

On OS-X, if you have an existing Jupyter installation (e.g. via Anaconda) and would like Julia to use it (instead of an internal Conda installation, see above), use (e.g.)

```
ENV["JUPYTER"] = "$(homedir())/opt/anaconda3/bin/jupyter"; pkg"add IJulia"
```

On Windows, if would like Julia to use an existing Jupyter installation (see above), use something like

```
ENV["JUPYTER"] = "DRIVE:/path/to/your/anaconda/.../jupyter.exe"; pkg"add IJulia"
```

Julia will remember the chosen Jupyter installation permanently, `ENV["JUPYTER"]` only needs to be set the first time you run `pkg"add IJulia"`.


## Environment variables

You may want/need to set the following environment variables:

* `$PATH`: Include the Julia `bin`-directory in your binary search path, see above. On OS-X and Windows, Visual Studio Code should detect the path your Julia binary automatically, if installed in the default location.

* [`$JULIA_NUM_THREADS`](https://docs.julialang.org/en/v1/manual/environment-variables/#JULIA_NUM_THREADS-1): Number of threads to use for Julia multi-threading

* [`$JULIA_DEPOT_PATH`](https://julialang.github.io/Pkg.jl/v1/glossary/) and [`JULIA_PKG_DEVDIR`](https://julialang.github.io/Pkg.jl/v1/managing-packages/#Developing-packages-1): If you want Julia to install packages in another location than `$HOME/.julia`.

See the Julia manual for a description of [other Julia-specific environment variables](https://docs.julialang.org/en/v1/manual/environment-variables/).


## Additional customization options

Note: If you want Julia to install packages in another location than `$HOME/.julia`, set the environment variables [`JULIA_DEPOT_PATH`](https://julialang.github.io/Pkg.jl/v1/glossary/) and [`JULIA_PKG_DEVDIR`](https://julialang.github.io/Pkg.jl/v1/managing-packages/#Developing-packages-1) (see above).
