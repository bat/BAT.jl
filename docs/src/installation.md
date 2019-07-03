# Installation

BAT.jl is written in the [Julia programming language](https://julialang.org/). To use BAT.jl, you will need to install Julia itself, the BAT.jl Julia package, and some additional Julia packages. Depending on your use case, you may also want to need a Python installation with certain Python packages (optional).

Table of contents:

```@contents
Pages = ["installation.md"]
Depth = 3
```

## Prerequisites

### OS-Specific Prerequisites

#### Mac OS-X

TODO:

On OS-X, you need to install the Apple command line developer tools. They are required by the Julia package Homebrew.jl, which in turn is required by lot's of other Julia packages.


### Installing Julia

Julia is available for Linux, OS-X and Windows, and easy to install:

* [Download Julia](https://julialang.org/downloads/).

* Extract the archive, resp. run the installer.

* You'll probably want to add the Julia `bin` directory to your `$PATH`

!!! note

    BAT.jl requires Julia v1.2.


### Installing Jupyter and matplotlib/pyplot (Optional)

If you plan to use [Jupyter](https://jupyter.org/) notebooks and/or the [matplotlib/pyplot backend](http://docs.juliaplots.org/latest/backends/) of the Julia [Plots.jl](http://docs.juliaplots.org/) package, you will also need a Python installation and certain Python packages (see below). BAT.jl is fully usable without Jupyter and matplotlib/pyplot, but both can come in very handy.

Julia can either use existing installations of Jupyter and pyplot , or install both internally by creating an internal Conda installation within `$HOME/.julia/conda`. We recommend the first approach (especially using Anaconda), since Julia will otherwise have to download over 1 GB of software, the `$HOME/.julia` directory will grow very large, and require you to start Jupyter in an indirect fashion via Julia (only to have Jupyter then start additional Julia instances as Jupyter kernels in return).

For details, see the [IJulia.jl](https://github.com/JuliaLang/IJulia.jl#installation), [PyCall.jl](https://github.com/JuliaPy/PyCall.jl#specifying-the-python-version) and [PyPlot.jl](https://github.com/JuliaPy/PyPlot.jl) documentation (you should not need to if you follow the steps below).

On Linux, Julia (more specifically the Julia packages [IJulia.jl](https://github.com/JuliaLang/IJulia.jl), [PyCall.jl](https://github.com/JuliaPy/PyCall.jl), and [PyPlot.jl](https://github.com/JuliaPy/PyPlot.jl)) will by default try to use the matplotlib/pyplot installation associated with the `python3` (resp. `python`) executable on your `$PATH`. Likewise, Julia will by default try to use the Jupyter installation associated with the `jupyter` executable on your `$PATH`.

However, on OS-X and Windows, both IJulia.jl and PyCall.jl by default always create a Julia-internal Conda installation (see above), even if Jupyter and matplotlib/pyplot are available (apparently broken Jupyter/Python installations on these platforms caused frequent support requests).  In contrast to this default behavior, we recommend to use a standalone Jupyter and Python installation on all OS platforms. Set the environment variables [`$JUPYTER`](https://github.com/JuliaLang/IJulia.jl#installation) and [`$PYTHON`](https://github.com/JuliaPy/PyCall.jl#specifying-the-python-version) to point to your Jupyter and Python executable to force Julia to use the existing installation.

We recommend that you install the [Anaconda](https://www.anaconda.com/) Python distribution, it includes both Jupyter and pyplot (it is of course possible to use non-Anaconda Jupyter and pyplot installations instead).


#### Installing Anaconda (Optional)

To install Anaconda

* [Download Anaconda](https://www.anaconda.com/distribution/).

* Run the installer

* Set the environment variables [`$JUPYTER`](https://github.com/JuliaLang/IJulia.jl#installation) and [`$PYTHON`](https://github.com/JuliaPy/PyCall.jl#specifying-the-python-version) to the full path of the Jupyter and Python executables (see above).


## Environment Variables

You may want/need to set the following environment variables:

* `$PATH`: Include the Julia `bin`-directory in your binary search path, see above.
If you intend to use Jupyter, you will probably want to include the directory containing the `jupyter` binary to your `PATH` as well.

* [`$JUPYTER`](https://github.com/JuliaLang/IJulia.jl#installation): See above, only when using Jupyter

* [`$PYTHON`](https://github.com/JuliaPy/PyCall.jl#specifying-the-python-version): See above, only when using the Plots.jl matplotlib/pyplot backend.

* [`$JULIA_NUM_THREADS`](https://docs.julialang.org/en/v1/manual/environment-variables/#JULIA_NUM_THREADS-1): Number of threads to use for Julia multi-threading

* [`$JULIA_DEPOT_PATH`](https://julialang.github.io/Pkg.jl/v1/glossary/) and [`JULIA_PKG_DEVDIR`](https://julialang.github.io/Pkg.jl/v1/managing-packages/#Developing-packages-1): If you want Julia to install packages in another location than `$HOME/.julia`.

See the Julia manual for a description of [other Julia-specific environment variables](https://docs.julialang.org/en/v1/manual/environment-variables/).


## Installing BAT.jl and Other Julia Packages

BAT.jl is provided as a registered Julia package. To install it, simply run

```julia
julia> using Pkg
julia> pkg"add BAT"
```

However, you will likely need other Julia packages too. We recommend that you install certain statistics, plotting, I/O and array packages as well:

```
julia> using Pkg
julia> pkg"add BAT ArraysOfArrays Distributions ElasticArrays HDF5 IntervalSets Plots ShapesOfVariables StatsBase Tables TypedTables"
```

To install the latest development version of BAT (master branch), instead of the latest stable release, use

```julia
julia> pkg"add BAT#master"
```

TODO: PRECOMPILE


To use the the Jupyter Julia kernel, you also need to install the package "IJulia":

```
pkg"add IJulia"
```

To use the Plots.jl matplotlib/pyplot backend (see above), you also need to add the package "PyPlot":

```
pkg"add PyPlot"
```

Depending on your application, you may want to install [additional Julia packages](https://juliaobserver.com/).

Note: If you want Julia to install packages in another location than `$HOME/.julia`, set the environment variables [`JULIA_DEPOT_PATH`](https://julialang.github.io/Pkg.jl/v1/glossary/) and [`JULIA_PKG_DEVDIR`](https://julialang.github.io/Pkg.jl/v1/managing-packages/#Developing-packages-1) (see above).
