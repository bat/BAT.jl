import Pkg

@assert !isempty(ARGS)
sysimgprj = first(ARGS)
@assert !isempty(sysimgprj)
@assert !ispath(sysimgprj)

Pkg.activate(sysimgprj)

#if isfile("src/BAT.jl")
#    Pkg.develop(Pkg.PackageSpec(path=pwd()))
#else
    Pkg.add(name="BAT", rev="master")
#end

pkgs = [
    "ArraysOfArrays",
    "Cuba",
    "Distributions",
    "ElasticArrays",
    "EmpiricalDistributions",
    "FileIO",
    "ForwardDiff",
    "HDF5",
    "IntervalSets",
    "JLD2",
    "Measurements",
    "ParallelProcessingTools",
    "Parameters",
    "Plots",
    "PDMats",
    "Random123",
    "RecipesBase",
    "SpecialFunctions",
    "StableRNGs",
    "StatsBase",
    "Tables",
    "TypedTables",
    "ValueShapes",
    "Zeros",
    "Zygote",
]

Pkg.add(pkgs)

Pkg.add("PackageCompiler")

Pkg.instantiate()
Pkg.precompile()


import PackageCompiler, Libdl

import BAT
bat_path = dirname(dirname(pathof(BAT)))

custom_sysimg = joinpath(sysimgprj, "JuliaSysimage." * Libdl.dlext)

PackageCompiler.create_sysimage(
    Symbol.(vcat(["BAT"], pkgs)),
    sysimage_path = custom_sysimg,
    precompile_execution_file = [
        joinpath(bat_path, "test", "runtests.jl"),
        # joinpath(bat_path, "docs", "src", "tutorial_lit.jl"),  # Causes trouble
    ],
    cpu_target = PackageCompiler.default_app_cpu_target()
)


import Markdown

default_sysimg = abspath(Sys.BINDIR, "..", "lib", "julia", "sys." * Libdl.dlext)

show(Markdown.parse("
BAT Julia system image created.

Default Julia system image is \"$default_sysimg\", to use BAT Julia system image, run

```shell
julia --project=\"$sysimgprj\" --sysimage=\"$custom_sysimg\"
```

Run

```julia
julia> import IJulia
julia> IJulia.installkernel(\"BAT Julia\", \"--project=$sysimgprj\", \"--sysimage=$custom_sysimg\")
```

to install a Jupyter kernel that will use the BAT Julia system image.
"))
