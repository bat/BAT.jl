using Pkg, TOML

# Remove this bootstrap when the upgrade dependencies are registered.
const ROOT = dirname(@__DIR__)
const OPTIONAL = joinpath(ROOT, "test", "optional")
const ADAPTER = TOML.parsefile(joinpath(OPTIONAL, "importancesamplers", "Project.toml"))
const DEV_DEPS = ("HeterogeneousComputing", "MeasureBase", "ValueShapes")

function prepare_sources(env)
    path = joinpath(env, "Project.toml")
    project = TOML.parsefile(path)
    deps = project["deps"]
    sources = get!(project, "sources", Dict{String,Any}())
    sources["BAT"] = Dict("path" => relpath(ROOT, env))
    for name in DEV_DEPS
        deps[name] = ADAPTER["deps"][name]
        sources[name] = ADAPTER["sources"][name]
    end
    if haskey(deps, "MGVI")
        sources["MGVI"] = TOML.parsefile(joinpath(OPTIONAL, "mgvi", "Project.toml"))["sources"]["MGVI"]
    end
    open(path, "w") do io
        TOML.print(io, project; sorted = true)
    end
end

withenv("JULIA_PKG_PRECOMPILE_AUTO" => "0") do
    env = isempty(ARGS) ? ROOT : joinpath(ROOT, only(ARGS))
    if env != ROOT && VERSION >= v"1.11"
        # Docs and optional projects can need different registered versions.
        prepare_sources(env)
        Pkg.activate(env)
        Pkg.instantiate()
    else
        Pkg.activate(ROOT)
        Pkg.add([
            PackageSpec(; url = ADAPTER["sources"][name]["url"], rev = ADAPTER["sources"][name]["rev"])
            for name in DEV_DEPS
        ])
        if env != ROOT
            # Julia 1.10 does not read [sources].
            cp(joinpath(ROOT, "Manifest.toml"), joinpath(env, "Manifest.toml"))
            Pkg.activate(env)
            Pkg.develop(; path = ROOT)
        end
    end
end
