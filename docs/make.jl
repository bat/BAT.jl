# Use
#
#     DOCUMENTER_DEBUG=true julia --color=yes make.jl local [fixdoctests]
#
# for local builds.

using Documenter
using BAT

makedocs(
    sitename = "BAT",
    modules = [BAT],
    format = Documenter.HTML(
        prettyurls = !("local" in ARGS),
        canonical = "https://bat.github.io/BAT.jl/stable/"
    ),
    authors = "Oliver Schulz, Frederik Beaujean, and contributors",
    pages=[
        "Home" => "index.md",
        "Tutorial" => "tutorial.md",
        "Basics" => "basics.md",
        "API" => "api.md",
        "LICENSE" => "LICENSE.md",
    ],
    doctest = ("fixdoctests" in ARGS) ? :fix : true,
)

deploydocs(
    repo = "github.com/bat/BAT.jl.git",
    forcepush = true
)
