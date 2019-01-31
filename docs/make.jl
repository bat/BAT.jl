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
    format = :html,
    authors = "Oliver Schulz, Frederik Beaujean, and contributors",
    pages=[
        "Home" => "index.md",
        "Tutorial" => "tutorial.md",
        "Basics" => "basics.md",
        "API" => "api.md",
        "LICENSE" => "LICENSE.md",
    ],
    doctest = ("fixdoctests" in ARGS) ? :fix : true,
    html_prettyurls = !("local" in ARGS),
    html_canonical = "https://bat.github.io/BAT.jl/stable/",
)

deploydocs(
    repo = "github.com/bat/BAT.jl.git"
)
