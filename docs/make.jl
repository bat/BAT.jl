# Use
#
#     DOCUMENTER_DEBUG=true julia --color=yes make.jl local [nonstrict] [fixdoctests]
#
# for local builds.

using Documenter
using Literate
using BAT

using ValueShapes
import HDF5


function fix_literate_output(content)
    content = replace(content, "EditURL = \"@__REPO_ROOT_URL__/\"" => "")
    return content
end

gen_content_dir = joinpath(@__DIR__, "src")
tutorial_src = joinpath(@__DIR__, "src", "tutorial_lit.jl")
Literate.markdown(tutorial_src, gen_content_dir, name = "tutorial", documenter = true, credit = true, postprocess = fix_literate_output)
#Literate.markdown(tutorial_src, gen_content_dir, name = "tutorial", codefence = "```@repl tutorial" => "```", documenter = true, credit = true)
Literate.notebook(tutorial_src, gen_content_dir, execute = false, name = "bat_tutorial", documenter = true, credit = true)
Literate.script(tutorial_src, gen_content_dir, keep_comments = false, name = "bat_tutorial", documenter = true, credit = false)

makedocs(
    sitename = "BAT",
    modules = [BAT],
    format = Documenter.HTML(
        prettyurls = !("local" in ARGS),
        canonical = "https://bat.github.io/BAT.jl/stable/"
    ),
    authors = "The BAT development team",
    pages=[
        "Home" => "index.md",
        "Installation" => "installation.md",
        "Tutorial" => "tutorial.md",
        "API Documentation" => "stable_api.md",
        "Plotting" => "plotting.md",
        "Experimental Features" => "experimental_api.md",
        "Internal API" => "internal_api.md",
        "Developer instructions" => "developing.md",
        "License" => "license.md",
    ],
    doctest = ("fixdoctests" in ARGS) ? :fix : true,
    linkcheck = ("linkcheck" in ARGS),
    strict = !("nonstrict" in ARGS),
)

deploydocs(
    repo = "github.com/bat/BAT.jl.git",
    forcepush = true,
    push_preview = true,
)
