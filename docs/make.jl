push!(LOAD_PATH,"../src/")
using BAT
using Documenter

makedocs(
    modules = [BAT],
    clean = true,
    format = :html,
    sitename = "BAT",
    authors = "Oliver Schulz, Frederik Beaujean, and contributors",
    # linkcheck = !("skiplinks" in ARGS),
    pages = ["Home" => "index.md",
             "Manual" => Any[
                 "man/tutorial.md",
                 "man/basics.md",
             ],
             "auto.md",
             ],
    # Use clean URLs, unless built as a "local" build
    html_prettyurls = !("local" in ARGS),
    strict = true,
    checkdocs = :all,
)
