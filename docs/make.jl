using BoostedCDE
using Documenter

DocMeta.setdocmeta!(BoostedCDE, :DocTestSetup, :(using BoostedCDE); recursive=true)

makedocs(;
    modules=[BoostedCDE],
    authors="Daniel Ward",
    repo="https://github.com/danielward27/BoostedCDE.jl/blob/{commit}{path}#{line}",
    sitename="BoostedCDE.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://danielward27.github.io/BoostedCDE.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/danielward27/BoostedCDE.jl",
    devbranch="main",
)
