using BoostingMVN
using Documenter

DocMeta.setdocmeta!(BoostingMVN, :DocTestSetup, :(using BoostingMVN); recursive=true)

makedocs(;
    modules=[BoostingMVN],
    authors="Daniel Ward",
    repo="https://github.com/danielward27/BoostingMVN.jl/blob/{commit}{path}#{line}",
    sitename="BoostingMVN.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://danielward27.github.io/BoostingMVN.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/danielward27/BoostingMVN.jl",
    devbranch="main",
)
