using Documenter, AlphaStable

makedocs(;
    modules=[AlphaStable],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/baggepinnen/AlphaStable.jl/blob/{commit}{path}#L{line}",
    sitename="AlphaStable.jl",
    authors="Fredrik Bagge Carlson",
    assets=String[],
)

deploydocs(;
    repo="github.com/baggepinnen/AlphaStable.jl",
)
