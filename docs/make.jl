using Documenter, NumCompressibleFlows

makedocs(;
    sitename = "NumCompressibleFlows.jl",
    modules = [NumCompressibleFlows],
    clean = true,
    doctest = false,
    checkdocs = :exports,
    authors = "M. Zainelabdeen, Ch. Merdon",
    repo = Remotes.GitHub("marwaz93", "NumCompressibleFlows"),
    format = Documenter.HTML(;
        mathengine = MathJax3(),
    ),
    pages = [
        "Home" => "index.md",
        "Problem and scheme" => "problem.md",
        "Benchmark problems" => "benchmarks.md",
        "Solver and study pipeline" => "pipeline.md",
        "API" => "api.md",
    ],
)

deploydocs(;
    repo = "github.com/marwaz93/NumCompressibleFlows.jl.git",
)
