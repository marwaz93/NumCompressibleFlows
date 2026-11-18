# Project description

## Aim
NumCompressibleFlows is a research code for finite element discretizations of stationary
compressible flow model problems (compressible Stokes/Navier-Stokes type equations with
manufactured solutions). It builds on the ExtendableFEM/ExtendableGrids ecosystem.
The lowest order schem employs a Bernardi--Raugel velocity, P0 pressure/density
and optional RT/BDM flux reconstruction for gradient-robustness. There are several
manufactured-solution examples (p7 vortex, rigid body rotation, Ethier-Steinman,
Taylor-Green, ...), a fixed-point/pseudo-time solver, error calculation in a postprocessing step
and convergence studies organized by DrWatson.

## Layout
- `src/` — the NumCompressibleFlows.jl package: `problem_definitions.jl` (exact solutions,
  densities, equations of state, grids), `kernels.jl` (FE kernels), `compressible_stokes.jl`
  (solver, `default_args`/`load_data`/`run_single`), `postprocessing.jl`, `plotting.jl`
- `scripts/` — one standalone study driver per example (`rigidbodyrotation.jl`,
  `p7vortex_powerlaw.jl`, ...) sharing the DrWatson layer `study_pipeline.jl`, plus
  `run.jl` as command line entry point for batch jobs
- `test/` — package tests (`run_tests`) based on manufactured fixed-point solutions
- `data/`, `plots/` — DrWatson-generated data (`.jld2`) and plot files of the studies,
  cached per study folder; do not hand-edit or commit them casually
- batch workflow: `create_parameters.sh`, `study.dag`, `julia_job.sub`, `run_julia.sh`
  (HTCondor); only use when the maintainer asks for cluster runs
- `ExtendableFEMBase` is a local path dependency (`../dev/ExtendableFEMBase`); read its
  source there when FEM assembly details matter
- `ExtendableFEM` is a local path dependency (`../dev/ExtendableFEM`); read its
  source there when FEM assembly details matter
- `ExtendableGrids` is a local path dependency (`../dev/ExtendableGrids`); read its
  source there when FEM assembly details matter

# General rules
- Keep maintaining a changelog following  [Keep a Changelog](https://keepachangelog.com/) with  ISO 8601 dates
- Please don't commit to the current branch - this should be done only by a human. 
  If you must, ask the maintainer for it or create a new branch in a temporary repo. E.g. if you want to make pull requests.
- Please sign pull requests with your model name.
- Never make symlinks

# Julia rules

## Julia code organization
- Organize this project as a Julia package. All common code should go into the common package code in src
- Adhere to the DRY (don't repeat yourself) principle. If code patterns repeat, put them into functions and move
  them to a common place (package src). 
- Study scripts in `scripts/` only configure studies (problem data, defaults, plotting);
  reusable solver/kernel/plotting logic belongs into `src/` (e.g. via `default_args` in
  `compressible_stokes.jl` or the sweep machinery of `setup_pipeline!`).

## Julia formatting
- Use runic formatting. Runic is installed as julia app
- Invoke the standalone `runic` binary on PATH (or `julia -m Runic`)
- Use `runic -i file.jl` to format in place, `runic -c file.jl` to just check
  (nonzero exit if unformatted), `runic -d` to show a diff

## Julia execution
- Use the Kaimon MCP tools for Julia work, not `julia -e ...`:
  `ex` (code goes in `e`, `q=false` prints), `run_tests`,
  `search_methods`/`type_info`, `pkg_add` for dependencies.
- If no session is connected: ask the maintainer to choose between `start_session(project_path=<project root>)` 
  and a KaimonGate supported session in the Julia REPL of the maintainer 
  (note: `start_session` requires this project to be listed in the allowed-projects
  config — Kaimon TUI Config tab [p] or ~/.config/kaimon/projects.json)
- Keep the final check a clean run_tests -- sessions keep state.
- Rely on Revise.jl  in the Julia sessions - use `includet` or `using` to load code.
- `ex` cells running longer than ~30 s are auto-promoted to a background job;
  collect the result later with `check_eval(eval_id=...)` and do other work
  meanwhile rather than repolling rapidly. Full-size study runs
  (e.g. `run_study(nrefs = 1:9)`) will always be promoted - several minutes are normal.
- The studies in `scripts/` are standalone scripts outside Revise's reach: loading a file
  only defines `run_study()`/`run_comparison()`, nothing is computed yet. Load with
  `include("scripts/<example>.jl")`, then call `run_study()` or `run_comparison()`;
  verify correctness on small `nrefs` first, full refinement ranges are for benchmarking.
  Studies reuse cached data files under `data/<study>/`, so use `force`/`force_recompute`
  only after changing code that affects results.
- If `ex` with `q=false` returns `null`, the evaluation failed; wrap exploratory
  probes in `try ... catch` and return a single expression
  (e.g. `"ERROR: " * sprint(showerror, e)`) to get a diagnosis back
- Never make symlinks, in particular between files in $HOME/.julia
