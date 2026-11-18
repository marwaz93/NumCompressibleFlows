# Solver and study pipeline

## Single solve

`default_args` is a `Dict` of all configuration parameters (problem parameters
`μ, λ, c, M`, solver options `nrefs, order, τfac, stab1, stab2, reconstruct,
target_residual, mass_balance, ...` and problem data `velocitytype, densitytype,
convectiontype, eostype, gridtype, ...`). [`load_data`](@ref) copies it and applies
keyword overrides, [`run_single`](@ref) solves the problem and stores `data["solution"]`
and the residuals back into the dict, and `compute_errors!` computes the error norms
(keys `"Error(L2,u)"`, `"Error(H1,u)"`, `"Error(L2,ϱ)"`, ...) from it.

```julia
data = load_data(velocitytype = TaylorGreenVelocity, gridtype = UnstructuredUnitCube,
                 densitytype = ExponentialZDensity, eostype = PowerLaw{1.4},
                 nrefs = 2, mass_balance = true, τfac = 0.5)
run_single(data)
```

## Study pipeline

Example scripts share the DrWatson layer `scripts/study_pipeline.jl`. Including a
script only defines `run_study()`/`run_comparison()`; nothing is computed yet:

```julia
include("scripts/rigidbodyrotation.jl")
run_study(nrefs = 2:5)                 # convergence history over refinement levels
run_study(nrefs = 3, μ = 0.01)         # single run with modified configuration
run_comparison()                       # one curve per convectiontype
```

Data are cached as `.jld2` files under `data/<studyname>/`, plots (PNG + convergence
history) under `plots/<studyname>/convergence_history/` (parameter studies in
`plots/<studyname>/parameter_studies_<param>/`). Every data/plot file gets an
accompanying `.txt` sidecar listing the full configuration and the computed values.
Cached files are reused unless `force`/`force_recompute = true` is passed.

Plotable quantities are defined in `CONV_QUANTITIES` (e.g. `L2u, H1u, L2ϱ, L2uR,
L2divu, L2divuR, H1u0, H1u1, L2u_inc, nits`) and selected via
`run_study(quantities = (:L2u, :H1u))`. Parameter sweeps over a config field are
configured with `SWEEP_CHOICES` / `set_sweep_choices!`.

## Batch runs

`scripts/run.jl` is the command line entry point (used by the HTCondor workflow
`create_parameters.sh`, `study.dag`, `julia_job.sub`, `run_julia.sh`):

```
julia --project=. scripts/run.jl mode=solve example=p7vortex_powerlaw nrefs=4 μ=0.001
julia --project=. scripts/run.jl mode=plot example=p7vortex_powerlaw
```
