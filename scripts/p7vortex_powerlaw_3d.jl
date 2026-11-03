# Convergence history for the 3D P7-vortex example (z-invariant embedding of
# the 2D manufactured solution on the tetrahedral unit cube):
#   velocitytype = P7VortexVelocity, densitytype = ExponentialDensity,
#   eostype = PowerLaw{1.4}, gridtype = UnstructuredUnitCube
# The exact velocity is (curl ξ / ϱ, 0); u ⋅ n = 0 on all boundary faces, the
# nonzero tangential velocity on the z-faces enters as interpolated Dirichlet
# data via wall_regions (see problem_definitions.jl).
# Runs in its own study folder (data/p7vortex_powerlaw_3d,
# plots/p7vortex_powerlaw_3d/convergence_history). Including this file only
# defines run_study() and run_comparison(), nothing is computed yet. Run e.g.
#   include("scripts/p7vortex_powerlaw_3d.jl")
#   run_study()
#   run_study(nrefs = 2:4, quantities = [:L2u, :L2uR, :H1u])
#   run_study(μ = 0.01)                       # small μ = small pseudo-time step
#       (as in the archived 2D study runs; for μ = 1 with closed characteristics
#       the upwind density iteration converges very slowly and relaxes ϱ toward
#       the collapsed closed-orbit state - the same behavior as in 2D)
#   run_study(maxsteps = 1e4, force = true)   # additional solver/plot options
#   run_comparison()                          # convectiontypes, default quantities
#       (RotationForm and NewConvection still use the 2D curl kernels and are
#       not yet available in 3D)
# Note: Error(H1,u0) (divergence-free projection error) is 2D-only so far.

include(joinpath(@__DIR__, "study_pipeline.jl"))

studyname = "p7vortex_powerlaw_3d"
study = setup_study!(studyname)  # convergence history plot folder only

"""
    run_study(; kwargs...)

Plot the convergence history of the 3D P7-vortex example. All keyword arguments
below are the study configuration (defaults override `default_args` via
`load_data`); any further keyword arguments (e.g. `maxsteps`, `target_residual`,
`initial_values`, `pressure_in_f`, `force`, `force_recompute`, `Plotter`) are
passed through to `plot_convergencehistory` / `load_data`.
"""
function run_study(;
    # problem parameters
    nrefs = 1:3,
    μ = 1,
    λ = 0,
    c = 1,
    M = 1,
    τfac = 4,
    ufac = 1,
    # discretization options
    order = 1,
    reconstruct = :RT,           # :RT, :BDM or :none
    # data of the problem
    velocitytype = P7VortexVelocity,
    densitytype = ExponentialDensity,
    eostype = PowerLaw{1.4},
    gridtype = UnstructuredUnitCube,
    convectiontype = NoConvection,
    upwindtype = StandardUpwind,
    coriolistype = NoCoriolis,
    stab1 = (1 - 0.1, 0),
    stab2 = (1.5, 0),
    # plot options
    quantities = :default,       # e.g. [:L2u, :H1u, :L2ϱ] or :all
    xquantity = :ndofs,          # :ndofs or :h
    slopes = (1, 2),             # reference slopes O(h^k)
    kwargs...,
)
    return plot_convergencehistory(;
        nrefs = nrefs,
        quantities = quantities,
        xquantity = xquantity,
        slopes = slopes,
        μ = μ,
        λ = λ,
        c = c,
        M = M,
        τfac = τfac,
        ufac = ufac,
        order = order,
        reconstruct = reconstruct,
        velocitytype = velocitytype,
        densitytype = densitytype,
        eostype = eostype,
        gridtype = gridtype,
        convectiontype = convectiontype,
        upwindtype = upwindtype,
        coriolistype = coriolistype,
        stab1 = stab1,
        stab2 = stab2,
        kwargs...,
    )
end

"""
    run_comparison(; param = :convectiontype, values = nothing, kwargs...)

Compare convergence histories of several configurations of the 3D P7-vortex
example, one per value of the config field `param` (default: convectiontype
with the `SWEEP_CHOICES` values). All other keyword arguments are the study
configuration and plot options as in `run_study`. Plots go to
`plots/p7vortex_powerlaw_3d/parameter_studies_<param>`.
"""
function run_comparison(;
    # study configuration (fixed for all compared runs, defaults as in run_study)
    nrefs = 1:3,
    μ = 1,
    λ = 0,
    c = 1,
    M = 1,
    τfac = 4,
    ufac = 1,
    order = 1,
    reconstruct = :RT,           # :RT, :BDM or :none (overwritten if swept)
    velocitytype = P7VortexVelocity,
    densitytype = ExponentialDensity,
    eostype = PowerLaw{1.4},
    gridtype = UnstructuredUnitCube,
    convectiontype = NoConvection,  # overwritten if swept
    upwindtype = StandardUpwind,
    coriolistype = NoCoriolis,
    stab1 = (1 - 0.1, 0),
    stab2 = (1.5, 0),
    # comparison options
    param = :convectiontype,
    values = nothing,
    quantities = (:L2u, :H1u, :L2ϱ),  # e.g. [:L2u, :H1u], :default or :all
    xquantity = :ndofs,
    slopes = (1, 2),
    kwargs...,
)
    haskey(Dict(kwargs), String(param)) && error(":$param must not be fixed in run_comparison, it is swept over values")
    return plot_parameter_study(;
        param = param,
        values = values,
        nrefs = nrefs,
        quantities = quantities,
        xquantity = xquantity,
        slopes = slopes,
        μ = μ,
        λ = λ,
        c = c,
        M = M,
        τfac = τfac,
        ufac = ufac,
        order = order,
        reconstruct = reconstruct,
        velocitytype = velocitytype,
        densitytype = densitytype,
        eostype = eostype,
        gridtype = gridtype,
        convectiontype = convectiontype,
        upwindtype = upwindtype,
        coriolistype = coriolistype,
        stab1 = stab1,
        stab2 = stab2,
        kwargs...,
    )
end
