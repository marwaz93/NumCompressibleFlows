# Convergence history for the 3D trig-vortex example (genuinely 3D manufactured
# solution on the tetrahedral unit cube):
#   velocitytype = TrigVortexVelocity, densitytype = ExponentialZDensity,
#   eostype = IdealGasLaw, gridtype = UnstructuredUnitCube
# The velocity comes from the curl of a 3D vector stream function
# Ψ = (ufac/π)(cos πy, cos πz, cos πx), so that ϱu = curl Ψ = ufac(sin πz, sin πx,
# sin πy) is exactly divergence-free and u = curl Ψ / ϱ has all three components
# nonzero (in particular u3 ≠ 0). On the unit cube every component is >= 0: the
# flow enters through x=0, y=0, z=0 (inflow regions 4,1,5) and leaves through
# x=1, y=1, z=1 (outflow regions 2,3,6); all characteristics are open segments
# (no closed-orbit density kernel, unlike the embedded planar P7 vortex), and max|u| = O(ufac)
# (sqrt(3)/min ϱ). This example converges also for NoConvection and large c,
# where closed-ring vortices fail (velocity error dominated by the
# c-amplified pressure response to the discrete density).
# Runs in its own study folder (data/trigvortex_3d,
# plots/trigvortex_3d/convergence_history). Including this file only defines
# run_study() and run_comparison(), nothing is computed yet. Run e.g.
#   include("scripts/trigvortex_3d.jl")
#   run_study()
#   run_study(nrefs = 2:4, quantities = [:L2u, :L2uR, :H1u])
#   run_comparison()                          # convectiontypes, default quantities
# Note: Error(H1,u0) (divergence-free projection error) is 2D-only so far.

include(joinpath(@__DIR__, "study_pipeline.jl"))

studyname = "trigvortex_3d"
study = setup_study!(studyname)  # convergence history plot folder only

"""
    run_study(; kwargs...)

Plot the convergence history of the 3D trig-vortex example. All keyword arguments
below are the study configuration (defaults override `default_args` via
`load_data`); any further keyword arguments (e.g. `maxsteps`, `target_residual`,
`initial_values`, `pressure_in_f`, `force`, `force_recompute`, `Plotter`) are
passed through to `plot_convergencehistory` / `load_data`.
"""
function run_study(;
    # problem parameters
    nrefs = 1:4,
    μ = 1,
    λ = 0,
    c = 1,
    M = 1,
    τfac = 1,              # pseudo-time step τ = μ/(c·order²·M·τfac·ufac); the
                           # upwind CFL bound in the callback limits it further for
                           # the O(ufac) fluxes of this example
    ufac = 1,
    # discretization options
    order = 1,
    reconstruct = :RT,           # :RT, :BDM or :none
    # data of the problem
    velocitytype = TrigVortexVelocity,
    densitytype = ExponentialZDensity,
    eostype = IdealGasLaw,
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

Compare convergence histories of several configurations of the 3D trig-vortex
example, one per value of the config field `param` (default: convectiontype
with the `SWEEP_CHOICES` values). All other keyword arguments are the study
configuration and plot options as in `run_study`. Plots go to
`plots/trigvortex_3d/parameter_studies_<param>`.
"""
function run_comparison(;
    # study configuration (fixed for all compared runs, defaults as in run_study)
    nrefs = 1:4,
    μ = 1,
    λ = 0,
    c = 1,
    M = 1,
    τfac = 1,
    ufac = 1,
    order = 1,
    reconstruct = :RT,           # :RT, :BDM or :none (overwritten if swept)
    velocitytype = TrigVortexVelocity,
    densitytype = ExponentialZDensity,
    eostype = IdealGasLaw,
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
