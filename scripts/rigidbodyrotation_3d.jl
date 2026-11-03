# Convergence history for the 3D rigid body rotation example (z-invariant
# embedding of the 2D manufactured solution on the tetrahedral unit cube):
#   velocitytype = RigidBodyRotation, densitytype = ExponentialDensityRBR,
#   eostype = IdealGasLaw, gridtype = UnstructuredUnitCube
# The exact velocity is ufac ⋅ (-y, x, 0) (exactly divergence-free) with
# ϱ! = M exp((x² + y²)/2c); inflow/outflow faces are y=0/x=1 and x=0/y=1
# (same convention as the unit square), the z-faces carry the nonzero
# tangential velocity as interpolated Dirichlet data via wall_regions.
# Runs in its own study folder (data/rigidbodyrotation_3d,
# plots/rigidbodyrotation_3d/convergence_history). Including this file only
# defines run_study() and run_comparison(), nothing is computed yet. Run e.g.
#   include("scripts/rigidbodyrotation_3d.jl")
#   run_study()
#   run_study(nrefs = 2:4, quantities = [:L2u, :L2uR, :H1u])
#   run_comparison()                          # convectiontypes, default quantities
#       (RotationForm and NewConvection still use the 2D curl kernels and are
#       not yet available in 3D)
# Note: Error(H1,u0) (divergence-free projection error) is 2D-only so far.

include(joinpath(@__DIR__, "study_pipeline.jl"))

studyname = "rigidbodyrotation_3d"
study = setup_study!(studyname)  # convergence history plot folder only

"""
    run_study(; kwargs...)

Plot the convergence history of the 3D rigid body rotation example. All keyword
arguments below are the study configuration (defaults override `default_args`
via `load_data`); any further keyword arguments (e.g. `maxsteps`,
`target_residual`, `initial_values`, `pressure_in_f`, `force`,
`force_recompute`, `Plotter`) are passed through to `plot_convergencehistory` /
`load_data`.
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
    velocitytype = RigidBodyRotation,
    densitytype = ExponentialDensityRBR,
    eostype = IdealGasLaw,
    gridtype = UnstructuredUnitCube,
    convectiontype = OseenConvection,
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

Compare convergence histories of several configurations of the 3D rigid body
rotation example, one per value of the config field `param` (default:
convectiontype with the `SWEEP_CHOICES` values). All other keyword arguments are
the study configuration and plot options as in `run_study`. Plots go to
`plots/rigidbodyrotation_3d/parameter_studies_<param>`.
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
    velocitytype = RigidBodyRotation,
    densitytype = ExponentialDensityRBR,
    eostype = IdealGasLaw,
    gridtype = UnstructuredUnitCube,
    convectiontype = OseenConvection,  # overwritten if swept
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
