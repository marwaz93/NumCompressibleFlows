# Convergence history for the manufactured rigid body rotation example:
#   velocitytype = RigidBodyRotation, densitytype = ExponentialDensityRBR, eostype = IdealGasLaw
# Runs in its own study folder (data/rigidbodyrotation,
# plots/rigidbodyrotation/convergence_history). Including this file only
# defines run_study() and run_comparison(), nothing is computed yet. Run e.g.
#   include("scripts/rigidbodyrotation.jl")
#   run_study()
#   run_study(nrefs = 2:5, quantities = [:L2u, :L2uR, :H1u])
#   run_study(maxsteps = 1e4, force = true)   # additional solver/plot options
#   run_comparison()                          # convectiontypes, default quantities
#   run_comparison(param = :reconstruct, values = [:RT, :BDM, :none], quantities = [:L2u, :H1u])
#   run_comparison(param = :μ, values = [1e-3, 1e-1, 1, 10], nrefs = 4, xquantity = :param)  # error over μ
#   set_sweep_choices!(convectiontype = (OseenConvection, NewConvection))  # change run_comparison defaults

include(joinpath(@__DIR__, "study_pipeline.jl"))

studyname = "rigidbodyrotation"
study = setup_study!(studyname)  # convergence history plot folder only

"""
    run_study(; kwargs...)

Plot the convergence history of the rigid body rotation example. All keyword
arguments below are the study configuration (defaults override `default_args`
via `load_data`); any further keyword arguments (e.g. `maxsteps`,
`target_residual`, `initial_values`, `pressure_in_f`, `force`,
`force_recompute`, `Plotter`) are passed through to `plot_convergencehistory`
/ `load_data`.
"""
function run_study(;
    # problem parameters
    nrefs = 1:4,
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
    gridtype = UnstructuredUnitSquare,
    convectiontype = OseenConvection,
    upwindtype = StandardUpwind,
    coriolistype = NoCoriolis,
    stab1 = (1 - 0.1, 0),
    stab2 = (1.5, 0),
    # plot options
    quantities = :default,       # e.g. [:L2u, :H1u, :L2ϱ, :H1u0] or :all
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

Compare convergence histories of several configurations of the rigid body
rotation example, one per value of the config field `param` (default:
convectiontype with the `SWEEP_CHOICES` values). `values` is the list of
values for that field; for categorical fields it may be omitted. All other
keyword arguments are the study configuration and plot options as in
`run_study` (in particular `nrefs`, `quantities = [:L2u, :H1u]`, `slopes`,
`force`); fixed study parameters are the defaults of `run_study`, the swept
field takes the values from `values`. With `xquantity = :param` the values
form the x-axis at a fixed mesh, e.g.

    run_comparison(param = :μ, values = [1e-3, 1e-1, 1, 10], nrefs = 4, xquantity = :param)

Plots go to `plots/rigidbodyrotation/parameter_studies_<param>`.
"""
function run_comparison(;
    # study configuration (fixed for all compared runs, defaults as in run_study)
    nrefs = 1:4,
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
    gridtype = UnstructuredUnitSquare,
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
    ## the swept fields of run_study get their values from `values` (the
    ## remaining discretization options are those of run_study)
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
