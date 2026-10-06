# Convergence history for the manufactured P7-vortex example:
#   velocitytype = P7VortexVelocity, densitytype = ExponentialDensity, eostype = PowerLaw{1.4}
# Runs in its own study folder (data/projects/p7vortex_powerlaw,
# plots/p7vortex_powerlaw/convergence_history). Including this file only
# defines run_study(), nothing is computed yet. Run e.g.
#   include("scripts/p7vortex_convergencehistory.jl")
#   run_study()
#   run_study(nrefs = 2:5, quantities = [:L2u, :L2uR, :H1u])
#   run_study(maxsteps = 1e4, force = true)   # additional solver/plot options

include(joinpath(@__DIR__, "study_pipeline.jl"))

studyname = "p7vortex_powerlaw"
study = setup_study!(studyname)  # convergence history plot folder only

"""
    run_study(; kwargs...)

Plot the convergence history of the P7-vortex example. All keyword arguments
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
    τfac = 4,
    ufac = 1,
    # discretization options
    order = 1,
    reconstruct = :RT,           # :RT, :BDM or :none
    # data of the problem
    velocitytype = P7VortexVelocity,
    densitytype = ExponentialDensity,
    eostype = PowerLaw{1.4},
    gridtype = UnstructuredUnitSquare,
    convectiontype = NoConvection,
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
