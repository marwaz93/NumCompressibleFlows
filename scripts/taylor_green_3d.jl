# Convergence history for the fully 3D Taylor-Green vortex example (Taylor &
# Green 1937, quarter-period shifted workshop form on a cube with impermeable
# walls, https://en.wikipedia.org/wiki/Taylor–Green_vortex; on the unit cube,
# u = curl(Psi)/ϱ form):
#   velocitytype = TaylorGreenVelocity, densitytype = ExponentialZDensity,
#   eostype = PowerLaw{1.4}, gridtype = UnstructuredUnitCube
# The mass flux is the divergence-free trigonometric triad (A + B + C = 0)
#   ϱu = curl(Psi) = (ufac/pi)(1/2 sin(pi x) cos(pi y) cos(pi z),
#                             1/2 cos(pi x) sin(pi y) cos(pi z),
#                                 - cos(pi x) cos(pi y) sin(pi z)),
# (the workshop case is C = 0, i.e. zero third component). Psi has three
# nonzero components (see problem_definitions.jl). All three flux components
# are nonzero and u = curl(Psi)/ϱ! is compressive for ExponentialZDensity,
# while the density source ∇⋅(ϱu) = ∇⋅curl(Psi) vanishes identically (legacy
# zero-source pipeline). The normal flux vanishes on all six cube faces (the
# sin(pi .) of the own component) while the tangential velocity does not: all
# faces are slip walls with inhomogeneous Dirichlet data (wall_regions), mass
# is conserved to machine precision. Unlike the old z-invariant variant
# Psi = (0, 0, sin sin cos) the interior streamlines leave every horizontal
# plane (u_z != 0): the closed vortex rings - and with them the kernel modes of
# the upwind continuity operator and the pseudo-time density concentration
# instability reported for the old field - are gone, and the discrete steady
# density is unique.
# Runs in its own study folder (data/taylor_green_3d,
# plots/taylor_green_3d/convergence_history). Including this file only defines
# run_study() and run_comparison(), nothing is computed yet. Run e.g.
#   include("scripts/taylor_green_3d.jl")
#   run_study()                          # defaults: mu = 1e-3, reconstruct = :RT
#   run_study(reconstruct = :none)       # without flux reconstruction
#   run_study(nrefs = 2, τfac = 30, mass_balance = true)
#   run_comparison()                     # convectiontypes, default quantities
#       (RotationForm and NewConvection still use the 2D curl kernels and are
#       not yet available in 3D)
# Note: Error(H1,u0) (divergence-free projection error) is 2D-only so far.

include(joinpath(@__DIR__, "study_pipeline.jl"))

studyname = "taylor_green_3d"
study = setup_study!(studyname)  # convergence history plot folder only

"""
    run_study(; kwargs...)

Plot the convergence history of the 3D Taylor-Green-type vortex example. All
keyword arguments below are the study configuration (defaults override
`default_args` via `load_data`); any further keyword arguments (e.g. `maxsteps`,
`target_residual`, `mass_balance`, `no_continuity_update`, `initial_values`,
`pressure_in_f`, `force`, `force_recompute`, `Plotter`) are passed through to
`plot_convergencehistory` / `load_data`.
"""
function run_study(;
    # problem parameters
    nrefs = 1:4,
    μ = 1.0e-3,
    λ = 0,
    c = 1,
    M = 1,
    τfac = 8.0,            # pseudo-time step τ = μ/(c·order²·M·τfac·ufac); keeps τ
                           # far below the upwind CFL bound (the callback clamps it
                           # further); same regime as the converged ES study
    ufac = 1,
    # discretization options
    order = 1,
    reconstruct = :RT,           # :RT, :BDM or :none
    # data of the problem
    velocitytype = TaylorGreenVelocity,
    densitytype = ExponentialZDensity,
    eostype = PowerLaw{1.4},
    gridtype = UnstructuredUnitCube,
    convectiontype = NoConvection,
    upwindtype = StandardUpwind,
    coriolistype = NoCoriolis,
    stab1 = (1 - 0.1, 0),
    stab2 = (1.5, 0),
    # iteration budget (the unstable coupled case explores the full budget)
    maxsteps = 40000,
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
        maxsteps = maxsteps,
        kwargs...,
    )
end

"""
    run_comparison(; param = :convectiontype, values = nothing, kwargs...)

Compare convergence histories of several configurations of the 3D
Taylor-Green-type vortex example, one per value of the config field `param`
(default: convectiontype with the `SWEEP_CHOICES` values). All other keyword
arguments are the study configuration and plot options as in `run_study`. Plots
go to `plots/taylor_green_3d/parameter_studies_<param>`.
"""
function run_comparison(;
    # study configuration (fixed for all compared runs, defaults as in run_study)
    nrefs = 1:4,
    μ = 1.0e-3,
    λ = 0,
    c = 1,
    M = 1,
    τfac = 8.0,
    ufac = 1,
    order = 1,
    reconstruct = :RT,           # :RT, :BDM or :none (overwritten if swept)
    velocitytype = TaylorGreenVelocity,
    densitytype = ExponentialZDensity,
    eostype = PowerLaw{1.4},
    gridtype = UnstructuredUnitCube,
    convectiontype = NoConvection,  # overwritten if swept
    upwindtype = StandardUpwind,
    coriolistype = NoCoriolis,
    stab1 = (1 - 0.1, 0),
    stab2 = (1.5, 0),
    maxsteps = 40000,
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
        maxsteps = maxsteps,
        kwargs...,
    )
end
