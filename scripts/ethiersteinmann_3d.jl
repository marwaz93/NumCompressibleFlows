# Convergence history for the Ethier-Steinmann-type 3D example (genuinely 3D
# manufactured solution on the tetrahedral unit cube, nonzero Dirichlet data):
#   velocitytype = EthierSteinmannVelocity, densitytype = ExponentialZDensity,
#   eostype = PowerLaw{1.4}, gridtype = UnstructuredUnitCube
# The compressible momentum flux is the curl of a vector stream function Psi,
# the vector potential of the exact divergence-free Stokes solution v_ES of the
# Ethier-Steinmann family (Ethier & Steinmann, IJNMF 1994):
#   v_ES = -a * (e^(ax) sin(ay + dz) + e^(az) cos(ax + dy),
#                e^(ay) sin(az + dx) + e^(ax) cos(ay + dz),
#                e^(az) sin(ax + dy) + e^(ay) cos(az + dx))
#   Psi  = (e^(ay) cos(az + dx), e^(az) cos(ax + dy), e^(ax) cos(ay + dz)),
#   rho*u = curl Psi = ufac * v_ES,   u = curl Psi / rho,
# so the momentum flux rho*u is exactly divergence-free and equals the exact ES
# Stokes velocity up to the scaling ufac (the decay factor exp(-d^2*t) of the
# unsteady ES solution is dropped since the solver is stationary). For the
# defaults a = pi/4, d = pi/2 the flow enters through x=1, y=1, z=1 and leaves
# through x=0, y=0, z=0, so all six faces carry inhomogeneous Dirichlet velocity
# data and the characteristics are open (no closed-orbit density kernel as for
# the P11 vortex). All three velocity components are nonzero and max|u| = O(ufac).
# Runs in its own study folder (data/ethiersteinmann_3d,
# plots/ethiersteinmann_3d/convergence_history). Including this file only defines
# run_study() and run_comparison(), nothing is computed yet. Run e.g.
#   include("scripts/ethiersteinmann_3d.jl")
#   run_study(nrefs = 2:3)
#   run_study(nrefs = 2:3, quantities = (:L2u, :H1u, :L2ϱ, :L2divu))
#   run_comparison()                          # convectiontypes, default quantities
# Note: Error(H1,u0) (divergence-free projection error) is 2D-only so far.

include(joinpath(@__DIR__, "study_pipeline.jl"))

studyname = "ethiersteinmann_3d"
study = setup_study!(studyname)  # convergence history plot folder only

"""
    run_study(; kwargs...)

Plot the convergence history of the 3D Ethier-Steinmann-type 3D example. All keyword arguments
below are the study configuration (defaults override `default_args` via
`load_data`); any further keyword arguments (e.g. `maxsteps`, `target_residual`,
`initial_values`, `pressure_in_f`, `force`, `force_recompute`, `Plotter`) are
passed through to `plot_convergencehistory` / `load_data`.
"""
function run_study(;
    # problem parameters (defaults = settings of the converged convergence study:
    # mu=1e-3, ufac=0.4, PowerLaw{1.4}, stab1=(0.9,0.01), mass_balance=true;
    # tau-fac 2.0 for levels 1-3, use taufac = 8.0 for level 4)
    nrefs = 1:4,
    μ = 1.0e-3,
    λ = 0,
    c = 1,
    M = 1,
    τfac = 2.0,            # pseudo-time step τ = μ/(c·order²·M·τfac·ufac); the
                           # upwind CFL bound in the callback limits it further for
                           # the O(ufac) fluxes of this example
    ufac = 0.4,
    # discretization options
    order = 1,
    reconstruct = :RT,           # :RT, :BDM or :none
    mass_balance = true,         # uniform density source lambda pins total mass to M_exact
    # data of the problem
    velocitytype = EthierSteinmannVelocity,
    densitytype = ExponentialZDensity,
    eostype = PowerLaw{1.4},
    gridtype = UnstructuredUnitCube,
     convectiontype = NoConvection,
     upwindtype = StandardUpwind,
     coriolistype = NoCoriolis,
     stab1 = (0.9, 0.01),
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
         mass_balance = mass_balance,
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

Compare convergence histories of several configurations of the 3D Ethier-Steinmann-type 3D vortex
example, one per value of the config field `param` (default: convectiontype
with the `SWEEP_CHOICES` values). All other keyword arguments are the study
configuration and plot options as in `run_study`. Plots go to
`plots/ethiersteinmann_3d/parameter_studies_<param>`.
"""
function run_comparison(;
    # study configuration (fixed for all compared runs, defaults as in run_study)
    nrefs = 1:4,
    μ = 1.0e-3,
    λ = 0,
    c = 1,
    M = 1,
    τfac = 2.0,
    ufac = 0.4,
    order = 1,
    reconstruct = :RT,           # :RT, :BDM or :none (overwritten if swept)
    mass_balance = true,
    velocitytype = EthierSteinmannVelocity,
    densitytype = ExponentialZDensity,
    eostype = PowerLaw{1.4},
    gridtype = UnstructuredUnitCube,
    convectiontype = NoConvection,  # overwritten if swept
    upwindtype = StandardUpwind,
    coriolistype = NoCoriolis,
    stab1 = (0.9, 0.01),
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
        mass_balance = mass_balance,
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
