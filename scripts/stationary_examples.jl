using NumCompressibleFlows
using ExtendableFEM
using ExtendableFEMBase
using ExtendableGrids
using Triangulate
using SimplexGridFactory
using GridVisualize
using Symbolics: Symbolics, @variables, build_function
using LinearAlgebra
using UnicodePlots
using Term

using DrWatson
using JLD2
using LaTeXStrings
using Colors
using ColorTypes
using Latexify
using Plots

# ==============================================================================
# Default configuration & helpers
# ==============================================================================

"""
    safe_produce_or_load(data; force = false, kwargs...)

Wrapper around `produce_or_load` that calls `run_single` to compute errors.
"""
function safe_produce_or_load(data; force = false, kwargs...)
    return produce_or_load(run_single, data; filename = filename, force = force, kwargs...)
end

default_args = Dict(
    # problem parameters
    "μ" => 1,
    "λ" => 0,
    "c" => 1,
    "M" => 1,
    # solving options
    "τfac" => 4,
    "ufac" => 1,
    "nrefs" => 4,
    "order" => 1,
    "pressure_stab" => 0,
    "bonus_quadorder" => 4,
    "maxsteps" => 8000,
    "subiterations_momentum" => :auto,
    "target_residual" => 1.0e-11,
    "reconstruct" => :RT, # choose from :RT :BDM :none
    # data of the problem
    "velocitytype" => ZeroVelocity,
    "densitytype" => ExponentialDensity,
    "convectiontype" => NoConvection,
    "upwindtype" => StandardUpwind,
    "initial_values" => :interpolate, # choose from :interpolate, :stokes,
    "no_continuity_update" => false,
    "coriolistype" => NoCoriolis,
    "eostype" => IdealGasLaw,
    "gridtype" => Mountain2D,
    "pressure_in_f" => true,
    "others_in_f" => true,
    "stab1" => (1-0.1, 0),
    "stab2" => (1.5, 0),
)

"""
    filename(data) -> String

Build a DrWatson-compatible filename string for a given `data` dict.
All key parameters are abbreviated and the result is prefixed with
`"data/projects/compressible_stokes/"`.
"""
function filename(data; prefix = "data/projects/compressible_stokes_repeat/")
    μ = data["μ"]
    λ = data["λ"]
    EOSType = data["eostype"]
    γ = gamma(EOSType)
    c = data["c"]
    M = data["M"]
    τfac = data["τfac"]
    ufac = data["ufac"]
    nrefs = data["nrefs"]
    order = data["order"]
    reconstruct = data["reconstruct"]
    target_residual = data["target_residual"]
    maxsteps = data["maxsteps"]
    pressure_stab = data["pressure_stab"]
    bonus_quadorder = data["bonus_quadorder"]

    # Abbreviate type names for savename
    vtype = replace(string(data["velocitytype"]), "Velocity" => "V")
    dtype = replace(string(data["densitytype"]), "Density" => "D")
    etype = replace(string(data["eostype"]), "Law" => "")
    gtype = replace(string(data["gridtype"]), "2D" => "")
    ctype = replace(string(data["convectiontype"]), "Convection" => "Conv")
    cortype = replace(string(data["coriolistype"]), "Coriolis" => "Cor")
    pressure_in_f = data["pressure_in_f"]
    stab1 = data["stab1"]
    stab2 = data["stab2"]
    convectiontype = data["convectiontype"]


    essential_params = @dict μ λ γ c M τfac ufac nrefs order reconstruct vtype dtype etype gtype ctype cortype pressure_in_f stab1 stab2 convectiontype

    sname = savename(essential_params;
                     allowedtypes = (Real, String, SubString, Symbol,
                                     Tuple{Real, Real}))
    sname = prefix * sname
    return sname
end

function load_data(; kwargs...)
    data = deepcopy(default_args)
    for (k, v) in kwargs
        data[String(k)] = v
    end
    return data
end

# ==============================================================================
# Registry of plottable convergence quantities (used by plot_convergencehistory)
# ==============================================================================

"""
Registry of convergence curves selectable by symbol in
`plot_convergencehistory`. Each entry has:
- `data`: key in the per-level data dict, or a function `data -> value`
- `label`: legend label
- optional `style`: Plots keyword arguments (linestyle, marker, markersize, color)
- optional `xinc = true`: plot against the DOF count of the incompressible system
- optional `incompressible = true`: selecting it triggers `run_incompressible!`
  (via `compute_errors(..., compare_incompressible = true)`)
"""
const CONV_QUANTITIES = (
    L2u = (; data = "Error(L2,u)", label = L"|| \mathbf{u} - \mathbf{u}_h \,||"),
    L2uR = (; data = "Error(L2,uR)", label = L"|| \mathbf{u} - \Pi\mathbf{u}_h \,||"),
    H1u = (; data = "Error(H1,u)", label = L"|| ∇(\mathbf{u} - \mathbf{u}_h)\,||"),
    L2ϱ = (; data = "Error(L2,ϱ)", label = L"|| {ϱ}-ϱ_h \, ||"),
    L2ϱu = (; data = "Error(L2,ϱu)", label = L"|| {ϱ\mathbf{u}}-ϱ_h \mathbf{u}_h \, ||"),
    H1u0 = (; data = "Error(H1,u0)", label = L"||  ∇( \mathbf{u}^0 - \mathbf{u}^0_h ) \,||"),
    H1u1 = (; data = d -> sqrt(d["Error(H1,u)"]^2 - d["Error(H1,u0)"]^2),
              label = L"||  ∇( \mathbf{u}^1 - \mathbf{u}^1_h ) \,||"),
    nits = (; data = "nits", label = L"nits"),
    H1u_inc = (; data = "Error(H1,u_inc)", label = L"|| ∇(\mathbf{u} - \mathbf{u}_h^{inc})\,||",
                  incompressible = true, xinc = true,
                  style = (linestyle = :dashdot, marker = :xcross, markersize = 7, color = :orange)),
    L2u_inc = (; data = "Error(L2,u_inc)", label = L"|| \mathbf{u} - \mathbf{u}_h^{inc}\,||",
                  incompressible = true, xinc = true,
                  style = (linestyle = :dashdot, marker = :xcross, markersize = 7, color = :purple)),
    L2u_diff = (; data = "Error(L2,u-u_inc)", label = L"|| \mathbf{u}_h - \mathbf{u}_h^{inc}\,||",
                  incompressible = true, xinc = true,
                  style = (linestyle = :dashdot, marker = :xcross, markersize = 7, color = :green)),
    res_momentum = (; data = "res_momentum", label = "residual momentum"),
    res_continuity = (; data = "res_continuity", label = "residual continuity"),
)

const DEFAULT_QUANTITIES = (:L2u, :H1u, :L2ϱ, :L2ϱu, :H1u0)

## value of a CONV_QUANTITIES entry for a given level's data dict
function conv_value(entry, d)
    if entry.data isa String
        return get(d, entry.data, NaN)
    else
        try
            return entry.data(d)
        catch
            return NaN
        end
    end
end

"""
    run_incompressible!(data; gauge = 1.0e-6, kwargs...)

Solve the incompressible reference problem (limit c = Inf) of the manufactured
problem defined by `data` and store the result in the same dict (keys
`Error(L2,u_inc)`, `Error(H1,u_inc)`, `res_incompressible`,
`incompressible_solution`). Uses prepare_data with the incompressible switch
(ϱ ≡ M, all 1/c terms ignored; the dropped ∇p is a pure gradient absorbed by
the pressure Lagrange multiplier). Exact solution: (curl ξ / M, p = 0).
The pressure mean is fixed via a small penalty `gauge`.
"""
function run_incompressible!(data; kwargs...)
    @info "running incompressible solver (c = Inf)"

    # -- problem parameters --
    μ  = data["μ"]
    λ  = data["λ"]
    γ  = data["γ"]
    M  = data["M"]
    c  = Inf
    ufac = data["ufac"]

    # -- solving options --
    nrefs           = data["nrefs"]
    order           = data["order"]
    reconstruct     = data["reconstruct"]
    target_residual = data["target_residual"]
    bonus_quadorder = data["bonus_quadorder"]

    # -- data of the problem --
    velocitytype   = data["velocitytype"]
    densitytype    = data["densitytype"]
    eostype        = data["eostype"]
    gridtype       = data["gridtype"]
    pressure_in_f  = true
    others_in_f    = true
    convectiontype = data["convectiontype"]
    stab1          = data["stab1"]

    ## prepare data with incompressible switch (ρ ≡ M, 1/c terms ignored)
    ϱ!, kernel_gravity!, kernel_rhs!, u!, ∇u! =
        prepare_data(velocitytype, densitytype, eostype;
                     others_in_f, pressure_in_f, M, c, μ, λ, γ,
                     ufac, convectiontype, incompressible = true, kwargs...)
    xgrid = NumCompressibleFlows.grid(gridtype; nref = nrefs)

    ## in/outflow regions (same construction as in run_single)
    testgrid = NumCompressibleFlows.grid(gridtype; nref = 1)
    rinflow  = inflow_regions(velocitytype, gridtype)
    routflow = outflow_regions(velocitytype, gridtype)
    rhom     = setdiff(unique!(testgrid[BFaceRegions]), union(rinflow, routflow))

    ## define unknowns
    u = Unknown("u"; name = "velocity", dim = 2)
    p = Unknown("p"; name = "pressure", dim = 1)
    ϱ = Unknown("ϱ"; name = "density", dim = 1) # dummy, only referenced by convection kernels

    ## define FE types and reconstruction operator (same velocity space as run_single)
    if order == 1
        FETypes = [H1BR{2}, L2P0{1}, L2P0{1}]
        ReconstSpace = reconstruct == :RT ? HDIVRT0{2} : reconstruct == :BDM ? HDIVBDM1{2} : nothing
        id_u    = ReconstSpace !== nothing ? apply(u, Reconstruct{ReconstSpace, Identity}) : id(u)
        div_u   = ReconstSpace !== nothing ? apply(u, Reconstruct{ReconstSpace, Divergence}) : div(u)
    elseif order == 2
        FETypes = [H1P2B{2, 2}, L2P1{1}, L2P1{1}]
        ReconstSpace = reconstruct == :RT ? HDIVRT1{2} : reconstruct == :BDM ? HDIVBDM2{2} : nothing
        id_u    = ReconstSpace !== nothing ? apply(u, Reconstruct{ReconstSpace, Identity}) : id(u)
        div_u   = ReconstSpace !== nothing ? apply(u, Reconstruct{ReconstSpace, Divergence}) : div(u)
    else
        throw(ArgumentError("order must be 1 or 2"))
    end
    FES = [FESpace{FETypes[j]}(xgrid) for j in 1:2]


    ## define incompressible Stokes problem
    PD = ProblemDescription("Incompressible Stokes problem (limit c = Inf)")
    assign_unknown!(PD, u)
    assign_unknown!(PD, p)

    assign_operator!(PD, BilinearOperator(stokes_kernel!, [grad(u), id(p)]; params = [μ], kwargs...))

    ## fix pressure dof
    assign_operator!(PD, FixDofs(p; dofs = [1], kwargs...))

    if convectiontype == StandardConvection
        assign_operator!(PD, LinearOperator(
        kernel_standardconvection_incompressible_linearoperator!, [id_u],
        [id_u, grad(u)]; quadorder = 2*order + 1,
        factor = -M, kwargs...))
    elseif convectiontype == OseenConvection
        assign_operator!(PD, BilinearOperator(
        kernel_oseenconvection!(u!, ϱ!), [id_u], [grad(u)]; quadorder = 2*order + 1,
        factor = 1, kwargs...))
    elseif !(convectiontype === NoConvection)
        error("convectiontype $(convectiontype) not yet supported by run_incompressible!")
    end

    assign_operator!(PD, LinearOperator(kernel_rhs!, [id_u]; kwargs...))

    ## boundary data (same as in run_single)
    if length(rhom) > 0
        assign_operator!(PD, HomogeneousBoundaryData(u; regions = rhom, kwargs...))
    end
    if length(rinflow) > 0 || length(routflow) > 0
        assign_operator!(PD, InterpolateBoundaryData(
            u, u!; bonus_quadorder, regions = union(rinflow, routflow), kwargs...))
    end

    ## the problem is linear (convection kernels would use the exact u! and ϱ! = M);
    ## is_linear prevents spurious Newton iterations on the coupled saddle point system
    sol, SC = solve(PD, FES; init = FEVector(FES; tags = [u, p]), maxiterations = 10,
        target_residual, constant_matrix = true, return_config = true)

    data["res_incompressible"] = residual(SC)

    ## errors against exact solution (u = curl ξ / M, ϱ = M, p = 0 up to gauge)
    ErrorIntegratorExact = ItemIntegrator(
        exact_error_incompressible!(u!, ∇u!, Float64(M)), [id(u), grad(u)];
        resultdim = 6, quadorder = 10, kwargs...)
    error = evaluate(ErrorIntegratorExact, sol)
    data["Error(L2,u_inc)"]  = sqrt(sum(error[1, :]) + sum(error[2, :]))
    data["Error(H1,u_inc)"]  = sqrt(sum(error[3, :]) + sum(error[4, :]) +
                                 sum(error[5, :]) + sum(error[6, :]))

    ## save data
    data["incompressible_solution"]  = sol
    data["unknown_u_incompressible"] = u
    data["unknown_p_incompressible"] = p # not comparable to pressure from compressible problem

    if length(sol.entries) < 1e5
        ExtendableFEM.plot([id(u)], sol; Plotter = UnicodePlots)
    end

    return data
end

## exact error kernel for the incompressible reference problem (exact u!, ∇u! minus discrete u, ∇u)
function exact_error_incompressible!(u!, ∇u!, ϱval)
    return function closure(result, args, qpinfo)
        u!(view(result, 1:2), qpinfo)
        ∇u!(view(result, 3:6), qpinfo)
        view(result, 1:6) .-= view(args, 1:6)
        return result .= result .^ 2
    end
end


# ==============================================================================
# Convection helper methods
# ==============================================================================

"""
    _add_convection!(PD, u, ϱ, id_u, grad, div_u, u!, ϱ!, convectiontype, order, kwargs...)

Dispatch on convectiontype to add the appropriate convection operator.
"""
function _add_convection!(PD, u, ϱ, id_u, grad, div_u, u!, ϱ!,
                          ::Type{<:StandardConvection}, order, xgrid, stab1, FluxIntegrator, fluxes, FES, kwargs...)
    assign_operator!(PD, LinearOperator(
        kernel_standardconvection_linearoperator!, [id_u],
        [id_u, grad(u), id(ϱ)]; quadorder = 2*order + 1,
        factor = -1, kwargs...))
end

function _add_convection!(PD, u, ϱ, id_u, grad, div_u, u!, ϱ!,
                          ::Type{<:OseenConvection}, order, xgrid, stab1, FluxIntegrator, fluxes, FES, kwargs...)
    assign_operator!(PD, BilinearOperator(kernel_oseenconvection!(u!, ϱ!), [id_u], [grad(u)]; quadorder = 2*order + 2, store = true, factor = 1, kwargs...))
end

function _add_convection!(PD, u, ϱ, id_u, grad, div_u, u!, ϱ!,
                          ::Type{<:RotationForm}, order, xgrid, stab1, FluxIntegrator, fluxes, FES, kwargs...)
    assign_operator!(PD, LinearOperator(
        kernel_rotationform_linearoperator!, [id_u, div_u],
        [id_u, curl2(u), id(ϱ)]; quadorder = 2*order + 1,
        factor = -1, kwargs...))
end

function _add_convection!(PD, u, ϱ, id_u, grad, div_u, u!, ϱ!,
                          ::Type{NoConvection}, order, xgrid, stab1, FluxIntegrator, fluxes, FES, kwargs...)
    nothing
end

function _add_convection!(PD, u, ϱ, id_u, grad, div_u, u!, ϱ!,
                          ::Type{<:KarperConvection}, order, xgrid, stab1, FluxIntegrator, fluxes, FES,  kwargs...)

    # project current velocity to P0
    FES_P0 = FESpace{L2P0{2}}(xgrid)
    u0 = FEVector(FES_P0)    
    bconv = FEVector(FES_P0)

    T = nothing 
    

    function callback_karper!(A, b, args; assemble_matrix = true,
                       assemble_rhs = true, time = 0, kwargs...)

        if assemble_rhs
            if isnothing(T)
                T = compute_lazy_interpolation_jacobian(FES_P0, args[1].FES)
            end
            # project current velocity (=args[1]) onto P0 --> ̂u
            u0.entries .= T * view(args[1])
            #lazy_interpolate!(u0[1], args, [id(1)]; quadorder = 2)

            ## computes integrals of u ⋅ n on all faces and use them for upwinding
            fill!(fluxes, 0)
            evaluate!(fluxes, FluxIntegrator, [args[1]])
            view(fluxes,:) ./= xgrid[FaceVolumes]

            fill!(bconv.entries, 0)
            assemble!(bconv, LinearOperatorDG(
            kernel_upwind_convection!, [jump(id(1))], [id(1), this(id(2)), other(id(2)), this(id(3)), other(id(3))];
            factor = -1, quadorder = 0, entities = ON_IFACES, params = [fluxes]), [args[1], args[2], u0[1]])
            
            
            if stab1[2] > 0
                assemble!(bconv, LinearOperatorDG(
                    velocity_jump_stab_kernel!(stab1[1], 3.0),
                    [jump(id(1))], [jump(id(2)), average(id(3))];
                    factor = stab1[2]*2, entities = ON_IFACES, kwargs...), [args[1], args[2], u0[1]])
            end
            
            
            b .+= view(bconv.entries' * T,:)

        end

    end                
    assign_operator!(PD, CallbackOperator(
        callback_karper!, [u, ϱ]; linearized_dependencies = [u],
        modifies_rhs = true, modifies_matrix = false, kwargs...,
        name = "upwind convection term"))
end

function _add_convection!(PD, u, ϱ, id_u, grad, div_u, u!, ϱ!,
                          ::Type{<:NewConvection}, order, xgrid, stab1, FluxIntegrator, fluxes, FES, kwargs...)
    assign_operator!(PD, LinearOperator(
        kernel_new_rotationform_linearoperator!, [id_u],
        [id_u, curl2(u), id(ϱ)]; quadorder = 2*order + 1,
        factor = -1, kwargs...))
         # project current velocity to P0
    FES_P0 = FESpace{L2P0{2}}(xgrid)
    u0 = FEVector(FES_P0)    
    bconv = FEVector(FES[1])
    

    function callback_newconvection!(A, b, args; assemble_matrix = true,
                       assemble_rhs = true, time = 0, kwargs...)

        if assemble_rhs
            lazy_interpolate!(u0[1], args, [id(1)]; postprocess = (result, input, qpinfo) -> (result[1] = input[1]^2+input[2]^2;), quadorder = 4)

            ## computes integrals of u ⋅ n on all faces and use them for upwinding
            fill!(fluxes, 0)
            evaluate!(fluxes, FluxIntegrator, [args[1]])
            view(fluxes,:) ./= xgrid[FaceVolumes]

            fill!(bconv.entries, 0)
            assemble!(bconv, LinearOperatorDG(
            kernel_upwind_newconvection!, [normalflux(1)], [this(id(2)), other(id(2)), jump(id(3))];
            factor = -1/2, quadorder = 0, entities = ON_IFACES, params = [fluxes]), [args[1], args[2], u0[1]])
            
            
            if stab1[2] > 0
                assemble!(bconv, LinearOperatorDG(
                    velocity_jump_stab_kernel!(stab1[1], 3.0),
                    [jump(id(1))], [jump(id(2)), average(id(3))];
                    factor = stab1[2]*2, entities = ON_IFACES, kwargs...), [args[1], args[2], u0[1]])
            end

            b .+= bconv.entries
            
        end

    end                
    assign_operator!(PD, CallbackOperator(
        callback_newconvection!, [u, ϱ]; linearized_dependencies = [u],
        modifies_rhs = true, modifies_matrix = false, kwargs...,
        name = "upwind convection term"))
end



# ==============================================================================
# Core solver: run_single
# ==============================================================================

function run_single(data; kwargs...)
    # -- problem parameters --
    μ      = data["μ"]
    λ      = data["λ"]
    eostype= data["eostype"]
    γ      = gamma(eostype)
    c      = data["c"]
    M      = data["M"]
    ufac   = data["ufac"]

    # -- solving options --
    τfac           = data["τfac"]
    nrefs          = data["nrefs"]
    order          = data["order"]
    reconstruct    = data["reconstruct"]
    target_residual = data["target_residual"]
    maxsteps       = data["maxsteps"]
    pressure_stab  = data["pressure_stab"]
    bonus_quadorder = data["bonus_quadorder"]
    subiterations_momentum = data["subiterations_momentum"]

    # -- data of the problem --
    velocitytype   = data["velocitytype"]
    densitytype    = data["densitytype"]
    gridtype       = data["gridtype"]
    pressure_in_f  = data["pressure_in_f"]
    initial_values = data["initial_values"]
    others_in_f = data["others_in_f"]
    convectiontype = data["convectiontype"]
    upwindtype      = data["upwindtype"]
    coriolistype   = data["coriolistype"]
    stab1          = data["stab1"]
    stab2          = data["stab2"]

    if target_residual / max(stab1[2], stab2[2]) < 1e-15
        target_residual = 1e-15 * max(stab1[2], stab2[2])
        @warn "reset target residual to $(target_residual) due to very large stabilization constants"
    end

    ## prepare data and grid
    ϱ!, kernel_gravity!, kernel_rhs!, u!, ∇u! =
        prepare_data(velocitytype, densitytype, eostype;
                     others_in_f, pressure_in_f, M, c, μ, λ, γ,
                     ufac, τfac, nrefs, kwargs...)
    xgrid = NumCompressibleFlows.grid(gridtype; nref = nrefs)

    M_exact = integrate(xgrid, ON_CELLS, ϱ!, 1; quadorder = 30)
    τ = μ / (c * order^2 * M * τfac * ufac)
    @info "M = $M, M_exact = $M_exact τ = $τ"

    ## define unknowns
    u = Unknown("u"; name = "velocity", dim = 2)
    ϱ = Unknown("ϱ"; name = "density", dim = 1)
    p = Unknown("p"; name = "pressure", dim = 1)

    ## define FE types and reconstruction operator
    if order == 1
        FETypes = [H1BR{2}, L2P0{1}, L2P0{1}]
        ReconstSpace = reconstruct == :RT ? HDIVRT0{2} : reconstruct == :BDM ? HDIVBDM1{2} : nothing
        id_u    = ReconstSpace !== nothing ? apply(u, Reconstruct{ReconstSpace, Identity}) : id(u)
        div_u   = ReconstSpace !== nothing ? apply(u, Reconstruct{ReconstSpace, Divergence}) : div(u)
    elseif order == 2
        FETypes = [H1P2B{2, 2}, L2P1{1}, L2P1{1}]
        ReconstSpace = reconstruct == :RT ? HDIVRT1{2} : reconstruct == :BDM ? HDIVBDM2{2} : nothing
        id_u    = ReconstSpace !== nothing ? apply(u, Reconstruct{ReconstSpace, Identity}) : id(u)
        div_u   = ReconstSpace !== nothing ? apply(u, Reconstruct{ReconstSpace, Divergence}) : div(u)
    end

    ## define FE spaces
    FES::Array{FESpace{Float64, Int32},1}  = [FESpace{FETypes[j]}(xgrid) for j in 1:3]

    ## in/outflow regions
    testgrid   = NumCompressibleFlows.grid(gridtype; nref = 1)
    rinflow    = inflow_regions(velocitytype, gridtype)
    routflow   = outflow_regions(velocitytype, gridtype)
    rhom       = setdiff(unique!(testgrid[BFaceRegions]), union(rinflow, routflow))
    @info rinflow, routflow, rhom

    ## define Stokes problem# name depending on convection type
    if convectiontype === NoConvection
        pdname = "Compressible Stokes problem"
    elseif convectiontype === OseenConvection
        pdname = "Compressible Oseen problem"
    elseif convectiontype === RotationForm
        pdname = "Compressible Navier-Stokes problem (rotation form)"
    else
        pdname = "Compressible Navier-Stokes problem"
    end
    PD = ProblemDescription(pdname)
    assign_unknown!(PD, u)
    assign_operator!(PD, BilinearOperator([grad(u)]; factor = μ, store = true, kwargs...))
    assign_operator!(PD, BilinearOperator([div_u]; factor = λ, store = true, kwargs...))

    if coriolistype !== NoCoriolis
        assign_operator!(PD, LinearOperator(
            kernel_coriolis_linearoperator!(coriolistype), [id_u],
            [id_u, id(ϱ)]; quadorder = 2*order + 1, factor = -1, kwargs...))
    end

    ## prepare upwind flux storage
    FluxIntegrator = ItemIntegrator([normalflux(1)]; quadorder = 2, entities = ON_FACES)
    fluxes = zeros(Float64, 1, size(xgrid[FaceCells], 2))

    ## add convection term (dispatched by convectiontype)
    _add_convection!(PD, u, ϱ, id_u, grad, div_u, u!, ϱ!, convectiontype, order, xgrid, stab1, FluxIntegrator, fluxes, FES, kwargs...)

    ## boundary data and source terms
    assign_operator!(PD, LinearOperator(
        eos!(eostype), [div(u)], [id(ϱ)]; factor = c, quadorder = order + 1, kwargs...))
    if length(rhom) > 0
        assign_operator!(PD, HomogeneousBoundaryData(u; regions = rhom, kwargs...))
    end
    if length(rinflow) > 0 || length(routflow) > 0
        assign_operator!(PD, InterpolateBoundaryData(
            u, u!; bonus_quadorder, regions = union(rinflow, routflow), kwargs...))
    end
    if kernel_rhs! !== nothing
        assign_operator!(PD, LinearOperator(
            kernel_rhs!, [id_u]; factor = 1, store = true,
            bonus_quadorder, kwargs...))
    end
    assign_operator!(PD, LinearOperator(
        kernel_gravity!, [id_u], [id(ϱ)]; factor = 1,
        bonus_quadorder, kwargs...))

    ## FVM for continuity equation
    @info "timestep = $τ"
    PDT = ProblemDescription("continuity equation")

    assign_unknown!(PDT, ϱ)
    if order > 1
        assign_operator!(PDT, BilinearOperator(
            kernel_continuity!, [grad(ϱ)], [id(ϱ)], [id(u)];
            quadorder = 2 * order, factor = -1, kwargs...))
    end
    if pressure_stab > 0
        assign_operator!(PDT, BilinearOperator(
            stab_kernel!, [jump(id(ϱ))], [jump(id(ϱ))], [id(u)];
            entities = ON_IFACES, factor = pressure_stab, kwargs...))
    end

    ## operators for implicit Euler time stepping
    assign_operator!(PDT, BilinearOperator(
        [id(ϱ)]; quadorder = 2 * (order - 1), factor = 1, store = true, kwargs...))
    assign_operator!(PDT, LinearOperator(
        [id(ϱ)], [id(ϱ)]; quadorder = 2 * (order - 1), factor = 1, kwargs...))

    D = nothing
    brho = nothing
    one_vector = nothing
    rowsums = nothing
    sol = nothing
    rho_mean = M_exact / sum(xgrid[CellVolumes])
    

    """
    callback!(A, b, args; assemble_matrix = true, assemble_rhs = true, time = 0, kwargs...)

    Pseudo-time-step callback for the continuity equation.  Assembled at each
    stationarity iteration:

    * updates upwind mass matrix `D` discretizing `∇·(ϱu)` on interior faces (DG)
    * update inflow source vector `brho` and outflow matrix correction from boundary data
    * updates jump stabilisation on interior faces (`stab1`) and mean-density
      stabilisation pulling toward `rho_mean` (`stab2`)
    * accumulates `A += τ·D`, `b += τ·brho` → implicit step `(I + τD)ϱ = b`
      with time-step `tau = min(V_cell / ‖rowsum(D)‖) / 2`
    """
    function callback!(A, b, args; assemble_matrix = true,
                       assemble_rhs = true, time = 0, kwargs...)

        fill!(D.entries.cscmatrix.nzval, 0)
        fill!(brho.entries, 0)
        
        if upwindtype === StandardUpwind
            ## computes integrals of u ⋅ n on all faces and use them for upwinding
            fill!(fluxes, 0)
            evaluate!(fluxes, FluxIntegrator, [args[1]])
            view(fluxes,:) ./= xgrid[FaceVolumes]

            assemble!(D, BilinearOperatorDG(
                kernel_upwind2!, [jump(id(1))],
                [this(id(1)), other(id(1))];
                factor = 1, quadorder = order, entities = ON_IFACES, params = [fluxes]))
        elseif upwindtype === PointwiseUpwind
            ## computes u ⋅ n at quadrature points and use them for upwinding
            assemble!(D, BilinearOperatorDG(kernel_upwind!, [jump(id(1))],
                [this(id(1)), other(id(1))], [id(1)];
                factor = 1, quadorder = order+1, entities = ON_IFACES), sol)
        end
        
        ## check diagonal dominance
        mul!(rowsums, D.entries, one_vector)

        ## adjust τ if needed
        tau = min(extrema(abs.(xgrid[CellVolumes]./rowsums))[1]/2, τ)
        print(" (τ = $τ) ")

        if length(rinflow) > 0
            assemble!(brho, LinearOperatorDG(
                kernel_inflow!(u!, ϱ!), [id(1)];
                factor = -1, bonus_quadorder, entities = ON_BFACES,
                regions = rinflow, kwargs...))
        end
        if length(routflow) > 0
            assemble!(D, BilinearOperatorDG(
                kernel_outflow!(u!), [id(1)];
                factor = 1, bonus_quadorder, entities = ON_BFACES,
                regions = routflow, kwargs...))
        end

        ## density jump stabilisation
        if stab1[2] > 0
            assemble!(D, BilinearOperatorDG(
                density_jump_stab_kernel!(stab1[1], γ),
                [jump(id(1))], [jump(id(1))], [average(id(1))];
                factor = stab1[2]*2, entities = ON_IFACES, bonus_quadorder = order, kwargs...), sol)
        end

        ## density mean stabilisation
        if stab2[2] > 0
            hmean = sum(xgrid[FaceVolumes]) / length(xgrid[FaceVolumes])
            assemble!(D, BilinearOperator(
                [id(1)]; factor = hmean^stab2[1]*stab2[2], kwargs...))
            assemble!(brho, LinearOperator(
                [id(1)]; factor = hmean^stab2[1]*rho_mean*stab2[2], kwargs...))
        end

        ExtendableFEMBase.add!(A, D.entries; factor = tau)
        b .+= tau * brho.entries
    end
    assign_operator!(PDT, CallbackOperator(
        callback!, [u]; linearized_dependencies = [ϱ, ϱ],
        modifies_rhs = length(rinflow) + length(routflow) > 0, kwargs...,
        name = "upwind matrix D scaled by tau"))

    EnergyIntegrator = ItemIntegrator(
        energy_kernel!, [id(u)]; resultdim = 1,
        quadorder = 2 * (order + 1), kwargs...)

    ## prepare error calculation
    MassIntegrator = ItemIntegrator([id(ϱ)]; resultdim = 1, kwargs...)

    sol = nothing

    ## finite element spaces and solution vector
    sol  = FEVector(FES; tags = [u, ϱ, p])
    
    ## initial guess
    if initial_values == :stokes
        fill!(sol[ϱ], M)
        @info "starting with constant density (-> first momentum update is Stokes like solution)..."
    elseif initial_values == :interpolate
        @info "interpolating exact solution for initial values..."
        interpolate!(sol[u], u!; bonus_quadorder)
        interpolate!(sol[ϱ], ϱ!; bonus_quadorder)
    else
        @error "Choose a valid initial value option: :stokes or :interpolate"
    end
    

    D        = FEMatrix(FES[2], FES[2])
    brho     = FEVector(FES[2])
    one_vector = ones(Float64, size(D.entries, 1))
    rowsums  = zeros(Float64, size(D.entries, 1))

    M_start  = sum(evaluate(MassIntegrator, sol))
    nonlinear_convection = !(convectiontype === NoConvection || convectiontype === OseenConvection)
        maxiterations_momentum = subiterations_momentum == :auto ?  (nonlinear_convection ? 1 : 1) : subiterations_momentum
    SC1 = SolverConfiguration(PD; init = sol, maxiterations = maxiterations_momentum,
        target_residual, constant_matrix = true, kwargs...)
    SC2 = SolverConfiguration(PDT; init = sol, maxiterations = 1,
        target_residual, kwargs...)
   
   if data["no_continuity_update"]
        @warn "continuity update is switched off, density stays at initial value" 
        sol, nits = iterate_until_stationarity([SC1];
            energy_integrator = EnergyIntegrator, maxsteps, init = sol, kwargs...)
   else
        sol, nits = iterate_until_stationarity([SC1, SC2];
            energy_integrator = EnergyIntegrator, maxsteps, init = sol, kwargs...)
   end

    ## calculate mass conservation
    Mend    = sum(evaluate(MassIntegrator, sol))
    @info "M_exact/M_start/M_end/difference = $(M_exact)/$M_start/$Mend/$(M_start-Mend)"

    ## save data
    data["ndofs"]    = length(sol.entries)
    data["nits"]     = nits
    data["solution"] = sol
    data["grid"]     = xgrid
    data["unknown_u"] = u
    data["unknown_ϱ"] = ϱ

    # residuals printing
    data["res_momentum"] = residual(SC1)
    data["res_continuity"] = data["no_continuity_update"] ? 0 : residual(SC2)

    ## plot unicode plot
    if length(sol.entries) < 1e5
        ExtendableFEM.plot([id(u), id(ϱ)], sol; Plotter = UnicodePlots)
    end

    return data
end

# ==============================================================================
# Error computation
# ==============================================================================

function compute_errors(config; force_recompute = false, compare_incompressible = true, kwargs...)
    fpath = filename(config) * ".jld2"
    @info "loading data from $fpath"
    data = wload(fpath)
    # problem parameters
    μ      = data["μ"]
    λ      = data["λ"]
    eostype= data["eostype"]
    γ      = gamma(eostype)
    c      = data["c"]
    M      = data["M"]
    ufac   = data["ufac"]

    if haskey(data, "solution")
        sol = data["solution"]
        u   = sol.tags[1]
        ϱ   = sol.tags[2]
        repair_grid!(sol[u].FES.xgrid)
    else
        @error "solution not found in data, cannot compute errors"
        return nothing
    end

    # data of the problem
    reconstruct    = data["reconstruct"]
    order          = data["order"]
    velocitytype   = data["velocitytype"]
    densitytype    = data["densitytype"]
    eostype        = data["eostype"]
    gridtype       = data["gridtype"]
    pressure_in_f  = data["pressure_in_f"]
    others_in_f = data["others_in_f"]
    convectiontype = data["convectiontype"]
    coriolistype   = data["coriolistype"]

    ϱ!, kernel_gravity!, kernel_rhs!, u!, ∇u! =
        prepare_data(velocitytype, densitytype, eostype;
                     others_in_f, pressure_in_f, M, c, μ, λ, γ,
                     ufac, kwargs...)
    if compare_incompressible && (force_recompute || !haskey(data, "Error(L2,u-u_inc)"))
        if !haskey(data, "incompressible_solution")
            run_incompressible!(data; kwargs...)
        end
        sol_inc = data["incompressible_solution"]
        diff_kernel = (result, input, qpinfo) -> (result .= (input[1] - input[3])^2 + (input[2] - input[4])^2)
        DiffIntegrator = ItemIntegrator(diff_kernel, [id(1), id(2)]; quadorder = 2 * (data["order"]+1), kwargs...)
        error_diff = evaluate(DiffIntegrator, [sol[u], sol_inc[1]])
        data["Error(L2,u-u_inc)"] = sqrt(sum(view(error_diff, :)))
        
    end
    if force_recompute || !haskey(data, "Error(H1,u0)")
        @info "computing divergence-free part of u - u_h"

        ## compute error of divergence-free part by solving
        ## incompressible Stokes problem with rhs (∇(u-uh), ∇v)

        ## compile-time pre-compilation of lazy_interpolate on a coarse grid
        compile_grid = simplexgrid(0:0.5:1, 0:0.5:1)
        cg_refined = barycentric_refine(compile_grid)
        test_fe1 = FEVector(FESpace{eltype(sol[u].FES)}(compile_grid); tags = [u])
        test_fe2 = FEVector(FESpace{H1Pk{2,2,2}}(cg_refined); tags = [u])
        @time lazy_interpolate!(test_fe2[1], test_fe1, [id(u)]; quadorder = 0)

        ## prepare FESpace
        xgrid      = sol[u].FES.xgrid
        xgridSP    = barycentric_refine(xgrid)
        FES_SP     = [FESpace{H1Pk{2,2,2}}(xgridSP),
                      FESpace{H1Pk{1,2,1}}(xgridSP; broken = true)]

        ## unknowns of divergence-free projection
        uzero = Unknown("u"; name = "Stokes projection")
        pzero = Unknown("p"; name = "pressure of projection")

        solSP = FEVector(FES_SP; tags = [uzero, pzero])
        append!(solSP, FES_SP[1]; tag = u)
        @time interpolate_BR_to_P2!(solSP[u], sol[u])

        ## velocity-update problem for iterated penalty method
        PDSP_u = ProblemDescription("Stokes projection problem - u update")
        assign_unknown!(PDSP_u, uzero)
        β = 1e+3  # div-penalty
        assign_operator!(PDSP_u, BilinearOperator([grad(uzero)]; factor = 1, store = true, kwargs...))
        assign_operator!(PDSP_u, BilinearOperator([div(uzero)]; store = true, factor = β, kwargs...))
        assign_operator!(PDSP_u, LinearOperator([grad(uzero)], [grad(u)]; factor = -1, kwargs...))
        assign_operator!(PDSP_u, LinearOperator(∇u!, [grad(uzero)]; kwargs...))
        assign_operator!(PDSP_u, LinearOperator([div(uzero)], [id(pzero)]; factor = 1, kwargs...))
        assign_operator!(PDSP_u, HomogeneousBoundaryData(uzero; regions = 1:4, kwargs...))

        ## pressure-update problem
        PDSP_p = ProblemDescription("Stokes projection problem - p update")
        assign_unknown!(PDSP_p, pzero)
        assign_operator!(PDSP_p, BilinearOperator([id(pzero)]; store = true, kwargs...))
        assign_operator!(PDSP_p, LinearOperator(div_projection!, [id(pzero)],
            [id(pzero), div(uzero)]; params = [β], factor = 1, kwargs...))

        ## run iterated penalty method
        SC1 = SolverConfiguration(PDSP_u; init = solSP, maxiterations = 1,
            target_residual = 1.0e-10, constant_matrix = true, kwargs...)
        SC2 = SolverConfiguration(PDSP_p; init = solSP, maxiterations = 1,
            target_residual = 1.0e-10, constant_matrix = true, kwargs...)
        solSP, nits = iterate_until_stationarity([SC1, SC2]; init = solSP, kwargs...)
        @info "converged after $nits iterations"

        ## norm of div-free projection of the error
        error0 = evaluate(L2NormIntegrator([grad(uzero)]), solSP)
        data["Error(H1,u0)"] = sqrt(
            sum(error0[1, :]) + sum(error0[2, :]) +
            sum(error0[3, :]) + sum(error0[4, :]))
        @info data["Error(H1,u0)"]
    else
        @info "skipping divergence-free error (already computed)"
    end

    if force_recompute || !haskey(data, "Error(L2,u)")
        @info "computing errors of u, ϱ and ϱu"
        order = data["order"]
        ErrorIntegratorExact = ItemIntegrator(
            exact_error!(u!, ∇u!, ϱ!), [id(u), grad(u), id(ϱ)];
            resultdim = 9, quadorder = 10, kwargs...)
        error = evaluate(ErrorIntegratorExact, sol)
        data["Error(L2,u)"]  = sqrt(sum(error[1, :]) + sum(error[2, :]))
        data["Error(H1,u)"]  = sqrt(sum(error[3, :]) + sum(error[4, :]) +
                                     sum(error[5, :]) + sum(error[6, :]))
        data["Error(L2,ϱ)"]  = sqrt(sum(error[7, :]))
        data["Error(L2,ϱu)"] = sqrt(sum(error[8, :]) + sum(error[9, :]))

        @assert reconstruct in [:none, :RT, :BDM]
        if data["reconstruct"] !== :none
            if order == 1
                ReconstSpace = reconstruct == :RT ? HDIVRT0{2} : reconstruct == :BDM ? HDIVBDM1{2} : nothing
                id_u    = ReconstSpace !== nothing ? apply(u, Reconstruct{ReconstSpace, Identity}) : id(u)
                div_u   = ReconstSpace !== nothing ? apply(u, Reconstruct{ReconstSpace, Divergence}) : div(u)
            elseif order == 2
                ReconstSpace = reconstruct == :RT ? HDIVRT1{2} : reconstruct == :BDM ? HDIVBDM2{2} : nothing
                id_u    = ReconstSpace !== nothing ? apply(u, Reconstruct{ReconstSpace, Identity}) : id(u)
                div_u   = ReconstSpace !== nothing ? apply(u, Reconstruct{ReconstSpace, Divergence}) : div(u)
            end
            ErrorIntegratorExactReconstruct = ItemIntegrator(
                exact_error!(u!, ∇u!, ϱ!), [id_u, grad(u), id(ϱ)];
                resultdim = 9, quadorder = 10, kwargs...)
            error = evaluate(ErrorIntegratorExactReconstruct, sol)
            data["Error(L2,uR)"]  = sqrt(sum(error[1, :]) + sum(error[2, :]))
        end
    else
        @info "skipping error calculation (already computed)"
    end

    ## save
    fpath = filename(data) * ".jld2"
    @info "saving data to $fpath"
    wsave(fpath, data)

    return data
end

# ==============================================================================
# Plotting infrastructure (filenames, directories, single solution plot)
# ==============================================================================

quickactivate(@__DIR__, "NumCompressibleFlows")
for p in [
    "compressible_stokes_paper_repeat/convergence_history",
    #"compressible_stokes_paper_repeat/penalty_convergence_history",
    "compressible_stokes_paper_repeat/parameter_studies_μ",
    "compressible_stokes_paper_repeat/parameter_studies_γ",
    "compressible_stokes_paper_repeat/parameter_studies_c",
    "compressible_stokes_paper_repeat/parameter_studies_cμ",
    "compressible_stokes_paper_repeat/parameter_studies_c1",
    "compressible_stokes_paper_repeat/parameter_studies_α",
    "compressible_stokes_paper_repeat/parameter_studies_c2",
]
    mkpath(plotsdir(p))
end

"""
    filename_plots(data; prefix = "", free_parameter = "")

Build a filename for a plot using `savename` with parameters selected by
`free_parameter`.  Each free parameter freezes a different subset of the
remaining parameters for inclusion in the savename. Returns a relative
path string ending in `.png`.
"""
function filename_plots(data; prefix = "", free_parameter = "")
    μ = data["μ"]
    c = data["c"]
    EOSType = data["eostype"]
    γ = gamma(EOSType)
    stab1 = data["stab1"]
    stab2 = data["stab2"]
    ϵ = 1 - stab1[1]
    α = stab2[1]
    c1 = stab1[2]
    c2 = stab2[2]
    nrefs = data["nrefs"]
    reconstruct = data["reconstruct"]
    convectiontype = string(data["convectiontype"])
    pressure_in_f = data["pressure_in_f"]
    upwindtype = data["upwindtype"]

    # Select which params go into savename depending on free_parameter
    essential_params = if free_parameter == "μ"
        @dict c γ ϵ c1 nrefs reconstruct convectiontype upwindtype
    elseif free_parameter == "γ"
        @dict μ c ϵ c1 nrefs reconstruct convectiontype upwindtype
    elseif free_parameter == "c"
        @dict μ γ ϵ c1 nrefs reconstruct convectiontype upwindtype
    elseif free_parameter == "cμ"
        @dict γ ϵ c1 nrefs reconstruct convectiontype upwindtype
    elseif free_parameter == "c1"
        @dict μ c γ ϵ nrefs reconstruct convectiontype upwindtype
    elseif free_parameter in ("c2", "α")
        @dict μ c γ ϵ c1 nrefs reconstruct convectiontype upwindtype
    else
        @dict μ c γ ϵ c1 nrefs reconstruct convectiontype pressure_in_f upwindtype
    end
    sname = savename(essential_params;
                     allowedtypes = (Real, String, SubString, Symbol,
                                     Tuple{Real, Real}))

    if free_parameter !== ""
    else
        sname = "plots/compressible_stokes_paper_repeat/convergence_history/" * sname * prefix * ".png"
    end

    return sname
end

"""
    plot_single(; Plotter = PyPlot, force = false, kwargs...)

Solve a single configuration and plot velocity (with quiver) and density.
Requires `unknown_u` and `unknown_ϱ` keys in the loaded data.
"""
function plot_single(; Plotter = PyPlot, force = false, kwargs...)
    data = load_data(; kwargs...)
    @debug "loading config" data
    data, ~ = produce_or_load(run_single, data, filename = filename, force = force)
    xgrid = data["grid"]
    sol = data["solution"]
    u = data["unknown_u"]
    ϱ = data["unknown_ϱ"]
    @debug "solution loaded" sol
    repair_grid!(xgrid)
    repair_grid!(sol[u].FES.xgrid)
    repair_grid!(sol[ϱ].FES.xgrid)

    ## plot
    pl = GridVisualizer(; Plotter = Plotter, layout = (1,2), clear = true,
        show = true, resolution = (1000, 500))
    scalarplot!(pl[1,1], xgrid, view(nodevalues(sol[u]; abs = true), 1, :),
        levels = 0, colorbarticks = 7, fontsize = 60)
    vectorplot!(pl[1,1], xgrid, eval_func_bary(PointEvaluator([id(u)], sol)),
        clear = false, fontsize = 60)
    scalarplot!(pl[1,2], xgrid, view(nodevalues(sol[ϱ]), 1, :), levels = 11,
        fontsize = 60)

    ## save
    scene = GridVisualize.reveal(pl)
    GridVisualize.save(filename_plots(data; prefix = "_Solutions"), scene;
        Plotter = Plotter)

    return data
end

# ==============================================================================
# Convergence history plot
# ==============================================================================

"""
    log_ticks(v; maxticks = 12)

Powers of ten covering the positive finite values in `v` (first tick below
the minimum, last tick above the maximum). If the range spans more than
`maxticks` decades, every k-th decade is used and the top decade is kept.
"""
function log_ticks(v; maxticks = 12)
    v = filter(x -> isfinite(x) && x > 0, collect(Float64, vec(v)))
    isempty(v) && return [1.0e-2, 1.0, 1.0e+2]
    lo = floor(Int, log10(minimum(v)))
    hi = ceil(Int, log10(maximum(v)))
    hi = max(hi, lo + 2)
    step = max(1, ceil(Int, (hi - lo + 1) / maxticks))
    ts = collect(lo:step:hi)
    ts[end] == hi || push!(ts, hi)
    return 10.0 .^ ts
end

"""
    plot_convergencehistory(; nrefs = 1:6, quantities = :default, xquantity = :ndofs,
        slopes = (1, 2), with_incompressible = false, kwargs...)

Plots the convergence history of the compressible solver. The plotted curves
are selected via `quantities`, a vector of symbols resolved against the
`CONV_QUANTITIES` registry (e.g. `[:L2u, :H1u, :L2ϱ, :H1u0, :H1u1, :nits]`).
`:default` gives `collect(DEFAULT_QUANTITIES)`, `:all` the whole registry.
One-off quantities can be passed inline as NamedTuples with fields `name`,
`data` (dict key or function `data -> value`) and `label`.

Quantities with `incompressible = true` (`:H1u_inc`, `:L2u_inc`, `:L2u_diff`)
trigger the incompressible reference solver `run_incompressible!` via
`compute_errors` and are plotted against the DOF count of the incompressible
system. For backward compatibility, `with_incompressible = true` (deprecated)
appends them to the `:default` selection.

`xquantity` selects the x-axis (`:ndofs` or `:h = ndofs^(-1/2)`), `slopes`
adds reference lines O(h^k).
"""
function plot_convergencehistory(; nrefs = 1:6, Plotter = Plots, force = false, force_recompute = false,
        quantities = :default, with_incompressible = false, xquantity = :ndofs, slopes = (1, 2), kwargs...)

    @info "Plotting convergence history for nrefs = $nrefs, quantities = $quantities, xquantity = $xquantity, slopes = $slopes..."
    ## resolve quantity selection against registry
    if quantities === :default
        qlist = collect(DEFAULT_QUANTITIES)
        with_incompressible && append!(qlist, (:H1u_inc, :L2u_inc, :L2u_diff))
    elseif quantities === :all
        qlist = collect(keys(CONV_QUANTITIES))
    else
        qlist = collect(quantities)
        if with_incompressible
            @warn "with_incompressible = true is ignored when quantities are given explicitly; add e.g. :H1u_inc, :L2u_inc, :L2u_diff to quantities instead"
        end
    end
    isempty(qlist) && error("quantities must not be empty")
    unknown = [q for q in qlist if q isa Symbol && !haskey(CONV_QUANTITIES, q)]
    isempty(unknown) || error("unknown quantities $unknown; available: $(collect(keys(CONV_QUANTITIES)))")
    entries = [(q isa Symbol ? q : get(q, :name, :custom), q isa Symbol ? CONV_QUANTITIES[q] : q) for q in qlist]
    for (name, e) in entries
        hasproperty(e, :data) && hasproperty(e, :label) || error("quantity $name needs fields :data and :label")
    end
    needs_inc = any(get(e, :incompressible, false) for (_, e) in entries)

    data = load_data(; kwargs...)
    #@show data
    nl = length(nrefs)
    vals = [zeros(Float64, nl) for _ in entries]
    NDoFs = zeros(Int, nl)
    NDoFsInc = zeros(Int, nl)
    Residuals = zeros(Float64, nl, 2)

    for (j, lvl) in enumerate(nrefs)
        _data = deepcopy(data)
        _data["nrefs"] = lvl
        _data, ~ = safe_produce_or_load(_data; force = force)
        NDoFs[j] = _data["ndofs"]
        _data = compute_errors(_data; force_recompute = force_recompute, compare_incompressible = needs_inc)

        for (k, (_, e)) in enumerate(entries)
            vals[k][j] = conv_value(e, _data)
        end
        if needs_inc
            NDoFsInc[j] = haskey(_data, "incompressible_solution") ? length(_data["incompressible_solution"].entries) : NDoFs[j]
        end

        if haskey(_data, "res_momentum")
            Residuals[j,1] = _data["res_momentum"]
            Residuals[j,2] = _data["res_continuity"]
        else
            @warn "residual information not found, consider rerunning"
            Residuals[j,1] = 1e30
            Residuals[j,2] = 1e30
        end

        @show Residuals

        ## console table of the first up to four selected quantities
        sel = min(4, length(entries))
        print_convergencehistory(NDoFs[:], hcat(vals[1:sel]...); X_to_h = X -> X.^(-1/2),
            ylabels = [string(entries[k][2].label) for k in 1:sel],
            xlabel = xquantity === :h ? "h" : "ndof", latex_mode = true)
    end

    ## plot
    #Plotter.rc("font", size=20)
    if !(xquantity in (:ndofs, :h))
        error("xquantity must be :ndofs or :h")
    end
    hvals = NDoFs[:].^(-1/2)
    xof = xquantity === :h ? hvals : Float64.(NDoFs[:])

    ## collect all curves first, so that the axis ticks can cover the plotted data
    ## (built via vcat instead of push! into a growing vector: quantity curves and slope
    ## reference lines are different concrete NamedTuple types, which breaks push! growth)
    series = vcat(
        [ (; x = Float64.(get(e, :xinc, false) ? (xquantity === :h ? NDoFsInc[:].^(-1/2) : NDoFsInc[:]) : xof),
            y = vals[k], label = e.label, style = get(e, :style, NamedTuple()))
          for (k, (_, e)) in enumerate(entries) ],
        [ (; x = xof, y = (m == 1 ? 0.5 : m == 2 ? 1e+1 : 1.0) .* hvals.^m,
            label = m == 1 ? L"\mathcal{O}(h)" : latexstring("\\mathcal{O}(h^{$m})"),
            style = (linestyle = :dash, color = :gray, marker = :none))
          for m in slopes ],
    )

    ## axis ticks as powers of ten covering all plotted values
    yticks = log_ticks(reduce(vcat, [s.y for s in series]))
    xticks = log_ticks(reduce(vcat, [s.x for s in series]))
    xlabelv = xquantity === :h ? "mesh size h" : "degrees of freedom"

    Plotter.plot(; show = true, size = (1000,1000), margin = 1Plots.cm, legendfontsize = 20, tickfontsize = 22, guidefontsize = 26, grid=true)
    for s in series
        Plotter.plot!(s.x, s.y; xscale = :log10, yscale = :log10, linewidth = 3,
            marker = :circle, markersize = 5, label = s.label, s.style...)
    end

    Plotter.plot!(; legend = :bottomleft, xtick = xticks, yticks = yticks, ylim = (yticks[1]/2, 2*yticks[end]), xlim = (xticks[1], xticks[end]), xlabel = xlabelv,gridalpha = 0.7,grid=true, background_color_legend = RGBA(1,1,1,0.7))
    ## save
    prefix = if quantities === :default && !with_incompressible
        ""
    elseif quantities === :all
        "_all_quantities"
    elseif with_incompressible
        "_with_incompressible"
    else
        "_" * join([string(n) for (n, _) in entries], "-")
    end
    Plotter.savefig(filename_plots(data; prefix))
end

# ==============================================================================
# Parameter study plots
# ==============================================================================

function plot_parameter_study_viscosity(; nrefs = [3], μ = [1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100,1000], Plotter = Plots, kwargs...)
    nrefs = nrefs isa AbstractVector ? nrefs : [nrefs]
    μ = μ isa AbstractVector ? μ : [μ]
    data = load_data(; kwargs...)
    @debug "loading config" data
    L2u = zeros(Float64, length(μ), length(nrefs))
    H1u = zeros(Float64, length(μ), length(nrefs))
    L2ϱ = zeros(Float64, length(μ), length(nrefs))

    for n = 1 : length(nrefs)
        data["nrefs"] = nrefs[n]
        for j = 1 : length(μ)
            data["μ"] = μ[j]
            data, ~ = safe_produce_or_load(data)
            L2u[j,n] = data["Error(L2,u)"]
            H1u[j,n] = data["Error(H1,u)"]
            L2ϱ[j,n] = data["Error(L2,ϱ)"]
        end
    end

    ## plot
    labels = [" level $n" for n in nrefs]
    yticks = [1e-5,1e-4,1e-3,1e-2,1e-1,1,10]
    xticks = μ
    Plotter.plot(; show = true, size = (1600,1000), margin = 1Plots.cm, legendfontsize = 20, tickfontsize = 16, guidefontsize = 22)
    for n = 1 : length(nrefs)
        Plotter.plot!(μ, L2ϱ[:,n]; xscale = :log10, yscale = :log10, linewidth = 3, marker = :circle, markersize = 5, label = L"|| {ϱ}-ϱ_h \, || \mathrm{level} = %$(nrefs[n])")
    end
    for n = 1 : length(nrefs)
        Plotter.plot!(μ, L2u[:,n]; xscale = :log10, yscale = :log10, linewidth = 3, marker = :circle, markersize = 5, label = L"||\mathbf{u} - \mathbf{u}_h \, || \mathrm{level} = %$(nrefs[n]) ")    
    end
    Plotter.plot!(; legend = :topright, xtick = xticks, yticks = yticks, ylim = (yticks[1]/2, 2*yticks[end]), xlabel = "μ", gridalpha = 0.5, grid=true)
        
    ##
    print_table(μ, L2u; xlabel = "μ", ylabels = "|| u - u_h || ".* labels)
        
    ## save
    Plotter.savefig(filename_plots(data; free_parameter = "μ"))
end

function plot_parameter_study_gamma(; nrefs = [3], γ = [1,1e+1,1e+2,1e+3], Plotter = Plots, kwargs...)
    nrefs = nrefs isa AbstractVector ? nrefs : [nrefs]
    γ = γ isa AbstractVector ? γ : [γ]
    data = load_data(; kwargs...)
    @debug "loading config" data
    L2u = zeros(Float64, length(γ), length(nrefs))
    H1u = zeros(Float64, length(γ), length(nrefs))
    L2ϱ = zeros(Float64, length(γ), length(nrefs))

    for n = 1 : length(nrefs)
        data["nrefs"] = nrefs[n]
        for j = 1 : length(γ)
            data["γ"] = γ[j]
            data, ~ = safe_produce_or_load(data)
            L2u[j,n] = data["Error(L2,u)"]
            H1u[j,n] = data["Error(H1,u)"]
            L2ϱ[j,n] = data["Error(L2,ϱ)"]
        end
    end

    ## plot
    labels = [" level $n" for n in nrefs]
    yticks = [1e-5,1e-4,1e-3,1e-2,1e-1,1,10]
    xticks = γ
    Plotter.plot(; show = true, size = (1600,1000), margin = 1Plots.cm, legendfontsize = 20, tickfontsize = 16, guidefontsize = 22)
    for n = 1 : length(nrefs)
        Plotter.plot!(γ, L2ϱ[:,n]; xscale = :log10, yscale = :log10, linewidth = 3, marker = :circle, markersize = 5, label = L"|| {ϱ}-ϱ_h \, || \mathrm{level} = %$(nrefs[n])")
    end
    for n = 1 : length(nrefs)
        Plotter.plot!(γ, L2u[:,n]; xscale = :log10, yscale = :log10, linewidth = 3, marker = :circle, markersize = 5, label = L"||\mathbf{u} - \mathbf{u}_h \, || \mathrm{level} = %$(nrefs[n]) ")    
    end
    Plotter.plot!(; legend = :topright, xtick = xticks, yticks = yticks, ylim = (yticks[1]/2, 2*yticks[end]), xlabel = "γ", gridalpha = 0.5, grid=true)
        
    ##
    print_table(γ, L2u; xlabel = "γ", ylabels = "|| u - u_h || ".* labels)
        
    ## save
    Plotter.savefig(filename_plots(data; free_parameter = "γ"))
end

function plot_parameter_study_mach_number(; nrefs = [3], c = [1,1e+1,1e+2,1e+3,1e+4,1e+5], Plotter = Plots, kwargs...)
    nrefs = nrefs isa AbstractVector ? nrefs : [nrefs]
    c = c isa AbstractVector ? c : [c]
    data = load_data(; kwargs...)
    @debug "loading config" data
    L2u = zeros(Float64, length(c), length(nrefs))
    H1u = zeros(Float64, length(c), length(nrefs))
    L2ϱ = zeros(Float64, length(c), length(nrefs))

    for n = 1 : length(nrefs)
        data["nrefs"] = nrefs[n]
        for j = 1 : length(c)
            data["c"] = c[j]
            data, ~ = safe_produce_or_load(data)
            L2u[j,n] = data["Error(L2,u)"]
            H1u[j,n] = data["Error(H1,u)"]
            L2ϱ[j,n] = data["Error(L2,ϱ)"]
        end
    end

    ## plot
    labels = [" level $n" for n in nrefs]
    yticks = [1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10,1e+1,1e+2]
    xticks = c
    Plotter.plot(; show = true, size = (1600,1000), margin = 1Plots.cm, legendfontsize = 20, tickfontsize = 16, guidefontsize = 22)
    for n = 1 : length(nrefs)
        Plotter.plot!(c, L2ϱ[:,n]; xscale = :log10, yscale = :log10, linewidth = 3, marker = :circle, markersize = 5, label = L"|| {ϱ}-ϱ_h \, || \mathrm{level} = %$(nrefs[n])")
    end
    for n = 1 : length(nrefs)
        Plotter.plot!(c, L2u[:,n]; xscale = :log10, yscale = :log10, linewidth = 3, marker = :circle, markersize = 5, label = L"||\mathbf{u} - \mathbf{u}_h \, || \mathrm{level} = %$(nrefs[n]) ")    
    end
    Plotter.plot!(; legend = :topright, xtick = xticks, yticks = yticks, ylim = (yticks[1]/2, 2*yticks[end]), xlabel = L"$c_M", gridalpha = 0.5, grid=true)
        
    ##
    print_table(c, L2u; xlabel = "c", ylabels = "|| u - u_h || ".* labels)
    print_table(c, L2ϱ; xlabel = "c", ylabels = "|| ϱ - ϱ_h || ".* labels)
        
    ## save
    Plotter.savefig(filename_plots(data; free_parameter = "c"))
end

function plot_parameter_study_mach_viscosity(; nrefs = 3,c = [1,1e+1,1e+2,1e+3,1e+4,1e+5,1e+6], μ = [1e-4,1e-3,1e-2,1e-1,1] , Plotter = Plots, kwargs...)
    c = c isa AbstractVector ? c : [c]
    μ = μ isa AbstractVector ? μ : [μ]
    data = load_data(; kwargs...)
    data["nrefs"] = nrefs
    @debug "loading config" data
    L2u = zeros(Float64, length(c), length(μ))
    H1u = zeros(Float64, length(c), length(μ))
    L2ϱ = zeros(Float64, length(c), length(μ))

    for n = 1 : length(μ)
        data["μ"] = μ[n]
        for j = 1 : length(c)
            data["c"] = c[j]
            data, ~ = safe_produce_or_load(data)
            L2u[j,n] = data["Error(L2,u)"]
            H1u[j,n] = data["Error(H1,u)"]
            L2ϱ[j,n] = data["Error(L2,ϱ)"]
        end
    end

    ## plot
    labels = [" μ =  $μk" for μk in μ]
    yticks = [1e-12,1e-11,1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1e+0,1e+1,1e+2]
    xticks = c
    Plotter.plot(; show = true, size = (1600,1000), margin = 1Plots.cm, legendfontsize = 20, tickfontsize = 16, guidefontsize = 22)
    for n = 1 : length(μ)
        Plotter.plot!(c, L2u[:,n]; xscale = :log10, yscale = :log10, linewidth = 3, marker = :circle, markersize = 5, label = L"||\mathbf{u} - \mathbf{u}_h \, || \mathrm{μ} = %$(μ[n]) ")
    end
    Plotter.plot!(; legend = :topright, xtick = xticks, yticks = yticks, ylim = (yticks[1]/2, 2*yticks[end]), xlabel = L"c_\mathrm{Ma}", gridalpha = 0.5, grid=true)

    ##
    print_table(c, L2u; xlabel = "c", ylabels = "|| u - u_h || ".* labels)

    ## save
    Plotter.savefig(filename_plots(data; free_parameter = "cμ"))
end

function plot_parameter_study_stab1(;  nrefs = [3,4,5],c1 = [1e-5,1e-4,1e-3,1e-2,1e-1,1], Plotter = Plots, kwargs...)
    nrefs = nrefs isa AbstractVector ? nrefs : [nrefs]
    c1 = c1 isa AbstractVector ? c1 : [c1]
    data = load_data(; kwargs...)
    @debug "loading config" data
    L2u = zeros(Float64, length(c1), length(nrefs))
    H1u = zeros(Float64, length(c1), length(nrefs))
    L2ϱ = zeros(Float64, length(c1), length(nrefs))

    for n = 1 : length(nrefs)
        data["nrefs"] = nrefs[n]
        for j = 1 : length(c1)
                data["stab1"] = (data["stab1"][1], c1[j])
                data, ~ = safe_produce_or_load(data)
                L2u[j,n] = data["Error(L2,u)"]
                H1u[j,n] = data["Error(H1,u)"]
                L2ϱ[j,n] = data["Error(L2,ϱ)"]
        end
    end

    ## plot
    labels = [" level $n" for n in nrefs]
    yticks = [1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10]
    xticks = c1
    Plotter.plot(; show = true, size = (1600,1000), margin = 1Plots.cm, legendfontsize = 20, tickfontsize = 16, guidefontsize = 22)
    for n = 1 : length(nrefs)
        Plotter.plot!(c1, H1u[:,n]; xscale = :log10, yscale = :log10, linewidth = 3, marker = :circle, markersize = 5, label = L"||∇(\mathbf{u} - \mathbf{u}_h) \,|| \mathrm{level} = %$(nrefs[n])")
        Plotter.plot!(c1, L2ϱ[:,n]; xscale = :log10, yscale = :log10, linewidth = 3, marker = :circle, markersize = 5, label = L"|| {ϱ}-ϱ_h \, || \mathrm{level} = %$(nrefs[n])")
    end
    for n = 1 : length(nrefs)
        Plotter.plot!(c1, L2u[:,n]; xscale = :log10, yscale = :log10, linewidth = 3, marker = :circle, markersize = 5, label = L"||\mathbf{u} - \mathbf{u}_h \, || \mathrm{level} = %$(nrefs[n]) ")    
    end
    Plotter.plot!(; legend = :bottomright, xtick = xticks, yticks = yticks, ylim = (yticks[1]/2, 2*yticks[end]), xlabel = "c1", gridalpha = 0.5, grid=true)
        
    ##
    print_table(c1, L2u; xlabel = "c1", ylabels = "|| u - u_h || ".* labels)
        
    ## save
    Plotter.savefig(filename_plots(data; free_parameter = "c1"))
end

function plot_parameter_study_stab2(;  nrefs = [3,4,5], c2  =[1e-4,1e-2,1,1e+2,1e+4], Plotter = Plots, kwargs...)
    nrefs = nrefs isa AbstractVector ? nrefs : [nrefs]
    c2 = c2 isa AbstractVector ? c2 : [c2]
    data = load_data(; kwargs...)
    @debug "loading config" data
    L2u = zeros(Float64, length(c2), length(nrefs))
    H1u = zeros(Float64, length(c2), length(nrefs))
    L2ϱ = zeros(Float64, length(c2), length(nrefs))

    for n = 1 : length(nrefs)
        data["nrefs"] = nrefs[n]
        for j = 1 : length(c2)
                data["stab2"] = (1.5, c2[j])
                data, ~ = safe_produce_or_load(data)
                L2u[j,n] = data["Error(L2,u)"]
                H1u[j,n] = data["Error(H1,u)"]
                L2ϱ[j,n] = data["Error(L2,ϱ)"]
        end
    end

    ## plot
    labels = [" level $n" for n in nrefs]
    yticks = [1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10]
    xticks = c2
    Plotter.plot(; show = true, size = (1600,1000), margin = 1Plots.cm, legendfontsize = 20, tickfontsize = 16, guidefontsize = 22)
    for n = 1 : length(nrefs)
        Plotter.plot!(c2, H1u[:,n]; xscale = :log10, yscale = :log10, linewidth = 3, marker = :circle, markersize = 5, label = L"||∇(\mathbf{u} - \mathbf{u}_h) \,|| \mathrm{level} = %$(nrefs[n])")
        Plotter.plot!(c2, L2ϱ[:,n]; xscale = :log10, yscale = :log10, linewidth = 3, marker = :circle, markersize = 5, label = L"|| {ϱ}-ϱ_h \, || \mathrm{level} = %$(nrefs[n])")
    end
    for n = 1 : length(nrefs)
        Plotter.plot!(c2, L2u[:,n]; xscale = :log10, yscale = :log10, linewidth = 3, marker = :circle, markersize = 5, label = L"||\mathbf{u} - \mathbf{u}_h \, || \mathrm{level} = %$(nrefs[n]) ")    
    end
    Plotter.plot!(; legend = :bottomright, xtick = xticks, yticks = yticks, ylim = (yticks[1]/2, 2*yticks[end]), xlabel = "c2", gridalpha = 0.5, grid=true)
        
    ##
    print_table(c2, L2u; xlabel = "c2", ylabels = "|| u - u_h || ".* labels)
        
    ## save
    Plotter.savefig(filename_plots(data; free_parameter = "c2"))
end

function plot_parameter_study_stab1_reconstruction(;  reconstruct = [true,false], c1 = [1e-5,1e-4,1e-3,1e-2,1e-1,1,1e1,1e2,1e3,1e4,1e5], Plotter = Plots, kwargs...)
    reconstruct = reconstruct isa AbstractVector ? reconstruct : [reconstruct]
    c1 = c1 isa AbstractVector ? c1 : [c1]
    data = load_data(; kwargs...)
    @debug "loading config" data
    L2u = zeros(Float64, length(c1), length(reconstruct))
    H1u = zeros(Float64, length(c1), length(reconstruct))
    L2ϱ = zeros(Float64, length(c1), length(reconstruct))

    for n = 1 : length(reconstruct)
        data["reconstruct"] = reconstruct[n]
        for j = 1 : length(c1)
                data["stab1"] = (data["stab1"][1], c1[j])
                data, ~ = safe_produce_or_load(data)
                L2u[j,n] = data["Error(L2,u)"]
                H1u[j,n] = data["Error(H1,u)"]
                L2ϱ[j,n] = data["Error(L2,ϱ)"]
        end
    end

    ## plot
    pi_names = [r ? "\\Pi = I_h^{\\mathrm{RT_0}}" : "\\Pi = \\mathrm{Id}" for r in reconstruct]
    col = [r ? colorant"#389826" : colorant"#CB3C33" for r in reconstruct]  # Julia green / red
    allvals = vcat(vec(H1u), vec(L2ϱ), vec(L2u))
    allvals = filter(x -> x > 0 && isfinite(x), allvals)
    ylo = 10.0^floor(log10(minimum(allvals)) - 0.5)
    yhi = 10.0^ceil(log10(maximum(allvals)) + 1.0)
    yticks = 10.0 .^ (floor(Int, log10(ylo)):ceil(Int, log10(yhi)))
    xticks = 10.0 .^ (-5:5)
    Plotter.plot(; show = true, size = (1200,900), margin = 1Plots.cm, legendfontsize = 14, tickfontsize = 16, guidefontsize = 20)
    # grouped by quantity; plot red (false) first, green (true) on top so both visible
    order = sortperm(reconstruct)  # false first, true second
    for n in order
        Plotter.plot!(c1, H1u[:,n]; xscale = :log10, yscale = :log10, linewidth = 3, linestyle = :dashdot, marker = :circle, markersize = 5, color = col[n], label = latexstring("||\\nabla(\\mathbf{u} - \\mathbf{u}_h)||,\\; ", pi_names[n]))
    end
    for n in order
        Plotter.plot!(c1, L2ϱ[:,n]; xscale = :log10, yscale = :log10, linewidth = 3, linestyle = :dot, marker = :diamond, markersize = 5, color = col[n], label = latexstring("||\\varrho - \\varrho_h||,\\; ", pi_names[n]))
    end
    for n in order
        Plotter.plot!(c1, L2u[:,n]; xscale = :log10, yscale = :log10, linewidth = 3, linestyle = :solid, marker = :square, markersize = 5, color = col[n], label = latexstring("||\\mathbf{u} - \\mathbf{u}_h||,\\; ", pi_names[n]))
    end
    Plotter.plot!(; legend = :topleft, xtick = xticks, yticks = yticks, ylim = (ylo, yhi), xlabel = L"c_s", gridalpha = 0.5, grid = true, background_color_legend = RGBA(1,1,1,0.7))

    ##
    labels = [" Pi=$(reconstruct[n] ? "RT0" : "Gamma")" for n in 1:length(reconstruct)]
    print_table(c1, L2u; xlabel = "c_s", ylabels = "|| u - u_h || " .* labels)

    ## save
    Plotter.savefig(filename_plots(data; free_parameter = "c1"))
end

function plot_parameter_study_alpha_reconstruction(;  reconstruct = [true,false], alpha = [0,5e-1,1,1e-0,1+5e-1,2e-0], Plotter = Plots, kwargs...)
    reconstruct = reconstruct isa AbstractVector ? reconstruct : [reconstruct]
    alpha = alpha isa AbstractVector ? alpha : [alpha]
    data = load_data(; kwargs...)
    @debug "loading config" data
    L2u = zeros(Float64, length(alpha), length(reconstruct))
    H1u = zeros(Float64, length(alpha), length(reconstruct))
    L2ϱ = zeros(Float64, length(alpha), length(reconstruct))

    for n = 1 : length(reconstruct)
        data["reconstruct"] = reconstruct[n]
        for j = 1 : length(alpha)
                data["stab1"] = (alpha[j]-1, data["stab1"][2])
                @info "α = $(alpha[j])"
                data, ~ = safe_produce_or_load(data)
                L2u[j,n] = data["Error(L2,u)"]
                H1u[j,n] = data["Error(H1,u)"]
                L2ϱ[j,n] = data["Error(L2,ϱ)"]
        end
    end

    ## plot
    pi_names = [r ? "\\Pi = I_h^{\\mathrm{RT_0}}" : "\\Pi = \\mathrm{Id}" for r in reconstruct]
    col = [r ? colorant"#389826" : colorant"#CB3C33" for r in reconstruct]  # Julia green / red
    allvals = vcat(vec(H1u), vec(L2ϱ), vec(L2u))
    allvals = filter(x -> x > 0 && isfinite(x), allvals)
    ylo = 10.0^floor(log10(minimum(allvals)) - 0.5)
    yhi = 10.0^ceil(log10(maximum(allvals)) + 1.0)
    yticks = 10.0 .^ (floor(Int, log10(ylo)):ceil(Int, log10(yhi)))
    Plotter.plot(; show = true, size = (1200,900), margin = 1Plots.cm, legendfontsize = 14, tickfontsize = 16, guidefontsize = 20)
    # grouped by quantity; plot red (false) first, green (true) on top so both visible
    order = sortperm(reconstruct)  # false first, true second
    for n in order
        Plotter.plot!(alpha, H1u[:,n]; yscale = :log10, linewidth = 3, linestyle = :dashdot, marker = :circle, markersize = 5, color = col[n], label = latexstring("||\\nabla(\\mathbf{u} - \\mathbf{u}_h)||,\\; ", pi_names[n]))
    end
    for n in order
        Plotter.plot!(alpha, L2ϱ[:,n]; yscale = :log10, linewidth = 3, linestyle = :dot, marker = :diamond, markersize = 5, color = col[n], label = latexstring("||\\varrho - \\varrho_h||,\\; ", pi_names[n]))
    end
    for n in order
        Plotter.plot!(alpha, L2u[:,n]; yscale = :log10, linewidth = 3, linestyle = :solid, marker = :square, markersize = 5, color = col[n], label = latexstring("||\\mathbf{u} - \\mathbf{u}_h||,\\; ", pi_names[n]))
    end
    Plotter.plot!(; legend = :topright, xtick = alpha, yticks = yticks, ylim = (ylo, yhi), xlim = (-0.05, 2.05), xlabel = L"\alpha", gridalpha = 0.5, grid = true, background_color_legend = RGBA(1,1,1,0.7))

    ##
    labels = [" Pi=$(reconstruct[n] ? "RT0" : "Gamma")" for n in 1:length(reconstruct)]
    print_table(alpha, L2u; xlabel = "α", ylabels = "|| u - u_h || " .* labels)

    ## save
    Plotter.savefig(filename_plots(data; free_parameter = "α"))
end
