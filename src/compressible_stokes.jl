# ==============================================================================
# Default configuration & filename helper
# ==============================================================================

const default_args = Dict(
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
    "stab1" => (1-0.1, 0),
    "stab2" => (1.5, 0),
    "bonus_quadorder" => 4,
    "maxsteps" => 8000,
    "subiterations_momentum" => :auto,
    "target_residual" => 1.0e-11,
    "reconstruct" => :RT, # choose from :RT :BDM :none
    "initial_values" => :interpolate, # choose from :interpolate, :stokes,
    "no_continuity_update" => false,
    # data of the problem
    "velocitytype" => ZeroVelocity,
    "densitytype" => ExponentialDensity,
    "convectiontype" => NoConvection,
    "upwindtype" => StandardUpwind,
    "coriolistype" => NoCoriolis,
    "eostype" => IdealGasLaw,
    "gridtype" => Mountain2D,
    "pressure_in_f" => true,
    "others_in_f" => true,
)

"""
    load_data(; kwargs...) -> Dict

Start from `default_args` and override entries by the given keyword arguments.
"""
function load_data(; kwargs...)
    data = deepcopy(default_args)
    for (k, v) in kwargs
        data[String(k)] = v
    end
    return data
end

# ==============================================================================
# Incompressible reference solver (limit c = Inf)
# ==============================================================================

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
    γ  = gamma(data["eostype"])
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

    ## the direct linear solve may not record any nonlinear residuals
    res = ExtendableFEM.residuals(SC)
    data["res_incompressible"] = isempty(res) ? 0.0 : res[end]

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
        try
            ExtendableFEM.plot([id(u)], sol; Plotter = UnicodePlots)
        catch err
            @warn "debug plot failed" exception = (err, catch_backtrace())
        end
    end

    return data
end

# ==============================================================================
# Convection operator assembly (dispatched by convectiontype)
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
                    factor = stab1[2]/2, entities = ON_IFACES, kwargs...), [args[1], args[2], u0[1]])
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
                    factor = stab1[2]/2, entities = ON_IFACES, kwargs...), [args[1], args[2], u0[1]])
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
                factor = stab1[2], entities = ON_IFACES, bonus_quadorder = order, kwargs...), sol)
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
        try
            ExtendableFEM.plot([id(u), id(ϱ)], sol; Plotter = UnicodePlots)
        catch err
            @warn "debug plot failed" exception = (err, catch_backtrace())
        end
    end

    return data
end
