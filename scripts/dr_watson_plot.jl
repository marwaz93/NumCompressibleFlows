using NumCompressibleFlows
using ExtendableFEM
using ExtendableFEMBase
using ExtendableGrids
using Triangulate
using SimplexGridFactory
using GridVisualize
using Symbolics
using LinearAlgebra
#using Test #hide

# new packages
using DrWatson
using JLD2
using LaTeXStrings
using Colors
using ColorTypes
using Latexify
using Plots
#gr()

# Safe loader that reads only scalar keys from JLD2 files, avoiding
# deserialization of complex FE types that break across package versions.
_safe_keys = ["Error(L2,u)", "Error(H1,u)", "Error(L2,ϱ)", "Error(L2,ϱu)",
              "ndofs", "nits", "M_exact", "M_start", "M_end","Error(H1,u0)"]
              
              # , "res_momentum", "res_continuity"]

function safe_produce_or_load(data; force = false, kwargs...)
    #fpath = filename(data) * ".jld2"
    #if !force && isfile(fpath)
    #    result = Dict{String, Any}()
    #    jldopen(fpath, "r") do f
    #        for k in keys(f)
    #           if k in _safe_keys
   #                 result[k] = f[k]
    #            end
    #        end
    #    end
    #    merge!(data, result)
    #    return data, fpath
    #else
        return produce_or_load(run_single, data, filename = filename; force = force, kwargs...)
    #end
end

default_args = Dict(
    # problem parameters
    "μ" => 1,
    "λ" => 0,
    "γ" => 1,
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
    "target_residual" => 1.0e-11,
    "reconstruct" => true,
    # data of the problem
    "velocitytype" => ZeroVelocity,
    "densitytype" => ExponentialDensity,
    "convectiontype" => NoConvection,
    "coriolistype" => NoCoriolis,
    "eostype" => IdealGasLaw,
    "gridtype" => Mountain2D,
    "pressure_in_f" => false,
    "laplacian_in_rhs" => true,
    "stab1" => (1-0.1,0),
    "stab2" => (1.5,0),
)

function filename(data)
    # problem parameters
    μ = data["μ"]
    λ = data["λ"]
    γ = data["γ"]
    c = data["c"]
    M = data["M"]
    # solving options
    τfac = data["τfac"]
    ufac = data["ufac"]
    nrefs = data["nrefs"]
    order = data["order"]
    reconstruct = data["reconstruct"]
    target_residual = data["target_residual"]
    maxsteps = data["maxsteps"]
    pressure_stab = data["pressure_stab"]
    bonus_quadorder = data["bonus_quadorder"]
    
    # data of the problem (convert types to abbreviated strings for savename)
    # Create shorter abbreviations for the types
    vtype = replace(string(data["velocitytype"]), "Velocity" => "V")
    dtype = replace(string(data["densitytype"]), "Density" => "D")
    etype = replace(string(data["eostype"]), "Law" => "")
    gtype = replace(string(data["gridtype"]), "2D" => "")
    ctype = replace(string(data["convectiontype"]), "Convection" => "Conv")
    cortype = replace(string(data["coriolistype"]), "Coriolis" => "Cor")
    
    #laplacian_in_rhs = data["laplacian_in_rhs"]
    pressure_in_f = data["pressure_in_f"]
    stab1 = data["stab1"]
    stab2 = data["stab2"]
    
    # Use shorter parameter names and only include essential ones
    essential_params = @dict μ λ γ c M τfac ufac nrefs order reconstruct vtype dtype etype gtype ctype cortype pressure_in_f stab1 stab2
   
   

    # sname
    # sname = savename((@dict μ λ γ c M τfac ufac nrefs order reconstruct target_residual maxsteps pressure_stab bonus_quadorder velocitytype densitytype eostype gridtype convectiontype coriolistype laplacian_in_rhs stab1 stab2))
    sname = savename(essential_params; allowedtypes = (Real, String, SubString, Symbol, Tuple{Real, Real}))
    sname = "data/projects/compressible_stokes/" * sname # the only place changing the data file
    return sname
end


function run_single(data; kwargs...)
    # problem parameters
    μ = data["μ"]
    λ = data["λ"]
    γ = data["γ"]
    c = data["c"]
    M = data["M"]
    ufac = data["ufac"]
    # solving options
    τfac = data["τfac"]
    nrefs = data["nrefs"]
    order = data["order"]
    reconstruct = data["reconstruct"]
    target_residual = data["target_residual"]
    maxsteps = data["maxsteps"]
    pressure_stab = data["pressure_stab"]
    bonus_quadorder = data["bonus_quadorder"]
    # data of the problem
    velocitytype = data["velocitytype"]
    densitytype = data["densitytype"]
    eostype = data["eostype"]
    gridtype = data["gridtype"]
    pressure_in_f = data["pressure_in_f"]
    laplacian_in_rhs = data["laplacian_in_rhs"]
    convectiontype = data["convectiontype"]
    coriolistype = data["coriolistype"]
    stab1 = data["stab1"]
    stab2 = data["stab2"]
    
    data = Dict{String, Any}(data)
    @show data, typeof(data)

    if target_residual / max(stab1[2],stab2[2]) < 1e-15
        target_residual = 1e-15 * max(stab1[2],stab2[2])
        @warn "reset target residual to $(target_residual) due to very large stabilization constants"
    end

    ## load data for testcase
    ϱ!, kernel_gravity!, kernel_rhs!, u!, ∇u! = prepare_data( velocitytype, densitytype , eostype  ; laplacian_in_rhs = laplacian_in_rhs, pressure_in_f = pressure_in_f, M = M, c = c, μ = μ, λ = λ,γ=γ, ufac = ufac,τfac = τfac , nrefs = nrefs , kwargs...)
    # added new for the type version
    xgrid = NumCompressibleFlows.grid(gridtype; nref = nrefs)
    #xgrid = grid(gridtype; nref = nrefs)

    # From here :D 


    M_exact = integrate(xgrid, ON_CELLS, ϱ!, 1; quadorder = 30)
    # M = M_exact # We could devide by M_exact directly in calculating τ instead of overwriting 
    τ = μ / (c*order^2 * M * τfac * ufac) # time step for pseudo timestepping
    #τ = μ / (4*order^2 * M * sqrt(τfac)) 
    @info "M = $M, M_exact = $M_exact τ = $τ"
    sleep(1)


    ## define unknowns
    u = Unknown("u"; name = "velocity", dim = 2)
    ϱ = Unknown("ϱ"; name = "density", dim = 1)
    p = Unknown("p"; name = "pressure", dim = 1)

    ## define reconstruction operator
    if order == 1
        FETypes = [H1BR{2}, L2P0{1}, L2P0{1}] # H1BR Bernardi-Raugel 2 is the dimension, L2P0 is P0 finite element
        id_u = reconstruct ? apply(u, Reconstruct{HDIVRT0{2}, Identity}) : id(u)# if reconstruct is true call apply, if false call id
        div_u = reconstruct ? apply(u, Reconstruct{HDIVRT0{2}, Divergence}) : div(u) # Marwa div term 
    # RT of lowest order reconstruction 
    elseif order == 2
        FETypes = [H1P2B{2, 2}, L2P1{1}, L2P1{1}] #H1P2B add additional cell bubbles, not Bernardi-Raugel? L2P1 is P1 finite element
        id_u = reconstruct ? apply(u, Reconstruct{HDIVRT1{2}, Identity}) : id(u) # RT of order 1 reconstruction
        div_u = reconstruct ? apply(u, Reconstruct{HDIVRT1{2}, Divergence}) : div(u) # Marwa div term 
    
    end

    ## in/outflow regions
    testgrid = NumCompressibleFlows.grid(gridtype; nref = 1)
    rinflow = inflow_regions(velocitytype, gridtype)
    routflow = outflow_regions(velocitytype, gridtype)
    rhom = setdiff(unique!(testgrid[BFaceRegions]), union(rinflow,routflow))
    @info rinflow, routflow, rhom
    sleep(1)

    ## define first sub-problem: Stokes equations to solve for velocity u
    PD = ProblemDescription("Stokes problem")

    assign_unknown!(PD, u)
    assign_operator!(PD, BilinearOperator([grad(u)]; factor = μ, store = true, kwargs...))
    assign_operator!(PD, BilinearOperator([div_u]; factor = λ, store = true, kwargs...)) # Marwa div term 
    if coriolistype !== NoCoriolis
        assign_operator!(PD, LinearOperator(kernel_coriolis_linearoperator!(coriolistype), [
        id_u], [id_u,id(ϱ)]; quadorder = 2*order + 1, factor = -1, kwargs...))
    end
    if convectiontype == StandardConvection
        assign_operator!(PD, LinearOperator(kernel_standardconvection_linearoperator!, [
        id_u], [id_u,grad(u),id(ϱ)]; quadorder = 2*order + 1, factor = -1, kwargs...))
    elseif convectiontype == OseenConvection
        assign_operator!(PD, BilinearOperator(kernel_oseenconvection!(u!, ϱ!), [
        id_u], [grad(u)]; quadorder = 2*order + 1, factor = 1, kwargs...))
    elseif convectiontype == RotationForm
        assign_operator!(PD, LinearOperator(kernel_rotationform_linearoperator!, [
        id_u, div_u], [id_u,curl2(u),id(ϱ)]; quadorder = 2*order + 1, factor = -1, kwargs...))
    elseif convectiontype == NoConvection
    else
        @error "discretization of convectiontype=$convectiontype not defined"
    end

    # Start adding hom boundary data 
    assign_operator!(PD, LinearOperator(eos!(eostype), [div(u)], [id(ϱ)]; factor = c, kwargs...))
    if length(rhom) > 0 
        assign_operator!(PD, HomogeneousBoundaryData(u; regions = rhom, kwargs...))
    end
    if length(rinflow) > 0 || length(routflow) > 0
        assign_operator!(PD, InterpolateBoundaryData(u, u!; bonus_quadorder = bonus_quadorder, regions = union(rinflow,routflow), kwargs...))
    end
    # 
    if kernel_rhs! !== nothing
        assign_operator!(PD, LinearOperator(kernel_rhs!, [id_u]; factor = 1, store = true, bonus_quadorder = bonus_quadorder, kwargs...))
    end
    assign_operator!(PD, LinearOperator(kernel_gravity!, [id_u], [id(ϱ)]; factor = 1, bonus_quadorder = bonus_quadorder, kwargs...))


     

   ## FVM for continuity equation
    @info "timestep = $τ"
    PDT = ProblemDescription("continuity equation")
    assign_unknown!(PDT, ϱ)
    if order > 1
        assign_operator!(PDT, BilinearOperator(kernel_continuity!, [grad(ϱ)], [id(ϱ)], [id(u)]; quadorder = 2 * order, factor = -1, kwargs...))
        # entities = ON_CELLS since it is the default
    end
    if pressure_stab > 0
        psf = pressure_stab #* xgrid[CellVolumes][1]
        assign_operator!(PDT, BilinearOperator(stab_kernel!, [jump(id(ϱ))], [jump(id(ϱ))], [id(u)]; entities = ON_IFACES, factor = psf, kwargs...))
    end
    assign_operator!(PDT, BilinearOperator([id(ϱ)]; quadorder = 2 * (order - 1), factor = 1, store = true, kwargs...)) # for (1/τ) (ϱ^n+1,λ)
    assign_operator!(PDT, LinearOperator([id(ϱ)], [id(ϱ)]; quadorder = 2 * (order - 1), factor = 1, kwargs...)) # for (1/τ) (ϱ^n,λ) on the rhs 
    # assign_operator!(PDT, BilinearOperatorDG(kernel_upwind!, [jump(id(ϱ))], [this(id(ϱ)), other(id(ϱ))], [id(u)]; quadorder = order + 1, factor = 1, entities = ON_IFACES, kwargs...))
    # [jump(id(ϱ))]is test function lambda [λ] , [this(id(ϱ)), other(id(ϱ))] is the the flux multlplied by lambda_upwind. [id(u)] is the function u that is needed
    # Start adding inflow 
    ## upwind operator with variable time step
    D = nothing
    brho = nothing
    one_vector = nothing
    rowsums = nothing
    sol = nothing
    rho_mean = M_exact/sum(xgrid[CellVolumes])

    function callback!(A, b, args; assemble_matrix = true, assemble_rhs = true, time = 0, kwargs...)
    fill!(D.entries.cscmatrix.nzval, 0)
    fill!(brho.entries, 0)
    assemble!(D, BilinearOperatorDG(kernel_upwind!, [jump(id(1))], 
     [this(id(1)), other(id(1))], [id(1)]; factor = 1, quadorder = order+1, entities = 
     ON_IFACES), sol)

    ## check if matrix is diagonally dominant
    mul!(rowsums, D.entries, one_vector)

    ## if not, use the smaller τ
     tau = min(extrema(abs.(xgrid[CellVolumes]./rowsums))[1]/2, τ)
     print(" (τ = $τ) ")

    if length(rinflow) > 0
        assemble!(brho, LinearOperatorDG(kernel_inflow!(u!,ϱ!), [id(1)]; factor = -1 , bonus_quadorder = bonus_quadorder, entities = ON_BFACES, regions = rinflow, kwargs...))    
    end
    if length(routflow) > 0
        assemble!(D, BilinearOperatorDG(kernel_outflow!(u!), [id(1)]; factor = 1, bonus_quadorder = bonus_quadorder, entities = ON_BFACES, regions = routflow, kwargs...)) 
    end

    ## densityjump stabilisation
    if stab1[2] > 0
        assemble!(D, BilinearOperatorDG(density_jump_stab_kernel!(stab1[1], γ),[jump(id(1))],[jump(id(1))], [average(id(1))]; factor = stab1[2]*2, entities = ON_IFACES, kwargs...), sol) 
    end
    ## density mean stabilisation
    if stab2[2] > 0
        hmean = sum(xgrid[FaceVolumes])/length(xgrid[FaceVolumes])
        assemble!(D, BilinearOperator([id(1)]; factor = hmean^stab2[1]*stab2[2], kwargs...)) 
        assemble!(brho, LinearOperator([id(1)]; factor = hmean^stab2[1]*rho_mean*stab2[2], kwargs...)) 
    end
    
    ExtendableFEMBase.add!(A, D.entries; factor = tau)
    b .+= tau * brho.entries
    end
    assign_operator!(PDT, CallbackOperator(callback!, [u]; linearized_dependencies = [ϱ,ϱ], modifies_rhs = false, kwargs..., name = "upwind matrix D scaled by tau"))

    EnergyIntegrator = ItemIntegrator(energy_kernel!, [id(u)]; resultdim = 1, quadorder = 2 * (order + 1), kwargs...)
    
    #NDofs = zeros(Int, nrefs)
    #Results = zeros(Float64, nrefs, 5) # it is a matrix whose rows are levels and columns are 
    ## prepare error calculation
    EnergyIntegrator = ItemIntegrator(energy_kernel!, [id(u)]; resultdim = 1, quadorder = 2 * (order + 1), kwargs...)
    MassIntegrator = ItemIntegrator([id(ϱ)]; resultdim = 1, kwargs...)
    NDofs = zeros(Int, nrefs)
    Results = zeros(Float64, nrefs, 5) # it is a matrix whose rows are levels and columns are various errors 

    sol = nothing
    #xgrid = nothing
    op_upwind = 0
    # here we run only in one level instead of for Loop over the levels
    FES = [FESpace{FETypes[j]}(xgrid) for j in 1:3] # 3 because we have dim(FETypes)=3
    sol = FEVector(FES; tags = [u, ϱ, p]) # create solution vector and tag blocks with the unknowns (u,ρ,p) that has the same order as FETypes

    ## initial guess
    fill!(sol[ϱ], M) # fill block corresbonding to unknown ρ with initial value M, in Algorithm it is M/|Ω|?? We could write it as M/|Ω| and delete area from down there
    interpolate!(sol[u], u!)
    interpolate!(sol[ϱ], ϱ!)
    #NDofs[lvl] = length(sol.entries)

    D = FEMatrix(FES[2], FES[2])
    brho = FEVector(FES[2])
    one_vector = ones(Float64, size(D.entries,1))
    rowsums = zeros(Float64, size(D.entries,1))

    M_start = sum(evaluate(MassIntegrator, sol))
    ## solve the two problems iteratively [1] >> [2] >> [1] >> [2] ...
    SC1 = SolverConfiguration(PD; init = sol, maxiterations = 1, target_residual = target_residual, constant_matrix = true, kwargs...)
    SC2 = SolverConfiguration(PDT; init = sol, maxiterations = 1, target_residual = target_residual, kwargs...)
    sol, nits = iterate_until_stationarity([SC1, SC2]; energy_integrator = EnergyIntegrator, maxsteps = maxsteps, init = sol, kwargs...)

    ## calculate mass
    Mend = sum(evaluate(MassIntegrator, sol))
    @info "M_exact/M_start/M_end/difference = $(M_exact)/$M_start/$Mend/$(M_start-Mend)"

    # Untill here :D 

    ## save data
    data["ndofs"] = length(sol.entries)
    data["solution"] = sol
    data["grid"] = xgrid
    data["unknown_u"] = u
    data["unknown_ϱ"] = ϱ

    ## save to data folder
    return data
end


function compute_errors(config; force_recompute = false, kwargs...)


    fpath = filename(config) * ".jld2"
    @info "loading data from $fpath"
    #@save fpath data
    data = wload(fpath)


    # problem parameters
    μ = data["μ"]
    λ = data["λ"]
    γ = data["γ"]
    c = data["c"]
    M = data["M"]
    ufac = data["ufac"]
    if haskey(data, "solution")
        sol = data["solution"]
        u = sol.tags[1]
        ϱ = sol.tags[2]
        repair_grid!(sol[u].FES.xgrid)
    else
        @error "solution not found in data, cannot compute errors"
        return nothing  
    end

    # data of the problem
    velocitytype = data["velocitytype"]
    densitytype = data["densitytype"]
    eostype = data["eostype"]
    gridtype = data["gridtype"]
    pressure_in_f = data["pressure_in_f"]
    laplacian_in_rhs = data["laplacian_in_rhs"]
    convectiontype = data["convectiontype"]
    coriolistype = data["coriolistype"]

    ϱ!, kernel_gravity!, kernel_rhs!, u!, ∇u! = prepare_data( velocitytype, densitytype , eostype  ; laplacian_in_rhs = laplacian_in_rhs, pressure_in_f = pressure_in_f, M = M, c = c, μ = μ, λ = λ,γ=γ, ufac = ufac, kwargs...)
    
    if force_recompute || !haskey(data, "Error(H1,u0)")
        @info "computing divergence-free part of u - u_h"

        ## compute error of divergence-free part by solving
        ## incompressible Stokes problem with rhs (∇(u-uh), ∇v)
        
        ## for pre-compilation of lazy_interpolate use a coarse grid
        compile_grid = simplexgrid(0:0.5:1,0:0.5:1)
        cg_refined = barycentric_refine(compile_grid)
        test_fe1 = FEVector(FESpace{eltype(sol[u].FES)}(compile_grid); tags = [u])
        test_fe2 = FEVector(FESpace{H1Pk{2,2,2}}(cg_refined); tags = [u])
        @time lazy_interpolate!(test_fe2[1], test_fe1, [id(u)]; quadorder = 0)
        
        ## prepare FESpace
        xgrid = sol[u].FES.xgrid
        xgridSP = barycentric_refine(xgrid)
        FES_SP = [FESpace{H1Pk{2,2,2}}(xgridSP), FESpace{H1Pk{1,2,1}}(xgridSP; broken = true)]

        ## unknowns of divergence-free projection
        uzero  = Unknown("u"; name = "Stokes projection")
        pzero = Unknown("p"; name = "pressure of projection")

        ## interpolate discrete compressible velocity solution into SV space
        solSP = FEVector(FES_SP; tags = [uzero, pzero])
        append!(solSP, FES_SP[1]; tag = u)
        #@time lazy_interpolate!(solSP[u], sol, [id(u)]; quadorder = 0) # now solSP[u] is the interpolated discrete compressibe solution

        @time interpolate_BR_to_P2!(solSP[u], sol[u])
        

        ## problem description for velocity update of iterated penalty method of div-free projection
        PDSP_u = ProblemDescription("Stokes projection problem - u update")
        assign_unknown!(PDSP_u, uzero)
        β = 1e+3 # div-penalty of iterated penalty method
        assign_operator!(PDSP_u, BilinearOperator([grad(uzero)]; factor = 1, store = true, kwargs...))
        assign_operator!(PDSP_u, BilinearOperator([div(uzero)]; store = true, factor = β , kwargs...))
        assign_operator!(PDSP_u, LinearOperator([grad(uzero)], [grad(u)]; factor = -1, kwargs...))    # ∇ u_h part     
        assign_operator!(PDSP_u, LinearOperator(∇u!, [grad(uzero)]; kwargs...)) # ∇ u_part
        assign_operator!(PDSP_u, LinearOperator([div(uzero)], [id(pzero)]; factor = 1, kwargs...))
        assign_operator!(PDSP_u, HomogeneousBoundaryData(uzero; regions = 1:4, kwargs...))

        ## problem description for pressure update of iterated penalty method of div-free projection
        PDSP_p = ProblemDescription("Stokes projection problem - p update")
        assign_unknown!(PDSP_p, pzero)
        assign_operator!(PDSP_p, BilinearOperator([id(pzero)]; store = true, kwargs...))
        assign_operator!(PDSP_p, LinearOperator(div_projection!, [id(pzero)], [id(pzero), div(uzero)]; params = [β], factor = 1, kwargs...))

        ## run iterated penalty method
        SC1 = SolverConfiguration(PDSP_u; init = solSP, maxiterations = 1, target_residual = 1.0e-10, constant_matrix = true, kwargs...)
        SC2 = SolverConfiguration(PDSP_p; init = solSP, maxiterations = 1, target_residual = 1.0e-10, constant_matrix = true, kwargs...)
        solSP, nits = iterate_until_stationarity([SC1, SC2]; init = solSP, kwargs...)
        @info "converged after $nits iterations"

        ## compute norm of div-free projection of the error
        error0 = evaluate(L2NormIntegrator([grad(uzero)]), solSP)
        data["Error(H1,u0)"] = sqrt(sum(view(error0, 1, :)) + sum(view(error0, 2, :)) +  + sum(view(error0, 3, :)) +  + sum(view(error0, 4, :))) # grad u = (u_1,u_2)
        @info data["Error(H1,u0)"]
    else
        @info "skipping computation of divergence-free part of u - u_h since already computed"
    end

    if force_recompute || !haskey(data, "Error(L2,u)")
        @info "computing errors of u, ϱ and ϱu"
        ## calculate error
        order = data["order"]
        ErrorIntegratorExact = ItemIntegrator(exact_error!(u!, ∇u!, ϱ!), [id(u), grad(u), id(ϱ)]; resultdim = 9, quadorder = 2 * (order + 1), kwargs...)
        error = evaluate(ErrorIntegratorExact, sol)
        data["Error(L2,u)"] = sqrt(sum(view(error, 1, :)) + sum(view(error, 2, :))) # u = (u_1,u_2)
        data["Error(H1,u)"] = sqrt(sum(view(error, 3, :)) + sum(view(error, 4, :)) + sum(view(error, 5, :)) + sum(view(error, 6, :))) # ∇u = (∂_x u_1,∂_y u_1, ∂_x u_2, ∂_y u_2 )
        data["Error(L2,ϱ)"] = sqrt(sum(view(error, 7, :))) # ϱ
        data["Error(L2,ϱu)"]  = sqrt(sum(view(error, 8, :)) + sum(view(error, 9, :))) # (ρ u_1 - ρ u_2)
        data["nits"] = nits
    else
        @info "skipping error calculation since already computed"
    end

    ## save
    fpath = filename(data) * ".jld2"
    @info "saving data to $fpath"
    #@save fpath data
    wsave(fpath, data)

    return data
end


quickactivate(@__DIR__, "NumCompressibleFlows")
mkpath(plotsdir("compressible_stokes_paper/convegence_history"))
mkpath(plotsdir("compressible_stokes_paper/panlty_convegence_history"))

mkpath(plotsdir("compressible_stokes_paper/parameter_studies_μ/"))
mkpath(plotsdir("compressible_stokes_paper/parameter_studies_γ/"))
mkpath(plotsdir("compressible_stokes_paper/parameter_studies_c/"))
mkpath(plotsdir("compressible_stokes_paper/parameter_studies_cμ/"))
mkpath(plotsdir("compressible_stokes_paper/parameter_studies_c1/"))
mkpath(plotsdir("compressible_stokes_paper/parameter_studies_α/"))
mkpath(plotsdir("compressible_stokes_paper/parameter_studies_c2/"))

function filename_plots(data; prefix = "", free_parameter = "")
    μ = data["μ"] 
    c = data["c"]
    γ = data["γ"]
    stab1 = data["stab1"]
    stab2 = data["stab2"]
    ϵ = 1 - stab1[1]
    α = stab2[1]
    c1 = stab1[2]
    c2 = stab2[2]
    velocitytype = string(data["velocitytype"])
    densitytype = string(data["densitytype"])
    reconstruct = data["reconstruct"]
    pressure_in_f = data["pressure_in_f"]
    eostype = data["eostype"]
    gridtype = data["gridtype"]
    convectiontype = data["convectiontype"]
    nrefs = data["nrefs"]

    if free_parameter == "μ"
        sname = savename((@dict c γ ϵ c1 nrefs reconstruct ))
    elseif free_parameter == "γ"
        sname = savename((@dict μ c ϵ c1 nrefs reconstruct ))
    elseif free_parameter == "c"
        sname = savename((@dict μ γ ϵ c1 nrefs reconstruct ))
    elseif free_parameter == "cμ"
        sname = savename((@dict γ ϵ c1 nrefs reconstruct ))
    elseif free_parameter == "c1"
        sname = savename((@dict μ c γ ϵ nrefs reconstruct ))
    elseif free_parameter == "c2"
        sname = savename((@dict μ c γ ϵ c1 nrefs reconstruct ))
    elseif free_parameter == "α"
        sname = savename((@dict μ c γ ϵ c1 nrefs reconstruct ))
    else
        sname = savename((@dict μ c γ ϵ c1 nrefs reconstruct pressure_in_f))
    end

    if free_parameter !== ""
        sname = "plots/compressible_stokes_paper/parameter_studies_$(free_parameter)/" * sname * prefix * ".png"
  else
        sname = "plots/compressible_stokes_paper/panlty_convegence_history/" * sname * prefix * ".png"
        
    end
    
    return sname
end

# default_args = Dict(
#     # problem parameters
#     "μ" => 1,
#     "λ" => 0,
#     "γ" => 1,
#     "c" => 1,
#     "M" => 1,
#     # solving options
#     "τfac" => 1,
#     "ufac" => 1,
#     "nrefs" => 4,
#     "order" => 1,
#     "pressure_stab" => 0,
#     "bonus_quadorder" => 4,
#     "maxsteps" => 2000,
#     "target_residual" => 1.0e-11,
#     "reconstruct" => true,
#     # data of the problem
#     "velocitytype" => ZeroVelocity,
#     "densitytype" => ExponentialDensity,
#     "convectiontype" => NoConvection,
#     "coriolistype" => NoCoriolis,
#     "eostype" => IdealGasLaw,
#     "gridtype" => Mountain2D,
#     "pressure_in_f" => false,
#     "laplacian_in_rhs" => true,
#     "stab1" => (1-0.1,0),
#     "stab2" => (1.5,0),
# )


function load_data(; kwargs...)
    data = deepcopy(default_args) 
    for (k,v) in kwargs 
        data[String(k)] = v 
    end
    return data
end


function plot_single(; Plotter = PyPlot, force = false, kwargs...) 

    data = load_data(; kwargs...)
    @show data
    data, ~ = produce_or_load(run_single, data, filename = filename, force = force) # ~ is used to ignore the second returned variable of produce_or_load function
    xgrid = data["grid"]
    sol = data["solution"]
    u = data["unknown_u"]
    ϱ = data["unknown_ϱ"]
    @show sol
    repair_grid!(xgrid)
    repair_grid!(sol[1].FES.xgrid)
    repair_grid!(sol[2].FES.xgrid)

    #plt = GridVisualizer(; Plotter = Plotter, layout = (1, 1), clear = true, size = (1000, 1000))
    #scalarplot!(plt[1, 1], xgrid, view(nodevalues(sol[u]; abs = true), 1, :), levels = 0, colorbarticks = 7)
    # vectorplot!(plt[1, 1], xgrid, eval_func_bary(PointEvaluator([id(u)], sol)), rasterpoints = 10, clear = false, title = "u_h (abs + quiver)")
    #scalarplot!(plt[1, 1], xgrid, view(nodevalues(sol[ϱ]), 1, :), levels = 11, title = "ϱ_h")


    ## plot
   #PyPlot.rc("font", size=60)
    pl = GridVisualizer(; Plotter = Plotter, layout = (1,2), clear = true,show=true, resolution = (1000,500))
    scalarplot!(pl[1,1], xgrid, view(nodevalues(sol[u]; abs = true),1,:), levels = 0, colorbarticks = 7, fontsize = 60)
    vectorplot!(pl[1,1], xgrid, eval_func_bary(PointEvaluator([id(u)], sol)), clear = false, fontsize = 60) # title = L"\mathbf{u}_h "
    scalarplot!(pl[1,2], xgrid, view(nodevalues(sol[ϱ]),1,:), levels = 11, fontsize = 60)  # title = L"ϱ_h" 
    #gridplot!(pl[2,2], xgrid)

    ## save
    
    #Plotter.savefig(filename_plots(data; prefix = "_Solutions"))

    scene = GridVisualize.reveal(pl)
    GridVisualize.save(filename_plots(data; prefix = "_Solutions"), scene; Plotter = Plotter)

    return data
end


function interpolate_BR_to_P2!(u_P2::FEVectorBlock, u_BR::FEVectorBlock)
    FES_BR = u_BR.FES
    FES_P2 = u_P2.FES
    xgrid_bary = u_P2.FES.xgrid
    xgrid = u_BR.FES.xgrid
    cellparents = xgrid_bary[CellParents]
    cellnodes_bary = xgrid_bary[CellNodes]

    dofs_BR = view(u_BR)
    dofs_P2 = view(u_P2)
    facedofs_P2 = FES_P2[FaceDofs] # NNNNNNFFF

    # node dofs of coarse grid remain unchanged
    nnodes = num_nodes(xgrid)
    nnodes_bary = num_nodes(xgrid_bary)
    nfaces_bary = num_sources(facedofs_P2)
    for n = 1 : nnodes
        dofs_P2[n] = dofs_BR[n]
        dofs_P2[n+(nnodes_bary+nfaces_bary)] = dofs_BR[n + nnodes]
    end

    ## define PointEvaluator for uBR
    PE = PointEvaluator([id(1)], [u_BR])
    evalBR = zeros(Float64, 2)
    xref_center = [1/3, 1/3]
    xref_outer = [[1/2, 0], [1/2, 1/2], [0, 1/2]]
    xref_inner = [[1/6, 1/6], [4/6, 1/6], [1/6, 4/6]]
    cell::Int = 0

    ncells_bary = num_cells(xgrid_bary)
    cellfaces_bary = xgrid_bary[CellFaces]
    node_offset = (nnodes_bary+nfaces_bary)
    face_offset = node_offset + nnodes_bary
    for cell_bary = 1 : ncells_bary
        cell = cellparents[cell_bary]

        # determine on which part of the coarse triangle we are
        child_type = mod(cell_bary - 1, 3) + 1 

        if child_type == 1
            # set new vertex in center (xref = [1/3, 1/3, 1/3])
            new_node = cellnodes_bary[3, cell_bary]
            evaluate_bary!(evalBR, PE, xref_center, cell)
            dofs_P2[new_node] = evalBR[1]
            dofs_P2[node_offset + new_node] = evalBR[2]
        end

        # set dof on outer face
        face_outer = cellfaces_bary[1, cell_bary]
        evaluate_bary!(evalBR, PE, xref_outer[child_type], cell)
        dofs_P2[nnodes_bary + face_outer] = evalBR[1]
        dofs_P2[face_offset + face_outer] = evalBR[2]
    
        # set dof on inner face
        face_inner = cellfaces_bary[3, cell_bary]
        evaluate_bary!(evalBR, PE, xref_inner[child_type], cell)
        dofs_P2[nnodes_bary + face_inner] = evalBR[1]
        dofs_P2[face_offset + face_inner] = evalBR[2]
    end
end


function plot_convergencehistory(; nrefs = 1:6, Plotter = Plots, force = false, force_recompute = false, kwargs...)

    data = load_data(; kwargs...)
    #@show data
    Results = zeros(Float64, length(nrefs), 7)
    NDoFs = zeros(Int, length(nrefs))
    #Residuals = zeros(Float64, length(nrefs), 2)

    for lvl in nrefs
        _data = deepcopy(data)
        _data["nrefs"] = lvl
        _data, ~ = safe_produce_or_load(_data; force = force)
        NDoFs[lvl] = _data["ndofs"]
        _data = compute_errors(_data; force_recompute = force_recompute)
        Results[lvl,1] = _data["Error(L2,u)"]
        Results[lvl,2] = _data["Error(H1,u)"] 
        Results[lvl,3] = _data["Error(L2,ϱ)"]
        Results[lvl,4] = _data["Error(L2,ϱu)"]
        Results[lvl,5] = haskey(_data, "Error(H1,u0)") ? _data["Error(H1,u0)"] : NaN
        Results[lvl,6] = haskey(_data, "Error(H1,u0)") ? sqrt(_data["Error(H1,u)"]^2 - _data["Error(H1,u0)"]^2) : NaN
        Results[lvl,7] = _data["nits"]
       

        #=
        if haskey(_data, "res_momentum")
            Residuals[lvl,1] = _data["res_momentum"]
           Residuals[lvl,2] = _data["res_continuity"]
        else
            @warn "residual information not found, consider rerunning"
            Residuals[lvl,1] = 1e30
            Residuals[lvl,2] = 1e30
        end

=#
    
        #print_convergencehistory(NDoFs[1:lvl], Results[1:lvl, :]; X_to_h = X -> X .^ (-1 / 2), ylabels = ["|| u - u_h ||", "|| ∇(u - u_h) ||", "|| ϱ - ϱ_h ||", "|| ϱu - ϱu_h ||", "#its"], xlabel = "ndof")
         print_convergencehistory(NDoFs[:], Results[:,[1,2,5,3]]; X_to_h = X -> 
            X.^(-1/2), ylabels = [L"\lt{\bu - \uh}" , L"\lt{\nabla \(\bu - \uh\)}", L"\lt{\nabla \(\bu^0 - \uh^0\)}",
            L"\lt{\varrho - \varrho_h}"], xlabel = "ndof", latex_mode = true)
    end
    # L"\lt{\nabla \(\varrho \bu - \varrho_h \uh\)}",
    
    #print(latexify(Results[:,[6,1,2,3]]; env = :table, booktabs = false, latex = false, fmt = "%.2e"))

    ## plot
    #Plotter.rc("font", size=20)
    yticks = [1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,1e1,1e2]
    xticks = [1e1,1e2,1e3,1e4,1e5,1e6,1e7]
    Plotter.plot(; show = true, size = (1000,1000), margin = 1Plots.cm, legendfontsize = 20, tickfontsize = 22, guidefontsize = 26, grid=true)
    Plotter.plot!(NDoFs, Results[:,1]; xscale = :log10, yscale = :log10, linewidth = 3, marker = :circle, markersize = 5, label = L"|| \mathbf{u} - \mathbf{u}_h \,||", grid=true)
    Plotter.plot!(NDoFs, Results[:,2]; xscale = :log10, yscale = :log10, linewidth = 3, marker = :circle, markersize = 5, label = L"|| ∇(\mathbf{u} - \mathbf{u}_h)\,||", grid=true)
    Plotter.plot!(NDoFs, Results[:,3]; xscale = :log10, yscale = :log10, linewidth = 3, marker = :circle, markersize = 5, label = L"|| {ϱ}-ϱ_h \, ||", grid=true)
    Plotter.plot!(NDoFs, Results[:,4]; xscale = :log10, yscale = :log10, linewidth = 3, marker = :circle, markersize = 5, label = L"|| {ϱ\mathbf{u}}-ϱ_h \mathbf{u}_h \, ||", grid=true)
    Plotter.plot!(NDoFs, Results[:,5]; xscale = :log10, yscale = :log10, linewidth = 3, marker = :circle, markersize = 5, label = L"||  ∇( \mathbf{u}^0 - \mathbf{u}^0_h ) \,||", grid=true)
    Plotter.plot!(NDoFs, Results[:,7]; xscale = :log10, yscale = :log10, linewidth = 3, marker = :circle, markersize = 5, label = L"nits", grid=true)
    Plotter.plot!(NDoFs, 0.5*NDoFs.^(-0.5); xscale = :log10, yscale = :log10, linestyle = :dash, linewidth = 3, color = :gray, label = L"\mathcal{O}(h)", grid=true)
    Plotter.plot!(NDoFs, (1e+1)*NDoFs.^(-1.0); xscale = :log10, yscale = :log10, linestyle = :dash, linewidth = 3, color = :gray, label = L"\mathcal{O}(h^2)", grid=true)

    #Plotter.plot!(NDofs, 0.5*NDofs.^(-0.5); xscale = :log10, yscale = :log10, linestyle = :dash, linewidth = 3, color = :gray, label = L"\mathcal{O}(h)", grid=true)
    #Plotter.plot!(NDofs, 0.5*NDofs.^(-1.0); xscale = :log10, yscale = :log10, linestyle = :dash, linewidth = 3, color = :gray, label = L"\mathcal{O}(h^2)", grid=true)
    #Plotter.plot!(NDoFs, 100*NDoFs.^(-1.25); xscale = :log10, yscale = :log10, linestyle = :dash, linewidth = 3, color = :gray, label = L"\mathcal{O}(h^{2.5})", grid=true)
    
    Plotter.plot!(; legend = :bottomleft, xtick = xticks, yticks = yticks, ylim = (yticks[1]/2, 2*yticks[end]), xlim = (xticks[1], xticks[end]), xlabel = "degrees of freedom",gridalpha = 0.7,grid=true, background_color_legend = RGBA(1,1,1,0.7))
    ## save
    Plotter.savefig(filename_plots(data))
  #  return Residuals
end

function plot_parameter_study_viscosity(; nrefs = [3], μ = [1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100,1000], Plotter = Plots, kwargs...)
    nrefs = nrefs isa AbstractVector ? nrefs : [nrefs]
    μ = μ isa AbstractVector ? μ : [μ]
    data = load_data(; kwargs...)
    @show data
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
        #Plotter.plot!(μ, H1u[:,n]; xscale = :log10, yscale = :log10, linewidth = 3, marker = :circle, markersize = 5, label = L"||∇(\mathbf{u} - \mathbf{u}_h) \,|| \mathrm{level} = %$(nrefs[n])") # "||∇(u-u_h)|| level = $(nrefs[n])"
        Plotter.plot!(μ, L2ϱ[:,n]; xscale = :log10, yscale = :log10, linewidth = 3, marker = :circle, markersize = 5, label = L"|| {ϱ}-ϱ_h \, || \mathrm{level} = %$(nrefs[n])") # "||ϱ - ϱ_h|| level = $(nrefs[n])"
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

function plot_parameter_study_stab1(;  nrefs = [3,4,5],c1 = [1e-5,1e-4,1e-3,1e-2,1e-1,1], Plotter = Plots, kwargs...)
    nrefs = nrefs isa AbstractVector ? nrefs : [nrefs]
    c1 = c1 isa AbstractVector ? c1 : [c1]
    data = load_data(; kwargs...)
    @show data
    L2u = zeros(Float64, length(c1), length(nrefs))
    H1u = zeros(Float64, length(c1), length(nrefs))
    L2ϱ = zeros(Float64, length(c1), length(nrefs))
   

    for n = 1 : length(nrefs)
        data["nrefs"] = nrefs[n]
        for j = 1 : length(c1)
                stab1 = data["stab1"]
                @show stab1[1]
                data["stab1"] = (stab1[1], c1[j])
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
        Plotter.plot!(c1, H1u[:,n]; xscale = :log10, yscale = :log10, linewidth = 3, marker = :circle, markersize = 5, label = L"||∇(\mathbf{u} - \mathbf{u}_h) \,|| \mathrm{level} = %$(nrefs[n])") # "||∇(u-u_h)|| level = $(nrefs[n])"
         Plotter.plot!(c1, L2ϱ[:,n]; xscale = :log10, yscale = :log10, linewidth = 3, marker = :circle, markersize = 5, label = L"|| {ϱ}-ϱ_h \, || \mathrm{level} = %$(nrefs[n])") # "||ϱ - ϱ_h|| level = $(nrefs[n])"
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
# Plotting c_s for reconstruction
function plot_parameter_study_stab1_reconstruction(;  reconstruct = [true,false], c1 = [1e-5,1e-4,1e-3,1e-2,1e-1,1,1e1,1e2,1e3,1e4,1e5], Plotter = Plots, kwargs...)
    reconstruct = reconstruct isa AbstractVector ? reconstruct : [reconstruct]
    c1 = c1 isa AbstractVector ? c1 : [c1]
    data = load_data(; kwargs...)
    @show data
    L2u = zeros(Float64, length(c1), length(reconstruct))
    H1u = zeros(Float64, length(c1), length(reconstruct))
    L2ϱ = zeros(Float64, length(c1), length(reconstruct))
   

    for n = 1 : length(reconstruct)
        data["reconstruct"] = reconstruct[n]
        for j = 1 : length(c1)
               stab1 = data["stab1"]
                @show stab1[1]
                data["stab1"] = (stab1[1], c1[j])
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
    @show data
    L2u = zeros(Float64, length(alpha), length(reconstruct))
    H1u = zeros(Float64, length(alpha), length(reconstruct))
    L2ϱ = zeros(Float64, length(alpha), length(reconstruct))
   

    for n = 1 : length(reconstruct)
        data["reconstruct"] = reconstruct[n]
        for j = 1 : length(alpha)
               stab1 = data["stab1"]
                @show stab1[1]
                data["stab1"] = (alpha[j]-1, stab1[2])
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
function plot_parameter_study_stab2(;  nrefs = [3,4,5], c2  =[1e-4,1e-2,1,1e+2,1e+4], Plotter = Plots, kwargs...)
    nrefs = nrefs isa AbstractVector ? nrefs : [nrefs]
    c2 = c2 isa AbstractVector ? c2 : [c2]
    data = load_data(; kwargs...)
    @show data
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
        Plotter.plot!(c2, H1u[:,n]; xscale = :log10, yscale = :log10, linewidth = 3, marker = :circle, markersize = 5, label = L"||∇(\mathbf{u} - \mathbf{u}_h) \,|| \mathrm{level} = %$(nrefs[n])") # "||∇(u-u_h)|| level = $(nrefs[n])"
         Plotter.plot!(c2, L2ϱ[:,n]; xscale = :log10, yscale = :log10, linewidth = 3, marker = :circle, markersize = 5, label = L"|| {ϱ}-ϱ_h \, || \mathrm{level} = %$(nrefs[n])") # "||ϱ - ϱ_h|| level = $(nrefs[n])"
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

function plot_parameter_study_gamma(; nrefs = [3], γ = [1,1e+1,1e+2,1e+3], Plotter = Plots, kwargs...)
    nrefs = nrefs isa AbstractVector ? nrefs : [nrefs]
    γ = γ isa AbstractVector ? γ : [γ]
    data = load_data(; kwargs...)
    @show data
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
        #Plotter.plot!(γ, H1u[:,n]; xscale = :log10, yscale = :log10, linewidth = 3, marker = :circle, markersize = 5, label = L"||∇(\mathbf{u} - \mathbf{u}_h) \,|| \mathrm{level} = %$(nrefs[n])") # "||∇(u-u_h)|| level = $(nrefs[n])"
        Plotter.plot!(γ, L2ϱ[:,n]; xscale = :log10, yscale = :log10, linewidth = 3, marker = :circle, markersize = 5, label = L"|| {ϱ}-ϱ_h \, || \mathrm{level} = %$(nrefs[n])") # "||ϱ - ϱ_h|| level = $(nrefs[n])"
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
    @show data
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
        #Plotter.plot!(c, H1u[:,n]; xscale = :log10, yscale = :log10, linewidth = 3, marker = :circle, markersize = 5, label = L"||∇(\mathbf{u} - \mathbf{u}_h) \,|| \mathrm{level} = %$(nrefs[n])") # "||∇(u-u_h)|| level = $(nrefs[n])"
        Plotter.plot!(c, L2ϱ[:,n]; xscale = :log10, yscale = :log10, linewidth = 3, marker = :circle, markersize = 5, label = L"|| {ϱ}-ϱ_h \, || \mathrm{level} = %$(nrefs[n])") # "||ϱ - ϱ_h|| level = $(nrefs[n])"
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
    @show data
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
    yticks = [1e+2,1e+1,1e+0,1e-12,1e-11,1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2]
    xticks = c
    Plotter.plot(; show = true, size = (1600,1000), margin = 1Plots.cm, legendfontsize = 20, tickfontsize = 16, guidefontsize = 22)
    #for n = 1 : length(μ)
        #Plotter.plot!(c, H1u[:,n]; xscale = :log10, yscale = :log10, linewidth = 3, marker = :circle, markersize = 5, label = L"||∇(\mathbf{u} - \mathbf{u}_h) \,|| \mathrm{μ} = %$(μ[n])") # "||∇(u-u_h)|| μ = $(μ[n])"
        #Plotter.plot!(c, L2ϱ[:,n]; xscale = :log10, yscale = :log10, linewidth = 3, marker = :circle, markersize = 5, label = L"|| {ϱ}-ϱ_h \, || \mathrm{μ} = %$(μ[n])") # "||ϱ - ϱ_h|| μ= $(μ[n])"
    #end
    for n = 1 : length(μ)
        Plotter.plot!(c, L2u[:,n]; xscale = :log10, yscale = :log10, linewidth = 3, marker = :circle, markersize = 5, label = L"||\mathbf{u} - \mathbf{u}_h \, || \mathrm{μ} = %$(μ[n]) ")    
    end
    Plotter.plot!(; legend = :topright, xtick = xticks, yticks = yticks, ylim = (yticks[1]/2, 2*yticks[end]), xlabel = L"c_\mathrm{Ma}", gridalpha = 0.5, grid=true)
        
    ##
    print_table(c, L2u; xlabel = "c", ylabels = "|| u - u_h || ".* labels)
    #print_table(c, L2ϱ; xlabel = "c", ylabels = "|| ϱ - ϱ_h || ".* labels)
        
    ## save
    Plotter.savefig(filename_plots(data; free_parameter = "cμ"))
end
