using NumCompressibleFlows
using ExtendableFEM
using ExtendableFEMBase
using ExtendableGrids
using Triangulate
using SimplexGridFactory
using GridVisualize
using Symbolics
using LinearAlgebra

abstract type TestDensity end
abstract type LinearDensity <: TestDensity end
abstract type ExponentialDensity <: TestDensity end
abstract type ExponentialDensityRBR <: TestDensity end

abstract type TestVelocity end
abstract type ZeroVelocity <: TestVelocity end
abstract type ConstantVelocity <: TestVelocity end
abstract type P7VortexVelocity <: TestVelocity end
abstract type RigidBodyRotation <: TestVelocity end

abstract type EOSType end
abstract type IdealGasLaw <: EOSType end
abstract type PowerLaw{γ} <: EOSType end

abstract type GridFamily end
abstract type Mountain2D <: GridFamily end
abstract type UnitSquare <: GridFamily end
abstract type UniformUnitSquare <: UnitSquare end # no grid function yet
abstract type UnstructuredUnitSquare <: UnitSquare end

abstract type ConvectionType end
abstract type NoConvection <: ConvectionType end
abstract type StandardConvection <: ConvectionType end
abstract type OseenConvection <: ConvectionType end
abstract type RotationForm <: ConvectionType end

abstract type CoriolisType end
abstract type NoCoriolis <: CoriolisType end
abstract type BetaPlaneApproximation{β0} <: CoriolisType where {β0} end

function grid(::Type{<:Mountain2D}; nref = 1, kwargs...)
    grid_builder = (nref) -> SimplexGridFactory.simplexgrid(
            Triangulate;
            points = [0 0; 0.2 0; 0.3 0.2; 0.45 0.05; 0.55 0.35; 0.65 0.2; 0.7 0.3; 0.8 0; 1 0; 1 1 ; 0 1]',
            bfaces = [1 2; 2 3; 3 4; 4 5; 5 6; 6 7; 7 8; 8 9; 9 10; 10 11; 11 1]',
            bfaceregions = ones(Int, 11),
            regionpoints = [0.5 0.5;]',
            regionnumbers = [1],
            regionvolumes = [4.0^-(nref) / 2]
        )
    return grid_builder(nref)
end

function grid(::Type{<:UnstructuredUnitSquare}; nref = 1, kwargs...)
    grid_builder = (nref) -> SimplexGridFactory.simplexgrid(
            Triangulate;
            points = [0 0; 1.0 0; 1.0 1.0; 0.0 1.0]',
            bfaces = [1 2; 2 3; 3 4; 4 1]',
            bfaceregions = [1,2,3,4],
            regionpoints = [0.5 0.5;]',
            regionnumbers = [1],
            regionvolumes = [4.0^-(nref) / 2]
        )
    return grid_builder(nref)
end

function grid(::Type{<:UniformUnitSquare}; nref = 1, kwargs...)
    grid_builder = (nref) -> uniform_refine(simplexgrid(0:0.5:1,0:0.5:1), nref)
    return grid_builder(nref)
end

streamfunction(::Type{<:ZeroVelocity};ufac = 1, kwargs...) = 0*x
streamfunction(::Type{<:ConstantVelocity};ufac = 1, kwargs...) = - ufac * y
streamfunction(::Type{<:P7VortexVelocity};ufac = 1, kwargs...) = ufac * x^2 * y^2 * (x - 1)^2 * (y - 1)^2
streamfunction(::Type{<:RigidBodyRotation};ufac = 1, kwargs...) = (ufac/2) * (x^2 + y^2)

already_divfree(::Type{<:TestVelocity}) = false
already_divfree(::Type{<:RigidBodyRotation}) = true

inflow_regions(::Type{<:ZeroVelocity}, gridtype) = []
inflow_regions(::Type{<:ConstantVelocity}, gridtype) = [1,3,4]
inflow_regions(::Type{<:P7VortexVelocity}, gridtype) = []
inflow_regions(::Type{<:RigidBodyRotation}, ::Type{<:UnitSquare}) = [1,2]
outflow_regions(::Type{<:ZeroVelocity}, gridtype) = []
outflow_regions(::Type{<:ConstantVelocity}, gridtype) = [2]
outflow_regions(::Type{<:P7VortexVelocity}, gridtype) = []
outflow_regions(::Type{<:RigidBodyRotation}, ::Type{<:UnitSquare}) = [3,4]


angular_velocity(::Type{<:NoCoriolis}; kwargs...) = 0*x
angular_velocity(::Type{<:BetaPlaneApproximation{β0}}; kwargs...) where {β0} = β0*y
angular_velocity(::Type{<:BetaPlaneApproximation{β0}}, qpinfo) where {β0} = β0*qpinfo.x[2]

density(::Type{<:ExponentialDensity}; c = 1, M = 1, kwargs...) = M *  exp(- y^3 / (3 * c)) 
#density(::Type{<:ExponentialDensity}; c = 1, M = 1, kwargs...) = exp(- y /  c) / M  # e^(-y/c_M)/ M  ... Objection: whrere is x^3 ?
density(::Type{<:ExponentialDensityRBR}; c = 1, M = 1, kwargs...) =  M * exp( (x^2+y^2) /  (2*c)) 
density(::Type{<:LinearDensity}; c = 1, M = 1, kwargs...) = ( 1+(x-(1/2))/c )/M 


function prepare_data(
    TVT::Type{<:TestVelocity},
    TDT::Type{<:TestDensity},
    EOSType::Type{<:EOSType};
    M = 1,
    c = 1,
    μ = 1,
    ufac = 1,
    laplacian_in_rhs = true,
    pressure_in_f = false,
    λ = -2*μ / 3,
    convectiontype = NoConvection,
    coriolistype = NoCoriolis,
    kwargs...)

    ## get stream function and density for test types
    ξ = streamfunction(TVT;ufac = ufac, kwargs...)
    ϱ = density(TDT; c = c, M = M, kwargs...) 
    
    ## gradient of stream function
    ∇ξ = Symbolics.gradient(ξ, [x, y])

    ## velocity u = curl ξ / ϱ 

    if already_divfree(TVT)
        u = [-∇ξ[2], ∇ξ[1]]
    else 
        u = [-∇ξ[2], ∇ξ[1]] ./ ϱ
    end
    #u = [-∇ξ[2], ∇ξ[1]] ./ ϱ
    ## gradient of velocity
    ∇u = Symbolics.jacobian(u, [x, y])
    ∇u_reshaped = [∇u[1, 1], ∇u[1, 2], ∇u[2, 1], ∇u[2, 2]] # [∇u1,∇u2]

    ## Laplacian
    Δu = [
        (Symbolics.gradient(∇u[1, 1], [x]) + Symbolics.gradient(∇u[1, 2], [y]))[1],
        (Symbolics.gradient(∇u[2, 1], [x]) + Symbolics.gradient(∇u[2, 2], [y]))[1],
    ]
    ## for the λ∇(∇⋅u) term 
    ∇divu = [
        (Symbolics.gradient(∇u[1, 1], [x]) + Symbolics.gradient(∇u[2, 2], [x]))[1],
        (Symbolics.gradient(∇u[1, 1], [y]) + Symbolics.gradient(∇u[2, 2], [y]))[1],
    ]

    ## for the convection term ϱ(u ⋅∇)u
    if convectiontype !== NoConvection
        conv = [ u[1] * ∇u[1, 1] + u[2] * ∇u[1, 2],
                 u[1] * ∇u[2, 1] +  u[2] * ∇u[2, 2] ] .* ϱ
    else
        conv = [0*x, 0*x]
    end
    @show conv 

    if coriolistype !== NoCoriolis
        ω = angular_velocity(coriolistype; kwargs...)
        conv += 2*ϱ*ω*[-u[2],  u[1]] # Marwa 30 May: conv += 2*ϱ*ω*[u[2], - u[1]] ? 
    end
    

    # L(u) + ∇p = f + ϱg with L(u) = -μ Δu - λ ∇(∇⋅u) + ϱ(u.∇)u 
    if pressure_in_f # Gradient_robustness 
        if EOSType <: IdealGasLaw
            f = c * Symbolics.gradient(ϱ, [x, y]) # f = ∇p 
        elseif EOSType <: PowerLaw
            γ = EOSType.parameters[1]
            @assert γ > 1
            f =  c * Symbolics.gradient(ϱ^γ, [x, y]) # f = ∇p 
        end
        if laplacian_in_rhs
            g =  0 * Δu 
            f += - μ * Δu - λ*∇divu + conv   # f = L(u) + ∇p (everything in f)
        else
            g = - μ * Δu / ϱ - λ*∇divu / ϱ + conv /ϱ  # ϱg = L(u)
        end
           
    else # Well_balancedness 
        if EOSType <: IdealGasLaw
            g = c * Symbolics.gradient(log(ϱ), [x, y]) # ϱg = ∇p  
        elseif EOSType <: PowerLaw
            γ = EOSType.parameters[1]
            @assert γ > 1
            g =  c* γ*ϱ^(γ-2) * Symbolics.gradient(ϱ, [x, y]) # ϱg = ∇p 
        end
        if laplacian_in_rhs 
            f =  0 * Δu 
            g += - μ * Δu / ϱ  - λ*∇divu / ϱ + conv /ϱ  # ϱg = L(u) + ∇p (everything in g)
            
        else
            f = - μ * Δu  - λ*∇divu + conv  # f = L(u)

        
    end
end
@info f, g

## Christian's def of f & g 
   #=
    if EOSType <: IdealGasLaw
        g = c * Symbolics.gradient(log(ϱ), [x, y])
    elseif EOSType <: PowerLaw
        γ = EOSType.parameters[1]
        @assert γ > 1
        g =  c* γ*ϱ^(γ-2) * Symbolics.gradient(ϱ, [x, y]) # testing welll-balance
    end

    ## gravity ϱg = - Δu + ϱ∇log(ϱ)
    if laplacian_in_rhs 
        f = - μ * Δu  - λ*∇divu   
        
    else
        g -= - μ * Δu / ϱ  - λ*∇divu / ϱ
        f = 0 * Δu 
    end
   =#

    #Δu = Symbolics.derivative(∇u[1,1], [x]) + Symbolics.derivative(∇u[2,2], [y])

    ϱ_eval = build_function(ϱ, x, y, expression = Val{false})
    u_eval = build_function(u, x, y, expression = Val{false})[2]
    ∇u_eval = build_function(∇u_reshaped, x, y, expression = Val{false})[2]
    g_eval = build_function(g, x, y, expression = Val{false})[2]
    f_eval = build_function(f, x, y, expression = Val{false})[2]

    ϱ!(result, qpinfo) = (result[1] = ϱ_eval(qpinfo.x[1], qpinfo.x[2]);)
    function kernel_gravity!(result, input, qpinfo)
        g_eval(result, qpinfo.x[1], qpinfo.x[2]) # qpinfo.x[1] is x and qpinfo.x[2] is y
        return result .*= input[1] # what does it mean ?
    end
    function kernel_rhs!(result, qpinfo)
        return f_eval(result, qpinfo.x[1], qpinfo.x[2])
    end

    u!(result, qpinfo) = (u_eval(result, qpinfo.x[1], qpinfo.x[2]);)
    ∇u!(result, qpinfo) = (∇u_eval(result, qpinfo.x[1], qpinfo.x[2]);)
    return ϱ!, kernel_gravity!, kernel_rhs!, u!, ∇u!
end


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
    # data of the problem
    velocitytype = data["velocitytype"]
    densitytype = data["densitytype"]
    eostype = data["eostype"]
    gridtype = data["gridtype"]
    convectiontype = data["convectiontype"]
    coriolistype = data["coriolistype"]
    laplacian_in_rhs = data["laplacian_in_rhs"]
    stab1 = data["stab1"]
    stab2 = data["stab2"]
    
    # sname
    sname = savename((@dict μ λ γ c M τfac ufac nrefs order reconstruct target_residual maxsteps pressure_stab bonus_quadorder velocitytype densitytype eostype gridtype convectiontype coriolistype laplacian_in_rhs stab1 stab2))
    sname = "data/projects/compressible_stokes/" * sname
    return sname
end

function run_single(data; kwargs...)
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

    ## load data for testcase
    ϱ!, kernel_gravity!, kernel_rhs!, u!, ∇u! = prepare_data(velocitytype, densitytype, eostype; laplacian_in_rhs = laplacian_in_rhs, pressure_in_f = pressure_in_f, M = M, c = c, μ = μ, λ = λ,γ=γ, ufac = ufac , nrefs = nrefs)
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
        assemble!(brho, LinearOperatorDG(kernel_inflow!(u!,ϱ!), [id(1)]; factor = -1 , bonus_quadorder = bonus_quadorder_bnd, entities = ON_BFACES, regions = rinflow, kwargs...))    
    end
    if length(routflow) > 0
        assemble!(D, BilinearOperatorDG(kernel_outflow!(u!), [id(1)]; factor = 1, bonus_quadorder = bonus_quadorder_bnd, entities = ON_BFACES, regions = routflow, kwargs...)) 
    end

    if stab1[2] > 0
        assemble!(D, BilinearOperatorDG(multiply_h_bilinear!(stab1[1]),[jump(id(1))]; factor = stab1[2], entities = ON_IFACES, kwargs...)) 
    end
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
    ErrorIntegratorExact = ItemIntegrator(exact_error!(u!, ∇u!, ϱ!), [id(u), grad(u), id(ϱ)]; resultdim = 9, quadorder = 2 * (order + 1), kwargs...)
    #NDofs = zeros(Int, nrefs)
    #Results = zeros(Float64, nrefs, 5) # it is a matrix whose rows are levels and columns are 
    ## prepare error calculation
    EnergyIntegrator = ItemIntegrator(energy_kernel!, [id(u)]; resultdim = 1, quadorder = 2 * (order + 1), kwargs...)
    ErrorIntegratorExact = ItemIntegrator(exact_error!(u!, ∇u!, ϱ!), [id(u), grad(u), id(ϱ)]; resultdim = 9, quadorder = 2 * (order + 1), kwargs...)
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

    ## calculate error
    error = evaluate(ErrorIntegratorExact, sol)
    data["Error(L2,u)"] = sqrt(sum(view(error, 1, :)) + sum(view(error, 2, :))) # u = (u_1,u_2)
    data["Error(H1,u)"] = sqrt(sum(view(error, 3, :)) + sum(view(error, 4, :)) + sum(view(error, 5, :)) + sum(view(error, 6, :))) # ∇u = (∂_x u_1,∂_y u_1, ∂_x u_2, ∂_y u_2 )
    data["Error(L2,ϱ)"] = sqrt(sum(view(error, 7, :))) # ϱ
    data["Error(L2,ϱu)"]  = sqrt(sum(view(error, 8, :)) + sum(view(error, 9, :))) # (ρ u_1 - ρ u_2)
    data["nits"] = nits


    ## save to data folder
    return data


end
