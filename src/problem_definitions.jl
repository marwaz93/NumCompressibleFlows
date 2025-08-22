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
    γ = 1,
    ufac = 1,
    laplacian_in_rhs = true,
    pressure_in_f = false,
    λ = 0,
    convectiontype = NoConvection,
    coriolistype = NoCoriolis,
    kwargs...)

    @info "Marwaaa Velocity is $TVT"
    
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
        if EOSType <: IdealGasLaw && γ == 1
            f = c * Symbolics.gradient(ϱ, [x, y]) # f = ∇p 
        elseif EOSType <: PowerLaw || γ > 1
            if EOSType <: PowerLaw
                γ = EOSType.parameters[1]
            end
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
        if EOSType <: IdealGasLaw && γ == 1
            g = c * Symbolics.gradient(log(ϱ), [x, y]) # ϱg = ∇p  
        elseif EOSType <: PowerLaw || γ > 1
            if EOSType <: PowerLaw
                γ = EOSType.parameters[1]
            end
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


