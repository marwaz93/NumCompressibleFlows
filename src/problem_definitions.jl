using NumCompressibleFlows
using ExtendableFEM
using ExtendableFEMBase
using ExtendableGrids
using Triangulate
using SimplexGridFactory
using GridVisualize
import Symbolics
using LinearAlgebra

abstract type TestDensity end
abstract type ConstantDensity <: TestDensity end
abstract type LinearDensity <: TestDensity end
abstract type ExponentialDensity <: TestDensity end
abstract type ExponentialDensityRBR <: TestDensity end

abstract type TestVelocity end
abstract type ZeroVelocity <: TestVelocity end
abstract type ConstantVelocity <: TestVelocity end
abstract type LinearVelocity <: TestVelocity end
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
abstract type UnitCube <: GridFamily end
abstract type UnstructuredUnitCube <: UnitCube end

abstract type ConvectionType end
abstract type NoConvection <: ConvectionType end
abstract type StandardConvection <: ConvectionType end
abstract type OseenConvection <: ConvectionType end
abstract type RotationForm <: ConvectionType end
abstract type KarperConvection <: ConvectionType end
abstract type NewConvection <: ConvectionType end

abstract type UpwindType end
abstract type StandardUpwind <: UpwindType end
abstract type PointwiseUpwind <: UpwindType end

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

## tetrahedral grid of the unit cube via TetGen
## boundary face regions: 1: y=0, 2: x=1, 3: y=1, 4: x=0, 5: z=0, 6: z=1
function grid(::Type{<:UnstructuredUnitCube}; nref = 1, kwargs...)
    grid_builder = (nref) -> SimplexGridFactory.simplexgrid(
            TetGen;
            points = [0 0 0; 1 0 0; 1 1 0; 0 1 0; 0 0 1; 1 0 1; 1 1 1; 0 1 1]',
            bfaces = [1 2 6; 1 6 5; 2 3 7; 2 7 6; 3 4 8; 3 8 7; 4 1 5; 4 5 8; 1 2 3; 1 3 4; 5 6 7; 5 7 8]',
            bfaceregions = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
            regionpoints = [0.5 0.5 0.5]',
            regionnumbers = [1],
            regionvolumes = [8.0^-(nref) / 6]
        )
    return grid_builder(nref)
end

dim(::Type{<:GridFamily}) = 2
dim(::Type{<:UnitCube}) = 3

gamma(::Type{<:PowerLaw{γ}}) where {γ} = γ
gamma(::Type{<:IdealGasLaw}) = 1

streamfunction(::Type{<:ZeroVelocity};ufac = 1, kwargs...) = 0*x
streamfunction(::Type{<:ConstantVelocity};ufac = 1, kwargs...) = - ufac * y
streamfunction(::Type{<:LinearVelocity};ufac = 1, kwargs...) = -  ufac * x * y
streamfunction(::Type{<:P7VortexVelocity};ufac = 1, kwargs...) = ufac * x^2 * y^2 * (x - 1)^2 * (y - 1)^2
streamfunction(::Type{<:RigidBodyRotation};ufac = 1, kwargs...) = (ufac/2) * (x^2 + y^2)

already_divfree(::Type{<:TestVelocity}) = false
already_divfree(::Type{<:RigidBodyRotation}) = true

inflow_regions(::Type{<:ZeroVelocity}, gridtype) = []
inflow_regions(::Type{<:ConstantVelocity}, gridtype) = [1,3,4]
inflow_regions(::Type{<:ConstantVelocity}, ::Type{<:UnitCube}) = [1,3,4,5,6]
inflow_regions(::Type{<:LinearVelocity}, gridtype) = [3]
inflow_regions(::Type{<:P7VortexVelocity}, gridtype) = []
inflow_regions(::Type{<:RigidBodyRotation}, ::Type{<:UnitSquare}) = [1,2]
inflow_regions(::Type{<:RigidBodyRotation}, ::Type{<:UnitCube}) = [1,2]
outflow_regions(::Type{<:ZeroVelocity}, gridtype) = []
outflow_regions(::Type{<:ConstantVelocity}, gridtype) = [2]
outflow_regions(::Type{<:ConstantVelocity}, ::Type{<:UnitCube}) = [2]
outflow_regions(::Type{<:P7VortexVelocity}, gridtype) = []
outflow_regions(::Type{<:RigidBodyRotation}, ::Type{<:UnitSquare}) = [3,4]
outflow_regions(::Type{<:RigidBodyRotation}, ::Type{<:UnitCube}) = [3,4]

## boundary regions with u ⋅ n = 0 but (interpolated) Dirichlet velocity data
## (e.g. the z-faces of the unit cube for the z-invariant embedded vortices,
## where the exact velocity is tangential and nonzero)
wall_regions(::Type{<:TestVelocity}, gridtype) = Int[]
wall_regions(::Type{<:P7VortexVelocity}, ::Type{<:UnitCube}) = [5,6]
wall_regions(::Type{<:RigidBodyRotation}, ::Type{<:UnitCube}) = [5,6]


angular_velocity(::Type{<:NoCoriolis}; kwargs...) = 0*x
angular_velocity(::Type{<:BetaPlaneApproximation{β0}}; kwargs...) where {β0} = β0*y
angular_velocity(::Type{<:BetaPlaneApproximation{β0}}, qpinfo) where {β0} = β0*qpinfo.x[2]

density(::Type{<:ExponentialDensity}; c = 1, M = 1, kwargs...) = M *  exp(- y^3 / (3 * c)) 
#density(::Type{<:ExponentialDensity}; c = 1, M = 1, kwargs...) = exp(- y /  c) / M  # e^(-y/c_M)/ M  ... Objection: whrere is x^3 ?
density(::Type{<:ExponentialDensityRBR}; c = 1, M = 1, kwargs...) =  M * exp( (x^2+y^2) /  (2*c)) 
density(::Type{<:LinearDensity}; c = 1, M = 1, kwargs...) = ( 1+(x-(1/2))/c )/M 
density(::Type{<:ConstantDensity}; c = 1, M = 1, kwargs...) = M 


function prepare_data(
    TVT::Type{<:TestVelocity},
    TDT::Type{<:TestDensity},
    EOSType::Type{<:EOSType};
    M = 1,
    c = 1,
    μ = 1,
    ufac = 1,
    others_in_f  = true,
    pressure_in_f = false,
    λ = 0,
    convectiontype = NoConvection,
    coriolistype = NoCoriolis,
    incompressible = isinf(c),
    dim = 2,
    kwargs...)
    
    ## get stream function and density for test types
    ## in the incompressible limit (c = Inf) the density is the constant ϱ = M
    γ = gamma(EOSType)
    ξ = streamfunction(TVT;ufac = ufac, kwargs...)
    ϱ = incompressible ? M + 0*x : density(TDT; c = c, M = M, kwargs...)
    xs = dim == 2 ? [x, y] : [x, y, z]
    ∂(f, v) = (g = Symbolics.gradient(f, [v]); isempty(g) ? 0 * x : g[1])

    ## gradient of stream function (always in the (x,y)-plane)
    ∇ξ = Symbolics.gradient(ξ, [x, y])

    ## velocity u = curl ξ / ϱ 
    if already_divfree(TVT)
        u2 = [-∇ξ[2], ∇ξ[1]]
    else 
        u2 = [-∇ξ[2], ∇ξ[1]] ./ ϱ
    end
    ## in 3D the planar field is embedded (z-invariant) as u = (u1, u2, 0),
    ## which is still exactly divergence-free
    u = dim == 2 ? u2 : [u2[1], u2[2], 0 * x]

    ## gradient of velocity (row-major: [∇u_1, ∇u_2, (∇u_3)] matching grad(u) layout)
    ∇u = Symbolics.jacobian(u, xs)
    ∇u_reshaped = [∇u[i, j] for i in 1:dim for j in 1:dim]

    ## Laplacian
    Δu = [sum(∂(∇u[i, j], xs[j]) for j in 1:dim) for i in 1:dim]
    ## for the λ∇(∇⋅u) term 
    divu = sum(∇u[k, k] for k in 1:dim)
    ∇divu = [∂(divu, xs[i]) for i in 1:dim]

    ## for the convection term ϱ(u ⋅∇)u
    if convectiontype !== NoConvection
        conv = [ϱ * sum(u[j] * ∇u[i, j] for j in 1:dim) for i in 1:dim]
    else
        conv = [0 * x for _ in 1:dim]
    end

    if coriolistype !== NoCoriolis
        if dim == 3
            @error "Coriolis terms are only supported in 2D so far (β-plane)."
        end
        ω = angular_velocity(coriolistype; kwargs...)
        conv += 2*ϱ*ω*[-u[2],  u[1]] # Marwa 30 May: conv += 2*ϱ*ω*[u[2], - u[1]] ? 
    end
    

    # L(u) + ∇p = f + ϱg with L(u) = -μ Δu - λ ∇(∇⋅u) + ϱ(u.∇)u 
    f = [0 * x for _ in 1:dim]
    g = [0 * x for _ in 1:dim]
    if incompressible
        # in the limit c → Inf the manufactured density is constant (ϱ = M) and
        # all pressure-gradient terms ∇p = c∇ϱ (or c∇ϱ^γ, and ϱg = ∇p in the
        # well-balanced variant) are pure gradients which are ignored here;
        # they are absorbed by the pressure Lagrange multiplier of the
        # incompressible problem and do not change the exact velocity
    elseif pressure_in_f # Gradient_robustness 
        if EOSType <: IdealGasLaw && γ == 1
            f = c * Symbolics.gradient(ϱ, xs) # f = ∇p
        elseif EOSType <: IdealGasLaw && γ > 1
            @error "IdealGasLaw with γ > 1 is not supported, please change to PowerLaw{γ}."
        elseif EOSType <: PowerLaw || γ > 1
            if EOSType <: PowerLaw
                γ = EOSType.parameters[1]
            end
            @assert γ > 1
            f =  c * Symbolics.gradient(ϱ^γ, xs) # f = ∇p
        end
    else # Well_balancedness 
        if EOSType <: IdealGasLaw && γ == 1
            g = c * Symbolics.gradient(log(ϱ), xs) # ϱg = ∇p  
        elseif EOSType <: PowerLaw || γ > 1
            if EOSType <: PowerLaw
                γ = EOSType.parameters[1]
            end
            @assert γ > 1
            g =  c* γ*ϱ^(γ-2) * Symbolics.gradient(ϱ, xs) # ϱg = ∇p 
        end
    end

    if others_in_f
        f += - μ * Δu - λ*∇divu + conv   # f also has L(u) 
    else
        g += - μ * Δu / ϱ - λ*∇divu / ϱ + conv /ϱ  # ϱg also has L(u) 
    end

    @info "conv [$convectiontype] = ", conv
    @info "f = ", f
    @info "g = ", g

    if dim == 2
        ϱ_eval = build_function(ϱ, x, y, expression = Val{false})
        u_eval = build_function(u, x, y, expression = Val{false})[2]
        ∇u_eval = build_function(∇u_reshaped, x, y, expression = Val{false})[2]
        g_eval = build_function(g, x, y, expression = Val{false})[2]
        f_eval = build_function(f, x, y, expression = Val{false})[2]
    else
        ϱ_eval = build_function(ϱ, x, y, z, expression = Val{false})
        u_eval = build_function(u, x, y, z, expression = Val{false})[2]
        ∇u_eval = build_function(∇u_reshaped, x, y, z, expression = Val{false})[2]
        g_eval = build_function(g, x, y, z, expression = Val{false})[2]
        f_eval = build_function(f, x, y, z, expression = Val{false})[2]
    end
    coords(qpinfo) = dim == 2 ? (qpinfo.x[1], qpinfo.x[2]) : (qpinfo.x[1], qpinfo.x[2], qpinfo.x[3])

    ϱ!(result, qpinfo) = (result[1] = ϱ_eval(coords(qpinfo)...);)
    function kernel_gravity!(result, input, qpinfo)
        g_eval(result, coords(qpinfo)...) # qpinfo.x[1] is x, qpinfo.x[2] is y, (qpinfo.x[3] is z)
        return result .*= input[1] # multiply with discrete density ϱ
    end
    function kernel_rhs!(result, qpinfo)
        return f_eval(result, coords(qpinfo)...)
    end

    u!(result, qpinfo) = (u_eval(result, coords(qpinfo)...);)
    ∇u!(result, qpinfo) = (∇u_eval(result, coords(qpinfo)...);)
    return ϱ!, kernel_gravity!, kernel_rhs!, u!, ∇u!
end


