abstract type TestDensity end
abstract type ExponentialDensity <: TestDensity end
abstract type LinearDensity <: TestDensity end

abstract type TestVelocity end
abstract type ZeroVelocity <: TestVelocity end
abstract type P7VortexVelocity <: TestVelocity end

abstract type EOSType end
abstract type IdealGasLaw <: EOSType end
abstract type PowerLaw{γ} <: EOSType end


abstract type GridFamily end
abstract type Mountain2D <: GridFamily end
abstract type UniformUnitSquare <:GridFamily end
abstract type UnstructuredUnitSquare <: GridFamily end

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
            bfaceregions = ones(Int, 4),
            regionpoints = [0.5 0.5;]',
            regionnumbers = [1],
            regionvolumes = [4.0^-(nref) / 2]
        )
    return grid_builder(nref)
end


streamfunction(::Type{<:ZeroVelocity}; kwargs...) = 0*x
streamfunction(::Type{<:P7VortexVelocity}; kwargs...) = x^2 * y^2 * (x - 1)^2 * (y - 1)^2

density(::Type{<:ExponentialDensity}; c = 1, M = 1, kwargs...) = exp(- y / c) / M
density(::Type{<:LinearDensity}; c = 1, M = 1, kwargs...) = ( 1+(x-(1/2))/c )/M 


function prepare_data(TVT::Type{<:TestVelocity}, TDT::Type{<:TestDensity}, EOSType::Type{<:EOSType}; M = 1, c = 1, μ = 1, ufac = 100, laplacian_in_rhs = true, kwargs...)

    ## get stream function and density for test types
    ξ = streamfunction(TVT; kwargs...)
    ϱ = density(TDT; kwargs...)
    
    ## gradient of stream function
    ∇ξ = Symbolics.gradient(ξ, [x, y])

    ## velocity u = curl ξ / ϱ
    u = [-∇ξ[2], ∇ξ[1]] ./ ϱ

    ## gradient of velocity
    ∇u = Symbolics.jacobian(u, [x, y])
    ∇u_reshaped = [∇u[1, 1], ∇u[1, 2], ∇u[2, 1], ∇u[2, 2]]

    ## Laplacian
    Δu = [
        (Symbolics.gradient(∇u[1, 1], [x]) + Symbolics.gradient(∇u[1, 2], [y]))[1],
        (Symbolics.gradient(∇u[2, 1], [x]) + Symbolics.gradient(∇u[2, 2], [y]))[1],
    ]

    if EOSType <: IdealGasLaw
        g = c * Symbolics.gradient(log(ϱ), [x, y])
    elseif EOSType <: PowerLaw
        γ = EOSType.parameters[1]
        @assert γ > 1
        g =  c* γ*ϱ^(γ-2) * Symbolics.gradient(ϱ, [x, y])
    end

    ## gravity ϱg = - Δu + ϱ∇log(ϱ)
    if laplacian_in_rhs
        f = - μ * Δu
        # Marwa therfore, /gradient p = c * /rho * /gradient /psi 
    else
        g -= - μ * Δu / ϱ
        f = 0
    end

    #Δu = Symbolics.derivative(∇u[1,1], [x]) + Symbolics.derivative(∇u[2,2], [y])

    ϱ_eval = build_function(ϱ, x, y, expression = Val{false})
    u_eval = build_function(u, x, y, expression = Val{false})[2]
    ∇u_eval = build_function(∇u_reshaped, x, y, expression = Val{false})[2]
    g_eval = build_function(g, x, y, expression = Val{false})[2]
    f_eval = build_function(f, x, y, expression = Val{false})[2]

    ϱ!(result, qpinfo) = (result[1] = ϱ_eval(qpinfo.x[1], qpinfo.x[2]);)
    function kernel_gravity!(result, input, qpinfo)
        g_eval(result, qpinfo.x[1], qpinfo.x[2]) # qpinfo.x[1] is x and qpinfo.x[2] is y
        return result .*= input[1] # input is [id_u], [id(ϱ)] ??
    end
    function kernel_rhs!(result, qpinfo)
        return f_eval(result, qpinfo.x[1], qpinfo.x[2])
    end

    u!(result, qpinfo) = (u_eval(result, qpinfo.x[1], qpinfo.x[2]);)
    ∇u!(result, qpinfo) = (∇u_eval(result, qpinfo.x[1], qpinfo.x[2]);)
    return ϱ!, kernel_gravity!, kernel_rhs!, u!, ∇u!
end
