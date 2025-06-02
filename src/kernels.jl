
function stab_kernel!(result, p, u, qpinfo)
    return result[1] = p[1] #*abs(u[1] + u[2])
end

## kernel for (uϱ, ∇λ) ON_CELLS in continuity equation
function kernel_continuity!(result, ϱ, u, qpinfo)
    result[1] = ϱ[1] * u[1]
    return result[2] = ϱ[1] * u[2]
end

## kernel for ((ϱu ⋅ ∇)u, ∇v) ON_CELLS in momentum balance
function kernel_standardconvection_linearoperator!(result, args, qpinfo)
    u = view(args,1:2)
    ∇u = view(args, 3:6)
    ϱ = view(args, 7)
    result[1] = ϱ[1] * dot(u, view(∇u,1:2))
    result[2] = ϱ[1] * dot(u, view(∇u,3:4))
    return nothing
end

## kernel for (ϱ rotu × u, v) - 1/2 (ϱu⋅u,div(v)) ON_CELLS in momentum balance
function kernel_rotationform_linearoperator!(result, args, qpinfo)
    u = view(args,1:2)
    curlu = view(args, 3)
    ϱ = view(args, 4)
    result[1] = -ϱ[1] * curlu[1] * u[2]
    result[2] = ϱ[1] * curlu[1] * u[1]
    result[3] = -ϱ[1] * dot(u,u)*0.5
    return nothing
end

## kernel for (2ϱω×u, v) ON_CELLS in momentum balance
function kernel_coriolis_linearoperator!(coriolistype)
    function closure(result, args, qpinfo)
        ω = angular_velocity(coriolistype, qpinfo)
        u = view(args, 1:2)
        ϱ = view(args, 3)
        result[1] = -ϱ[1] * 2 * ω * u[2]
        result[2] =  ϱ[1] * 2 * ω * u[1]
        return nothing
    end
end

## kernel for ((ϱβ ⋅ ∇)u, ∇v) ON_CELLS in momentum balance
function kernel_oseenconvection_linearoperator!(β!)
    βval = zeros(Float64, 2)
    function closure(result, args, qpinfo)
        β!(βval, qpinfo)
        ∇u = view(args, 1:4)
        ϱ = view(args, 5)
        result[1] = ϱ[1] * dot(βval, view(∇u,1:2))
        result[2] = ϱ[1] * dot(βval, view(∇u,3:4))
        return nothing
    end
end


## kernel for ((β ⋅ ∇)u, ∇v) ON_CELLS in momentum balance
function kernel_oseenconvection!(β!, ϱ!)
    βval = zeros(Float64, 2)
    ϱval = zeros(Float64, 1)
    function closure(result, args, qpinfo)
        β!(βval, qpinfo)
        ϱ!(ϱval, qpinfo)
        ∇u = view(args, 1:4)
        result[1] = ϱval[1] * dot(βval, view(∇u,1:2))
        result[2] = ϱval[1] * dot(βval, view(∇u,3:4))
        return nothing
    end
end

## kernel for (u⋅n ϱ^upw, λ) ON_IFACES in continuity equation
function kernel_upwind!(result, input, u, qpinfo) # u = [id(u)], input = [this(id(ϱ)), other(id(ϱ))]
    flux = dot(u, qpinfo.normal) # u * n
    return if flux > 0
        result[1] = input[1] * flux # rho_left * flux 
    else
        result[1] = input[2] * flux # rho_right * flux
    end
end


## kernel for inflow boundary
function kernel_inflow!(u!,ϱ!) # test function is [id(ϱ)] = λ
    uval = zeros(Float64, 2)
    ϱval = zeros(Float64, 1)
    function closure(result, qpinfo)
        u!(uval, qpinfo)
        flux = dot(uval, qpinfo.normal)
        ϱ!(ϱval, qpinfo)
        result[1] = ϱval[1] * flux
    end
end


## kernel for outflow boundary, this function is not clear what it does ?
function kernel_outflow!(u!)
    uval = zeros(Float64, 2)
    function closure(result, args, qpinfo) # args = [id(ϱ)], [id(ϱ)] ?
        u!(uval, qpinfo)
        flux = dot(uval, qpinfo.normal)
        result[1] = args[1] * flux 
    end
end

## kernel for exact error calculation
function exact_error!(u!, ∇u!, ϱ!)
    return function closure(result, u, qpinfo)
        u!(view(result, 1:2), qpinfo)
        ∇u!(view(result, 3:6), qpinfo)
        ϱ!(view(result, 7), qpinfo)
        result[8] = result[1] * result[7]
        result[9] = result[2] * result[7]
        view(result, 1:7) .-= u
        result[8] -= u[1] * u[7]
        result[9] -= u[2] * u[7]
        return result .= result .^ 2
    end
end

## kernel for gravity term in testcase 1
function standard_gravity!(result, ϱ, qpinfo)
    result[1] = 0
    return result[2] = -ϱ[1] # Marwa: why ? g = ∇ψ = c_M ∇(log ρ)
end

function energy_kernel!(result, u, qpinfo)
    return result[1] = dot(u, u) / 2
end

function eos!(::Type{<:IdealGasLaw}; kawrgs...)
    function eos_idealgas!(result, input, qpinfo)
        return result[1] = input[1]
    end
    return eos_idealgas!
end


function eos!(::Type{<:PowerLaw{γ}}; kawrgs...) where {γ}
    function eos_powerlaw!(result, input, qpinfo)
        return result[1] = input[1]^γ
    end
    return eos_powerlaw!
end