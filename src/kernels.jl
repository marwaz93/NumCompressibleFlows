
function stab_kernel!(result, p, u, qpinfo)
    return result[1] = p[1] #*abs(u[1] + u[2])
end

## kernel for (uϱ, ∇λ) ON_CELLS in continuity equation
function kernel_continuity!(result, ϱ, u, qpinfo)
    result[1] = ϱ[1] * u[1]
    return result[2] = ϱ[1] * u[2]
end

## kernel for (u⋅n ϱ^upw, λ) ON_IFACES in continuity equation
function kernel_upwind!(result, input, u, qpinfo)
    flux = dot(u, qpinfo.normal) # u * n
    return if flux > 0
        result[1] = input[1] * flux # rho_left * flux
    else
        result[1] = input[2] * flux # rho_righ * flux
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