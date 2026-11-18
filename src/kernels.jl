
function stab_kernel!(result, p, u, qpinfo)
    return result[1] = p[1] #*abs(u[1] + u[2])
end

## kernel for (uϱ, ∇λ) ON_CELLS in continuity equation
function kernel_continuity!(result, ϱ, u, qpinfo)
    for i in eachindex(result)
        result[i] = ϱ[1] * u[i]
    end
    return nothing
end

## kernel for ((ϱu ⋅ ∇)u, ∇v) ON_CELLS in momentum balance
## args layout: [id(u)(d), grad(u)(d²), ϱ(1)]
function kernel_standardconvection_linearoperator!(result, args, qpinfo)
    d = length(result)
    u = view(args, 1:d)
    ∇u = view(args, d+1:d+d*d)
    ϱ = view(args, d+d*d+1)
    for i in 1:d
        result[i] = ϱ[1] * dot(u, view(∇u, (i-1)*d+1:i*d))
    end
    return nothing
end


## kernel for ((ϱu ⋅ ∇)u, ∇v) ON_CELLS in momentum balance
function kernel_standardconvection_incompressible_linearoperator!(result, args, qpinfo)
    d = length(result)
    u = view(args, 1:d)
    ∇u = view(args, d+1:d+d*d)
    for i in 1:d
        result[i] = dot(u, view(∇u, (i-1)*d+1:i*d))
    end
    return nothing
end

function density_jump_stab_kernel!(exponent, γ)
    function closure(result, input, args, qpinfo)
        ζ = max(0,2 - γ)
        result .= input * abs(args[1])^ζ * qpinfo.volume^exponent
    end
end

function velocity_jump_stab_kernel!(exponent, γ)
    function closure(result, args, qpinfo)
        ζ = max(0,2 - γ)
        result .= args[1] .* args[2] .* qpinfo.volume^exponent
    end
end

function multiply_h_linear!(power)
    function closure(result, qpinfo)
        result .*= qpinfo.volume^power
    end
end

function multiply_h_bilinear!(power)
    function closure(result, input, qpinfo)
        result .= input * qpinfo.volume^power
    end
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

## Bernoulli consistency correction for the rotation form: elementwise IBP of
## (ϱ∇(½‖u‖²), v) leaves, for broken ϱ, the additional face term
##   -Σ_F ∫_F ½‖u‖² jump(ϱ) v⋅n,
## which vanishes for smooth ϱ and restores algebraic consistency with the
## standard convection form (verified up to round-off by assembly identity tests).
## kernel for -∫_F ½‖u‖² ϱjump n ⋅ v on ON_FACES in momentum balance
function kernel_rotationform_bernoulli_jump!(result, args, qpinfo) # args = [id(u), jump(ϱ)]
    d = length(result)
    u = view(args, 1:d)
    ϱjump = args[d+1]
    c = -0.5 * dot(u, u) * ϱjump
    for i in 1:d
        result[i] = c * qpinfo.normal[i]
    end
    return nothing
end

## kernel for (ϱ rotu × u, v) ON_CELLS in momentum balance
function kernel_new_rotationform_linearoperator!(result, args, qpinfo)
    u = view(args,1:2)
    curlu = view(args, 3)
    ϱ = view(args, 4)
    result[1] = -ϱ[1] * curlu[1] * u[2]
    result[2] = ϱ[1] * curlu[1] * u[1]
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
        d = length(result)
        length(βval) == d || resize!(βval, d)
        β!(βval, qpinfo)
        ∇u = view(args, 1:d*d)
        ϱ = view(args, d*d+1)
        for i in 1:d
            result[i] = ϱ[1] * dot(βval, view(∇u, (i-1)*d+1:i*d))
        end
        return nothing
    end
end


## kernel for ((β ⋅ ∇)u, ∇v) ON_CELLS in momentum balance
function kernel_oseenconvection_linearoperator!(β!, ϱ!)
    βval = zeros(Float64, 2)
    ϱval = zeros(Float64, 1)
    function closure(result, args, qpinfo)
        d = length(result)
        length(βval) == d || resize!(βval, d)
        β!(βval, qpinfo)
        ϱ!(ϱval, qpinfo)
        ∇u = view(args, 1:d*d)
        for i in 1:d
            result[i] = ϱval[1] * dot(βval, view(∇u, (i-1)*d+1:i*d))
        end
        return nothing
    end
end

function kernel_oseenconvection!(β!, ϱ!)
    βval = zeros(Float64, 2)
    ϱval = zeros(Float64, 1)
    function closure(result, ∇u, qpinfo)
        d = length(result)
        length(βval) == d || resize!(βval, d)
        β!(βval, qpinfo)
        ϱ!(ϱval, qpinfo)
        for i in 1:d
            result[i] = ϱval[1] * dot(βval, view(∇u, (i-1)*d+1:i*d))
        end
        return nothing
    end
end

## kernel for (u⋅n ϱ^upw, λ) ON_IFACES in continuity equation
function kernel_upwind!(result, input, u, qpinfo) # u = [id(u)], input = [this(id(ϱ)), other(id(ϱ))]
    flux = dot(u, qpinfo.normal) # u * n (normal here is face normal)
    return if flux > 0
        result[1] = input[1] * flux # rho_left * flux 
    else
        result[1] = input[2] * flux # rho_right * flux
    end
end

## kernel for (u⋅n ϱ^upw, λ) ON_IFACES in continuity equation
function kernel_upwind2!(result, input, qpinfo) # input = [this(id(ϱ)), other(id(ϱ))]
    flux = qpinfo.params[1][qpinfo.item] # dot(u, qpinfo.normal) # u * n
    return if flux > 0
        result[1] = input[1] * flux # rho_left * flux 
    else
        result[1] = input[2] * flux # rho_right * flux
    end
end

## convection term Karper style
function kernel_upwind_convection!(result, args, qpinfo) # args = [id(u), this(id(ϱ)), other(id(ϱ)), this(id(û)), other(id(û))]
    d = length(result)
    ϱL = view(args, d+1)
    ϱR = view(args, d+2)
    u0L = view(args, d+3:2d+2)
    u0R = view(args, 2d+3:3d+2)
    flux = qpinfo.params[1][qpinfo.item] #dot(u, qpinfo.normal) # u * n
    return if flux > 0
        result .= ϱL .* u0L *  flux
    else
        result .= ϱR .* u0R * flux
    end
end

## KarperConvection face term on boundary faces; exterior state (ϱ,û) from Dirichlet data
function kernel_upwind_convection_bdry!(u!, ϱ!)
    uval = zeros(Float64, 2)
    ϱval = zeros(Float64, 1)
    function closure(result, args, qpinfo) # args = [this(id(ϱ)), this(id(û))]
        d = length(result)
        length(uval) == d || resize!(uval, d)
        ϱL = view(args, 1)
        u0L = view(args, 2:d+1)
        u!(uval, qpinfo)
        ϱ!(ϱval, qpinfo)
        flux = dot(uval, qpinfo.normal) # outward normal flux of the boundary data
        return if flux > 0             # outflow: upwind from inside
            result .= ϱL .* u0L .* flux
        else                           # inflow: upwind from boundary data
            result .= ϱval[1] .* uval .* flux
        end
    end
end

## convection face term with broken P0 test: result (dim 2) is multiplied with jump(μ)
function kernel_upwind_newconvection!(result, args, qpinfo) # args = [this(id(ϱ)), other(id(ϱ)), jump(id(‖u‖²))]
    ϱL = view(args, 1)
    ϱR = view(args, 2)
    u0_jump = view(args, 3) 
    flux = qpinfo.params[1][qpinfo.item] #dot(u, qpinfo.normal) # u * n
    return if flux > 0
        result .= ϱL .* u0_jump .* qpinfo.normal
    else
        result .= ϱR .* u0_jump .* qpinfo.normal
    end
end
## NewConvection face term on boundary faces; exterior state (ϱ,‖u‖²) from Dirichlet data
function kernel_upwind_newconvection_bdry!(u!, ϱ!)
    uval = zeros(Float64, 2)
    ϱval = zeros(Float64, 1)
    function closure(result, args, qpinfo) # args = [this(id(ϱ)), this(id(‖u‖²))]
        d = length(result)
        length(uval) == d || resize!(uval, d)
        ϱL = view(args, 1)
        u2L = view(args, 2)
        u!(uval, qpinfo)
        ϱ!(ϱval, qpinfo)
        jump2 = u2L .- dot(uval, uval)   # [‖u‖²] with ‖u!‖² as exterior trace
        flux = dot(uval, qpinfo.normal)  # outward normal flux of the boundary data
        return if flux > 0               # outflow: upwind from inside
            result .= ϱL .* jump2 .* qpinfo.normal
        else                             # inflow: upwind from boundary data
            result .= ϱval[1] .* jump2 .* qpinfo.normal
        end
    end
end


## kernel for inflow boundary
function kernel_inflow!(u!,ϱ!) # test function is [id(ϱ)] = λ
    uval = zeros(Float64, 2)
    ϱval = zeros(Float64, 1)
    function closure(result, qpinfo)
        d = length(qpinfo.normal)
        length(uval) == d || resize!(uval, d)
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
        d = length(qpinfo.normal)
        length(uval) == d || resize!(uval, d)
        u!(uval, qpinfo)
        flux = dot(uval, qpinfo.normal)
        result[1] = args[1] * flux 
    end
end


## kernel for exact error calculation
## result layout: [L2u(d), ∇u(d²), ϱ(1), ϱu(d)] → resultdim = (d+1)²
## (input args have the matching layout [id(u), grad(u), id(ϱ)] without ϱu part)
function exact_error!(u!, ∇u!, ϱ!; dim = 2)
    d = dim
    n = d + d*d # index of the ϱ slot
    return function closure(result, u, qpinfo)
        u!(view(result, 1:d), qpinfo)
        ∇u!(view(result, d+1:n), qpinfo)
        ϱ!(view(result, n+1), qpinfo)
        for i in 1:d
            result[n+1+i] = result[i] * result[n+1]
        end
        view(result, 1:n+1) .-= view(u, 1:n+1)
        for i in 1:d
            result[n+1+i] -= u[i] * u[n+1]
        end
        return result .= result .^ 2
    end
end

## exact error kernel for the incompressible reference problem
## (exact u!, ∇u! minus discrete u, ∇u)
function exact_error_incompressible!(u!, ∇u!, ϱval; dim = 2)
    d = dim
    return function closure(result, args, qpinfo)
        u!(view(result, 1:d), qpinfo)
        ∇u!(view(result, d+1:d+d*d), qpinfo)
        view(result, 1:d+d*d) .-= view(args, 1:d+d*d)
        return result .= result .^ 2
    end
end

## kernel for exact divergence error calculation
## (exact div u from ∇u! minus discrete divergence, e.g. from a [div(u)]
##  or [apply(u, Reconstruct{..., Divergence})] operator)
function div_error!(u!, ∇u!; dim = 2)
    d = dim
    ∇uval = zeros(Float64, d*d)
    return function closure(result, u, qpinfo)
        ∇u!(∇uval, qpinfo) # layout: [∂x u1, ∂y u1, (∂z u1), ∂x u2, ...]
        result[1] = (sum(∇uval[(i-1)*d+i] for i in 1:d) - u[1])^2
        return result
    end
end

## kernel for gravity term in testcase 1
function standard_gravity!(result, ϱ, qpinfo)
    result .= 0
    return result[end] = -ϱ[1] # Marwa: why ? g = ∇ψ = c_M ∇(log ρ)
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

## mixed Stokes kernel for [grad(u),id(p)] x [grad(v),id(q)], viscosity μ = qpinfo.params[1]
## grad(u) layout: [∂x u1, ∂y u1, (∂z u1), ∂x u2, ...] (componentwise rows), pressure
## acts on the diagonal entries (i-1)*d+i of ∇u, minus the divergence in the last slot
function stokes_kernel!(result, u_ops, qpinfo)
    μ = qpinfo.params[1]
    n = length(result) - 1
    d = round(Int, sqrt(n))
    ∇u, p = view(u_ops, 1:n), view(u_ops, n+1)
    for i in 1:d
        for j in 1:d
            k = (i-1)*d + j
            result[k] = μ * ∇u[k] - (i == j ? p[1] : 0)
        end
    end
    result[n+1] = -sum(∇u[(i-1)*d+i] for i in 1:d)
    # result is then multiplied with [grad(v),id(q)]
    return nothing
end

# kernel for div projection
function div_projection!(result, input, qpinfo)
    result[1] = input[1] - qpinfo.params[1] * input[2]
    return nothing
end