using Test
using NumCompressibleFlows
using Symbolics
using ExtendableFEM

## solve the manufactured constant velocity/density problem (ξ = -y, ϱ = M)
## with the given configuration options and return the velocity/density errors;
## the exact solution u = (1/M, 0), ϱ = M is contained in the discrete spaces
## (constants live in BR and P0) and solves all terms of the discrete scheme
## exactly: ∇(cϱ^γ) = 0, (u⋅∇)u = 0 and all density/velocity jumps vanish,
## so the errors should be at (near) machine precision level, independently
## of the equation of state and of the chosen convection discretization
function manufactured_errors(; kwargs...)
    data = load_data(velocitytype = ConstantVelocity, densitytype = ConstantDensity,
                     gridtype = UnstructuredUnitSquare, nrefs = 2, maxsteps = 20; kwargs...)
    data = run_single(data)

    ϱ!, ~, ~, u!, ∇u! = prepare_data(ConstantVelocity, ConstantDensity, data["eostype"];
                                     M = 1, c = 1, μ = 1, λ = 0, ufac = 1,
                                     pressure_in_f = true, others_in_f = true,
                                     convectiontype = data["convectiontype"])

    sol = data["solution"]
    u = data["unknown_u"]
    ϱ = data["unknown_ϱ"]

    ErrorIntegrator = ItemIntegrator(exact_error!(u!, ∇u!, ϱ!), [id(u), grad(u), id(ϱ)];
                                     resultdim = 9, quadorder = 10)
    error = evaluate(ErrorIntegrator, sol)
    L2u = sqrt(sum(error[1, :]) + sum(error[2, :]))
    H1u = sqrt(sum(error[3, :]) + sum(error[4, :]) +
               sum(error[5, :]) + sum(error[6, :]))
    L2ϱ = sqrt(sum(error[7, :]))

    return (; L2u, H1u, L2ϱ)
end

# tolerance for the manufactured-solution tests: the exact solution is a fixed
# point of the discrete scheme, so all errors are at machine precision level
tol = 1.0e-13

@testset "manufactured constant velocity and density" begin
    for eostype in (IdealGasLaw, PowerLaw{1.4})
        @testset "eostype = $eostype" begin
            e = manufactured_errors(eostype = eostype)
            println("errors ($eostype): L2u = $(e.L2u), H1u = $(e.H1u), L2ϱ = $(e.L2ϱ)")
            @test e.L2u < tol
            @test e.H1u < tol
            @test e.L2ϱ < tol
        end
    end
    # note: KarperConvection is not tested here since its upwind convection
    # term currently assumes homogeneous velocity boundary data, which
    # ConstantVelocity (inflow/outflow on the unit square) does not have
    for convectiontype in (StandardConvection, OseenConvection)
        @testset "convectiontype = $convectiontype" begin
            e = manufactured_errors(convectiontype = convectiontype)
            println("errors ($convectiontype): L2u = $(e.L2u), H1u = $(e.H1u), L2ϱ = $(e.L2ϱ)")
            @test e.L2u < tol
            @test e.H1u < tol
            @test e.L2ϱ < tol
        end
    end
    # starting from constant density instead of interpolating the exact solution
    # (first momentum update then acts as a Stokes-like solve) should converge
    # to the same exact discrete fixed point
    @testset "initial_values = :stokes" begin
        e = manufactured_errors(initial_values = :stokes)
        println("errors (initial_values = :stokes): L2u = $(e.L2u), H1u = $(e.H1u), L2ϱ = $(e.L2ϱ)")
        @test e.L2u < tol
        @test e.H1u < tol
        @test e.L2ϱ < tol
    end
end

