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

    ϱ!, ~, ~, u!, ∇u!, ~ = prepare_data(ConstantVelocity, ConstantDensity, data["eostype"];
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
    # note: KarperConvection and NewConvection are tested here as well since
    # their upwind face terms now include a ghost boundary contribution from
    # the Dirichlet data u!, ϱ!, which makes ConstantVelocity (inflow/outflow
    # on the unit square) an exact fixed point of the discrete scheme
    for convectiontype in (StandardConvection, OseenConvection, NewConvection, KarperConvection)
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

## 3D analogue: the z-invariant constant velocity field u = (1/M, 0, 0)
## (embedded planar stream function ξ = -y) with ϱ = M on the tetrahedral unit
## cube is likewise an exact discrete fixed point (constants live in BR{3} and
## P0, the H(div)-reconstruction HDIVRT0{3} is exactly divergence-free on
## constants and reproduces constants)
function manufactured_errors_3d(; kwargs...)
    data = load_data(velocitytype = ConstantVelocity, densitytype = ConstantDensity,
                     gridtype = UnstructuredUnitCube, nrefs = 1, maxsteps = 20; kwargs...)
    data = run_single(data)
    d = data["dim"]
    @test d == 3

    ϱ!, ~, ~, u!, ∇u!, ~ = prepare_data(ConstantVelocity, ConstantDensity, data["eostype"];
                                     M = 1, c = 1, μ = 1, λ = 0, ufac = 1,
                                     pressure_in_f = true, others_in_f = true,
                                     convectiontype = data["convectiontype"], dim = 3)

    sol = data["solution"]
    u = data["unknown_u"]
    ϱ = data["unknown_ϱ"]

    ErrorIntegrator = ItemIntegrator(exact_error!(u!, ∇u!, ϱ!; dim = 3), [id(u), grad(u), id(ϱ)];
                                     resultdim = (d+1)^2, quadorder = 10)
    error = evaluate(ErrorIntegrator, sol)
    n = d + d*d # error rows: [u(d), ∇u(d²), ϱ(1), ϱu(d)]
    L2u = sqrt(sum(error[1:d, :]))
    H1u = sqrt(sum(error[d+1:n, :]))
    L2ϱ = sqrt(sum(error[n+1, :]))

    return (; L2u, H1u, L2ϱ)
end

@testset "manufactured constant velocity and density (3D)" begin
    for eostype in (IdealGasLaw, PowerLaw{1.4})
        @testset "eostype = $eostype" begin
            e = manufactured_errors_3d(eostype = eostype)
            println("3D errors ($eostype): L2u = $(e.L2u), H1u = $(e.H1u), L2ϱ = $(e.L2ϱ)")
            @test e.L2u < tol
            @test e.H1u < tol
            @test e.L2ϱ < tol
        end
    end
    for convectiontype in (StandardConvection, OseenConvection)
        @testset "convectiontype = $convectiontype" begin
            e = manufactured_errors_3d(convectiontype = convectiontype)
            println("3D errors ($convectiontype): L2u = $(e.L2u), H1u = $(e.H1u), L2ϱ = $(e.L2ϱ)")
            @test e.L2u < tol
            @test e.H1u < tol
            @test e.L2ϱ < tol
        end
    end
end

