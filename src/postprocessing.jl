"""
    compute_errors!(data; force_recompute = false, compare_incompressible = true, kwargs...)

Compute and store error norms (`Error(L2,u)`, `Error(H1,u)`, `Error(L2,ϱ)`,
`Error(L2,ϱu)`, `Error(L2,uR)`, `Error(H1,u0)`, `Error(L2,div u)`,
`Error(L2,div uR)`) in the given dict of a solved
configuration and optionally compare against the incompressible reference
solution (`run_incompressible!`). The dict must contain a `solution` key, e.g.
as returned by `run_single` or by loading the corresponding JLD2 file.
File IO (loading/saving via DrWatson) is left to the caller.
"""
function compute_errors!(data; force_recompute = false, compare_incompressible = true, kwargs...)
    # problem parameters
    μ      = data["μ"]
    λ      = data["λ"]
    eostype= data["eostype"]
    γ      = gamma(eostype)
    c      = data["c"]
    M      = data["M"]
    ufac   = data["ufac"]

    if haskey(data, "solution")
        sol = data["solution"]
        u   = sol.tags[1]
        ϱ   = sol.tags[2]
        repair_grid!(sol[u].FES.xgrid)
    else
        @error "solution not found in data, cannot compute errors"
        return nothing
    end

    # data of the problem
    reconstruct    = data["reconstruct"]
    order          = data["order"]
    velocitytype   = data["velocitytype"]
    densitytype    = data["densitytype"]
    eostype        = data["eostype"]
    gridtype       = data["gridtype"]
    pressure_in_f  = data["pressure_in_f"]
    others_in_f = data["others_in_f"]
    convectiontype = data["convectiontype"]
    coriolistype   = data["coriolistype"]

    ϱ!, kernel_gravity!, kernel_rhs!, u!, ∇u! =
        prepare_data(velocitytype, densitytype, eostype;
                     others_in_f, pressure_in_f, M, c, μ, λ, γ,
                     ufac, kwargs...)
    if compare_incompressible && (force_recompute || !haskey(data, "Error(L2,u-u_inc)"))
        if !haskey(data, "incompressible_solution")
            run_incompressible!(data; kwargs...)
        end
        sol_inc = data["incompressible_solution"]
        diff_kernel = (result, input, qpinfo) -> (result .= (input[1] - input[3])^2 + (input[2] - input[4])^2)
        DiffIntegrator = ItemIntegrator(diff_kernel, [id(1), id(2)]; quadorder = 2 * (data["order"]+1), kwargs...)
        error_diff = evaluate(DiffIntegrator, [sol[u], sol_inc[1]])
        data["Error(L2,u-u_inc)"] = sqrt(sum(view(error_diff, :)))
        
    end
    if force_recompute || !haskey(data, "Error(H1,u0)")
        @info "computing divergence-free part of u - u_h"

        ## compute error of divergence-free part by solving
        ## incompressible Stokes problem with rhs (∇(u-uh), ∇v)

        ## compile-time pre-compilation of lazy_interpolate on a coarse grid
        compile_grid = simplexgrid(0:0.5:1, 0:0.5:1)
        cg_refined = barycentric_refine(compile_grid)
        test_fe1 = FEVector(FESpace{eltype(sol[u].FES)}(compile_grid); tags = [u])
        test_fe2 = FEVector(FESpace{H1Pk{2,2,2}}(cg_refined); tags = [u])
        @time lazy_interpolate!(test_fe2[1], test_fe1, [id(u)]; quadorder = 0)

        ## prepare FESpace
        xgrid      = sol[u].FES.xgrid
        xgridSP    = barycentric_refine(xgrid)
        FES_SP     = [FESpace{H1Pk{2,2,2}}(xgridSP),
                      FESpace{H1Pk{1,2,1}}(xgridSP; broken = true)]

        ## unknowns of divergence-free projection
        uzero = Unknown("u"; name = "Stokes projection")
        pzero = Unknown("p"; name = "pressure of projection")

        solSP = FEVector(FES_SP; tags = [uzero, pzero])
        append!(solSP, FES_SP[1]; tag = u)
        @time interpolate_BR_to_P2!(solSP[u], sol[u])

        ## velocity-update problem for iterated penalty method
        PDSP_u = ProblemDescription("Stokes projection problem - u update")
        assign_unknown!(PDSP_u, uzero)
        β = 1e+3  # div-penalty
        assign_operator!(PDSP_u, BilinearOperator([grad(uzero)]; factor = 1, store = true, kwargs...))
        assign_operator!(PDSP_u, BilinearOperator([div(uzero)]; store = true, factor = β, kwargs...))
        assign_operator!(PDSP_u, LinearOperator([grad(uzero)], [grad(u)]; factor = -1, kwargs...))
        assign_operator!(PDSP_u, LinearOperator(∇u!, [grad(uzero)]; bonus_quadorder = 4, kwargs...))
        assign_operator!(PDSP_u, LinearOperator([div(uzero)], [id(pzero)]; factor = 1, kwargs...))
        assign_operator!(PDSP_u, HomogeneousBoundaryData(uzero; regions = 1:4, kwargs...))

        ## pressure-update problem
        PDSP_p = ProblemDescription("Stokes projection problem - p update")
        assign_unknown!(PDSP_p, pzero)
        assign_operator!(PDSP_p, BilinearOperator([id(pzero)]; store = true, kwargs...))
        assign_operator!(PDSP_p, LinearOperator(div_projection!, [id(pzero)],
            [id(pzero), div(uzero)]; params = [β], factor = 1, kwargs...))

        ## run iterated penalty method
        SC1 = SolverConfiguration(PDSP_u; init = solSP, maxiterations = 1,
            target_residual = 1.0e-10, constant_matrix = true, kwargs...)
        SC2 = SolverConfiguration(PDSP_p; init = solSP, maxiterations = 1,
            target_residual = 1.0e-10, constant_matrix = true, kwargs...)
        solSP, nits = iterate_until_stationarity([SC1, SC2]; init = solSP, kwargs...)
        @info "converged after $nits iterations"

        ## norm of div-free projection of the error
        error0 = evaluate(L2NormIntegrator([grad(uzero)]), solSP)
        data["Error(H1,u0)"] = sqrt(
            sum(error0[1, :]) + sum(error0[2, :]) +
            sum(error0[3, :]) + sum(error0[4, :]))
        @info data["Error(H1,u0)"]
    else
        @info "skipping divergence-free error (already computed)"
    end

    if force_recompute || !haskey(data, "Error(L2,u)")
        @info "computing errors of u, ϱ and ϱu"
        order = data["order"]
        ErrorIntegratorExact = ItemIntegrator(
            exact_error!(u!, ∇u!, ϱ!), [id(u), grad(u), id(ϱ)];
            resultdim = 9, quadorder = 10, kwargs...)
        error = evaluate(ErrorIntegratorExact, sol)
        data["Error(L2,u)"]  = sqrt(sum(error[1, :]) + sum(error[2, :]))
        data["Error(H1,u)"]  = sqrt(sum(error[3, :]) + sum(error[4, :]) +
                                     sum(error[5, :]) + sum(error[6, :]))
        data["Error(L2,ϱ)"]  = sqrt(sum(error[7, :]))
        data["Error(L2,ϱu)"] = sqrt(sum(error[8, :]) + sum(error[9, :]))

        ## L2 error of the divergence: exact div u (from ∇u!) minus div u_h
        DivErrorIntegrator = ItemIntegrator(
            div_error!(u!, ∇u!), [div(u)]; resultdim = 1, quadorder = 10, kwargs...)
        data["Error(L2,div u)"] = sqrt(sum(evaluate(DivErrorIntegrator, sol)))

        @assert reconstruct in [:none, :RT, :BDM]
        if data["reconstruct"] !== :none
            if order == 1
                ReconstSpace = reconstruct == :RT ? HDIVRT0{2} : reconstruct == :BDM ? HDIVBDM1{2} : nothing
                id_u    = ReconstSpace !== nothing ? apply(u, Reconstruct{ReconstSpace, Identity}) : id(u)
                div_u   = ReconstSpace !== nothing ? apply(u, Reconstruct{ReconstSpace, Divergence}) : div(u)
            elseif order == 2
                ReconstSpace = reconstruct == :RT ? HDIVRT1{2} : reconstruct == :BDM ? HDIVBDM2{2} : nothing
                id_u    = ReconstSpace !== nothing ? apply(u, Reconstruct{ReconstSpace, Identity}) : id(u)
                div_u   = ReconstSpace !== nothing ? apply(u, Reconstruct{ReconstSpace, Divergence}) : div(u)
            end
            ErrorIntegratorExactReconstruct = ItemIntegrator(
                exact_error!(u!, ∇u!, ϱ!), [id_u, grad(u), id(ϱ)];
                resultdim = 9, quadorder = 10, kwargs...)
            error = evaluate(ErrorIntegratorExactReconstruct, sol)
            data["Error(L2,uR)"]  = sqrt(sum(error[1, :]) + sum(error[2, :]))
            ## L2 error of the divergence of the H(div)-conforming reconstruction
            DivErrorIntegratorReconstruct = ItemIntegrator(
                div_error!(u!, ∇u!), [div_u]; resultdim = 1, quadorder = 10, kwargs...)
            data["Error(L2,div uR)"] = sqrt(sum(evaluate(DivErrorIntegratorReconstruct, sol)))
        end
    else
        @info "skipping error calculation (already computed)"
    end

    return data
end
