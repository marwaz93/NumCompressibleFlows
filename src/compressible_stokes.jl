# testcase 3 added by Marwa




function load_testcase_data(testcase::Int = 1; laplacian_in_rhs = true,Akbas_example=1, M = 1, c = 1, μ = 1,γ=1,λ = -2*μ / 3, ufac = 100)
    if testcase == 1
        grid_builder = (nref) -> simplexgrid(
            Triangulate;
            points = [0 0; 0.2 0; 0.3 0.2; 0.45 0.05; 0.55 0.35; 0.65 0.2; 0.7 0.3; 0.8 0; 1 0; 1 1 ; 0 1]',
            bfaces = [1 2; 2 3; 3 4; 4 5; 5 6; 6 7; 7 8; 8 9; 9 10; 10 11; 11 1]',
            bfaceregions = ones(Int, 11),
            regionpoints = [0.5 0.5;]',
            regionnumbers = [1],
            regionvolumes = [4.0^-(nref) / 2]
        )
        xgrid = grid_builder(3)
        u1!(result, qpinfo) = (fill!(result, 0);)
        ∇u1!(result, qpinfo) = (fill!(result, 0);)
        M_exact = integrate(xgrid, ON_CELLS, (result, qpinfo) -> (result[1] = exp(-qpinfo.x[2] / c) / M;), 1; quadorder = 20) # e^(-y/c_M)/ M 
        area = sum(xgrid[CellVolumes])
        ϱ1!(result, qpinfo) = (result[1] = exp(-qpinfo.x[2] / c) / (M_exact / area);) # integral of ρ_1 is M*|Ω|
        return grid_builder, standard_gravity!, nothing, u1!, ∇u1!, ϱ1!, 1
    elseif testcase == 2
        grid_builder = (nref) -> simplexgrid(
            Triangulate;
            points = [0 0; 1 0; 1 1 ; 0 1]',
            bfaces = [1 2; 2 3; 3 4; 4 1]',
            bfaceregions = ones(Int, 4),
            regionpoints = [0.5 0.5;]',
            regionnumbers = [1],
            regionvolumes = [4.0^-(nref)]
        )

        xgrid = grid_builder(3)
        M_exact = integrate(xgrid, ON_CELLS, (result, qpinfo) -> (result[1] = exp(-qpinfo.x[1]^3 / (3 * c)) / M ;), 1; quadorder = 20) # I changed it by division over M
        ϱ_eval, g_eval, f_eval, u_eval, ∇u_eval = prepare_data2!(; laplacian_in_rhs = laplacian_in_rhs, M = M_exact, c = c, μ = μ, ufac = ufac)
        ϱ2!(result, qpinfo) = (result[1] = ϱ_eval(qpinfo.x[1], qpinfo.x[2]);)

        M_exact = integrate(xgrid, ON_CELLS, ϱ2!, 1)
        area = sum(xgrid[CellVolumes])

        function kernel_gravity!(result, input, qpinfo)
            g_eval(result, qpinfo.x[1], qpinfo.x[2]) # qpinfo.x[1] is x and qpinfo.x[2] is y
            return result .*= input[1] # input is [id_u], [id(ϱ)] ??
        end

        function kernel_rhs!(result, qpinfo)
            return f_eval(result, qpinfo.x[1], qpinfo.x[2])
        end

        u2!(result, qpinfo) = (u_eval(result, qpinfo.x[1], qpinfo.x[2]);)
        ∇u2!(result, qpinfo) = (∇u_eval(result, qpinfo.x[1], qpinfo.x[2]);)
        return grid_builder, kernel_gravity!, f_eval === nothing ? nothing : kernel_rhs!, u2!, ∇u2!, ϱ2!, ufac
    
    elseif testcase == 3
        grid_builder = (nref) -> simplexgrid(
            Triangulate;
            points = [0 0; 1 0; 1 1 ; 0 1]',
            bfaces = [1 2; 2 3; 3 4; 4 1]',
            bfaceregions = ones(Int, 4),
            regionpoints = [0.5 0.5;]',
            regionnumbers = [1],
            regionvolumes = [4.0^-(nref)]
        )

        xgrid = grid_builder(3)
        M_exact = integrate(xgrid, ON_CELLS, (result, qpinfo) -> (result[1] = ( 1 + (qpinfo.x[1]-(1/2))/c ) / M;), 1; quadorder = 20) # updated by Marwa
        ϱ_eval, g_eval, f_eval, u_eval, ∇u_eval = prepare_data3!(;Akbas_example=Akbas_example, laplacian_in_rhs = laplacian_in_rhs,λ=λ, M = M_exact, c = c, μ = μ,γ=γ, ufac = ufac)
        ϱ3!(result, qpinfo) = (result[1] = ϱ_eval(qpinfo.x[1], qpinfo.x[2]);)

        M_exact = integrate(xgrid, ON_CELLS, ϱ3!, 1)
        area = sum(xgrid[CellVolumes])

        function kernel_gravity3!(result, input, qpinfo)
            g_eval(result, qpinfo.x[1], qpinfo.x[2]) # qpinfo.x[1] is x and qpinfo.x[2] is y
            return result .*= input[1] # input is [id_u], [id(ϱ)] ??
        end

        function kernel_rhs3!(result, qpinfo)
            return f_eval(result, qpinfo.x[1], qpinfo.x[2])
        end

        u3!(result, qpinfo) = (u_eval(result, qpinfo.x[1], qpinfo.x[2]);)
        ∇u3!(result, qpinfo) = (∇u_eval(result, qpinfo.x[1], qpinfo.x[2]);)
        return grid_builder, kernel_gravity3!, f_eval === nothing ? nothing : kernel_rhs3!, u3!, ∇u3!, ϱ3!, ufac
    end
end




## exact data for testcase 2 computed by Symbolics
function prepare_data2!(; M = 1, c = 1, μ = 1, ufac = 100, laplacian_in_rhs = true)

    @variables x y

    ## density
    ϱ = exp(-x^3 / (3 * c)) / M # Marwa:  /rho = exp (.) / /int_exp(.) 

    ## stream function ξ
    ## sucht that ϱu = curl ξ
    ξ = x^2 * y^2 * (x - 1)^2 * (y - 1)^2 * ufac

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


     ## gravity ϱg = - Δu + ϱ∇log(ϱ)

    if laplacian_in_rhs
        f = - μ * Δu
        g = c * Symbolics.gradient(log(ϱ), [x, y])

        # Marwa therfore, /gradient p = c * /rho * /gradient /psi 
    else
        g = - μ * Δu / ϱ + c * Symbolics.gradient(log(ϱ), [x, y])
        f = 0
    end

    #Δu = Symbolics.derivative(∇u[1,1], [x]) + Symbolics.derivative(∇u[2,2], [y])

    ϱ_eval = build_function(ϱ, x, y, expression = Val{false})
    u_eval = build_function(u, x, y, expression = Val{false})
    ∇u_eval = build_function(∇u_reshaped, x, y, expression = Val{false})
    g_eval = build_function(g, x, y, expression = Val{false})
    f_eval = build_function(f, x, y, expression = Val{false})

    return ϱ_eval, g_eval[2], f == 0 ? nothing : f_eval[2], u_eval[2], ∇u_eval[2]
end
## implements examples in Akbas et el. sec 7.1 (Akbas_example=1) and sec 7.2 and 7.3 (Akbas_example=2)
function prepare_data3!(;Akbas_example=1, M = 1, c = 1, μ = 1, γ=1,λ = -2*μ / 3, ufac = 1, laplacian_in_rhs = true)

    @variables x y

    ## density
    ϱ = ( 1+(x-(1/2))/c )/M 
    p = c * ϱ^γ

    if Akbas_example==1
        ξ = x^2 * y^2 * (x - 1)^2 * (y - 1)^2 * ufac
    elseif Akbas_example==2
   
        ξ = 0
    end

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
    ∇divu = [
        (Symbolics.gradient(∇u[1, 1], [x]) + Symbolics.gradient(∇u[2, 2], [x]))[1],
        (Symbolics.gradient(∇u[1, 1], [y]) + Symbolics.gradient(∇u[2, 2], [y]))[1],
    ]

    if Akbas_example==1
        f = - μ * Δu + Symbolics.gradient(p, [x, y]) - λ*∇divu
        g = 0 * Symbolics.gradient(0, [x, y])
    elseif Akbas_example==2
        if laplacian_in_rhs
            f =  c* γ*ϱ^(γ-1) * Symbolics.gradient(ϱ, [x, y])
            g = 0 * Symbolics.gradient(0, [x, y])
            @show g 
            # also laplacian term missing 
        else
            # Marwa therfore, /gradient p = c * /rho * /gradient /psi
            g =  c* γ*ϱ^(γ-2) * Symbolics.gradient(ϱ, [x, y])
            f = 0

            # i think this case is wrong in my code, we miss the laplacian term
        end
    end
    ϱ_eval = build_function(ϱ, x, y, expression = Val{false})
    u_eval = build_function(u, x, y, expression = Val{false})
    ∇u_eval = build_function(∇u_reshaped, x, y, expression = Val{false})
    g_eval = build_function(g, x, y, expression = Val{false})
    f_eval = build_function(f, x, y, expression = Val{false})

    return ϱ_eval, g_eval[2], f == 0 ? nothing : f_eval[2], u_eval[2], ∇u_eval[2]
end

function filename(data)
    # problem parameters
    μ = data["μ"]
    λ = data["λ"]
    γ = data["γ"]
    c = data["c"]
    M = data["M"]
    # solving options
    nrefs = data["nrefs"]
    order = data["order"]
    reconstruct = data["reconstruct"]
    target_residual = data["target_residual"]
    maxsteps = data["maxsteps"]
    pressure_stab = data["pressure_stab"]
    # data of the problem
    ufac = data["ufac"]
    testcase = data["testcase"]
    laplacian_in_rhs = data["laplacian_in_rhs"]
    Akbas_example = data["Akbas_example"]
    # sname
    sname = savename((@dict μ λ γ c M order nrefs order reconstruct target_residual maxsteps pressure_stab ufac testcase laplacian_in_rhs Akbas_example))
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
    nrefs = data["nrefs"]
    order = data["order"]
    reconstruct = data["reconstruct"]
    target_residual = data["target_residual"]
    maxsteps = data["maxsteps"]
    pressure_stab = data["pressure_stab"]
    # data of the problem
    ufac = data["ufac"]
    testcase = data["testcase"]
    laplacian_in_rhs = data["laplacian_in_rhs"]
    Akbas_example = data["Akbas_example"]

    data = Dict{String, Any}(data)
    @show data, typeof(data)

    ## load data for testcase
    grid_builder, kernel_gravity!, kernel_rhs!, u!, ∇u!, ϱ!, τfac = load_testcase_data(testcase; laplacian_in_rhs = laplacian_in_rhs,Akbas_example=Akbas_example, M = M, c = c, μ = μ, λ=λ, γ=γ, ufac = ufac)
    xgrid = grid_builder(nrefs)

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
    ## define first sub-problem: Stokes equations to solve for velocity u
    PD = ProblemDescription("Stokes problem")
    assign_unknown!(PD, u)
    assign_operator!(PD, BilinearOperator([grad(u)]; factor = μ, store = true, kwargs...))
    assign_operator!(PD, BilinearOperator([div_u]; factor = λ, store = true, kwargs...)) # Marwa div term 
    assign_operator!(PD, LinearOperator([div(u)], [id(ϱ)]; factor = c, kwargs...)) 
    assign_operator!(PD, HomogeneousBoundaryData(u; regions = 1:4, kwargs...))
    if kernel_rhs! !== nothing
        assign_operator!(PD, LinearOperator(kernel_rhs!, [id_u]; factor = 1, store = true, bonus_quadorder = 3 * order, kwargs...))
    end
    assign_operator!(PD, LinearOperator(kernel_gravity!, [id_u], [id(ϱ)]; factor = 1, bonus_quadorder = 3 * order, kwargs...))

    ## FVM for continuity equation
    τ = μ / (c*order^2 * M * sqrt(τfac)) # time step for pseudo timestepping
    @info "timestep = $τ"
    PDT = ProblemDescription("continuity equation")
    assign_unknown!(PDT, ϱ)
    if order > 1
        assign_operator!(PDT, BilinearOperator(kernel_continuity!, [grad(ϱ)], [id(ϱ)], [id(u)]; quadorder = 2 * order, factor = -1, kwargs...))
    end
    if pressure_stab > 0
        psf = pressure_stab #* xgrid[CellVolumes][1]
        assign_operator!(PDT, BilinearOperator(stab_kernel!, [jump(id(ϱ))], [jump(id(ϱ))], [id(u)]; entities = ON_IFACES, factor = psf, kwargs...))
    end
    assign_operator!(PDT, BilinearOperator([id(ϱ)]; quadorder = 2 * (order - 1), factor = 1 / τ, store = true, kwargs...))
    assign_operator!(PDT, LinearOperator([id(ϱ)], [id(ϱ)]; quadorder = 2 * (order - 1), factor = 1 / τ, kwargs...))
    assign_operator!(PDT, BilinearOperatorDG(kernel_upwind!, [jump(id(ϱ))], [this(id(ϱ)), other(id(ϱ))], [id(u)]; quadorder = order + 1, entities = ON_IFACES, kwargs...))
    #  [jump(id(ϱ))]is test function lambda , [this(id(ϱ)), other(id(ϱ))] is the the flux multlplied by lambda_upwind. [id(u)] is the function u that is needed 
    ## prepare error calculation
    EnergyIntegrator = ItemIntegrator(energy_kernel!, [id(u)]; resultdim = 1, quadorder = 2 * (order + 1), kwargs...)
    ErrorIntegratorExact = ItemIntegrator(exact_error!(u!, ∇u!, ϱ!), [id(u), grad(u), id(ϱ)]; resultdim = 9, quadorder = 2 * (order + 1), kwargs...)
    #NDofs = zeros(Int, nrefs)
    #Results = zeros(Float64, nrefs, 5) # it is a matrix whose rows are levels and columns are 

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

    ## solve the two problems iteratively [1] >> [2] >> [1] >> [2] ...
    SC1 = SolverConfiguration(PD; init = sol, maxiterations = 1, target_residual = target_residual, constant_matrix = true, kwargs...)
    SC2 = SolverConfiguration(PDT; init = sol, maxiterations = 1, target_residual = target_residual, kwargs...)
    sol, nits = iterate_until_stationarity([SC1, SC2]; energy_integrator = EnergyIntegrator, maxsteps = maxsteps, init = sol, kwargs...)
  

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
             


        




    


