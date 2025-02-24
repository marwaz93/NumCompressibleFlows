# testcase 3 added by Marwa
function load_testcase_data(testcase::Int = 1; laplacian_in_rhs = true,Akbas_example=1, M = 1, c = 1, μ = 1,γ=1, ufac = 100)
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
        ϱ_eval, g_eval, f_eval, u_eval, ∇u_eval = prepare_data3!(;Akbas_example=Akbas_example, laplacian_in_rhs = laplacian_in_rhs, M = M_exact, c = c, μ = μ,γ=γ, ufac = ufac)
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
# Marwa what is this [1] in front of above line ? 

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
function prepare_data3!(;Akbas_example=1, M = 1, c = 1, μ = 1, γ=1, ufac = 1, laplacian_in_rhs = true)

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
    if Akbas_example==1
        f = -2* μ * Δu + Symbolics.gradient(p, [x, y])
        g = 0 * Symbolics.gradient(0, [x, y])
    elseif Akbas_example==2
        if laplacian_in_rhs
            f =  c* γ*ϱ^(γ-1) * Symbolics.gradient(ϱ, [x, y])
            g = 0 * Symbolics.gradient(0, [x, y])
            @show g 
        else
            # Marwa therfore, /gradient p = c * /rho * /gradient /psi
            g =  c* γ*ϱ^(γ-2) * Symbolics.gradient(ϱ, [x, y])
            f = 0
        end
    end
    ϱ_eval = build_function(ϱ, x, y, expression = Val{false})
    u_eval = build_function(u, x, y, expression = Val{false})
    ∇u_eval = build_function(∇u_reshaped, x, y, expression = Val{false})
    g_eval = build_function(g, x, y, expression = Val{false})
    f_eval = build_function(f, x, y, expression = Val{false})

    return ϱ_eval, g_eval[2], f == 0 ? nothing : f_eval[2], u_eval[2], ∇u_eval[2]
end
             


        




    


