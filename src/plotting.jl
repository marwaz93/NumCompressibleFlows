# ==============================================================================
# Registry of plottable convergence quantities (used by plot_convergencehistory)
# ==============================================================================

"""
Registry of convergence curves selectable by symbol in
`plot_convergencehistory`. Each entry has:
- `data`: key in the per-level data dict, or a function `data -> value`
- `label`: legend label
- optional `style`: Plots keyword arguments (linestyle, marker, markersize, color)
- optional `xinc = true`: plot against the DOF count of the incompressible system
- optional `incompressible = true`: selecting it triggers `run_incompressible!`
  (via `compute_errors(..., compare_incompressible = true)`)
"""
const CONV_QUANTITIES = (
    L2u = (; data = "Error(L2,u)", label = L"|| \mathbf{u} - \mathbf{u}_h \,||"),                       # L2 error of velocity
    L2uR = (; data = "Error(L2,uR)", label = L"|| \mathbf{u} - \Pi\mathbf{u}_h \,||"),                  # L2 error of velocity projected to the FE space
    H1u = (; data = "Error(H1,u)", label = L"|| ∇(\mathbf{u} - \mathbf{u}_h)\,||"),                     # H1 error of velocity
    L2ϱ = (; data = "Error(L2,ϱ)", label = L"|| {ϱ}-ϱ_h \, ||"),                                         # L2 error of density
    L2ϱu = (; data = "Error(L2,ϱu)", label = L"|| {ϱ\mathbf{u}}-ϱ_h \mathbf{u}_h \, ||"),                # L2 error of momentum
    H1u0 = (; data = "Error(H1,u0)", label = L"||  ∇( \mathbf{u}^0 - \mathbf{u}^0_h ) \,||"),           # H1 error of div-free part
    H1u1 = (; data = d -> sqrt(d["Error(H1,u)"]^2 - d["Error(H1,u0)"]^2),                               # H1 error of curl-free part
              label = L"||  ∇( \mathbf{u}^1 - \mathbf{u}^1_h ) \,||"),
    nits = (; data = "nits", label = L"nits"),                                                          # number of nonlinear iterations
    H1u_inc = (; data = "Error(H1,u_inc)", label = L"|| ∇(\mathbf{u}^{inc} - \mathbf{u}_h^{inc})\,||",  # H1 error of incompressible limit
                  incompressible = true, xinc = true,
                  style = (linestyle = :dashdot, marker = :xcross, markersize = 7, color = :orange)),
    L2u_inc = (; data = "Error(L2,u_inc)", label = L"|| \mathbf{u}^{inc} - \mathbf{u}_h^{inc}\,||",     # L2 error of incompressible limit
                  incompressible = true, xinc = true,
                  style = (linestyle = :dashdot, marker = :xcross, markersize = 7, color = :purple)),
    L2u_diff = (; data = "Error(L2,u-u_inc)", label = L"|| \mathbf{u}_h - \mathbf{u}_h^{inc}\,||",      # L2 difference between compressible and incompressible solutions
                  incompressible = true, xinc = true,
                  style = (linestyle = :dashdot, marker = :xcross, markersize = 7, color = :green)),
    res_momentum = (; data = "res_momentum", label = "residual momentum"),                              # residual of momentum equation
    res_continuity = (; data = "res_continuity", label = "residual continuity"),                        # residual of continuity equation
)

const DEFAULT_QUANTITIES = (:L2u, :H1u, :L2ϱ, :L2ϱu, :H1u0)

## value of a CONV_QUANTITIES entry for a given level's data dict
function conv_value(entry, d)
    if entry.data isa String
        return get(d, entry.data, NaN)
    else
        try
            return entry.data(d)
        catch
            return NaN
        end
    end
end

# ==============================================================================
# Data pipeline (DrWatson layer, provided by the calling script)
# ==============================================================================

"""
Registry for the DrWatson-based data pipeline used by the plotting functions.
The module itself does not depend on DrWatson; file naming and file production
are provided by the script via

    NumCompressibleFlows.setup_pipeline!(
        filename = ...,        # data -> jld2 filename stem
        filename_plots = ...,  # (data; prefix, free_parameter) -> plot filename
        produce = ...,         # (data; force) -> (data, loaded) via produce_or_load(run_single, ...)
        compute_errors = ...,  # (config; kwargs...) -> data with error keys, saved to file
    )

before any plotting function is called (see `scripts/stationary_examples.jl`).
"""
const PIPELINE = Ref{Union{Nothing,NamedTuple}}(nothing)

function setup_pipeline!(; filename, filename_plots, produce, compute_errors, record_values = nothing)
    @info "setting up data pipeline" filename filename_plots produce compute_errors record_values
    PIPELINE[] = (; filename, filename_plots, produce, compute_errors, record_values)
    @info "data pipeline ready, plotting functions can now be called"
    return nothing
end

function pipeline()
    if PIPELINE[] === nothing
        error("data pipeline not configured: include a script that calls NumCompressibleFlows.setup_pipeline!(; filename, filename_plots, produce, compute_errors)")
    end
    return PIPELINE[]::NamedTuple
end

## delegators with the names used by the plotting functions below
safe_produce_or_load(data; kwargs...) = pipeline().produce(data; kwargs...)
compute_errors(config; kwargs...) = pipeline().compute_errors(config; kwargs...)
filename_plots(data; kwargs...) = pipeline().filename_plots(data; kwargs...)

"""
    save_plot_values(plotfile, data, columns)

After saving a plot to `plotfile`, append the computed values to its `.txt`
description file via the pipeline's `record_values` (if provided). `columns`
is a vector of `name => column_vector` pairs.
"""
function save_plot_values(plotfile, data, columns)
    record = pipeline().record_values
    record === nothing && return nothing
    record(plotfile, data, columns)
    return nothing
end

# ==============================================================================
# Single solution plot
# ==============================================================================

"""
    plot_single(; Plotter = Plots, force = false, kwargs...)

Solve a single configuration and plot velocity (with quiver) and density.
Requires `unknown_u` and `unknown_ϱ` keys in the loaded data.
"""
function plot_single(; Plotter = Plots, force = false, kwargs...)
    data = load_data(; kwargs...)
    @debug "loading config" data
    data, ~ = safe_produce_or_load(data; force)
    xgrid = data["grid"]
    sol = data["solution"]
    u = data["unknown_u"]
    ϱ = data["unknown_ϱ"]
    @debug "solution loaded" sol
    repair_grid!(xgrid)
    repair_grid!(sol[u].FES.xgrid)
    repair_grid!(sol[ϱ].FES.xgrid)

    ## plot
    pl = GridVisualizer(; Plotter = Plotter, layout = (1,2), clear = true,
        show = true, resolution = (1000, 500))
    scalarplot!(pl[1,1], xgrid, view(nodevalues(sol[u]; abs = true), 1, :),
        levels = 0, colorbarticks = 7, fontsize = 60)
    vectorplot!(pl[1,1], xgrid, eval_func_bary(PointEvaluator([id(u)], sol)),
        clear = false, fontsize = 60)
    scalarplot!(pl[1,2], xgrid, view(nodevalues(sol[ϱ]), 1, :), levels = 11,
        fontsize = 60)

    ## save
    scene = GridVisualize.reveal(pl)
    GridVisualize.save(filename_plots(data; prefix = "_Solutions"), scene;
        Plotter = Plotter)

    return data
end

# ==============================================================================
# Convergence history plot
# ==============================================================================

"""
    log_ticks(v; maxticks = 12)

Powers of ten covering the positive finite values in `v` (first tick below
the minimum, last tick above the maximum). If the range spans more than
`maxticks` decades, every k-th decade is used and the top decade is kept.
"""
function log_ticks(v; maxticks = 12)
    v = filter(x -> isfinite(x) && x > 0, collect(Float64, vec(v)))
    isempty(v) && return [1.0e-2, 1.0, 1.0e+2]
    lo = floor(Int, log10(minimum(v)))
    hi = ceil(Int, log10(maximum(v)))
    hi = max(hi, lo + 2)
    step = max(1, ceil(Int, (hi - lo + 1) / maxticks))
    ts = collect(lo:step:hi)
    ts[end] == hi || push!(ts, hi)
    return 10.0 .^ ts
end

"""
    plot_convergencehistory(; nrefs = 1:6, quantities = :default, xquantity = :ndofs,
        slopes = (1, 2), with_incompressible = false, kwargs...)

Plots the convergence history of the compressible solver. The plotted curves
are selected via `quantities`, a vector of symbols resolved against the
`CONV_QUANTITIES` registry (e.g. `[:L2u, :H1u, :L2ϱ, :H1u0, :H1u1, :nits]`).
`:default` gives `collect(DEFAULT_QUANTITIES)`, `:all` the whole registry.
One-off quantities can be passed inline as NamedTuples with fields `name`,
`data` (dict key or function `data -> value`) and `label`.

Quantities with `incompressible = true` (`:H1u_inc`, `:L2u_inc`, `:L2u_diff`)
trigger the incompressible reference solver `run_incompressible!` via
`compute_errors` and are plotted against the DOF count of the incompressible
system. For backward compatibility, `with_incompressible = true` (deprecated)
appends them to the `:default` selection.

`xquantity` selects the x-axis (`:ndofs` or `:h = ndofs^(-1/2)`), `slopes`
adds reference lines O(h^k).
"""
function plot_convergencehistory(; nrefs = 1:6, Plotter = Plots, force = false, force_recompute = false,
        quantities = :default, with_incompressible = false, xquantity = :ndofs, slopes = (1, 2), kwargs...)

    @info "Plotting convergence history for nrefs = $nrefs, quantities = $quantities, xquantity = $xquantity, slopes = $slopes..."
    ## resolve quantity selection against registry
    if quantities === :default
        qlist = collect(DEFAULT_QUANTITIES)
        with_incompressible && append!(qlist, (:H1u_inc, :L2u_inc, :L2u_diff))
    elseif quantities === :all
        qlist = collect(keys(CONV_QUANTITIES))
    else
        qlist = collect(quantities)
        if with_incompressible
            @warn "with_incompressible = true is ignored when quantities are given explicitly; add e.g. :H1u_inc, :L2u_inc, :L2u_diff to quantities instead"
        end
    end
    isempty(qlist) && error("quantities must not be empty")
    unknown = [q for q in qlist if q isa Symbol && !haskey(CONV_QUANTITIES, q)]
    isempty(unknown) || error("unknown quantities $unknown; available: $(collect(keys(CONV_QUANTITIES)))")
    entries = [(q isa Symbol ? q : get(q, :name, :custom), q isa Symbol ? CONV_QUANTITIES[q] : q) for q in qlist]
    for (name, e) in entries
        hasproperty(e, :data) && hasproperty(e, :label) || error("quantity $name needs fields :data and :label")
    end
    needs_inc = any(get(e, :incompressible, false) for (_, e) in entries)

    data = load_data(; kwargs...)
    #@show data
    nl = length(nrefs)
    vals = [zeros(Float64, nl) for _ in entries]
    NDoFs = zeros(Int, nl)
    NDoFsInc = zeros(Int, nl)
    Residuals = zeros(Float64, nl, 2)

    for (j, lvl) in enumerate(nrefs)
        _data = deepcopy(data)
        _data["nrefs"] = lvl
        _data, ~ = safe_produce_or_load(_data; force = force)
        NDoFs[j] = _data["ndofs"]
        _data = compute_errors(_data; force_recompute = force_recompute, compare_incompressible = needs_inc)

        for (k, (_, e)) in enumerate(entries)
            vals[k][j] = conv_value(e, _data)
        end
        if needs_inc
            NDoFsInc[j] = haskey(_data, "incompressible_solution") ? length(_data["incompressible_solution"].entries) : NDoFs[j]
        end

        if haskey(_data, "res_momentum")
            Residuals[j,1] = _data["res_momentum"]
            Residuals[j,2] = _data["res_continuity"]
        else
            @warn "residual information not found, consider rerunning"
            Residuals[j,1] = 1e30
            Residuals[j,2] = 1e30
        end

        @show Residuals

        ## console table of the first up to four selected quantities
        sel = min(4, length(entries))
        print_convergencehistory(NDoFs[:], hcat(vals[1:sel]...); X_to_h = X -> X.^(-1/2),
            ylabels = [string(entries[k][2].label) for k in 1:sel],
            xlabel = xquantity === :h ? "h" : "ndof", latex_mode = true)
    end

    ## plot
    #Plotter.rc("font", size=20)
    if !(xquantity in (:ndofs, :h))
        error("xquantity must be :ndofs or :h")
    end
    hvals = NDoFs[:].^(-1/2)
    xof = xquantity === :h ? hvals : Float64.(NDoFs[:])

    ## collect all curves first, so that the axis ticks can cover the plotted data
    ## (built via vcat instead of push! into a growing vector: quantity curves and slope
    ## reference lines are different concrete NamedTuple types, which breaks push! growth)
    series = vcat(
        [ (; x = Float64.(get(e, :xinc, false) ? (xquantity === :h ? NDoFsInc[:].^(-1/2) : NDoFsInc[:]) : xof),
            y = vals[k], label = e.label, style = get(e, :style, NamedTuple()))
          for (k, (_, e)) in enumerate(entries) ],
        [ (; x = xof, y = (m == 1 ? 0.5 : m == 2 ? 1e+1 : 1.0) .* hvals.^m,
            label = m == 1 ? L"\mathcal{O}(h)" : latexstring("\\mathcal{O}(h^{$m})"),
            style = (linestyle = :dash, color = :gray, marker = :none))
          for m in slopes ],
    )

    ## axis ticks as powers of ten covering all plotted values
    yticks = log_ticks(reduce(vcat, [s.y for s in series]))
    xticks = log_ticks(reduce(vcat, [s.x for s in series]))
    xlabelv = xquantity === :h ? "mesh size h" : "degrees of freedom"

    Plotter.plot(; show = true, size = (1000,1000), margin = 1Plots.cm, legendfontsize = 20, tickfontsize = 22, guidefontsize = 26, grid=true)
    for s in series
        Plotter.plot!(s.x, s.y; xscale = :log10, yscale = :log10, linewidth = 3,
            marker = :circle, markersize = 5, label = s.label, s.style...)
    end

    Plotter.plot!(; legend = :bottomleft, xtick = xticks, yticks = yticks, ylim = (yticks[1]/2, 2*yticks[end]), xlim = (xticks[1], xticks[end]), xlabel = xlabelv,gridalpha = 0.7,grid=true, background_color_legend = RGBA(1,1,1,0.7))
    ## save
    prefix = if quantities === :default && !with_incompressible
        ""
    elseif quantities === :all
        "_all_quantities"
    elseif with_incompressible
        "_with_incompressible"
    else
        "_" * join([string(n) for (n, _) in entries], "-")
    end
    ## record the refinement levels of this study (the single nrefs entry of
    ## the config dict is meaningless for a multi-level convergence plot)
    prefix *= "_nrefs-$(join(nrefs, '-'))"
    plotfile = filename_plots(data; prefix)
    Plotter.savefig(plotfile)
    ## append the computed values to the plot's .txt description
    columns = Pair{String, Vector}["ndofs" => vec(NDoFs)]
    for (k, (name, _)) in enumerate(entries)
        push!(columns, string(name) => vec(vals[k]))
    end
    if needs_inc
        push!(columns, "ndofs_inc" => vec(NDoFsInc))
    end
    save_plot_values(plotfile, data, columns)
end

# ==============================================================================
# Parameter study plots
# ==============================================================================

function plot_parameter_study_viscosity(; nrefs = [3], μ = [1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100,1000], Plotter = Plots, kwargs...)
    nrefs = nrefs isa AbstractVector ? nrefs : [nrefs]
    μ = μ isa AbstractVector ? μ : [μ]
    data = load_data(; kwargs...)
    @debug "loading config" data
    L2u = zeros(Float64, length(μ), length(nrefs))
    H1u = zeros(Float64, length(μ), length(nrefs))
    L2ϱ = zeros(Float64, length(μ), length(nrefs))

    for n = 1 : length(nrefs)
        data["nrefs"] = nrefs[n]
        for j = 1 : length(μ)
            data["μ"] = μ[j]
            data, ~ = safe_produce_or_load(data)
            L2u[j,n] = data["Error(L2,u)"]
            H1u[j,n] = data["Error(H1,u)"]
            L2ϱ[j,n] = data["Error(L2,ϱ)"]
        end
    end

    ## plot
    labels = [" level $n" for n in nrefs]
    yticks = [1e-5,1e-4,1e-3,1e-2,1e-1,1,10]
    xticks = μ
    Plotter.plot(; show = true, size = (1600,1000), margin = 1Plots.cm, legendfontsize = 20, tickfontsize = 16, guidefontsize = 22)
    for n = 1 : length(nrefs)
        Plotter.plot!(μ, L2ϱ[:,n]; xscale = :log10, yscale = :log10, linewidth = 3, marker = :circle, markersize = 5, label = L"|| {ϱ}-ϱ_h \, || \mathrm{level} = %$(nrefs[n])")
    end
    for n = 1 : length(nrefs)
        Plotter.plot!(μ, L2u[:,n]; xscale = :log10, yscale = :log10, linewidth = 3, marker = :circle, markersize = 5, label = L"||\mathbf{u} - \mathbf{u}_h \, || \mathrm{level} = %$(nrefs[n]) ")    
    end
    Plotter.plot!(; legend = :topright, xtick = xticks, yticks = yticks, ylim = (yticks[1]/2, 2*yticks[end]), xlabel = "μ", gridalpha = 0.5, grid=true)
        
    ##
    print_table(μ, L2u; xlabel = "μ", ylabels = "|| u - u_h || ".* labels)
        
    ## save
    plotfile = filename_plots(data; free_parameter = "μ")
    Plotter.savefig(plotfile)
    save_plot_values(plotfile, data, Pair{String, Vector}[
        "μ" => vec(repeat(μ, length(nrefs))), "nrefs" => vec(repeat(nrefs, inner = length(μ))),
        "L2u" => vec(L2u), "H1u" => vec(H1u), "L2ϱ" => vec(L2ϱ)])
end

function plot_parameter_study_gamma(; nrefs = [3], γ = [1,1e+1,1e+2,1e+3], Plotter = Plots, kwargs...)
    nrefs = nrefs isa AbstractVector ? nrefs : [nrefs]
    γ = γ isa AbstractVector ? γ : [γ]
    data = load_data(; kwargs...)
    @debug "loading config" data
    L2u = zeros(Float64, length(γ), length(nrefs))
    H1u = zeros(Float64, length(γ), length(nrefs))
    L2ϱ = zeros(Float64, length(γ), length(nrefs))

    for n = 1 : length(nrefs)
        data["nrefs"] = nrefs[n]
        for j = 1 : length(γ)
            data["γ"] = γ[j]
            data, ~ = safe_produce_or_load(data)
            L2u[j,n] = data["Error(L2,u)"]
            H1u[j,n] = data["Error(H1,u)"]
            L2ϱ[j,n] = data["Error(L2,ϱ)"]
        end
    end

    ## plot
    labels = [" level $n" for n in nrefs]
    yticks = [1e-5,1e-4,1e-3,1e-2,1e-1,1,10]
    xticks = γ
    Plotter.plot(; show = true, size = (1600,1000), margin = 1Plots.cm, legendfontsize = 20, tickfontsize = 16, guidefontsize = 22)
    for n = 1 : length(nrefs)
        Plotter.plot!(γ, L2ϱ[:,n]; xscale = :log10, yscale = :log10, linewidth = 3, marker = :circle, markersize = 5, label = L"|| {ϱ}-ϱ_h \, || \mathrm{level} = %$(nrefs[n])")
    end
    for n = 1 : length(nrefs)
        Plotter.plot!(γ, L2u[:,n]; xscale = :log10, yscale = :log10, linewidth = 3, marker = :circle, markersize = 5, label = L"||\mathbf{u} - \mathbf{u}_h \, || \mathrm{level} = %$(nrefs[n]) ")    
    end
    Plotter.plot!(; legend = :topright, xtick = xticks, yticks = yticks, ylim = (yticks[1]/2, 2*yticks[end]), xlabel = "γ", gridalpha = 0.5, grid=true)
        
    ##
    print_table(γ, L2u; xlabel = "γ", ylabels = "|| u - u_h || ".* labels)
        
    ## save
    plotfile = filename_plots(data; free_parameter = "γ")
    Plotter.savefig(plotfile)
    save_plot_values(plotfile, data, Pair{String, Vector}[
        "γ" => vec(repeat(γ, length(nrefs))), "nrefs" => vec(repeat(nrefs, inner = length(γ))),
        "L2u" => vec(L2u), "H1u" => vec(H1u), "L2ϱ" => vec(L2ϱ)])
end

function plot_parameter_study_mach_number(; nrefs = [3], c = [1,1e+1,1e+2,1e+3,1e+4,1e+5], Plotter = Plots, kwargs...)
    nrefs = nrefs isa AbstractVector ? nrefs : [nrefs]
    c = c isa AbstractVector ? c : [c]
    data = load_data(; kwargs...)
    @debug "loading config" data
    L2u = zeros(Float64, length(c), length(nrefs))
    H1u = zeros(Float64, length(c), length(nrefs))
    L2ϱ = zeros(Float64, length(c), length(nrefs))

    for n = 1 : length(nrefs)
        data["nrefs"] = nrefs[n]
        for j = 1 : length(c)
            data["c"] = c[j]
            data, ~ = safe_produce_or_load(data)
            L2u[j,n] = data["Error(L2,u)"]
            H1u[j,n] = data["Error(H1,u)"]
            L2ϱ[j,n] = data["Error(L2,ϱ)"]
        end
    end

    ## plot
    labels = [" level $n" for n in nrefs]
    yticks = [1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10,1e+1,1e+2]
    xticks = c
    Plotter.plot(; show = true, size = (1600,1000), margin = 1Plots.cm, legendfontsize = 20, tickfontsize = 16, guidefontsize = 22)
    for n = 1 : length(nrefs)
        Plotter.plot!(c, L2ϱ[:,n]; xscale = :log10, yscale = :log10, linewidth = 3, marker = :circle, markersize = 5, label = L"|| {ϱ}-ϱ_h \, || \mathrm{level} = %$(nrefs[n])")
    end
    for n = 1 : length(nrefs)
        Plotter.plot!(c, L2u[:,n]; xscale = :log10, yscale = :log10, linewidth = 3, marker = :circle, markersize = 5, label = L"||\mathbf{u} - \mathbf{u}_h \, || \mathrm{level} = %$(nrefs[n]) ")    
    end
    Plotter.plot!(; legend = :topright, xtick = xticks, yticks = yticks, ylim = (yticks[1]/2, 2*yticks[end]), xlabel = L"$c_M", gridalpha = 0.5, grid=true)
        
    ##
    print_table(c, L2u; xlabel = "c", ylabels = "|| u - u_h || ".* labels)
    print_table(c, L2ϱ; xlabel = "c", ylabels = "|| ϱ - ϱ_h || ".* labels)
        
    ## save
    plotfile = filename_plots(data; free_parameter = "c")
    Plotter.savefig(plotfile)
    save_plot_values(plotfile, data, Pair{String, Vector}[
        "c" => vec(repeat(c, length(nrefs))), "nrefs" => vec(repeat(nrefs, inner = length(c))),
        "L2u" => vec(L2u), "H1u" => vec(H1u), "L2ϱ" => vec(L2ϱ)])
end

function plot_parameter_study_mach_viscosity(; nrefs = 3,c = [1,1e+1,1e+2,1e+3,1e+4,1e+5,1e+6], μ = [1e-4,1e-3,1e-2,1e-1,1] , Plotter = Plots, kwargs...)
    c = c isa AbstractVector ? c : [c]
    μ = μ isa AbstractVector ? μ : [μ]
    data = load_data(; kwargs...)
    data["nrefs"] = nrefs
    @debug "loading config" data
    L2u = zeros(Float64, length(c), length(μ))
    H1u = zeros(Float64, length(c), length(μ))
    L2ϱ = zeros(Float64, length(c), length(μ))

    for n = 1 : length(μ)
        data["μ"] = μ[n]
        for j = 1 : length(c)
            data["c"] = c[j]
            data, ~ = safe_produce_or_load(data)
            L2u[j,n] = data["Error(L2,u)"]
            H1u[j,n] = data["Error(H1,u)"]
            L2ϱ[j,n] = data["Error(L2,ϱ)"]
        end
    end

    ## plot
    labels = [" μ =  $μk" for μk in μ]
    yticks = [1e-12,1e-11,1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1e+0,1e+1,1e+2]
    xticks = c
    Plotter.plot(; show = true, size = (1600,1000), margin = 1Plots.cm, legendfontsize = 20, tickfontsize = 16, guidefontsize = 22)
    for n = 1 : length(μ)
        Plotter.plot!(c, L2u[:,n]; xscale = :log10, yscale = :log10, linewidth = 3, marker = :circle, markersize = 5, label = L"||\mathbf{u} - \mathbf{u}_h \, || \mathrm{μ} = %$(μ[n]) ")
    end
    Plotter.plot!(; legend = :topright, xtick = xticks, yticks = yticks, ylim = (yticks[1]/2, 2*yticks[end]), xlabel = L"c_\mathrm{Ma}", gridalpha = 0.5, grid=true)

    ##
    print_table(c, L2u; xlabel = "c", ylabels = "|| u - u_h || ".* labels)

    ## save
    plotfile = filename_plots(data; free_parameter = "cμ")
    Plotter.savefig(plotfile)
    save_plot_values(plotfile, data, Pair{String, Vector}[
        "c" => vec(repeat(c, length(μ))), "μ" => vec(repeat(μ, inner = length(c))),
        "nrefs" => fill(nrefs, length(c) * length(μ)),
        "L2u" => vec(L2u), "H1u" => vec(H1u), "L2ϱ" => vec(L2ϱ)])
end

function plot_parameter_study_stab1(;  nrefs = [3,4,5],c1 = [1e-5,1e-4,1e-3,1e-2,1e-1,1], Plotter = Plots, kwargs...)
    nrefs = nrefs isa AbstractVector ? nrefs : [nrefs]
    c1 = c1 isa AbstractVector ? c1 : [c1]
    data = load_data(; kwargs...)
    @debug "loading config" data
    L2u = zeros(Float64, length(c1), length(nrefs))
    H1u = zeros(Float64, length(c1), length(nrefs))
    L2ϱ = zeros(Float64, length(c1), length(nrefs))

    for n = 1 : length(nrefs)
        data["nrefs"] = nrefs[n]
        for j = 1 : length(c1)
                data["stab1"] = (data["stab1"][1], c1[j])
                data, ~ = safe_produce_or_load(data)
                L2u[j,n] = data["Error(L2,u)"]
                H1u[j,n] = data["Error(H1,u)"]
                L2ϱ[j,n] = data["Error(L2,ϱ)"]
        end
    end

    ## plot
    labels = [" level $n" for n in nrefs]
    yticks = [1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10]
    xticks = c1
    Plotter.plot(; show = true, size = (1600,1000), margin = 1Plots.cm, legendfontsize = 20, tickfontsize = 16, guidefontsize = 22)
    for n = 1 : length(nrefs)
        Plotter.plot!(c1, H1u[:,n]; xscale = :log10, yscale = :log10, linewidth = 3, marker = :circle, markersize = 5, label = L"||∇(\mathbf{u} - \mathbf{u}_h) \,|| \mathrm{level} = %$(nrefs[n])")
        Plotter.plot!(c1, L2ϱ[:,n]; xscale = :log10, yscale = :log10, linewidth = 3, marker = :circle, markersize = 5, label = L"|| {ϱ}-ϱ_h \, || \mathrm{level} = %$(nrefs[n])")
    end
    for n = 1 : length(nrefs)
        Plotter.plot!(c1, L2u[:,n]; xscale = :log10, yscale = :log10, linewidth = 3, marker = :circle, markersize = 5, label = L"||\mathbf{u} - \mathbf{u}_h \, || \mathrm{level} = %$(nrefs[n]) ")    
    end
    Plotter.plot!(; legend = :bottomright, xtick = xticks, yticks = yticks, ylim = (yticks[1]/2, 2*yticks[end]), xlabel = "c1", gridalpha = 0.5, grid=true)
        
    ##
    print_table(c1, L2u; xlabel = "c1", ylabels = "|| u - u_h || ".* labels)
        
    ## save
    plotfile = filename_plots(data; free_parameter = "c1")
    Plotter.savefig(plotfile)
    save_plot_values(plotfile, data, Pair{String, Vector}[
        "c1" => vec(repeat(c1, length(nrefs))), "nrefs" => vec(repeat(nrefs, inner = length(c1))),
        "L2u" => vec(L2u), "H1u" => vec(H1u), "L2ϱ" => vec(L2ϱ)])
end

function plot_parameter_study_stab2(;  nrefs = [3,4,5], c2  =[1e-4,1e-2,1,1e+2,1e+4], Plotter = Plots, kwargs...)
    nrefs = nrefs isa AbstractVector ? nrefs : [nrefs]
    c2 = c2 isa AbstractVector ? c2 : [c2]
    data = load_data(; kwargs...)
    @debug "loading config" data
    L2u = zeros(Float64, length(c2), length(nrefs))
    H1u = zeros(Float64, length(c2), length(nrefs))
    L2ϱ = zeros(Float64, length(c2), length(nrefs))

    for n = 1 : length(nrefs)
        data["nrefs"] = nrefs[n]
        for j = 1 : length(c2)
                data["stab2"] = (1.5, c2[j])
                data, ~ = safe_produce_or_load(data)
                L2u[j,n] = data["Error(L2,u)"]
                H1u[j,n] = data["Error(H1,u)"]
                L2ϱ[j,n] = data["Error(L2,ϱ)"]
        end
    end

    ## plot
    labels = [" level $n" for n in nrefs]
    yticks = [1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10]
    xticks = c2
    Plotter.plot(; show = true, size = (1600,1000), margin = 1Plots.cm, legendfontsize = 20, tickfontsize = 16, guidefontsize = 22)
    for n = 1 : length(nrefs)
        Plotter.plot!(c2, H1u[:,n]; xscale = :log10, yscale = :log10, linewidth = 3, marker = :circle, markersize = 5, label = L"||∇(\mathbf{u} - \mathbf{u}_h) \,|| \mathrm{level} = %$(nrefs[n])")
        Plotter.plot!(c2, L2ϱ[:,n]; xscale = :log10, yscale = :log10, linewidth = 3, marker = :circle, markersize = 5, label = L"|| {ϱ}-ϱ_h \, || \mathrm{level} = %$(nrefs[n])")
    end
    for n = 1 : length(nrefs)
        Plotter.plot!(c2, L2u[:,n]; xscale = :log10, yscale = :log10, linewidth = 3, marker = :circle, markersize = 5, label = L"||\mathbf{u} - \mathbf{u}_h \, || \mathrm{level} = %$(nrefs[n]) ")    
    end
    Plotter.plot!(; legend = :bottomright, xtick = xticks, yticks = yticks, ylim = (yticks[1]/2, 2*yticks[end]), xlabel = "c2", gridalpha = 0.5, grid=true)
        
    ##
    print_table(c2, L2u; xlabel = "c2", ylabels = "|| u - u_h || ".* labels)
        
    ## save
    plotfile = filename_plots(data; free_parameter = "c2")
    Plotter.savefig(plotfile)
    save_plot_values(plotfile, data, Pair{String, Vector}[
        "c2" => vec(repeat(c2, length(nrefs))), "nrefs" => vec(repeat(nrefs, inner = length(c2))),
        "L2u" => vec(L2u), "H1u" => vec(H1u), "L2ϱ" => vec(L2ϱ)])
end

function plot_parameter_study_stab1_reconstruction(;  reconstruct = [true,false], c1 = [1e-5,1e-4,1e-3,1e-2,1e-1,1,1e1,1e2,1e3,1e4,1e5], Plotter = Plots, kwargs...)
    reconstruct = reconstruct isa AbstractVector ? reconstruct : [reconstruct]
    c1 = c1 isa AbstractVector ? c1 : [c1]
    data = load_data(; kwargs...)
    @debug "loading config" data
    L2u = zeros(Float64, length(c1), length(reconstruct))
    H1u = zeros(Float64, length(c1), length(reconstruct))
    L2ϱ = zeros(Float64, length(c1), length(reconstruct))

    for n = 1 : length(reconstruct)
        data["reconstruct"] = reconstruct[n]
        for j = 1 : length(c1)
                data["stab1"] = (data["stab1"][1], c1[j])
                data, ~ = safe_produce_or_load(data)
                L2u[j,n] = data["Error(L2,u)"]
                H1u[j,n] = data["Error(H1,u)"]
                L2ϱ[j,n] = data["Error(L2,ϱ)"]
        end
    end

    ## plot
    pi_names = [r ? "\\Pi = I_h^{\\mathrm{RT_0}}" : "\\Pi = \\mathrm{Id}" for r in reconstruct]
    col = [r ? colorant"#389826" : colorant"#CB3C33" for r in reconstruct]  # Julia green / red
    allvals = vcat(vec(H1u), vec(L2ϱ), vec(L2u))
    allvals = filter(x -> x > 0 && isfinite(x), allvals)
    ylo = 10.0^floor(log10(minimum(allvals)) - 0.5)
    yhi = 10.0^ceil(log10(maximum(allvals)) + 1.0)
    yticks = 10.0 .^ (floor(Int, log10(ylo)):ceil(Int, log10(yhi)))
    xticks = 10.0 .^ (-5:5)
    Plotter.plot(; show = true, size = (1200,900), margin = 1Plots.cm, legendfontsize = 14, tickfontsize = 16, guidefontsize = 20)
    # grouped by quantity; plot red (false) first, green (true) on top so both visible
    order = sortperm(reconstruct)  # false first, true second
    for n in order
        Plotter.plot!(c1, H1u[:,n]; xscale = :log10, yscale = :log10, linewidth = 3, linestyle = :dashdot, marker = :circle, markersize = 5, color = col[n], label = latexstring("||\\nabla(\\mathbf{u} - \\mathbf{u}_h)||,\\; ", pi_names[n]))
    end
    for n in order
        Plotter.plot!(c1, L2ϱ[:,n]; xscale = :log10, yscale = :log10, linewidth = 3, linestyle = :dot, marker = :diamond, markersize = 5, color = col[n], label = latexstring("||\\varrho - \\varrho_h||,\\; ", pi_names[n]))
    end
    for n in order
        Plotter.plot!(c1, L2u[:,n]; xscale = :log10, yscale = :log10, linewidth = 3, linestyle = :solid, marker = :square, markersize = 5, color = col[n], label = latexstring("||\\mathbf{u} - \\mathbf{u}_h||,\\; ", pi_names[n]))
    end
    Plotter.plot!(; legend = :topleft, xtick = xticks, yticks = yticks, ylim = (ylo, yhi), xlabel = L"c_s", gridalpha = 0.5, grid = true, background_color_legend = RGBA(1,1,1,0.7))

    ##
    labels = [" Pi=$(reconstruct[n] ? "RT0" : "Gamma")" for n in 1:length(reconstruct)]
    print_table(c1, L2u; xlabel = "c_s", ylabels = "|| u - u_h || " .* labels)

    ## save
    plotfile = filename_plots(data; free_parameter = "c1")
    Plotter.savefig(plotfile)
    save_plot_values(plotfile, data, Pair{String, Vector}[
        "c1" => vec(repeat(c1, length(reconstruct))), "reconstruct" => vec(repeat(reconstruct, inner = length(c1))),
        "L2u" => vec(L2u), "H1u" => vec(H1u), "L2ϱ" => vec(L2ϱ)])
end

function plot_parameter_study_alpha_reconstruction(;  reconstruct = [true,false], alpha = [0,5e-1,1,1e-0,1+5e-1,2e-0], Plotter = Plots, kwargs...)
    reconstruct = reconstruct isa AbstractVector ? reconstruct : [reconstruct]
    alpha = alpha isa AbstractVector ? alpha : [alpha]
    data = load_data(; kwargs...)
    @debug "loading config" data
    L2u = zeros(Float64, length(alpha), length(reconstruct))
    H1u = zeros(Float64, length(alpha), length(reconstruct))
    L2ϱ = zeros(Float64, length(alpha), length(reconstruct))

    for n = 1 : length(reconstruct)
        data["reconstruct"] = reconstruct[n]
        for j = 1 : length(alpha)
                data["stab1"] = (alpha[j]-1, data["stab1"][2])
                @info "α = $(alpha[j])"
                data, ~ = safe_produce_or_load(data)
                L2u[j,n] = data["Error(L2,u)"]
                H1u[j,n] = data["Error(H1,u)"]
                L2ϱ[j,n] = data["Error(L2,ϱ)"]
        end
    end

    ## plot
    pi_names = [r ? "\\Pi = I_h^{\\mathrm{RT_0}}" : "\\Pi = \\mathrm{Id}" for r in reconstruct]
    col = [r ? colorant"#389826" : colorant"#CB3C33" for r in reconstruct]  # Julia green / red
    allvals = vcat(vec(H1u), vec(L2ϱ), vec(L2u))
    allvals = filter(x -> x > 0 && isfinite(x), allvals)
    ylo = 10.0^floor(log10(minimum(allvals)) - 0.5)
    yhi = 10.0^ceil(log10(maximum(allvals)) + 1.0)
    yticks = 10.0 .^ (floor(Int, log10(ylo)):ceil(Int, log10(yhi)))
    Plotter.plot(; show = true, size = (1200,900), margin = 1Plots.cm, legendfontsize = 14, tickfontsize = 16, guidefontsize = 20)
    # grouped by quantity; plot red (false) first, green (true) on top so both visible
    order = sortperm(reconstruct)  # false first, true second
    for n in order
        Plotter.plot!(alpha, H1u[:,n]; yscale = :log10, linewidth = 3, linestyle = :dashdot, marker = :circle, markersize = 5, color = col[n], label = latexstring("||\\nabla(\\mathbf{u} - \\mathbf{u}_h)||,\\; ", pi_names[n]))
    end
    for n in order
        Plotter.plot!(alpha, L2ϱ[:,n]; yscale = :log10, linewidth = 3, linestyle = :dot, marker = :diamond, markersize = 5, color = col[n], label = latexstring("||\\varrho - \\varrho_h||,\\; ", pi_names[n]))
    end
    for n in order
        Plotter.plot!(alpha, L2u[:,n]; yscale = :log10, linewidth = 3, linestyle = :solid, marker = :square, markersize = 5, color = col[n], label = latexstring("||\\mathbf{u} - \\mathbf{u}_h||,\\; ", pi_names[n]))
    end
    Plotter.plot!(; legend = :topright, xtick = alpha, yticks = yticks, ylim = (ylo, yhi), xlim = (-0.05, 2.05), xlabel = L"\alpha", gridalpha = 0.5, grid = true, background_color_legend = RGBA(1,1,1,0.7))

    ##
    labels = [" Pi=$(reconstruct[n] ? "RT0" : "Gamma")" for n in 1:length(reconstruct)]
    print_table(alpha, L2u; xlabel = "α", ylabels = "|| u - u_h || " .* labels)

    ## save
    plotfile = filename_plots(data; free_parameter = "α")
    Plotter.savefig(plotfile)
    save_plot_values(plotfile, data, Pair{String, Vector}[
        "α" => vec(repeat(alpha, length(reconstruct))), "reconstruct" => vec(repeat(reconstruct, inner = length(alpha))),
        "L2u" => vec(L2u), "H1u" => vec(H1u), "L2ϱ" => vec(L2ϱ)])
end
