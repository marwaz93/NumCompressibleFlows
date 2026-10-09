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
    L2divu = (; data = "Error(L2,div u)", label = L"|| \mathrm{div}(\mathbf{u} - \mathbf{u}_h) \,||"),  # L2 error of divergence
    L2divuR = (; data = "Error(L2,div uR)", label = L"|| \mathrm{div}(\mathbf{u} - \Pi\mathbf{u}_h) \,||"), # L2 error of divergence of reconstruction
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

const DEFAULT_QUANTITIES = (:L2u, :H1u, :L2ϱ, :L2ϱu, :H1u0, :L2divu)
const DEFAULT_SWEEP_QUANTITIES = (:L2u, :H1u, :L2ϱ)

## built-in default sweep values for categorical config fields of
## plot_parameter_study (see SWEEP_CHOICES for the mutable current setting)
const DEFAULT_SWEEP_CHOICES = (
    convectiontype = (StandardConvection, OseenConvection, KarperConvection, NewConvection),
    upwindtype = (StandardUpwind, PointwiseUpwind),
    reconstruct = (:RT, :BDM, :none),
    coriolistype = (NoCoriolis, BetaPlaneApproximation),
    eostype = (IdealGasLaw, PowerLaw{1.4}),
)

## current default sweep values per config field of plot_parameter_study;
## modify with set_sweep_choices! / reset_sweep_choices! (or directly)
const SWEEP_CHOICES = Dict{Symbol, Any}(k => v for (k, v) in pairs(DEFAULT_SWEEP_CHOICES))

"""
    set_sweep_choices!(; param = values...)

Set the default sweep `values` that `plot_parameter_study` uses for
categorical config fields, e.g.

    set_sweep_choices!(convectiontype = (OseenConvection, NewConvection),
                       reconstruct = (:RT, :BDM))

Each `param` must be a config dict key (a key of `default_args`, except
`:nrefs`); the values are stored as a tuple, so vectors and tuples are both
accepted. New fields can be added this way too. See also
`reset_sweep_choices!`; `SWEEP_CHOICES` may also be edited directly.
"""
function set_sweep_choices!(; kwargs...)
    isempty(kwargs) && error("usage: set_sweep_choices!(param = values, ...)")
    for (param, values) in pairs(kwargs)
        haskey(default_args, String(param)) || error("unknown config field :$param; choose one of: $(sort(collect(keys(default_args))))")
        param === :nrefs && error(":nrefs is the x-axis of plot_parameter_study, it cannot be swept")
        vals = values isa Union{AbstractVector, Tuple} ? Tuple(values) : (values,)
        isempty(vals) && error("sweep choices for :$param must not be empty")
        SWEEP_CHOICES[param] = vals
    end
    return SWEEP_CHOICES
end

"""
    reset_sweep_choices!()

Restore the built-in defaults of `SWEEP_CHOICES`.
"""
function reset_sweep_choices!()
    empty!(SWEEP_CHOICES)
    for (param, values) in pairs(DEFAULT_SWEEP_CHOICES)
        SWEEP_CHOICES[param] = values
    end
    return SWEEP_CHOICES
end

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

## short display/file name of a sweep value (module qualification removed,
## common type suffixes abbreviated)
short_name(v) = replace(string(v), "NumCompressibleFlows." => "",
    "UnstructuredUnitSquare" => "SquareUnstruct", "UniformUnitSquare" => "SquareUniform",
    "UnitSquare" => "Square", "Convection" => "Conv", "Upwind" => "Upw",
    "Coriolis" => "Cor", "Approximation" => "")

"""
    resolve_quantities(quantities; with_incompressible = false)

Resolve a quantity selection against `CONV_QUANTITIES`: a vector of symbols
(`:default` = `DEFAULT_QUANTITIES`, `:all` = whole registry), or inline
NamedTuples with fields `name`, `data` and `label`. Returns a vector of
`(name, entry)` pairs.
"""
function resolve_quantities(quantities; with_incompressible = false)
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
    return entries
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
`CONV_QUANTITIES` registry (e.g. `[:L2u, :H1u, :L2ϱ, :H1u0, :H1u1, :L2divu, :nits]`).
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
    entries = resolve_quantities(quantities; with_incompressible)
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

## per-sweep-value colors, per-quantity linestyles and markers (cycled as needed)
const STUDY_COLORS = [colorant"#386cb0", colorant"#e6550d", colorant"#31a354", colorant"#756bb1",
                      colorant"#a6761d", colorant"#7b3294", colorant"#525252", colorant"#e7298a"]
const STUDY_LINESTYLES = (:solid, :dashdot, :dot, :dash, :dashdotdot)
const STUDY_MARKERS = (:circle, :rect, :diamond, :utriangle, :dtriangle, :pentagon)

"""
    plot_parameter_study(; param = :convectiontype, values = nothing, nrefs = 1:4,
        quantities = DEFAULT_SWEEP_QUANTITIES, xquantity = :ndofs, slopes = (1, 2),
        Plotter = Plots, force = false, force_recompute = false, kwargs...)

Compare convergence histories of several configurations, one per value of a
single entry of the config dict: `param` names the key (any key of
`default_args`, except `:nrefs`), `values` the list of values to compare, e.g.

    plot_parameter_study(param = :convectiontype,
        values = [NewConvection, OseenConvection, StandardConvection, KarperConvection],
        quantities = [:L2u, :H1u])

For categorical fields (`convectiontype`, `upwindtype`, `reconstruct`,
`coriolistype`, `eostype`) `values` defaults to the mutable `SWEEP_CHOICES`
registry (change it with `set_sweep_choices!`); for numeric fields (e.g.
`param = :μ`) a `values` vector is required. Every
keyword argument in `kwargs` is study configuration passed to `load_data`
(just like in `plot_convergencehistory`).

Each (value, refinement level) combination is run through the data pipeline
(`safe_produce_or_load` + `compute_errors`). The plot then shows one curve per
quantity/value pair against `xquantity`:
- `:ndofs` or `:h` (convergence comparison): x is the DOF count / mesh size,
  one curve per quantity/value pair, quantities distinguished by linestyle
  and marker, values by color; `slopes` adds reference lines O(h^k) anchored
  at the first curve.
- `:param`: the sweep values must be numeric and form the x-axis (log-scale
  if all positive), so one mesh or mesh level (`nrefs`, typically a single
  value like `nrefs = 4`) is fixed and the quantities are plotted against
  e.g. the viscosity `μ` or a stabilization factor; one curve per
  quantity/level pair, quantities by linestyle and marker, levels by color
  (`slopes` is ignored). For tuple-valued fields such as `stab1 = (α, c1)`,
  `xindex = 2` plots against the second component.

Incompressible reference quantities (`:H1u_inc`, `:L2u_inc`, `:L2u_diff`) are
supported as in `plot_convergencehistory`.
"""
function plot_parameter_study(; param = :convectiontype, values = nothing, nrefs = 1:4,
        quantities = DEFAULT_SWEEP_QUANTITIES, xquantity = :ndofs, xindex = nothing,
        slopes = (1, 2), Plotter = Plots, force = false, force_recompute = false, kwargs...)

    key = String(param)
    haskey(default_args, key) || error("unknown config field :$param; choose one of: $(sort(collect(keys(default_args))))")
    key == "nrefs" && error(":nrefs is the x-axis of this plot; sweep another field via param/values")
    !(xquantity in (:ndofs, :h, :param)) && error("xquantity must be :ndofs, :h or :param")

    if values === nothing
        choices = get(SWEEP_CHOICES, Symbol(key), nothing)
        choices !== nothing || error("no default sweep choices for :$param, pass values = [...]")
        values = collect(choices)
    else
        values = values isa AbstractVector ? collect(values) : [values]
    end
    length(values) == 1 && @warn "plot_parameter_study: only one value given for :$key"

    ## with xquantity = :param the (numeric) sweep values are the x-axis
    against_param = xquantity === :param
    if against_param
        xvals = map(values) do v
            v isa Real && return Float64(v)
            xindex === nothing && error("xquantity = :param needs numeric sweep values, or xindex = <i> to plot against the i-th component of tuple values")
            v[xindex] isa Real || error("xindex = $xindex does not select a number in the sweep values")
            return Float64(v[xindex])
        end
    end

    nrefs_vals = collect(nrefs)
    entries = resolve_quantities(quantities)
    needs_inc = any(get(e, :incompressible, false) for (_, e) in entries)

    @info "Plotting parameter study for $key = $(short_name.(values)), nrefs = $nrefs_vals, quantities = $(first.(entries)), xquantity = $xquantity, slopes = $slopes..."

    data = load_data(; kwargs...)
    nv, nl, nq = length(values), length(nrefs_vals), length(entries)
    vals = [fill(NaN, nv, nl) for _ in 1:nq]   # vals[k][i, j]: quantity k, value i, level j
    NDoFs = fill(0, nv, nl)
    NDoFsInc = fill(0, nv, nl)

    for (i, val) in enumerate(values)
        @info "study config: $key = $(short_name(val))"
        for (j, lvl) in enumerate(nrefs_vals)
            _data = deepcopy(data)
            _data[key] = val
            _data["nrefs"] = lvl
            _data, ~ = safe_produce_or_load(_data; force = force)
            NDoFs[i, j] = _data["ndofs"]
            _data = compute_errors(_data; force_recompute = force_recompute, compare_incompressible = needs_inc)
            for (k, (_, e)) in enumerate(entries)
                vals[k][i, j] = conv_value(e, _data)
            end
            if needs_inc
                NDoFsInc[i, j] = haskey(_data, "incompressible_solution") ? length(_data["incompressible_solution"].entries) : NDoFs[i, j]
            end
        end
        ## console table of the first up to four selected quantities for this value
        sel = min(4, nq)
        if !against_param
            print_convergencehistory(NDoFs[i, :], hcat([vals[k][i, :] for k in 1:sel]...); X_to_h = X -> X.^(-1/2),
                ylabels = [string(entries[k][2].label) * " (" * short_name(val) * ")" for k in 1:sel],
                xlabel = xquantity === :h ? "h" : "ndof", latex_mode = true)
        end
    end
    if against_param
        ## console table of values against the sweep parameter, per level
        sel = min(4, nq)
        for j in 1:nl
            print_table(xvals, hcat([vals[k][:, j] for k in 1:sel]...); xlabel = key,
                ylabels = [string(entries[k][2].label) * " level $(nrefs_vals[j])" for k in 1:sel])
        end
    end

    ## collect all curves; Vector{NamedTuple} avoids push! on growing concrete types
    series = NamedTuple[]
    logx = true
    if against_param
        ## one curve per quantity/level pair against the sweep parameter
        logx = all(isfinite, xvals) && all(>(0), xvals)
        for (k, (_, e)) in enumerate(entries)
            for (j, lvl) in enumerate(nrefs_vals)
                push!(series, (; x = xvals, y = vals[k][:, j],
                    label = LaTeXString(string(e.label) * ", level " * string(lvl)),
                    style = (linestyle = STUDY_LINESTYLES[mod1(k, end)], marker = STUDY_MARKERS[mod1(k, end)],
                             color = STUDY_COLORS[mod1(j, end)])))
            end
        end
    else
        ## one curve per quantity/value pair (linestyle and marker by
        ## quantity, color by value), plus slope reference lines anchored at
        ## the first curve
        for (k, (_, e)) in enumerate(entries)
            for i in 1:nv
                x0 = get(e, :xinc, false) ? NDoFsInc[i, :] : NDoFs[i, :]
                x = xquantity === :h ? Float64.(x0) .^ (-1/2) : Float64.(x0)
                push!(series, (; x, y = vals[k][i, :],
                    label = LaTeXString(string(e.label) * ", " * short_name(values[i])),
                    style = (linestyle = STUDY_LINESTYLES[mod1(k, end)], marker = STUDY_MARKERS[mod1(k, end)],
                             color = STUDY_COLORS[mod1(i, end)])))
            end
        end
        if !isempty(slopes)
            hvals = Float64.(NDoFs[1, :]) .^ (-1/2)
            j0 = findfirst(j -> isfinite(vals[1][1, j]) && vals[1][1, j] > 0, 1:nl)
            if j0 !== nothing
                y0 = vals[1][1, j0]
                append!(series, [ (; x = xquantity === :h ? hvals : Float64.(NDoFs[1, :]),
                    y = y0 .* (hvals ./ hvals[j0]) .^ m,
                    label = m == 1 ? L"\mathcal{O}(h)" : latexstring("\\mathcal{O}(h^{$m})"),
                    style = (linestyle = :dash, color = :gray, marker = :none)) for m in slopes ])
            end
        end
    end

    ## axis ticks: powers of ten covering all plotted values (or the sweep
    ## values themselves for a linear parameter axis)
    yticks = log_ticks(reduce(vcat, [s.y for s in series]))
    xticks = logx ? log_ticks(reduce(vcat, [s.x for s in series])) : unique!(sort!(reduce(vcat, [s.x for s in series])))
    xlabelv = against_param ? key : (xquantity === :h ? "mesh size h" : "degrees of freedom")

    Plotter.plot(; show = true, size = (1600, 1000), margin = 1Plots.cm, legendfontsize = 14, tickfontsize = 18, guidefontsize = 22, grid = true)
    for s in series
        Plotter.plot!(s.x, s.y; xscale = logx ? :log10 : :identity, yscale = :log10, linewidth = 3, markersize = 5,
            label = s.label, s.style...)
    end
    xlimv = logx ? (xticks[1], xticks[end]) : let lo = minimum(reduce(vcat, [s.x for s in series])), hi = maximum(reduce(vcat, [s.x for s in series])), w = max(hi - lo, oneunit(hi))
        (lo - 0.05w, hi + 0.05w)
    end
    Plotter.plot!(; legend = :best, xtick = xticks, yticks = yticks, ylim = (yticks[1]/2, 2*yticks[end]), xlim = xlimv, xlabel = xlabelv, gridalpha = 0.7, grid = true, background_color_legend = RGBA(1,1,1,0.7))

    ## save: filename freezes all study parameters except the swept field and
    ## nrefs (done by the pipeline's filename_plots via free_parameter = key)
    ## plot filename suffix: the swept field (its values are recorded in the
    ## .txt description), non-default quantity selection and refinement levels
    ## (a level range with unit step is written as first-last)
    prefix = "_" * key
    qnames = [string(n) for (n, _) in entries]
    qnames == [string(q) for q in DEFAULT_SWEEP_QUANTITIES] || (prefix *= "_" * join(qnames, "-"))
    nrefs_str = length(nrefs_vals) > 2 && all(==(1), diff(nrefs_vals)) ? "$(nrefs_vals[1])-$(nrefs_vals[end])" : join(nrefs_vals, '-')
    prefix *= (against_param ? "_levels-$nrefs_str" : "_nrefs-$nrefs_str")
    plotfile = filename_plots(data; prefix, free_parameter = key)
    Plotter.savefig(plotfile)
    ## append the computed values to the plot's .txt description (row order: value outer, level inner)
    columns = Pair{String, Vector}[
        key => repeat(short_name.(values), inner = nl),
        "nrefs" => repeat(nrefs_vals, outer = nv),
        "ndofs" => vec(NDoFs')]
    against_param && push!(columns, "x" => repeat(xvals, inner = nl))
    for (k, (name, _)) in enumerate(entries)
        push!(columns, string(name) => vec(vals[k]'))
    end
    if needs_inc
        push!(columns, "ndofs_inc" => vec(NDoFsInc'))
    end
    save_plot_values(plotfile, data, columns)
end
