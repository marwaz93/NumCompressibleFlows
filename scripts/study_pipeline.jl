# DrWatson layer shared by all example scripts. For a given study name this
# creates the study folders, builds the data/plot filename functions and the
# file production functions, and installs them into the NumCompressibleFlows
# module via setup_pipeline!. All data and plot files of one study share one
# joint base path. Usage from an example script:
#   include(joinpath(@__DIR__, "study_pipeline.jl"))
#   setup_study!("my_study_name"; plot_folders = ["", "μ", "c"])
#   plot_convergencehistory(nrefs = 2:5)

using DrWatson
using NumCompressibleFlows

quickactivate(@__DIR__, "NumCompressibleFlows")

## plot subfolder: convergence history, or one folder per parameter study
plot_folder(free_parameter) = free_parameter === "" ? "convergence_history" : "parameter_studies_" * free_parameter

"""
    write_description(fpath, data; header = "", values = nothing)

Write a text file `<fpath>.txt` listing all entries of the `data` dict, one
per line (`key = value`, long values abbreviated), followed by an optional
tab-separated table of computed values given as a vector of
`name => column_vector` pairs (unequal column lengths are padded). Used to
document data (`.jld2`) and plot (`.png`) files of a study.
"""
function write_description(fpath, data; header = "", values = nothing)
    open(fpath * ".txt", "w") do io
        isempty(header) || println(io, "# ", header)
        for key in sort(collect(keys(data)))
            value = data[key]
            desc = if value isa Union{Real, AbstractString, AbstractChar, Symbol, Bool, Nothing, DataType, Function, Tuple}
                string(value)
            elseif value isa AbstractArray
                "<$(typeof(value)), size $(size(value))>"
            else
                s = replace(repr(value), "\n" => " ")
                length(s) > 100 ? first(s, 100) * " …" : s
            end
            println(io, key, " = ", desc)
        end
        if values !== nothing && !isempty(values)
            println(io)
            println(io, "# computed values")
            println(io, join([string(name) for (name, _) in values], "\t"))
            n = maximum(length(column) for (_, column) in values)
            for i in 1:n
                println(io, join([i <= length(column) ? string(column[i]) : "" for (_, column) in values], "\t"))
            end
        end
    end
    return fpath * ".txt"
end

"""
    setup_study!(studyname; plot_folders = [""])

Prepare the folders of the study `studyname` below `datadir("projects", ...)`
and `plotsdir(...)`, build its DrWatson file naming and file production
functions, and register them in the NumCompressibleFlows module with
`setup_pipeline!`. `plot_folders` selects the plot subfolders: `""` is the
convergence history, any other string `fp` a parameter study folder
`parameter_studies_fp`. Returns `(; filename, filename_plots, produce,
compute_errors)` bound to this study.
"""
function setup_study!(studyname; plot_folders = [""])
    @info "setting up study" studyname
    dataprefix = datadir("projects", studyname) * "/"
    plots_prefix = plotsdir(studyname) * "/"
    for free_parameter in plot_folders
        mkpath(plots_prefix * plot_folder(free_parameter))
    end
    @info "study folders ready" dataprefix plots_prefix

    """
        filename(data; prefix = dataprefix) -> String

    Build a DrWatson-compatible filename string for a given `data` dict.
    All key parameters are abbreviated and the result is prefixed with the
    study's data path `<datadir>/projects/<studyname>/`.
    """
    function filename(data; prefix = dataprefix)
        μ = data["μ"]
        λ = data["λ"]
        c = data["c"]
        M = data["M"]
        τfac = data["τfac"]
        ufac = data["ufac"]
        nrefs = data["nrefs"]
        order = data["order"]
        reconstruct = data["reconstruct"]
        target_residual = data["target_residual"]

        # Abbreviate type names for savename
        vtype = replace(string(data["velocitytype"]), "Velocity" => "V")
        dtype = replace(string(data["densitytype"]), "Density" => "D")
        etype = replace(string(data["eostype"]), "Law" => "")  # carries γ, e.g. Power{1.4}
        gtype = replace(string(data["gridtype"]), "2D" => "")
        ctype = replace(string(data["convectiontype"]), "Convection" => "Conv")
        cortype = replace(string(data["coriolistype"]), "Coriolis" => "Cor")
        pressure_in_f = data["pressure_in_f"]
        stab1 = data["stab1"]
        stab2 = data["stab2"]
        convectiontype = data["convectiontype"]

        ## ordered by relevance: problem definition, physics parameters,
        ## discretization, numerical factors, solver tolerance, refinement
        ## level (savename keeps the order of a NamedTuple when sort = false)
        essential = (vtype = vtype, dtype = dtype, etype = etype, gtype = gtype,
                     ctype = ctype, cortype = cortype, pressure_in_f = pressure_in_f,
                     stab1 = stab1, stab2 = stab2,
                     μ = μ, λ = λ, c = c, M = M,
                     order = order, reconstruct = reconstruct,
                     τfac = τfac, ufac = ufac, tres = target_residual,
                     nrefs = nrefs)

        sname = savename(essential; sort = false,
                         allowedtypes = (Real, String, SubString, Symbol,
                                         Tuple{Real, Real}))
        sname = prefix * sname
        return sname
    end

    """
        filename_plots(data; prefix = "", free_parameter = "")

    Build a filename for a plot using `savename` with parameters selected by
    `free_parameter`.  Each free parameter freezes a different subset of the
    remaining parameters for inclusion in the savename. The plot is placed in
    the study's plot subfolder (`convergence_history` or
    `parameter_studies_<fp>`). Returns a path string ending in `.png`. As a
    side effect, writes a `.txt` file describing the parameters next to the
    (about to be saved) plot file.
    """
    function filename_plots(data; prefix = "", free_parameter = "")
        μ = data["μ"]
        c = data["c"]
        etype = replace(string(data["eostype"]), "Law" => "")  # carries γ, e.g. Power{1.4}
        ϵ = 1 - data["stab1"][1]
        c1 = data["stab1"][2]
        nrefs = data["nrefs"]
        reconstruct = data["reconstruct"]
        convectiontype = string(data["convectiontype"])
        pressure_in_f = data["pressure_in_f"]

        # Select which params go into savename depending on free_parameter,
        # ordered by relevance: problem definition, stabilization, physics,
        # discretization (savename keeps the order when sort = false)
        essential = if free_parameter == "μ"
            (convectiontype = convectiontype, ϵ = ϵ, c1 = c1, c = c, etype = etype,
             reconstruct = reconstruct, nrefs = nrefs)
        elseif free_parameter == "γ"
            (convectiontype = convectiontype, ϵ = ϵ, c1 = c1, μ = μ, c = c,
             reconstruct = reconstruct, nrefs = nrefs)
        elseif free_parameter == "c"
            (convectiontype = convectiontype, ϵ = ϵ, c1 = c1, μ = μ, etype = etype,
             reconstruct = reconstruct, nrefs = nrefs)
        elseif free_parameter == "cμ"
            (convectiontype = convectiontype, ϵ = ϵ, c1 = c1, etype = etype,
             reconstruct = reconstruct, nrefs = nrefs)
        elseif free_parameter == "c1"
            (convectiontype = convectiontype, ϵ = ϵ, μ = μ, c = c, etype = etype,
             reconstruct = reconstruct, nrefs = nrefs)
        elseif free_parameter in ("c2", "α")
            (convectiontype = convectiontype, ϵ = ϵ, c1 = c1, μ = μ, c = c, etype = etype,
             reconstruct = reconstruct, nrefs = nrefs)
        else
            ## convergence history: no single nrefs (the run spans a range of levels,
            ## appended to the prefix by plot_convergencehistory instead)
            (convectiontype = convectiontype, pressure_in_f = pressure_in_f,
             ϵ = ϵ, c1 = c1, μ = μ, c = c, etype = etype, reconstruct = reconstruct)
        end

        sname = savename(essential; sort = false,
                         allowedtypes = (Real, String, SubString, Symbol,
                                         Tuple{Real, Real}))

        sname = plots_prefix * plot_folder(free_parameter) * "/" * sname * prefix * ".png"
        write_description(sname[1:end-4], data; header = "plot file $sname, free_parameter = $(free_parameter), prefix = $(prefix)")
        return sname
    end

    """
        safe_produce_or_load(data; force = false, kwargs...)

    Wrapper around `produce_or_load` that calls `run_single` to compute errors.
    Additionally writes a `.txt` file describing all dict entries next to the
    data file.
    """
    function safe_produce_or_load(data; force = false, kwargs...)
        _data, loaded = produce_or_load(run_single, data; filename = filename, force = force, kwargs...)
        write_description(filename(data), _data; header = "data file $(filename(data)).jld2")
        return _data, loaded
    end

    """
        compute_errors(config; force_recompute = false, compare_incompressible = true, kwargs...)

    Load the JLD2 data of a solved configuration (via `filename`), compute and
    store error norms (via `compute_errors!`) and save the data back to the file.
    """
    function compute_errors(config; kwargs...)
        fpath = filename(config) * ".jld2"
        @info "loading data from $fpath"
        data = wload(fpath)
        compute_errors!(data; kwargs...) isa Nothing && return nothing
        fpath = filename(data) * ".jld2"
        @info "saving data to $fpath"
        wsave(fpath, data)
        write_description(filename(data), data; header = "data file $fpath (including error norms)")
        return data
    end

    """
        record_values(plotfile, data, columns)

    Rewrite the `.txt` description of the plot `plotfile` (a `.png` path) with
    the parameters of `data` plus a tab-separated table of computed values,
    given as a vector of `name => column_vector` pairs.
    """
    function record_values(plotfile, data, columns; header = "plot file $plotfile (with computed values)")
        write_description(plotfile[1:end-4], data; header, values = columns)
        return nothing
    end

    NumCompressibleFlows.setup_pipeline!(
        filename = filename,
        filename_plots = filename_plots,
        produce = safe_produce_or_load,
        compute_errors = compute_errors,
        record_values = record_values,
    )

    return (; filename, filename_plots, produce = safe_produce_or_load, compute_errors, record_values)
end
