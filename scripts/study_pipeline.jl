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

Prepare the folders of the study `studyname` below `datadir(...)` and
`plotsdir(...)`, build its DrWatson file naming and file production
functions, and register them in the NumCompressibleFlows module with
`setup_pipeline!`. `plot_folders` selects the plot subfolders: `""` is the
convergence history, any other string `fp` a parameter study folder
`parameter_studies_fp`. Returns `(; filename, filename_plots, produce,
compute_errors)` bound to this study.
"""
function setup_study!(studyname; plot_folders = [""])
    @info "setting up study" studyname
    dataprefix = datadir(studyname) * "/"
    plots_prefix = plotsdir(studyname) * "/"
    for free_parameter in plot_folders
        mkpath(plots_prefix * plot_folder(free_parameter))
    end
    @info "study folders ready" dataprefix plots_prefix

    ## type parameters (problem definition and discretization choices), as
    ## savename tag => (config dict key, name abbreviation)
    typeparams = (
        vtype   = ("velocitytype",   "Velocity" => "V"),
        dtype   = ("densitytype",    "Density" => "D"),
        etype   = ("eostype",        "Law" => ""),  # carries γ, e.g. Power{1.4}
        gtype   = ("gridtype",       ["UnstructuredUnitSquare" => "SquareUnstruct",
                                      "UniformUnitSquare" => "SquareUniform",
                                      "UnitSquare" => "Square"]),
        ctype   = ("convectiontype", "Convection" => "Conv"),
        uptype  = ("upwindtype",     "Upwind" => "Upw"),
        cortype = ("coriolistype",   "Coriolis" => "Cor"),
    )
    ## config dict key -> savename tag (for parameters with other names)
    tagmap = Dict("pressure_in_f" => "pif", "others_in_f" => "oif",
                  "target_residual" => "tres",
                  "velocitytype" => "vtype", "densitytype" => "dtype", "eostype" => "etype",
                  "gridtype" => "gtype", "convectiontype" => "ctype", "upwindtype" => "uptype",
                  "coriolistype" => "cortype")

    ## savename postprocessing: the (fixed-order) type tags appear as values
    ## only, i.e. "vtype=ConstantV" becomes "ConstantV"
    typekey_regex = r"(^|_)(vtype|dtype|etype|gtype|ctype|uptype|cortype)="
    strip_typekeys(sname) = replace(sname, typekey_regex => Base.SubstitutionString("\\1"))

    """
        essential(data; nrefs = true, drop = "") -> NamedTuple

    Ordered parameters for `savename` (which keeps the order when `sort =
    false`); `drop` is the config dict key of a parameter to omit (the field
    swept by a plot). The type tags lose their key name in the final name via
    `strip_typekeys`, so they are recognized by their position and the fixed
    order of this NamedTuple.
    """
    function essential(data; nrefs = true, drop = "")
        droptag = isempty(drop) ? "" : get(tagmap, drop, drop)
        parts = Pair{Symbol, Any}[]
        ## problem definition and discretization choices
        for (tag, (key, abbrev)) in pairs(typeparams)
            reps = abbrev isa Pair ? (abbrev,) : abbrev
            push!(parts, tag => replace(string(data[key]), reps...))
        end
        push!(parts,
            :pif => data["pressure_in_f"], :oif => data["others_in_f"],
            :stab1 => data["stab1"], :stab2 => data["stab2"],
            :μ => data["μ"], :λ => data["λ"], :c => data["c"], :M => data["M"],
            :order => data["order"], :reconstruct => data["reconstruct"],
            :τfac => data["τfac"], :ufac => data["ufac"],
            :tres => data["target_residual"])
        nrefs && push!(parts, :nrefs => data["nrefs"])
        return NamedTuple(filter(p -> String(p.first) != droptag, parts))
    end

    """
        filename(data; prefix = dataprefix) -> String

    Build a DrWatson-compatible filename string for a given `data` dict.
    All key parameters are abbreviated (see `essential`) and the result is
    prefixed with the study's data path `<datadir>/projects/<studyname>/`.
    """
    function filename(data; prefix = dataprefix)
        sname = savename(essential(data); sort = false,
                         allowedtypes = (Real, String, SubString, Symbol,
                                         Tuple{Real, Real}))
        return prefix * strip_typekeys(sname)
    end

    """
        filename_plots(data; prefix = "", free_parameter = "")

    Build a filename for a plot using `savename`. The name freezes exactly the
    parameters that identify the study's data files (see `filename`), except
    the refinement level `nrefs` (a plot spans a range of levels, appended to
    the `prefix` by the plotting function) and, for parameter studies, the
    swept config field `free_parameter` (a key of the config dict, e.g.
    `"convectiontype"`; empty for the plain convergence history). The plot is
    placed in the study's plot subfolder (`convergence_history` or
    `parameter_studies_<fp>`, created on demand). Returns a path string ending
    in `.png`. As a side effect, writes a `.txt` file describing the
    parameters next to the (about to be saved) plot file.
    """
    function filename_plots(data; prefix = "", free_parameter = "")
        sname = strip_typekeys(savename(essential(data; nrefs = false, drop = free_parameter); sort = false,
                         allowedtypes = (Real, String, SubString, Symbol,
                                         Tuple{Real, Real})))
        folder = plots_prefix * plot_folder(free_parameter)
        mkpath(folder)
        sname = folder * "/" * sname * prefix * ".png"
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
