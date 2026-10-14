#!/usr/bin/env julia
# Command line entry point for the HTCondor jobs (see create_parameters.sh).
# Arguments are key=value pairs, e.g.
#
#   julia --project=. scripts/run.jl mode=solve example=p7vortex_powerlaw \
#       convectiontype=OseenConvection mu=0.01 "stab1=(0.9,0)" "stab2=(1.5,0)" nrefs=9
#
#   julia --project=. scripts/run.jl mode=plot example=p7vortex_powerlaw \
#       convectiontypes=NoConvection+OseenConvection mu=0.01 "stab1=(0.9,0)" "stab2=(1.5,0)" nrefs=1:9
#
# mode=solve computes (or loads) one refinement level of one configuration and
# its errors, mode=plot draws the convergence history of every given
# convection type and a comparison plot of all of them from existing data
# files (configurations with missing levels are skipped, never recomputed).
# All parameters not given here are the defaults of default_args in
# src/compressible_stokes.jl.

## no display on the compute servers
ENV["GKSwstype"] = "100"

using NumCompressibleFlows
using Plots

# ============================
# examples: study script and fixed problem data (as in the run_study defaults
# of the corresponding script)
# ============================

const EXAMPLES = Dict(
    "p7vortex_powerlaw" => (
        velocitytype = P7VortexVelocity,
        densitytype = ExponentialDensity,
        eostype = PowerLaw{1.4},
        gridtype = UnstructuredUnitSquare,
    ),
    "rigidbodyrotation" => (
        velocitytype = RigidBodyRotation,
        densitytype = ExponentialDensityRBR,
        eostype = IdealGasLaw,
        gridtype = UnstructuredUnitSquare,
    ),
)

const CONVECTIONTYPES = Dict(
    "NoConvection" => NoConvection,
    "StandardConvection" => StandardConvection,
    "OseenConvection" => OseenConvection,
    "RotationForm" => RotationForm,
    "KarperConvection" => KarperConvection,
    "NewConvection" => NewConvection,
)

# ============================
# parsers
# ============================

## integers stay integers, so that e.g. mu=1 gives the same filenames as μ = 1 in the scripts
parse_number(s) = something(tryparse(Int, s), parse(Float64, s))

parse_tuple(s) = Tuple(parse_number(strip(v)) for v in split(strip(s, ['(', ')']), ","))

function parse_nrefs(s)
    occursin(":", s) || return parse(Int, s)
    a, b = split(s, ":")
    return parse(Int, a):parse(Int, b)
end

function parse_convectiontype(s)
    haskey(CONVECTIONTYPES, s) || error("unknown convectiontype $s, choose from $(sort(collect(keys(CONVECTIONTYPES))))")
    return CONVECTIONTYPES[s]
end

function parse_arguments(args)
    raw = Dict{String, String}()
    for arg in args
        key, value = split(arg, "="; limit = 2)
        raw[key] = value
    end
    get_arg(key) = haskey(raw, key) ? raw[key] : error("missing parameter: $key")

    mode = get_arg("mode")
    mode in ("solve", "plot") || error("mode must be solve or plot")
    example = get_arg("example")
    haskey(EXAMPLES, example) || error("unknown example $example, choose from $(sort(collect(keys(EXAMPLES))))")

    params = (;
        mode,
        example,
        nrefs = parse_nrefs(get_arg("nrefs")),
        convectiontypes = mode == "solve" ? [parse_convectiontype(get_arg("convectiontype"))] :
                          parse_convectiontype.(split(get_arg("convectiontypes"), "+")),
        μ = parse_number(get_arg("mu")),
        stab1 = parse_tuple(get_arg("stab1")),
        stab2 = parse_tuple(get_arg("stab2")),
    )
    return params
end

# ============================
# main
# ============================

function main(params, study)
    config(convectiontype) = (; EXAMPLES[params.example]..., convectiontype,
                                μ = params.μ, stab1 = params.stab1, stab2 = params.stab2)

    if params.mode == "solve"
        data = load_data(; config(only(params.convectiontypes))..., nrefs = params.nrefs)
        @info "solving" data
        ## loads instead of recomputing if the data file already exists
        study.produce(data)
        study.compute_errors(data; compare_incompressible = false)
    else
        complete = filter(params.convectiontypes) do convectiontype
            missing = [lvl for lvl in params.nrefs
                       if !isfile(study.filename(load_data(; config(convectiontype)..., nrefs = lvl)) * ".jld2")]
            isempty(missing) || @warn "skipping $convectiontype, missing levels $missing"
            return isempty(missing)
        end
        for convectiontype in complete
            plot_convergencehistory(; nrefs = params.nrefs, config(convectiontype)...)
        end
        if length(complete) > 1
            plot_parameter_study(; param = :convectiontype, values = complete,
                                   nrefs = params.nrefs, config(first(complete))...)
        end
    end
    return nothing
end

params = parse_arguments(ARGS)
@show params

## sets up the data/plots folders and filenames of the study, defines `study`
include(joinpath(@__DIR__, params.example * ".jl"))

main(params, study)
