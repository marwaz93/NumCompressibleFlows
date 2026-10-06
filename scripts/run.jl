#!/usr/bin/env julia

# --- Project setup ---
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

# --- Load your codebase ---
# This should be the file that defines:
#   plot_convergencehistory
#   P7VortexVelocity, LinearDensity, UnstructuredUnitSquare, PowerLaw, etc.
include("stationary_examples.jl")

# --- Read parameters (env vars with defaults) ---
#γ    = parse(Float64, ARGS[1])
#μ    = parse(Float64, get(ENV, "MU", "0.1"))
#τfac = parse(Float64, get(ENV, "TAUFAC", "1.0"))
#c    = parse(Int,     get(ENV, "C", "2"))

# nrefs = get(ENV, "NREFS", "1:2")
# nrefs = eval(Meta.parse(nrefs))   # safe here since you control input

# stab1 = get(ENV, "STAB1", "(0.1,1)")
# stab1 = eval(Meta.parse(stab1))
# function parse_arguments()
#     params = Dict{String, String}()
    
#     for arg in ARGS
#         key, value = split(arg, "=", limit=2)
#         params[key] = value
#     end

#     return params
# end

# ============================
# Type-specific parsers
# ============================

parse_parameter(::Type{Float64}, s::String) =
    parse(Float64, s)

parse_parameter(::Type{Bool}, s::String) =
    parse(Bool, s)

parse_parameter(::Type{Tuple{Float64,Float64}}, s::String) = begin
    s = strip(s, ['(', ')'])
    a, b = split(s, ",")
    (parse(Float64, strip(a)), parse(Float64, strip(b)))
end

parse_parameter(::Type{UnitRange{Int64}}, s::String) = begin
    a, b = split(s, ":")
    parse(Int, a):parse(Int, b)
end

parse_parameter(::Type{TestVelocity}, s::String) = begin
    velocity_types = Dict(
        "ZeroVelocity"       => ZeroVelocity,
        "ConstantVelocity"   => ConstantVelocity,
        "LinearVelocity"     => LinearVelocity,
        "P7VortexVelocity"   => P7VortexVelocity,
        "RigidBodyRotation"  => RigidBodyRotation
    )

    if !haskey(velocity_types, s)
        error("Unknown velocity type: $s")
    end

    velocity_types[s]
end

# ============================
# Parameter parser
# ============================

function parse_parameters()

    raw = Dict{String,String}()

    for arg in ARGS
        key, value = split(arg, "=", limit=2)
        raw[key] = value
    end

    parameter_types = Dict(
        "gamma"   => Float64,
        "mu"      => Float64,
        "c"       => Float64,
        "M"       => Float64,
        "tau_fac" => Float64,
        "stab1"  => Tuple{Float64,Float64},
        "nrefs" => UnitRange{Int64},
        "reconstruct"  => Bool,
        "velocitytype" => TestVelocity
    )

    params = Dict{String,Any}()

    for (name, T) in parameter_types

        if !haskey(raw, name)
            error("Missing parameter: $name")
        end

        params[name] = parse_parameter(T, raw[name])
    end

    return params
end

function main()

    params = parse_parameters()
    γ    = params["gamma"]
    μ    = params["mu"]
    c    = params["c"] 
    M = params["M"]
    τfac   = params["tau_fac"]
    stab1     = params["stab1"]
    nrefs    = params["nrefs"]
    velocitytype  = params["velocitytype"]
    #densitytype = params["densitytype"])
    #gridtype = params["gridtype"])
    #eostype = params["eostype"])
    reconstruct = params["reconstruct"]

   @show γ, typeof(γ)
   @show μ, typeof(μ)
   @show c, typeof(c) 
   @show M, typeof(M)
   @show τfac, typeof(τfac)
   @show stab1, typeof(stab1)
   @show nrefs, typeof(nrefs)
   @show reconstruct, typeof(reconstruct)
   @show velocitytype, typeof(velocitytype)

   # --- Run your function ---
    plot_convergencehistory(
        Plotter      = Plots,
        velocitytype = velocitytype,
        densitytype  = LinearDensity,
        gridtype     = UnstructuredUnitSquare,
        eostype      = PowerLaw{γ},
        c            = c,
        γ            = γ,
        μ            = μ,
        M            = M,
        τfac         = τfac,
        nrefs        = nrefs,
        stab1        = stab1,
        reconstruct  = reconstruct
    )
end

main()
