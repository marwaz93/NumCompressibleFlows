module NumCompressibleFlows

using ExtendableFEM
using ExtendableGrids
using Triangulate
using SimplexGridFactory
using GridVisualize
using Symbolics
using LinearAlgebra
#using Test #hide
using DrWatson

# global Symbolics variables for definition of exact solutions
@variables x y z t


include("kernels.jl")
export stab_kernel!, kernel_continuity!, kernel_upwind!, exact_error!, standard_gravity!, energy_kernel!, eos_powerlaw! # these functions  change the input data


include("compressible_stokes.jl")
export load_testcase_data, filename, run_single
export TestProblem, ExponentialDensity, LinearDensity
export GridFamily, Mountain2D, UnstructuredUnitSquare, UniformUnitSquare
export grid
export prepare_data


end # module NumCompressibleFlows
