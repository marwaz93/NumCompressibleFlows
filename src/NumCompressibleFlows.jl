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

include("problem_definitions.jl")
export TestVelocity, P7VortexVelocity, ZeroVelocity
export TestDensity, ExponentialDensity, LinearDensity
export EOSType, IdealGasLaw, PowerLaw
export GridFamily, Mountain2D, UnstructuredUnitSquare, UniformUnitSquare
export grid
export prepare_data, filename, run_single


include("kernels.jl")
export stab_kernel!, kernel_continuity!, kernel_upwind!, exact_error!, standard_gravity!, energy_kernel!, eos!, kernel_convection_linearoperator! # these functions  change the input data


#include("compressible_stokes.jl")
#export load_testcase_data, filename, run_single



end # module NumCompressibleFlows
