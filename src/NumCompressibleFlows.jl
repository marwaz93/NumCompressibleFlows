module NumCompressibleFlows

using ExtendableFEM
using ExtendableGrids
using Triangulate
using SimplexGridFactory
using GridVisualize
using Symbolics
using LinearAlgebra
#using Test #hide

include("kernels.jl")
export stab_kernel!, kernel_continuity!, kernel_upwind!, exact_error!, standard_gravity!, energy_kernel! # these functions  change the input data


include("compressible_stokes.jl")
export load_testcase_data


end # module NumCompressibleFlows
