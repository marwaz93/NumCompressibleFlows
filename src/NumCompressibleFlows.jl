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
export TestVelocity, P7VortexVelocity, ZeroVelocity, ConstantVelocity, RigidBodyRotation
export TestDensity, ExponentialDensity, LinearDensity, ExponentialDensityRBR
export EOSType, IdealGasLaw, PowerLaw
export ConvectionType, NoConvection, StandardConvection, OseenConvection, RotationForm
export CoriolisType, NoCoriolis, BetaPlaneApproximation
export GridFamily, Mountain2D, UnitSquare, UnstructuredUnitSquare, UniformUnitSquare
export inflow_regions, outflow_regions
export grid
export prepare_data, filename, run_single


include("kernels.jl")
export stab_kernel!
export kernel_continuity!
export kernel_upwind!
export exact_error!
export standard_gravity!
export energy_kernel!
export eos!
export kernel_standardconvection_linearoperator!
export kernel_rotationform_linearoperator!
export kernel_oseenconvection!
export kernel_coriolis_linearoperator!
export kernel_inflow!
export kernel_outflow!


#include("compressible_stokes.jl")
#export load_testcase_data, filename, run_single



end # module NumCompressibleFlows
