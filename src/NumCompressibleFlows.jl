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
export TestDensity, ConstantDensity, ExponentialDensity, LinearDensity, ExponentialDensityRBR
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
export density_jump_stab_kernel!
export eos!
export kernel_standardconvection_linearoperator!
export kernel_rotationform_linearoperator!
export kernel_oseenconvection!
export kernel_coriolis_linearoperator!
export kernel_inflow!
export kernel_outflow!
export multiply_h_bilinear!, multiply_h_linear!
export stokes_kernel
export div_projection!


#include("compressible_stokes.jl")
#export load_testcase_data, filename, run_single

## problem: loading and saving grids leads to ElementGeometries -> DataType conversion (by DrWarson/JLD2?) which has to be reverted
## after loading (until this is fixed ina more elegant way)
function repair_grid!(xgrid::ExtendableGrid)
    xgrid[CellGeometries] = VectorOfConstants{ElementGeometries,Int}(xgrid.components[CellGeometries][1], num_cells(xgrid))
    xgrid[FaceGeometries] = VectorOfConstants{ElementGeometries,Int}(xgrid.components[FaceGeometries][1], length(xgrid.components[FaceGeometries]))
    xgrid[BFaceGeometries] = VectorOfConstants{ElementGeometries,Int}(xgrid.components[BFaceGeometries][1], length(xgrid.components[BFaceGeometries]))

    xgrid[UniqueCellGeometries] = Vector{ElementGeometries}([xgrid.components[CellGeometries][1]])
    xgrid[UniqueFaceGeometries] = Vector{ElementGeometries}([xgrid.components[FaceGeometries][1]])
    xgrid[UniqueBFaceGeometries] = Vector{ElementGeometries}([xgrid.components[BFaceGeometries][1]])
end
export repair_grid!

end # module NumCompressibleFlows
