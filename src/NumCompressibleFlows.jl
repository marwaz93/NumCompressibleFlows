module NumCompressibleFlows

using ExtendableFEM
using ExtendableFEMBase
using ExtendableGrids
using Triangulate
using SimplexGridFactory
using GridVisualize
using Symbolics: Symbolics, @variables, build_function
using LinearAlgebra
#using Test #hide
using UnicodePlots
using Term
using Plots
using LaTeXStrings
using Latexify
using Colors
using ColorTypes

# global Symbolics variables for definition of exact solutions
@variables x y z t

include("problem_definitions.jl")
export TestVelocity, P7VortexVelocity, ZeroVelocity, ConstantVelocity, RigidBodyRotation
export TestDensity, ConstantDensity, ExponentialDensity, LinearDensity, ExponentialDensityRBR
export EOSType, IdealGasLaw, PowerLaw, gamma
export ConvectionType, NoConvection, StandardConvection, OseenConvection, RotationForm, KarperConvection, NewConvection
export UpwindType, StandardUpwind, PointwiseUpwind
export CoriolisType, NoCoriolis, BetaPlaneApproximation
export GridFamily, Mountain2D, UnitSquare, UnstructuredUnitSquare, UniformUnitSquare
export inflow_regions, outflow_regions
export grid
export prepare_data, run_single
export default_args, load_data, run_incompressible!


include("utilities.jl")

include("kernels.jl")
export stab_kernel!
export kernel_continuity!
export kernel_upwind!, kernel_upwind2!, kernel_upwind_convection!, kernel_upwind_newconvection!
export exact_error!, exact_error_incompressible!
export standard_gravity!
export energy_kernel!
export density_jump_stab_kernel!, velocity_jump_stab_kernel!
export eos!
export kernel_standardconvection_linearoperator!, kernel_standardconvection_incompressible_linearoperator!
export kernel_rotationform_linearoperator!, kernel_new_rotationform_linearoperator!
export kernel_oseenconvection!
export kernel_coriolis_linearoperator!
export kernel_inflow!
export kernel_outflow!
export multiply_h_bilinear!, multiply_h_linear!
export stokes_kernel!
export div_projection!


include("compressible_stokes.jl")

include("postprocessing.jl")
export compute_errors!

include("plotting.jl")
export CONV_QUANTITIES, DEFAULT_QUANTITIES
export setup_pipeline!, log_ticks, conv_value
export plot_single, plot_convergencehistory
export plot_parameter_study_viscosity, plot_parameter_study_gamma, plot_parameter_study_mach_number,
    plot_parameter_study_mach_viscosity, plot_parameter_study_stab1, plot_parameter_study_stab2,
    plot_parameter_study_stab1_reconstruction, plot_parameter_study_alpha_reconstruction

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
