# Benchmark problems

Manufactured solutions are selected by `velocitytype` (with matching density, grid and
in/outflow regions dispatched by `inflow_regions`/`outflow_regions`/`wall_regions`).

## Velocities

| `velocitytype` | dim | description |
| --- | --- | --- |
| `ZeroVelocity` | 2,3 | trivial solution (test of boundary data / stabilization) |
| `ConstantVelocity` | 2,3 | constant ``\mathbf{u} = (1/M, 0)``, exact fixed point at machine precision |
| `LinearVelocity` | 2 | linear velocity field |
| `P7VortexVelocity` | 2 | p7-vortex ``\xi = x^7 y^7 / 7^7`` type embedded vortex |
| `RigidBodyRotation` | 2,3 | rotating flow with ``\varrho\mathbf{u}`` from a stream function |
| `P11VortexVelocity` | 3 | embedded 3D vortex with vanishing trace on the cube boundary |
| `EthierSteinmanVelocity` | 3 | Ethier--Steinman-type Stokes solution (Ethier & Steinman, IJNMF 1994), nonzero Dirichlet data on all cube faces |
| `TaylorGreenVelocity` | 3 | Taylor--Green vortex array on the unit cube, slip walls |

The flux construction is ``\varrho \mathbf{u} = \mathrm{curl}\,\xi`` (2D stream function,
embedded z-invariantly in 3D), ``\varrho \mathbf{u} = \mathrm{curl}\,\Psi`` (3D vector stream
function, see `vector_streamfunction`) or a direct (compressive) prescription via `direct_velocity`.

## Densities

| `densitytype` | ``\varrho`` |
| --- | --- |
| `ConstantDensity` | ``M`` |
| `LinearDensity` | ``{(1 + (x - 1/2)/c)}/{M}`` |
| `ExponentialDensity` | ``M \exp(-y^3/(3c))`` |
| `ExponentialDensityRBR` | ``M \exp((x^2+y^2)/(2c))`` |
| `ExponentialZDensity` | ``M \exp((1/2 - z)/c)`` (3D isothermal atmosphere) |

## Grids

`Mountain2D`, `UnstructuredUnitSquare`, `UniformUnitSquare` (2D) and
`UnitCube`, `UnstructuredUnitCube` (3D), all refined via `nrefs` through `grid`.

## Example scripts

Each example in `scripts/` configures one study (problem data + defaults + plotting)
and provides `run_study()` / `run_comparison()`:

| script | example |
| --- | --- |
| `p7vortex_powerlaw.jl` | p7 vortex with `PowerLaw` EOS (2D) |
| `rigidbodyrotation.jl` | rigid body rotation (2D) |
| `rigidbodyrotation_3d.jl` | rigid body rotation, z-invariant on the cube (3D) |
| `ethiersteinman_3d.jl` | Ethier--Steinman-type vortex (genuinely 3D) |
| `p11vortex_3d.jl` | embedded P11 vortex (3D) |
| `taylor_green_3d.jl` | Taylor--Green vortex (3D) |
