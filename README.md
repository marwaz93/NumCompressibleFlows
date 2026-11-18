# NumCompressibleFlows.jl

[![Documentation](https://github.com/marwaz93/NumCompressibleFlows/actions/workflows/Documentation.yml/badge.svg)](https://marwaz93.github.io/NumCompressibleFlows/dev/)

Finite element discretizations of stationary compressible flow model problems
with manufactured solutions, built on the
[ExtendableFEM](https://github.com/WIAS-PDELib/ExtendableFEM.jl) /
[ExtendableGrids](https://github.com/WIAS-PDELib/ExtendableGrids.jl) ecosystem.

The package discretizes a steady compressible Stokes/Oseen/Navier--Stokes model

```math
\mathrm{div}(\varrho \mathbf{u}) = f_\varrho, \qquad
-\mu\,\Delta \mathbf{u} - \lambda\,\nabla \mathrm{div}\,\mathbf{u}
+ \varrho(\mathbf{u}\cdot\nabla)\mathbf{u} + \nabla p = \mathbf{f}, \qquad
p = c\,\varrho^\gamma
```

with Bernardi--Raugel velocity and P0 pressure/density (order 1; P2b/P1 for
order 2), optional RT/BDM flux reconstruction for gradient-robustness, an
upwind FVM continuity equation and a fixed-point/pseudo-time solver. Exact
solutions (vortices, rigid body rotation, Ethier--Steinman, Taylor--Green, ...)
are manufactured so that convergence and robustness studies can be run
end-to-end.

## Installation

The package is not registered. From the Julia prompt:

```
] add https://github.com/marwaz93/NumCompressibleFlows
```

## Quick start

```julia
using NumCompressibleFlows
data = load_data(velocitytype = RigidBodyRotation, densitytype = ExponentialDensityRBR,
                 eostype = PowerLaw{1.4}, gridtype = UnstructuredUnitSquare, nrefs = 3)
run_single(data)
compute_errors!(data)
data["Error(L2,u)"]
```

Convergence studies are configured in standalone scripts, e.g.

```julia
include("scripts/rigidbodyrotation.jl")
run_study(nrefs = 2:5)        # data -> data/rigidbodyrotation, plots -> plots/rigidbodyrotation
run_comparison()              # one curve per convectiontype
```

## Documentation

The [documentation](https://marwaz93.github.io/NumCompressibleFlows/dev/) covers
the model problem and scheme, the benchmark problems, and the solver and study
pipeline. Build it locally with

```
julia --project=docs docs/make.jl
```
