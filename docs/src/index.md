# NumCompressibleFlows.jl

Research code for finite element discretizations of stationary compressible flow
model problems with manufactured solutions, built on the
[ExtendableFEM](https://wias-pdelib.github.io/ExtendableFEM.jl) /
[ExtendableGrids](https://wias-pdelib.github.io/ExtendableGrids.jl) ecosystem.

It provides
- Bernardi--Raugel/P0 (order 1) and P2b/P1 (order 2) velocity/density discretizations
  with optional Raviart--Thomas / BDM flux reconstruction for gradient-robustness,
- a fixed-point / pseudo-time solver for the coupled momentum + continuity system,
- a library of manufactured benchmark problems (vortices, rigid body rotation,
  Ethier--Steinman, Taylor--Green, ...),
- a DrWatson-based study pipeline for convergence histories and parameter studies.

## Installation

The package is not registered. From the Julia prompt:

```
] add https://github.com/marwaz93/NumCompressibleFlows
# or with a local checkout
] develop /path/to/NumCompressibleFlows
```

## Quick start

```julia
using NumCompressibleFlows
data = load_data(velocitytype = RigidBodyRotation, densitytype = ExponentialDensityRBR,
                 eostype = PowerLaw{1.4}, gridtype = UnstructuredUnitSquare, nrefs = 3)
data = run_single(data)
compute_errors!(data)
data["Error(L2,u)"]
```

```@contents
Depth = 2
```
