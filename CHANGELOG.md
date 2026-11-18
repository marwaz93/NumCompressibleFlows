# Changelog

All notable changes to this project will be documented in this file.

This project adheres to the format of [Keep a Changelog](https://keepachangelog.com/)
with ISO 8601 dates. The changelog starts from the state of the `main` branch at
`59626a5` (merge of PR #6); changes on feature branches are collected under
`[Unreleased]` and moved into a version section when the branch is merged.

## [0.2.0] - 2026-11-18

Changes of branch `extend-3d` on top of `main` (`59626a5`), extending the solver
from 2D to 3D.

### Added
  - 3D support: tetrahedral unit cube grid, dimension-generic kernels
  - 3D manufactured solutions: `TaylorGreenVelocity`, `EthierSteinmanVelocity`, `ExponentialZDensity`
  - mass balance option for inflow/outflow boundary conditions: uniform source `lambda` pins the steady mass to `M_exact`

### Changed
  - adjusted h-exponents of jump stabilization terms in 3D
  - adjusted quadrature order for correct face-flux integrals in 3D


