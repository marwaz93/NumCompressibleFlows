# Problem and scheme

## Model problem

On a domain ``\Omega \subset \mathbb{R}^d``, ``d \in \{2,3\}``, find velocity
``\mathbf{u}``, density ``\varrho`` and pressure ``p`` with

```math
\begin{aligned}
  \mathrm{div}(\varrho \mathbf{u}) &= f_\varrho && \text{in } \Omega,\\
  -\mu\,\Delta \mathbf{u} - \lambda\,\nabla \mathrm{div}\, \mathbf{u}
    + \varrho (\mathbf{u} \cdot \nabla)\mathbf{u}
    + \varrho\, \mathbf{f}_{\mathrm{cor}} + \nabla p &= \mathbf{f} && \text{in } \Omega,\\
  p &= c\, \varrho^\gamma && \text{(equation of state)},
\end{aligned}
```

with Dirichlet velocity data on inflow/wall boundaries and upwinded outflow, and with

- ``\mu, \lambda``: viscosity coefficients, ``c``: speed-of-sound-type parameter, ``M``: reference density,
- equation of state `IdealGasLaw` (``\gamma = 1``) or `PowerLaw{γ}` (via `gamma(eostype)`),
- convection variants `NoConvection`, `StandardConvection`, `OseenConvection`,
  `RotationForm` (Bernoulli form), `KarperConvection`, `NewConvection`,
- optional Coriolis term `BetaPlaneApproximation{β0}` with ``f_{cor} = \beta_0 y\, \mathbf{e}_x^\perp``.

The problems are manufactured: an exact solution ``(\mathbf{u}, \varrho)`` is prescribed
(typically ``\varrho \mathbf{u} = \mathrm{curl}\, \xi`` or ``\mathrm{curl}\, \Psi``, hence exactly
mass-conserving), and the force ``\mathbf{f}`` and density source ``f_\varrho`` are computed
symbolically by `prepare_data`. The pressure gradient either enters through the unknown
pressure (`pressure_in_f = false`) or is absorbed into ``\mathbf{f}`` (`pressure_in_f = true`),
in which case it is handled by the flux reconstruction below (gradient-robustness).

## Discretization

| `order` | velocity | pressure / density | flux reconstruction |
| --- | --- | --- | --- |
| 1 | Bernardi--Raugel (`H1BR`) | P0 (`L2P0`) | `RT0` / `BDM1` (`reconstruct = :RT/:BDM`) |
| 2 | P2 bubble (`H1P2B`) | P1 (`L2P1`) | `RT1` / `BDM2` |

With `reconstruct = :none` the identity/divergence operators are used directly
(no gradient-robustness). The continuity equation is discretized as an upwind FVM
(DG with jump stabilization `stab1`, mean-value stabilization `stab2`, upwinding via
`upwindtype = StandardUpwind/PointwiseUpwind`) and advanced by pseudo-time stepping
with pseudo-time step ``\tau = \mu / (c \cdot \mathrm{order}^2 \cdot M \cdot \mathtt{τfac} \cdot \mathtt{ufac})``,
while the momentum equation is solved exactly in each fixed-point iteration
(`subiterations_momentum = :auto`). An optional mass balance correction
(`mass_balance = true`) adds a uniform density source ``\lambda`` that acts as a discrete
Lagrange multiplier pinning the steady mass to ``M_{\mathrm{exact}}``.

`run_incompressible!` solves the incompressible reference problem (``c = \infty``,
``\varrho \equiv M``) for comparison of the low/high ``c`` regimes.
