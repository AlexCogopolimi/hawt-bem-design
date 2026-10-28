# HAWT Blade Design & Off-Design Analysis (BEM)

Aerodynamic design and performance analysis of a 3-bladed, 10 MW-class horizontal-axis wind turbine using **Blade Element Momentum (BEM) theory**, written from scratch in Python.

Rotor geometry and operating parameters are based on the **DTU 10 MW Reference Wind Turbine** (rotor diameter 178.3 m, FFA-W3 airfoil family, max rotor speed 9.6 rpm).

*Academic project – Politecnico di Milano, M.Sc. Energy Engineering (2025).*

---

## What the code does

### 1. Optimal blade design – `blade_design.py`
- Computes the optimal axial and tangential induction factors (*a*, *a'*) along the span, including wake rotation
- Selects at each radial station the angle of attack that maximises lift-to-drag ratio, interpolating airfoil polars across **thickness** (FFA-W3-360 / 301 / 241) and **Reynolds number**
- Sizes the chord distribution with **Prandtl tip-loss correction**, iterating on the local Reynolds number until convergence
- Smooths the resulting chord distribution with a polynomial fit

### 2. On-design performance – `on_design.py`
- Power coefficient computed with two independent approaches (momentum theory and blade element theory) as a cross-check
- Rated power, torque, thrust and thrust coefficient at design conditions (V₀ = 9 m/s, TSR = 7.5)

### 3. Off-design operation & power regulation – `off_design.py`
- Iterative BEM solver with under-relaxation for each wind speed
- High-induction correction of the thrust coefficient (**Wilson**; Glauert and Buhl also implemented)
- **Below rated**: variable-speed operation at constant tip-speed ratio
- **Above rated**: pitch angle found with a bracketing + bisection search to hold rated power
- Outputs rotor speed, power, Cp, torque, thrust and Cx versus wind speed

### Plots – `plot_results.py`
Chord, twist and induction-factor distributions along the span.

### Shared library – `functions.py`
Polar interpolation (angle of attack, Reynolds, airfoil thickness), induction-factor solvers, Cp/power/thrust integration.

---

## Results at design point (V₀ = 9 m/s, TSR = 7.5)

| Quantity | Value |
|---|---|
| Power coefficient Cp | **0.49** (momentum and blade-element theory agree) |
| Aerodynamic power | ≈ 5.0 MW |
| Rotor torque | ≈ 6.4 MN·m |
| Rotor thrust | ≈ 1.0 MN |

![Blade design](figures/blade_design.png)

---

## Repository structure

```
├── functions.py        # BEM library
├── blade_design.py     # chord & twist design
├── on_design.py        # performance at design point
├── off_design.py       # power curve & pitch regulation
├── plot_results.py     # blade geometry figures
├── corda2.txt          # designed chord distribution (c/R)
├── result_on_design.txt# design-point BEM solution along the span
├── *_polars_RE*        # XFOIL polars of FFA-W3 airfoils at different Re
└── figures/
```

## How to run

Requires Python 3 with `numpy`, `scipy`, `matplotlib`.

```bash
python blade_design.py   # blade geometry
python on_design.py      # design-point performance
python off_design.py     # off-design power curve
python plot_results.py   # figures
```

Data files must be in the same folder as the scripts.

---

## Notes

Airfoil polars were generated with **XFOIL** for Reynolds numbers from 0.25 to 15 × 10⁶.

This is academic project code, kept as it was when the project was delivered: it is written for clarity of the physics rather than as a reusable package. Comments are partly in Italian.
