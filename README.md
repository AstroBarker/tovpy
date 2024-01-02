# tovpy
A Python based Tolman–Oppenheimer–Volkoff (TOV) solver for the structure of a simple neutron star.
Can be used to produce mass radius plots.
Supports both polytropic equation of state as well as tabulated nuclear equations of state.
For nuclear equations of state, use tables from [StellarCollapse](https://stellarcollapse.org/microphysics.html)

With the nuclear equation of state, we assume a cold neutron star (minimum temperature of the table) and beta equilibrium.

Includes forward Eular and RK4 temporal integration.

## Dependencies
 - numpy
 - matplotlib
 - astropy
 - numba
