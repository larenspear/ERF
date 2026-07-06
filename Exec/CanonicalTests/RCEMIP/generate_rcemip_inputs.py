#!/usr/bin/env python3
"""
Generate ERF input files for the RCEMIP case (Wing et al. 2018, GMD 11, 793-813).

Produces, for each SST (295 / 300 / 305 K):
  - input_sounding_rcemip_sst<SST> : analytic sounding (z, theta, qv, u, v)
    in ERF input_sounding format, from the RCEMIP analytic expressions
    (Eqs. 1-6 of Wing et al. 2018)
and prints inputs-file snippets that depend on the vertical grid:
  - erf.terrain_z_levels : staggered z levels (RCEMIP-style stretched grid,
    dz ~50 m at the surface, <=100 m below 3 km, growing to 500 m aloft,
    model top at 33 km)
  - amr.n_cell z-count that matches those levels
  - erf.o3vmr : the RCEMIP ozone profile evaluated at every cell center
    (Eq. 1 of Wing et al. 2018: o3 = g1 * p^g2 * exp(-p/g3) [ppmv], p in hPa)
  - erf.fixed_solar_zenith_angle = cos(42.05 deg)

Run:  python3 generate_rcemip_inputs.py [--plot]
"""

import math
import argparse

# ---------------------------------------------------------------------------
# RCEMIP protocol constants (Wing et al. 2018, Tables 1-3)
# ---------------------------------------------------------------------------
GRAV  = 9.79764        # gravity used by the RCEMIP analytic sounding [m/s2]
RD    = 287.04         # dry-air gas constant used by RCEMIP [J/kg/K]
P0    = 1014.8e2       # surface pressure [Pa]
GAMMA = 0.0067         # virtual temperature lapse rate [K/m]
ZT    = 15000.0        # tropopause height [m]
ZQ1   = 4000.0         # humidity decay height 1 [m]
ZQ2   = 7500.0         # humidity decay height 2 [m]
QT    = 1.0e-14        # specific humidity above z_t [kg/kg]
Q0    = {295: 12.00e-3, 300: 18.65e-3, 305: 24.00e-3}  # surface q [kg/kg]

# Ozone profile fit parameters (o3 in ppmv, p in hPa)
O3_G1 = 3.6478
O3_G2 = 0.83209
O3_G3 = 11.3515

# Insolation
SOLAR_CONSTANT = 551.58          # [W/m2]
ZENITH_DEG     = 42.05           # solar zenith angle [deg]

# ERF constants (Source/ERF_Constants.H) -- used ONLY to convert the analytic
# T(z), p(z) into the potential temperature ERF expects in the sounding file,
# so that ERF's EOS reproduces the intended temperature profile.
ERF_RD   = 287.0
ERF_CP   = 1004.5
ERF_PREF = 1.0e5

ZTOP = 33000.0  # model top [m]


# ---------------------------------------------------------------------------
# Analytic profiles (Wing et al. 2018, Eqs. 2-6)
# ---------------------------------------------------------------------------
def q_v(z, sst):
    """Specific humidity [kg/kg]."""
    q0 = Q0[sst]
    if z > ZT:
        return QT
    return q0 * math.exp(-z / ZQ1) * math.exp(-((z / ZQ2) ** 2))


def T_virtual(z, sst):
    """Virtual temperature [K]."""
    Tv0 = sst * (1.0 + 0.608 * Q0[sst])
    if z > ZT:
        return Tv0 - GAMMA * ZT
    return Tv0 - GAMMA * z


def pressure(z, sst):
    """Pressure [Pa] from the analytic hydrostatic solution."""
    Tv0 = sst * (1.0 + 0.608 * Q0[sst])
    Tvt = Tv0 - GAMMA * ZT
    if z <= ZT:
        return P0 * ((Tv0 - GAMMA * z) / Tv0) ** (GRAV / (RD * GAMMA))
    pt = P0 * (Tvt / Tv0) ** (GRAV / (RD * GAMMA))
    return pt * math.exp(-GRAV * (z - ZT) / (RD * Tvt))


def temperature(z, sst):
    """Absolute temperature [K]."""
    return T_virtual(z, sst) / (1.0 + 0.608 * q_v(z, sst))


def theta(z, sst):
    """Potential temperature [K] using ERF's constants and 1000-hPa reference."""
    return temperature(z, sst) * (ERF_PREF / pressure(z, sst)) ** (ERF_RD / ERF_CP)


def o3_vmr(p_pa):
    """RCEMIP ozone volume mixing ratio [mol/mol] at pressure p [Pa]."""
    p_hpa = p_pa / 100.0
    return O3_G1 * (p_hpa ** O3_G2) * math.exp(-p_hpa / O3_G3) * 1.0e-6


def qsat_surface(sst):
    """Saturation specific humidity [kg/kg] at (SST, P0), Bolton formula."""
    tc = sst - 273.15
    es = 611.2 * math.exp(17.67 * tc / (tc + 243.5))  # [Pa]
    return 0.622 * es / (P0 - 0.378 * es)


# ---------------------------------------------------------------------------
# Vertical grid: dz = 50 m at the surface, stretching (4.5 %/level) to 100 m,
# constant 100 m below 3 km, stretching (8 %/level) to 500 m, constant 500 m
# up to the model top; the constant-dz top section is rescaled so the last
# level lands exactly on ZTOP.
# ---------------------------------------------------------------------------
def build_z_levels():
    z = [0.0]
    dz = 50.0
    while True:
        znew = z[-1] + dz
        if dz >= 500.0 - 1.0e-9:
            break
        z.append(znew)
        if znew < 3000.0:
            dz = min(dz * 1.045, 100.0)
        else:
            dz = min(dz * 1.08, 500.0)
    # fill the rest with a constant dz that divides the remaining depth evenly
    ntop = round((ZTOP - z[-1]) / 500.0)
    dz_top = (ZTOP - z[-1]) / ntop
    for _ in range(ntop):
        z.append(z[-1] + dz_top)
    z[-1] = ZTOP
    return z


# The recommended CRM grid of Wing et al. (2018), Table 3: 74 scalar levels.
OFFICIAL_CENTERS = [37.0, 112.0, 194.0, 288.0, 395.0, 520.0, 667.0, 843.0,
                    1062.0, 1331.0, 1664.0, 2055.0, 2505.0, 3000.0] \
                   + [float(z) for z in range(3500, 33001, 500)]


def build_official_z_levels():
    """Staggered faces whose cell centers are exactly the Table 3 levels
    (midpoint recursion: face[k+1] = 2*center[k] - face[k]); top face 33250 m."""
    faces = [0.0]
    for c in OFFICIAL_CENTERS:
        faces.append(2.0 * c - faces[-1])
    return faces


def write_sounding(sst, fname):
    q0_gkg = Q0[sst] * 1.0e3
    th0 = sst * (ERF_PREF / P0) ** (ERF_RD / ERF_CP)
    lines = [f"{P0/100.0:12.4f} {th0:12.6f} {q0_gkg:14.8e}"]
    zs = [10.0] + [50.0 * k for k in range(1, int(34000.0 / 50.0) + 1)]
    for z in zs:
        th = theta(z, sst)
        qg = q_v(z, sst) * 1.0e3
        lines.append(f"{z:10.1f} {th:12.6f} {qg:14.8e} {0.0:6.1f} {0.0:6.1f}")
    with open(fname, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {fname}  ({len(zs)} levels, theta_sfc = {th0:.4f} K)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true",
                        help="plot the analytic profiles (requires matplotlib)")
    parser.add_argument("--official-grid", action="store_true",
                        help="use the 74-level grid of Wing et al. (2018) Table 3 "
                             "(top face at 33250 m) instead of the finer default")
    args = parser.parse_args()

    for sst in (295, 300, 305):
        write_sounding(sst, f"input_sounding_rcemip_sst{sst}")

    zlev = build_official_z_levels() if args.official_grid else build_z_levels()
    nz = len(zlev) - 1
    zmid = [0.5 * (zlev[k] + zlev[k + 1]) for k in range(nz)]

    print(f"\n# ----- vertical grid: {nz} cells, top at {zlev[-1]:.0f} m -----")
    if args.official_grid:
        print(f"# (geometry.prob_hi z must be {zlev[-1]:.0f})")
    print(f"amr.n_cell (z) = {nz}")
    print("erf.terrain_z_levels = " + " ".join(f"{z:.3f}" for z in zlev))

    # Ozone on the cell centers of that grid (bottom-up), SST = 300 K pressures
    o3 = [o3_vmr(pressure(z, 300)) for z in zmid]
    print(f"\n# ----- RCEMIP ozone profile on the {nz} cell centers (bottom-up) -----")
    print("erf.o3vmr = " + " ".join(f"{v:.6e}" for v in o3))

    mu0 = math.cos(math.radians(ZENITH_DEG))
    print(f"\n# ----- insolation -----")
    print(f"erf.fixed_total_solar_irradiance = {SOLAR_CONSTANT}")
    print(f"erf.fixed_solar_zenith_angle     = {mu0:.7f}   # cos({ZENITH_DEG} deg)")
    print(f"# -> TOA insolation = {SOLAR_CONSTANT*mu0:.2f} W/m2 (RCEMIP: 409.6)")

    print(f"\n# ----- surface reference values -----")
    for sst in (295, 300, 305):
        print(f"# SST {sst} K: qsat(SST,p0) = {qsat_surface(sst)*1e3:.3f} g/kg, "
              f"theta_sfc = {sst*(ERF_PREF/P0)**(ERF_RD/ERF_CP):.4f} K")

    if args.plot:
        import matplotlib.pyplot as plt
        zs = [25.0 * k for k in range(0, 1341)]
        fig, axs = plt.subplots(1, 4, figsize=(14, 6), sharey=True)
        for sst in (295, 300, 305):
            axs[0].plot([temperature(z, sst) for z in zs], zs, label=f"SST {sst}")
            axs[1].plot([theta(z, sst) for z in zs], zs)
            axs[2].plot([q_v(z, sst) * 1e3 for z in zs], zs)
            axs[3].plot([o3_vmr(pressure(z, sst)) * 1e6 for z in zs], zs)
        axs[0].set_xlabel("T [K]"); axs[1].set_xlabel("theta [K]")
        axs[2].set_xlabel("qv [g/kg]"); axs[3].set_xlabel("O3 [ppmv]")
        axs[0].set_ylabel("z [m]"); axs[0].legend()
        for ax in axs: ax.grid(alpha=0.3)
        fig.suptitle("RCEMIP analytic initial profiles (Wing et al. 2018)")
        fig.tight_layout()
        fig.savefig("rcemip_profiles.png", dpi=150)
        print("\nwrote rcemip_profiles.png")


if __name__ == "__main__":
    main()
