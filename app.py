"""
Shell-and-Tube Heat Exchanger Design Studio
===========================================
Graduation-project tool: thermal + mechanical (geometry) sizing of shell-and-tube
heat exchangers following Kern's method, TEMA classification and BS 3274.

Run:   streamlit run app.py

Method implemented (same sequence as the project report)
  1. Duty            Q  = sum(m*Cp*dT) + sum(m*lambda)            [kW]
  2. Utility flow    m_u = Q / lambda   (steam)  |  Q / (Cp*dT)    [kg/s]
  3. LMTD (counter)  dTlm = (dT1 - dT2) / ln(dT1/dT2)
  4. Correction      Ft from R and S  (1 shell pass / 2,4.. tube passes, N shells)
  5. Area            A  = Q / (U * Ft * dTlm)
  6. Tubes           Nt = ceil( A / (pi * do * L) )
  7. Bundle dia.     Db = do * (Nt / K1)^(1/n1)       (Coulson & Richardson / Sinnott)
  8. Shell dia.      Ds = Db + bundle-to-shell clearance
  9. Check           5 < L/Ds < 10
 10. Baffles         B = Ds/3,  As = (pt - do) * B * Ds / pt,  pt = 1.25*do
"""

import copy
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# =============================================================================
# 1. CONSTANTS AND PROJECT CASES
# =============================================================================

# Bundle-diameter constants  Db = do*(Nt/K1)^(1/n1)   (pitch = 1.25*do)
# Source: Coulson & Richardson Vol.6 / Sinnott, Table 12.4
K1_N1 = {
    ("Triangular", 1): (0.319, 2.142),
    ("Triangular", 2): (0.249, 2.207),
    ("Triangular", 4): (0.175, 2.285),
    ("Triangular", 6): (0.0743, 2.499),
    ("Triangular", 8): (0.0365, 2.675),
    ("Square", 1): (0.215, 2.207),
    ("Square", 2): (0.156, 2.291),
    ("Square", 4): (0.158, 2.263),
    ("Square", 6): (0.0402, 2.617),
    ("Square", 8): (0.0331, 2.643),
}

HEADS = ["Floating head (split-ring)", "Floating head (pull-through)", "Fixed tube sheet"]
SHELL_PHASES = ["Sensible (no phase change)", "Condensing", "Boiling"]
UTILITIES = ["Steam (condensing)", "Water (sensible)"]

# Approximate bundle-to-shell clearance [mm] vs bundle diameter [mm]
# (digitised from Sinnott Fig. 12.10 -> verify against the chart for the final report;
#  the project presets use the clearances quoted in the report instead).
CLEARANCE_TABLE = {
    "Fixed tube sheet": ([100, 500, 1000, 1500], [10, 13, 18, 22]),
    "Floating head (split-ring)": ([100, 250, 500, 1000, 1500], [52, 56, 62, 70, 78]),
    "Floating head (pull-through)": ([100, 500, 1000, 1500], [88, 92, 98, 104]),
}

STEEL_DENSITY = 7850.0   # kg/m3
TUBESHEET_THK = 25.0     # mm (assumed)
HEAD_ALLOWANCE = 1.15    # +15 % for heads, flanges, baffles (rough)

DEFAULTS = dict(
    title="Custom exchanger",
    process_role="Cooled",            # Heated (process = cold) / Cooled (process = hot)
    items=[],                         # process duty items
    t_proc_in=100.0, t_proc_out=40.0,
    utility_type="Water (sensible)",
    t_util_in=23.0, t_util_out=35.0,  # steam: t_util_in = saturation temperature
    util_cp=4.18, util_lambda=2153.0,
    U=300.0, do_mm=20.0, t_wall_mm=2.0, L_m=2.44,
    pitch="Triangular", passes=2, shells=1,
    tube_side="Utility", shell_phase=SHELL_PHASES[0], head=HEADS[0],
    clearance_mm=None,                # None -> automatic from CLEARANCE_TABLE
    baffle_div=3.0, baffle_cut_pct=25.0,
    p_design_bar=10.0, stress_nmm2=110.0, round_to_passes=False,
    notes="",
)


def _item(name, m, cp, dT=None, latent=0.0):
    """One line of process duty: Q = m*cp*dT + m*latent."""
    return dict(name=name, m=m, cp=cp, dT=dT, latent=latent)


def _case(**kw):
    d = copy.deepcopy(DEFAULTS)
    d.update(kw)
    return d


PRESETS = {
    "E-1": _case(
        title="E-1  Feed preheater (LP steam) - before C-1",
        process_role="Heated",
        items=[
            _item("Cyclohexane (sensible 70-81 C)", 0.007005, 2.237, 11),
            _item("Cyclohexane (sensible 81-117 C)", 0.007005, 1.635, 36),
            _item("Cyclohexane (latent at 81 C)", 0.007005, 0.0, 0, 355),
            _item("Limonene", 0.422045, 2.18),
            _item("alpha-Pinene", 0.019, 2.07),
        ],
        t_proc_in=70, t_proc_out=117,
        utility_type="Steam (condensing)", t_util_in=160, t_util_out=160, util_lambda=2153,
        U=300, do_mm=16, L_m=1.83, pitch="Square", passes=1,
        tube_side="Process", shell_phase="Condensing", clearance_mm=53,
        notes=("Report says steam at 140 C / 6.8 bar, but its dT1 = 90 and dT2 = 43 (LMTD 63.685) "
               "correspond to a 160 C steam temperature, so 160 C is used here. Saturated steam at "
               "6.8 bar is actually about 164 C. The report's shell diameter (314 mm) was built from "
               "Db = 261 mm; the K1/n1 equation gives Db = 145 mm, so Ds is about 198 mm here."),
    ),
    "E-2": _case(
        title="E-2  H2O / H2O2 feed preheater (hot water)",
        process_role="Heated",
        items=[_item("Water", 0.12026, 4.18), _item("Hydrogen peroxide", 0.122, 2.84)],
        t_proc_in=37, t_proc_out=60,
        utility_type="Water (sensible)", t_util_in=98, t_util_out=67, util_cp=4.19,
        U=800, do_mm=20, L_m=1.83, pitch="Square", passes=2,
        tube_side="Process", shell_phase=SHELL_PHASES[0], clearance_mm=52,
        notes=("Hot-water temperatures are not stated in the report; 98 -> 67 C was back-calculated "
               "from its dT1 = 38 and dT2 = 30. The report's R = 0.8 / P = 0.39 and its hot-water flow "
               "(0.3107 vs 0.3388 kg/s) are mutually inconsistent. Its L/Ds = 11.65 is marked as "
               "passing, but it is above 10."),
    ),
    "E-3": _case(
        title="E-3  C-1 bottoms cooler (178 -> 40 C)",
        process_role="Cooled",
        items=[
            _item("Limonene", 0.4094, 2.28),
            _item("alpha-Pinene (sensible 178-155 C)", 0.00836, 1.94, 23),
            _item("alpha-Pinene (sensible 155-40 C)", 0.00836, 1.65, 115),
            _item("alpha-Pinene (latent)", 0.00836, 0.0, 0, 272.9),
        ],
        t_proc_in=178, t_proc_out=40,
        utility_type="Water (sensible)", t_util_in=23, t_util_out=35, util_cp=4.18,
        U=300, do_mm=20, L_m=2.44, pitch="Triangular", passes=2,
        tube_side="Utility", clearance_mm=55,
        notes="Consistent with the report (the 'circle pattern' K1 = 0.249, n1 = 2.207 is the triangular, 2-pass set).",
    ),
    "E-4": _case(
        title="E-4  H2O2 / H2O cooler (68.5 -> 40 C)",
        process_role="Cooled",
        items=[_item("Hydrogen peroxide", 0.1213, 2.84), _item("Water", 0.0331, 4.18)],
        t_proc_in=68.5, t_proc_out=40,
        utility_type="Water (sensible)", t_util_in=23, t_util_out=30, util_cp=4.18,
        U=700, do_mm=20, L_m=1.83, pitch="Square", passes=2,
        tube_side="Utility", clearance_mm=52,
        notes=("The report takes Ft for 2 tube passes but uses the 1-pass square constants (K1 = 0.215). "
               "Here the 2-pass square set (0.156 / 2.291) is used consistently, so Db is slightly larger. "
               "The report also quotes L = 1.93 m, then uses 1.83 m."),
    ),
    "E-5": _case(
        title="E-5  Recycle cooler (limonene + limonene oxide, 182 -> 40 C)",
        process_role="Cooled",
        items=[_item("Limonene", 0.218, 2.288), _item("Limonene oxide", 0.077, 1.783)],
        t_proc_in=182, t_proc_out=40,
        utility_type="Water (sensible)", t_util_in=23, t_util_out=35, util_cp=4.18,
        U=400, do_mm=16, L_m=2.44, pitch="Triangular", passes=2,
        tube_side="Utility", clearance_mm=55,
        notes=("The report evaluates Db with Nt = 55 (copied from E-3) instead of 35, giving 184.6 mm; "
               "the correct value is about 150 mm. The limonene-oxide flow is 0.07 in its table but 0.077 in its calculation."),
    ),
    "E-6": _case(
        title="E-6  Limonene-oxide product cooler (197.5 -> 40 C)",
        process_role="Cooled",
        items=[_item("Limonene oxide", 0.45164, 1.783)],
        t_proc_in=197.5, t_proc_out=40,
        utility_type="Water (sensible)", t_util_in=23, t_util_out=35, util_cp=4.18,
        U=350, do_mm=20, L_m=2.44, pitch="Triangular", passes=2,
        tube_side="Utility", clearance_mm=56,
        notes=("The report prints dT1 = 147.5 C, but its LMTD of 64.47 C corresponds to dT1 = 162.5 C "
               "(197.5 - 35). The shell diameter in its summary table (2260.23 mm) is a typo for 260.23 mm."),
    ),
}


# =============================================================================
# 2. CALCULATION ENGINE
# =============================================================================

def lmtd_counter(dt1: float, dt2: float) -> float:
    """Log-mean temperature difference for counter-current flow."""
    if abs(dt1 - dt2) < 1e-9:
        return dt1
    return (dt1 - dt2) / math.log(dt1 / dt2)


def correction_factor(R: float, S: float, tube_passes: int, shells: int) -> float:
    """
    LMTD correction factor Ft for N shell passes with 2,4,... tube passes per shell.
    R = (T1 - T2) / (t2 - t1),   S = (t2 - t1) / (T1 - t1)
    Ft = NTU(pure counter-current) / NTU(actual arrangement)
    Raises ValueError if the duty is thermodynamically infeasible for that arrangement.
    """
    if tube_passes == 1 or R < 1e-6:        # pure counter-flow or isothermal shell side (steam)
        return 1.0
    if abs(R - 1.0) < 1e-6:                 # avoid the R = 1 singularity
        R = 1.0 + 1e-6
    if S * R >= 1.0 or S >= 1.0:
        raise ValueError("R*S >= 1: temperature cross - use more shells in series or a different layout.")
    n = shells
    x = ((1 - R * S) / (1 - S)) ** (1.0 / n)
    s1 = (1 - x) / (R - x)                  # effectiveness of ONE shell
    root = math.sqrt(R * R + 1)
    a = 2.0 / s1 - 1.0 - R
    if a - root <= 0:
        raise ValueError("Ft not computable (temperature cross) - add shells in series.")
    ntu_counter = math.log((1 - S) / (1 - R * S)) / (R - 1.0)
    ntu_actual = (n / root) * math.log((a + root) / (a - root))
    return ntu_counter / ntu_actual


def bundle_clearance(db_mm: float, head: str) -> float:
    xs, ys = CLEARANCE_TABLE[head]
    return float(np.interp(db_mm, xs, ys))


def design(p: dict) -> dict:
    """Full preliminary design of one exchanger. All lengths in mm unless noted."""
    r = {"warnings": []}
    heated = p["process_role"] == "Heated"
    tp_in, tp_out = float(p["t_proc_in"]), float(p["t_proc_out"])
    if heated and tp_out <= tp_in:
        raise ValueError("Process is being heated: outlet temperature must exceed inlet temperature.")
    if (not heated) and tp_out >= tp_in:
        raise ValueError("Process is being cooled: outlet temperature must be below inlet temperature.")
    if not p["items"]:
        raise ValueError("Add at least one process stream in the duty table.")

    # ---- 1. Heat duty:  Q = sum(m*Cp*dT) + sum(m*lambda) ----------------------
    dt_overall = abs(tp_out - tp_in)
    q_rows, Q = [], 0.0
    for it in p["items"]:
        m, cp = float(it["m"]), float(it.get("cp") or 0.0)
        dT = dt_overall if it.get("dT") is None else float(it["dT"])
        lat = float(it.get("latent") or 0.0)
        q_s, q_l = m * cp * dT, m * lat
        Q += q_s + q_l
        q_rows.append(dict(Stream=it["name"], m_kg_s=m, Cp_kJ_kgK=cp, dT_C=dT,
                           latent_kJ_kg=lat, Q_sensible_kW=q_s, Q_latent_kW=q_l, Q_kW=q_s + q_l))
    if Q <= 0:
        raise ValueError("Total duty must be positive.")
    r["Q"], r["q_table"] = Q, pd.DataFrame(q_rows)

    # ---- 2. Utility side -----------------------------------------------------
    steam = p["utility_type"].startswith("Steam")
    tu_in = float(p["t_util_in"])
    tu_out = tu_in if steam else float(p["t_util_out"])
    if steam and not heated:
        raise ValueError("Steam can only be used to heat the process stream.")
    if heated and tu_in < tu_out:
        raise ValueError("Hot utility: inlet must be hotter than outlet.")
    if (not heated) and tu_out <= tu_in:
        raise ValueError("Cold utility: outlet must be warmer than inlet.")
    if steam:
        m_util = Q / float(p["util_lambda"])
    else:
        m_util = Q / (float(p["util_cp"]) * abs(tu_out - tu_in))
    r["m_util"] = m_util

    if heated:
        th_in, th_out, tc_in, tc_out = tu_in, tu_out, tp_in, tp_out
    else:
        th_in, th_out, tc_in, tc_out = tp_in, tp_out, tu_in, tu_out
    if th_in <= tc_out or th_out <= tc_in:
        raise ValueError("Temperature cross: hot-end or cold-end approach is <= 0 K for counter-current flow.")
    r.update(th_in=th_in, th_out=th_out, tc_in=tc_in, tc_out=tc_out)

    # ---- 3. LMTD and correction factor --------------------------------------
    dt1, dt2 = th_in - tc_out, th_out - tc_in
    lm = lmtd_counter(dt1, dt2)
    R = (th_in - th_out) / (tc_out - tc_in)
    S = (tc_out - tc_in) / (th_in - tc_in)
    Ft = correction_factor(R, S, int(p["passes"]), int(p["shells"]))
    r.update(dt1=dt1, dt2=dt2, lmtd=lm, R=R, S=S, Ft=Ft, lmtd_c=Ft * lm)
    if Ft < 0.75:
        r["warnings"].append(f"Ft = {Ft:.3f} < 0.75: the arrangement is inefficient - add shells in series.")
    elif Ft < 0.8:
        r["warnings"].append(f"Ft = {Ft:.3f} is marginal (< 0.8).")

    # ---- 4. Area and tube count ---------------------------------------------
    U = float(p["U"])
    A = Q * 1000.0 / (U * r["lmtd_c"])                    # m2
    do = float(p["do_mm"]) / 1000.0
    L = float(p["L_m"])
    a_tube = math.pi * do * L
    Nt = math.ceil(A / a_tube - 1e-9)
    if p["round_to_passes"] and p["passes"] > 1:
        Nt = int(math.ceil(Nt / p["passes"]) * p["passes"])
    r.update(U=U, A=A, a_tube=a_tube, Nt=Nt, A_installed=Nt * a_tube,
             do=p["do_mm"], L=L, di=p["do_mm"] - 2 * p["t_wall_mm"])

    # ---- 5. Bundle and shell diameter ---------------------------------------
    K1, n1 = K1_N1[(p["pitch"], int(p["passes"]))]
    Db = p["do_mm"] * (Nt / K1) ** (1.0 / n1)
    clr = p["clearance_mm"] if p["clearance_mm"] is not None else bundle_clearance(Db, p["head"])
    Ds = Db + clr
    ratio = L * 1000.0 / Ds
    r.update(K1=K1, n1=n1, Db=Db, clearance=clr, Ds=Ds, L_over_Ds=ratio,
             ratio_ok=5.0 < ratio < 10.0)
    if ratio >= 10:
        r["warnings"].append(f"L/Ds = {ratio:.2f} > 10: shell too slender - shorten tubes (L <= {10*Ds/1000:.2f} m) "
                             "or use more shells.")
    if ratio <= 5:
        r["warnings"].append(f"L/Ds = {ratio:.2f} < 5: shell too stubby - use longer tubes (L >= {5*Ds/1000:.2f} m).")

    # ---- 6. Baffles and shell-side flow area --------------------------------
    pt = 1.25 * p["do_mm"]
    B = Ds / p["baffle_div"]
    As = (pt - p["do_mm"]) * B * Ds / pt * 1e-6           # m2
    n_baffles = max(int(L * 1000.0 // B) - 1, 0)
    if B < max(0.2 * Ds, 50.0):
        r["warnings"].append("Baffle spacing is below the TEMA minimum (max(Ds/5, 50 mm)).")
    if B > Ds:
        r["warnings"].append("Baffle spacing exceeds the shell diameter - tube vibration risk.")
    phase = p["shell_phase"]
    if phase == "Condensing":
        orient = "Vertical baffle cut (condensate drains to the outlet)"
    elif phase == "Boiling":
        orient = "Vertical or horizontal (depends on the service)"
    else:
        orient = "Horizontal baffle cut (prevents stratification)"
    r.update(pt=pt, B=B, As=As, n_baffles=n_baffles, orientation=orient,
             cut_height=p["baffle_cut_pct"] / 100.0 * Ds)

    # ---- 7. Tube-side flow area and mean temperature --------------------------
    a_bore = math.pi * r["di"] ** 2 / 4.0                 # mm2
    per_pass = Nt / p["passes"]
    r.update(a_bore=a_bore, tubes_per_pass=per_pass, tube_flow_area=per_pass * a_bore * 1e-6)
    if p["tube_side"] == "Process":
        r["tube_mean_T"] = 0.5 * (tp_in + tp_out)
    else:
        r["tube_mean_T"] = 0.5 * (tu_in + tu_out)

    # ---- 8. Rough steel mass (for the trade-off chart) -----------------------
    f, Pd = p["stress_nmm2"], p["p_design_bar"] * 0.1     # N/mm2
    t_shell = max(Pd * Ds / (2 * f - Pd), 6.0)            # mm, thin-wall (Sinnott eq. 13.1), min 6 mm
    m_shell = math.pi * (Ds + t_shell) / 1000 * (t_shell / 1000) * L * STEEL_DENSITY
    m_tubes = Nt * math.pi * ((p["do_mm"] / 1000) ** 2 - (r["di"] / 1000) ** 2) / 4 * L * STEEL_DENSITY
    m_sheets = 2 * math.pi / 4 * (Ds / 1000) ** 2 * (TUBESHEET_THK / 1000) * STEEL_DENSITY
    r.update(t_shell=t_shell, mass=HEAD_ALLOWANCE * (m_shell + m_tubes + m_sheets))
    return r


def temperature_profile(r: dict, n: int = 101):
    """
    Ideal counter-current profiles along the length (constant U, constant Cp):
        dT(z)  = dT_cold_end * (dT_hot_end/dT_cold_end)^z
        Tc(z)  = Tc_in + (Tc_out - Tc_in) * (rho^z - 1)/(rho - 1),   z = x/L, rho = dT1/dT2
    Cold fluid enters at z = 0, hot fluid enters at z = 1.
    """
    z = np.linspace(0, 1, n)
    d0, d1 = r["dt2"], r["dt1"]            # approach at z=0 (cold in) and z=1 (cold out)
    rho = d1 / d0
    if abs(rho - 1) < 1e-9:
        frac = z
    else:
        frac = (rho ** z - 1) / (rho - 1)
    tc = r["tc_in"] + (r["tc_out"] - r["tc_in"]) * frac
    th = tc + d0 * rho ** z
    return z * r["L"], th, tc


def sweep_lengths(p: dict, lengths) -> pd.DataFrame:
    """Re-run the design for several tube lengths (trade-off analysis)."""
    rows = []
    for L in lengths:
        q = dict(p)
        q["L_m"] = L
        try:
            d = design(q)
            rows.append(dict(L=L, Nt=d["Nt"], Ds=d["Ds"], ratio=d["L_over_Ds"], mass=d["mass"],
                             A_inst=d["A_installed"]))
        except ValueError:
            continue
    return pd.DataFrame(rows)


def all_cases() -> pd.DataFrame:
    rows = []
    for k, c in PRESETS.items():
        d = design(c)
        rows.append(dict(Case=k, A=d["A"], Nt=d["Nt"], Ds=d["Ds"], mass=d["mass"], Q=d["Q"]))
    return pd.DataFrame(rows)


# =============================================================================
# 3. REPORT TABLE (used by Tab 1 and by the CSV export)
# =============================================================================

def fmt(x, d=3):
    return f"{x:,.{d}f}"


def build_report(p: dict, r: dict) -> pd.DataFrame:
    rows = []

    def add(sec, name, val, unit="-"):
        rows.append((sec, name, val, unit))

    S = "Process data"
    add(S, "Exchanger", p["title"])
    add(S, "Process stream role", p["process_role"])
    add(S, "Process inlet temperature", fmt(p["t_proc_in"], 1), "°C")
    add(S, "Process outlet temperature", fmt(p["t_proc_out"], 1), "°C")
    add(S, "Utility", p["utility_type"])
    add(S, "Utility inlet temperature", fmt(r["tc_in"] if p["process_role"] == "Cooled" else r["th_in"], 1), "°C")
    add(S, "Utility outlet temperature", fmt(r["tc_out"] if p["process_role"] == "Cooled" else r["th_out"], 1), "°C")
    add(S, "Utility mass flow rate", fmt(r["m_util"], 4), "kg/s")

    S = "Thermal design"
    add(S, "Total heat duty Q", fmt(r["Q"], 3), "kW")
    add(S, "ΔT1 (hot in − cold out)", fmt(r["dt1"], 2), "°C")
    add(S, "ΔT2 (hot out − cold in)", fmt(r["dt2"], 2), "°C")
    add(S, "LMTD (counter-current)", fmt(r["lmtd"], 3), "°C")
    add(S, "R", fmt(r["R"], 3))
    add(S, "S", fmt(r["S"], 4))
    add(S, "Correction factor Ft", fmt(r["Ft"], 3))
    add(S, "Corrected LMTD", fmt(r["lmtd_c"], 3), "°C")
    add(S, "Overall heat-transfer coefficient U (assumed)", fmt(r["U"], 0), "W/m²·K")
    add(S, "Required heat-transfer area A", fmt(r["A"], 3), "m²")

    S = "Tubes and shell"
    add(S, "Tube outside diameter do", fmt(r["do"], 1), "mm")
    add(S, "Tube inside diameter di", fmt(r["di"], 1), "mm")
    add(S, "Tube length L", fmt(r["L"], 2), "m")
    add(S, "Area of one tube", fmt(r["a_tube"], 4), "m²")
    add(S, "Number of tubes Nt", f"{r['Nt']}")
    add(S, "Installed area", fmt(r["A_installed"], 3), "m²")
    add(S, "Tube pitch type", p["pitch"])
    add(S, "Tube pitch pt (1.25·do)", fmt(r["pt"], 1), "mm")
    add(S, "Tube passes / shell passes", f"{p['passes']} / {p['shells']}")
    add(S, "K1 / n1", f"{r['K1']} / {r['n1']}")
    add(S, "Bundle diameter Db", fmt(r["Db"], 1), "mm")
    add(S, "Bundle-to-shell clearance", fmt(r["clearance"], 1), "mm")
    add(S, "Shell diameter Ds", fmt(r["Ds"], 1), "mm")
    add(S, "L/Ds", fmt(r["L_over_Ds"], 2), "PASS (5-10)" if r["ratio_ok"] else "FAIL (5-10)")
    add(S, "Head type", p["head"])

    S = "Baffles and flow areas"
    add(S, "Baffle spacing B", fmt(r["B"], 1), "mm")
    add(S, "Baffle cut", fmt(p["baffle_cut_pct"], 0), "% of Ds")
    add(S, "Number of baffles (approx.)", f"{r['n_baffles']}")
    add(S, "Baffle orientation", r["orientation"])
    add(S, "Shell cross-flow area As", fmt(r["As"], 5), "m²")
    add(S, "Tube cross-section (bore)", fmt(r["a_bore"], 2), "mm²")
    add(S, "Tube-side flow area", fmt(r["tube_flow_area"], 5), "m²")
    add(S, "Tube-side mean temperature", fmt(r["tube_mean_T"], 1), "°C")

    S = "Mechanical (rough)"
    add(S, "Shell wall thickness (design P, min 6 mm)", fmt(r["t_shell"], 1), "mm")
    add(S, "Approx. steel mass", fmt(r["mass"], 0), "kg")

    S = "Standards"
    add(S, "Design codes", "BS 3274 + TEMA (Class B/C/R per service)")
    return pd.DataFrame(rows, columns=["Section", "Parameter", "Value", "Unit"])


# =============================================================================
# 4. FIGURES
# =============================================================================

def fig_profile(r):
    x, th, tc = temperature_profile(r)
    f = go.Figure()
    f.add_trace(go.Scatter(x=x, y=th, name="Hot fluid", line=dict(color="#d62728", width=3)))
    f.add_trace(go.Scatter(x=x, y=tc, name="Cold fluid", line=dict(color="#1f77b4", width=3)))
    f.add_annotation(x=0, y=tc[0], text=f"Cold in {tc[0]:.1f} °C", showarrow=True, ax=40, ay=30)
    f.add_annotation(x=x[-1], y=tc[-1], text=f"Cold out {tc[-1]:.1f} °C", showarrow=True, ax=-50, ay=30)
    f.add_annotation(x=x[-1], y=th[-1], text=f"Hot in {th[-1]:.1f} °C", showarrow=True, ax=-50, ay=-30)
    f.add_annotation(x=0, y=th[0], text=f"Hot out {th[0]:.1f} °C", showarrow=True, ax=50, ay=-30)
    f.update_layout(title="Temperature profile along the exchanger (counter-current, ideal)",
                    xaxis_title="Distance along tube length (m)", yaxis_title="Temperature (°C)",
                    hovermode="x unified", legend=dict(orientation="h", y=1.1), height=450)
    return f


def fig_cases(current_title):
    df = all_cases()
    f = make_subplots(specs=[[{"secondary_y": True}]])
    f.add_trace(go.Bar(x=df["Case"], y=df["A"], name="Required area (m²)", marker_color="#1f77b4",
                       text=[f"{v:.2f}" for v in df["A"]], textposition="outside"), secondary_y=False)
    f.add_trace(go.Scatter(x=df["Case"], y=df["mass"], name="Steel mass (kg)", mode="lines+markers",
                           line=dict(color="#ff7f0e", width=3)), secondary_y=True)
    f.update_yaxes(title_text="Heat-transfer area (m²)", secondary_y=False)
    f.update_yaxes(title_text="Approx. steel mass (kg)", secondary_y=True)
    f.update_layout(title="Graph B1 - Area vs. equipment mass across project cases (E-1 ... E-6)",
                    legend=dict(orientation="h", y=1.12), height=430)
    return f


def fig_sweep(p, r):
    lengths = [1.83, 2.44, 3.05, 3.66, 4.27, 4.88, 6.10]      # 6, 8, 10, 12, 14, 16, 20 ft
    df = sweep_lengths(p, lengths)
    f = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                      specs=[[{"secondary_y": True}], [{}]],
                      subplot_titles=("Steel mass and shell diameter", "L/Ds (acceptable window 5-10)"))
    f.add_trace(go.Bar(x=df["L"], y=df["mass"], name="Steel mass (kg)", marker_color="#2ca02c"), row=1, col=1)
    f.add_trace(go.Scatter(x=df["L"], y=df["Ds"], name="Ds (mm)", mode="lines+markers",
                           line=dict(color="#9467bd", width=3)), row=1, col=1, secondary_y=True)
    f.add_trace(go.Scatter(x=df["L"], y=df["ratio"], name="L/Ds", mode="lines+markers",
                           line=dict(color="#d62728", width=3)), row=2, col=1)
    f.add_hrect(y0=5, y1=10, fillcolor="green", opacity=0.12, line_width=0, row=2, col=1)
    f.update_yaxes(title_text="Mass (kg)", row=1, col=1, secondary_y=False)
    f.update_yaxes(title_text="Ds (mm)", row=1, col=1, secondary_y=True)
    f.update_yaxes(title_text="L/Ds", row=2, col=1)
    f.update_xaxes(title_text="Tube length (m)", row=2, col=1)
    f.update_layout(title=f"Graph B2 - Tube-length trade-off for the selected case (A_req = {r['A']:.2f} m²)",
                    height=620, legend=dict(orientation="h", y=1.08))
    return f


def tube_positions(r, p):
    """Tube centres on a square / triangular pitch, closest-to-centre first (illustrative layout)."""
    pt, do = r["pt"], r["do"]
    lim = r["Db"] / 2 - do / 2
    n = int(r["Ds"] / pt) + 3
    while True:
        pts = []
        for j in range(-n, n + 1):
            for i in range(-n, n + 1):
                if p["pitch"] == "Square":
                    x, y = i * pt, j * pt
                else:
                    x, y = (i + 0.5 * (j % 2)) * pt, j * pt * math.sqrt(3) / 2
                if math.hypot(x, y) <= lim:
                    pts.append((x, y))
        if len(pts) >= r["Nt"] or lim > r["Ds"]:
            break
        lim *= 1.02
    pts.sort(key=lambda t: math.hypot(*t))
    return pts[: r["Nt"]]


def fig_cross_section(p, r):
    f = go.Figure()
    Rs, Rb, do = r["Ds"] / 2, r["Db"] / 2, r["do"]
    f.add_shape(type="circle", x0=-Rs, y0=-Rs, x1=Rs, y1=Rs, line=dict(color="#333", width=3))
    f.add_shape(type="circle", x0=-Rb, y0=-Rb, x1=Rb, y1=Rb, line=dict(color="#888", dash="dash"))
    pts = tube_positions(r, p)
    for x, y in pts:
        f.add_shape(type="circle", x0=x - do / 2, y0=y - do / 2, x1=x + do / 2, y1=y + do / 2,
                    line=dict(color="#1f77b4", width=1), fillcolor="rgba(31,119,180,0.25)")
    # baffle cut segment (removed part of the plate)
    c = p["baffle_cut_pct"] / 100.0 * r["Ds"]
    y0 = Rs - c
    ys = np.linspace(y0, Rs, 40)
    xs = np.sqrt(np.maximum(Rs ** 2 - ys ** 2, 0))
    horizontal = p["shell_phase"] != "Condensing"
    if horizontal:
        px, py = np.r_[xs, -xs[::-1]], np.r_[ys, ys[::-1]]
    else:
        px, py = np.r_[ys, ys[::-1]], np.r_[xs, -xs[::-1]]
    f.add_trace(go.Scatter(x=px, y=py, fill="toself", mode="lines", line=dict(color="crimson"),
                           fillcolor="rgba(220,20,60,0.18)", name=f"Baffle cut ({p['baffle_cut_pct']:.0f} % Ds)"))
    # pass partition lanes
    if p["passes"] >= 2:
        f.add_shape(type="line", x0=-Rb, x1=Rb, y0=0, y1=0, line=dict(color="black", width=2, dash="dot"))
    if p["passes"] >= 4:
        f.add_shape(type="line", x0=0, x1=0, y0=-Rb, y1=Rb, line=dict(color="black", width=2, dash="dot"))
    f.update_xaxes(visible=False, range=[-Rs * 1.1, Rs * 1.1])
    f.update_yaxes(visible=False, scaleanchor="x", range=[-Rs * 1.1, Rs * 1.1])
    f.update_layout(title=f"Cross-section: {p['pitch']} pitch, {len(pts)} of {r['Nt']} tubes drawn "
                          f"(Ds = {r['Ds']:.0f} mm, Db = {r['Db']:.0f} mm)",
                    height=520, legend=dict(orientation="h", y=-0.02), margin=dict(l=10, r=10, t=50, b=10))
    return f


def fig_side_view(p, r):
    L, Ds, Db = r["L"] * 1000.0, r["Ds"], r["Db"]
    ch = 0.6 * Ds                                    # channel / head length
    f = go.Figure()
    shp = f.add_shape

    shp(type="rect", x0=0, x1=L, y0=-Ds / 2, y1=Ds / 2, line=dict(color="#333", width=3),
        fillcolor="rgba(255,235,205,0.35)")                                           # shell
    shp(type="rect", x0=0, x1=L, y0=-Db / 2, y1=Db / 2, line=dict(color="#1f77b4", width=1),
        fillcolor="rgba(31,119,180,0.18)")                                            # tube bundle
    for y in np.linspace(-Db / 2 * 0.85, Db / 2 * 0.85, 6):
        shp(type="line", x0=0, x1=L, y0=y, y1=y, line=dict(color="#1f77b4", width=1))
    shp(type="rect", x0=-ch, x1=0, y0=-Ds / 2, y1=Ds / 2, line=dict(color="#333", width=3),
        fillcolor="rgba(200,200,200,0.4)")                                            # front channel
    shp(type="rect", x0=-0.02 * Ds, x1=0.02 * Ds, y0=-Ds / 2, y1=Ds / 2, fillcolor="#555")  # tubesheet
    if p["passes"] >= 2:
        shp(type="line", x0=-ch, x1=0, y0=0, y1=0, line=dict(color="black", width=3))        # pass partition

    if p["head"] == "Fixed tube sheet":
        shp(type="rect", x0=L, x1=L + ch, y0=-Ds / 2, y1=Ds / 2, line=dict(color="#333", width=3),
            fillcolor="rgba(200,200,200,0.4)")
        shp(type="rect", x0=L - 0.02 * Ds, x1=L + 0.02 * Ds, y0=-Ds / 2, y1=Ds / 2, fillcolor="#555")
        rear = "Rear channel (fixed)"
    else:
        shp(type="rect", x0=L - 0.12 * Ds, x1=L - 0.09 * Ds, y0=-Db / 2, y1=Db / 2, fillcolor="#555")  # floating tubesheet
        shp(type="rect", x0=L - 0.09 * Ds, x1=L, y0=-Db / 2, y1=Db / 2, line=dict(color="#c0392b", width=2),
            fillcolor="rgba(192,57,43,0.25)")                                                       # floating head cover
        shp(type="rect", x0=L, x1=L + 0.5 * Ds, y0=-Ds / 2, y1=Ds / 2, line=dict(color="#333", width=3),
            fillcolor="rgba(200,200,200,0.4)")                                                      # shell cover
        rear = "Floating head" + (" (split ring)" if "split" in p["head"] else " (pull-through)")

    # baffles (limit drawn plates to keep the picture readable)
    nb = min(r["n_baffles"], 24)
    gap = p["baffle_cut_pct"] / 100.0 * Ds
    horizontal = p["shell_phase"] != "Condensing"
    for k in range(nb):
        x = (k + 1) * L / (nb + 1)
        if horizontal:
            y0, y1 = (-Ds / 2, Ds / 2 - gap) if k % 2 == 0 else (-Ds / 2 + gap, Ds / 2)
            shp(type="rect", x0=x - 0.006 * L, x1=x + 0.006 * L, y0=y0, y1=y1, fillcolor="#c0392b", line_width=0)
        else:
            shp(type="rect", x0=x - 0.006 * L, x1=x + 0.006 * L, y0=-Ds / 2, y1=Ds / 2,
                fillcolor="rgba(192,57,43,0.45)", line=dict(color="#c0392b", width=1, dash="dot"))

    # nozzles
    nz = 0.18 * Ds
    shell_in_top = p["process_role"] == "Heated" and p["tube_side"] == "Process"
    for x, top in ((0.12 * L, True), (0.88 * L, False)):
        y0, y1 = (Ds / 2, Ds / 2 + nz) if top else (-Ds / 2 - nz, -Ds / 2)
        shp(type="rect", x0=x - 0.02 * L, x1=x + 0.02 * L, y0=y0, y1=y1, fillcolor="#777", line_width=1)
    shp(type="rect", x0=-ch * 0.7, x1=-ch * 0.3, y0=Ds / 2, y1=Ds / 2 + nz, fillcolor="#777", line_width=1)
    if p["passes"] == 1:
        shp(type="rect", x0=L + 0.5 * Ds * 0.3 if p["head"] != "Fixed tube sheet" else L + ch * 0.3,
            x1=L + 0.5 * Ds * 0.7 if p["head"] != "Fixed tube sheet" else L + ch * 0.7,
            y0=-Ds / 2 - nz, y1=-Ds / 2, fillcolor="#777", line_width=1)
    else:
        shp(type="rect", x0=-ch * 0.7, x1=-ch * 0.3, y0=-Ds / 2 - nz, y1=-Ds / 2, fillcolor="#777", line_width=1)

    shell_label = "Shell side: " + ("steam / utility" if p["tube_side"] == "Process" else "process fluid")
    tube_label = "Tube side: " + ("process fluid" if p["tube_side"] == "Process" else "utility")
    f.add_annotation(x=L / 2, y=Ds / 2 + nz * 1.8, text=shell_label + "  (counter-current to tube side)", showarrow=False)
    f.add_annotation(x=-ch / 2, y=-Ds / 2 - nz * 1.9, text=f"Front channel<br>{tube_label}", showarrow=False)
    f.add_annotation(x=L + (ch if p["head"] == "Fixed tube sheet" else 0.5 * Ds) / 2, y=Ds / 2 + nz * 1.8,
                     text=rear, showarrow=False)
    f.add_annotation(x=L / 2, y=-Ds / 2 - nz * 1.9,
                     text=f"{r['n_baffles']} baffles @ B = {r['B']:.0f} mm, {p['baffle_cut_pct']:.0f} % cut "
                          f"({'horizontal' if horizontal else 'vertical'})", showarrow=False)
    xmax = L + (ch if p["head"] == "Fixed tube sheet" else 0.5 * Ds)
    f.update_xaxes(visible=False, range=[-ch * 1.15, xmax * 1.03])
    f.update_yaxes(visible=False, scaleanchor="x", range=[-Ds * 0.95, Ds * 0.95])
    f.update_layout(title=f"Longitudinal view - {p['head']}, {p['passes']} tube pass(es)",
                    height=430, margin=dict(l=10, r=10, t=50, b=10), showlegend=False)
    return f


def ascii_layout(p, r):
    """Text schematic of the exchanger (monospace)."""
    W = 46
    nb = max(1, min(r["n_baffles"], 7))
    horizontal = p["shell_phase"] != "Condensing"
    rows = [[" "] * W for _ in range(3)]
    for k in range(nb):
        pos = int((k + 1) * W / (nb + 1))
        for rr in range(3):
            if horizontal:
                gap_row = 0 if k % 2 == 0 else 2
                rows[rr][pos] = " " if rr == gap_row else "│"
            else:
                rows[rr][pos] = "┆"
    rear = "FIXED " if p["head"] == "Fixed tube sheet" else "FLOAT "
    body = []
    for rr in range(3):
        line = "".join("═" if c == " " else c for c in rows[rr]) if True else ""
        left = "║ ▌" if rr != 1 or p["passes"] == 1 else "╠═▌"
        right = "▐ ║" if p["head"] == "Fixed tube sheet" else "▐▒▒)"
        body.append(f"{left}{line}{right}")
    cut = "horizontal" if horizontal else "vertical"
    head = "Fixed tube sheet" if p["head"] == "Fixed tube sheet" else p["head"]
    txt = f"""
 Shell IN ↓                                              Shell OUT ↓
 ╔{'═'*3}╦{'═'*W}╦{'═'*3 if p['head']=='Fixed tube sheet' else '═'*4}╗
 {body[0]}
 {body[1]}
 {body[2]}
 ╚{'═'*3}╩{'═'*W}╩{'═'*3 if p['head']=='Fixed tube sheet' else '═'*4}╝
  ↑ front channel / tube-side nozzles      {rear.strip().lower()} end

 Legend:  ═ tubes   │ / ┆ baffle plates   ▌▐ tube sheets   ▒ floating-head cover
 Head type      : {head}
 Tube layout    : {p['pitch']} pitch, pt = {r['pt']:.1f} mm, {r['Nt']} tubes of {r['do']:.0f} mm OD x {r['L']:.2f} m
 Tube passes    : {p['passes']}   (partition plate in the front channel{' - none for 1 pass' if p['passes']==1 else ''})
 Baffles        : {r['n_baffles']} plates, B = {r['B']:.0f} mm, {p['baffle_cut_pct']:.0f} % cut, {cut} orientation
 Shell / bundle : Ds = {r['Ds']:.0f} mm,  Db = {r['Db']:.0f} mm,  L/Ds = {r['L_over_Ds']:.2f}
 Flow           : counter-current
"""
    return txt


# =============================================================================
# 5. STREAMLIT UI
# =============================================================================

st.set_page_config(page_title="Shell & Tube Heat Exchanger Design", page_icon="🔥", layout="wide")
st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {background:#f6f8fb; border:1px solid #d9e1ec; border-radius:10px;
                                 padding:14px 16px;}
    div[data-testid="stMetricLabel"] p {font-weight:600; color:#44546a;}
    .block-container {padding-top:1.8rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

CASE_LABELS = {k: v["title"] for k, v in PRESETS.items()}

# ---------------------------------------------------------------- sidebar ----
sb = st.sidebar
sb.title("⚙️ Control Panel")
mode = sb.radio("Input source", ["Pre-configured case (E-1 … E-6)", "Custom inputs"])

if mode.startswith("Pre"):
    key = sb.selectbox("Project case", list(PRESETS), format_func=lambda k: CASE_LABELS[k])
    collected = copy.deepcopy(PRESETS[key])
    with sb.expander("Case inputs (from the report)"):
        c = PRESETS[key]
        st.write(f"**Process:** {c['process_role']}  {c['t_proc_in']} → {c['t_proc_out']} °C")
        st.write(f"**Utility:** {c['utility_type']}  {c['t_util_in']} → {c['t_util_out']} °C")
        st.write(f"**U** = {c['U']} W/m²·K, **do** = {c['do_mm']} mm, **L** = {c['L_m']} m")
        st.write(f"**Layout:** {c['pitch']} pitch, {c['passes']} pass(es), {c['head']}")
else:
    base = PRESETS["E-3"]
    collected = copy.deepcopy(DEFAULTS)
    collected["title"] = sb.text_input("Exchanger name", "Custom exchanger")

    sb.subheader("Process stream")
    collected["process_role"] = sb.radio("Process stream is being", ["Heated", "Cooled"], index=1, horizontal=True)
    c1, c2 = sb.columns(2)
    collected["t_proc_in"] = c1.number_input("T in (°C)", value=178.0, step=1.0)
    collected["t_proc_out"] = c2.number_input("T out (°C)", value=40.0, step=1.0)
    sb.caption("Duty table: leave ΔT empty to use T out − T in. Latent in kJ/kg.")
    init = pd.DataFrame([
        dict(Component=i["name"], m_kg_s=i["m"], Cp_kJ_kgK=i["cp"],
             dT_C=(np.nan if i["dT"] is None else float(i["dT"])), latent_kJ_kg=float(i["latent"]))
        for i in base["items"]
    ])
    edited = sb.data_editor(init, num_rows="dynamic", key="items_editor", hide_index=True)
    items = []
    for _, row in edited.iterrows():
        if pd.isna(row["m_kg_s"]) or pd.isna(row["Cp_kJ_kgK"]):
            continue
        items.append(_item(str(row["Component"]) if pd.notna(row["Component"]) else "Stream",
                           float(row["m_kg_s"]), float(row["Cp_kJ_kgK"]),
                           None if pd.isna(row["dT_C"]) else float(row["dT_C"]),
                           0.0 if pd.isna(row["latent_kJ_kg"]) else float(row["latent_kJ_kg"])))
    collected["items"] = items

    sb.subheader("Utility")
    collected["utility_type"] = sb.selectbox("Utility", UTILITIES, index=1)
    if collected["utility_type"].startswith("Steam"):
        collected["t_util_in"] = sb.number_input("Steam saturation T (°C)", value=160.0)
        collected["t_util_out"] = collected["t_util_in"]
        collected["util_lambda"] = sb.number_input("Latent heat of steam (kJ/kg)", value=2153.0)
    else:
        c1, c2 = sb.columns(2)
        collected["t_util_in"] = c1.number_input("Utility in (°C)", value=23.0)
        collected["t_util_out"] = c2.number_input("Utility out (°C)", value=35.0)
        collected["util_cp"] = sb.number_input("Utility Cp (kJ/kg·K)", value=4.18)

    sb.subheader("Geometry")
    collected["U"] = sb.number_input("Assumed U (W/m²·K)", value=300.0, min_value=10.0, step=10.0,
                                     help="Pick from a typical range (e.g. Holman) and iterate.")
    collected["do_mm"] = sb.number_input("Tube OD (mm)", value=20.0, min_value=10.0, max_value=50.0, step=1.0)
    collected["L_m"] = sb.number_input("Tube length (m)", value=2.44, min_value=0.5, max_value=7.0, step=0.01,
                                       help="6 ft = 1.83 m, 8 ft = 2.44 m, 10 ft = 3.05 m, 12 ft = 3.66 m")
    collected["pitch"] = sb.selectbox("Pitch type", ["Triangular", "Square"])
    c1, c2 = sb.columns(2)
    collected["passes"] = c1.selectbox("Tube passes", [1, 2, 4, 6, 8], index=1)
    collected["shells"] = c2.selectbox("Shells in series", [1, 2, 3], index=0)
    collected["head"] = sb.selectbox("Exchanger type", HEADS)
    collected["tube_side"] = sb.radio("Tube-side fluid", ["Process", "Utility"], index=1, horizontal=True)
    collected["shell_phase"] = sb.selectbox("Shell-side behaviour", SHELL_PHASES)

    with sb.expander("Advanced"):
        collected["t_wall_mm"] = st.number_input("Tube wall thickness (mm)", value=2.0, step=0.1)
        manual = st.checkbox("Enter bundle clearance manually", value=False)
        collected["clearance_mm"] = st.number_input("Clearance (mm)", value=55.0) if manual else None
        collected["baffle_div"] = st.number_input("Baffle spacing = Ds / x, x =", value=3.0, min_value=1.0, max_value=5.0)
        collected["baffle_cut_pct"] = st.number_input("Baffle cut (% of Ds)", value=25.0, min_value=15.0, max_value=45.0)
        collected["p_design_bar"] = st.number_input("Design pressure (bar g)", value=10.0)
        collected["stress_nmm2"] = st.number_input("Allowable stress (N/mm²)", value=110.0)
        collected["round_to_passes"] = st.checkbox("Round Nt up to a multiple of the pass count", value=False)

if "applied" not in st.session_state:
    st.session_state.applied = copy.deepcopy(PRESETS["E-1"])
if sb.button("🔄 Update Calculations", type="primary", use_container_width=True):
    st.session_state.applied = copy.deepcopy(collected)
if collected != st.session_state.applied:
    sb.info("Inputs changed - press **Update Calculations** to apply.")

p = st.session_state.applied

# ----------------------------------------------------------------- engine ----
st.title("🔥 Shell-and-Tube Heat Exchanger Design Studio")
st.caption("Kern's method · TEMA · BS 3274 · counter-current flow · preliminary thermal & geometric sizing")
st.subheader(p["title"])

try:
    res = design(p)
except (ValueError, ZeroDivisionError, KeyError) as e:
    st.error(f"Design could not be completed: {e}")
    st.stop()

report = build_report(p, res)
sb.download_button("⬇️ Export report (CSV)", data=report.to_csv(index=False).encode("utf-8-sig"),
                   file_name=f"{p['title'].split()[0]}_design_report.csv", mime="text/csv",
                   use_container_width=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "Thermal Design & Results (النتائج التصميمية)",
    "Mechanical & Geometry Design (التصميم الميكانيكي وأبعاد الأنابيب)",
    "Advanced Graphs & Visualizations (الرسومات البيانية)",
    "Equipment Schematic & Layout (تمثيل المعدات)",
])

# ------------------------------------------------------------------ TAB 1 ----
with tab1:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Heat Duty Q", f"{res['Q']:.2f} kW")
    m2.metric("Corrected LMTD", f"{res['lmtd_c']:.2f} °C", help=f"Ft × LMTD = {res['Ft']:.3f} × {res['lmtd']:.2f}")
    m3.metric("Overall U (assumed)", f"{res['U']:.0f} W/m²·K")
    m4.metric("Heat-Transfer Area A", f"{res['A']:.3f} m²")
    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Correction factor Ft", f"{res['Ft']:.3f}", help=f"R = {res['R']:.3f},  S = {res['S']:.4f}")
    m6.metric("Utility flow", f"{res['m_util']:.4f} kg/s")
    m7.metric("ΔT1 / ΔT2", f"{res['dt1']:.1f} / {res['dt2']:.1f} °C")
    m8.metric("Installed area", f"{res['A_installed']:.3f} m²",
              delta=f"{(res['A_installed']/res['A']-1)*100:.1f} % margin", delta_color="off")
    for w in res["warnings"]:
        st.warning(w)

    st.markdown("#### All input / output parameters")
    st.dataframe(report, hide_index=True, height=560)
    st.markdown("#### Heat-duty breakdown")
    qt = res["q_table"].rename(columns={
        "m_kg_s": "ṁ (kg/s)", "Cp_kJ_kgK": "Cp (kJ/kg·K)", "dT_C": "ΔT (°C)",
        "latent_kJ_kg": "λ (kJ/kg)", "Q_sensible_kW": "Q sensible (kW)",
        "Q_latent_kW": "Q latent (kW)", "Q_kW": "Q (kW)"})
    st.dataframe(qt.style.format({c: "{:.4f}" for c in qt.columns if c != "Stream"}), hide_index=True)
    if p.get("notes"):
        with st.expander("📌 Reconciliation notes vs. the project report"):
            st.write(p["notes"])

# ------------------------------------------------------------------ TAB 2 ----
with tab2:
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Number of tubes Nt", f"{res['Nt']}")
    g2.metric("Bundle diameter Db", f"{res['Db']:.1f} mm", help=f"Db = do·(Nt/K1)^(1/n1), K1 = {res['K1']}, n1 = {res['n1']}")
    g3.metric("Shell clearance", f"{res['clearance']:.1f} mm")
    g4.metric("Shell diameter Ds", f"{res['Ds']:.1f} mm")

    st.markdown("#### Length-to-diameter check (5 < L/Ds < 10)")
    msg = f"L/Ds = {res['L']*1000:.0f} / {res['Ds']:.1f} = **{res['L_over_Ds']:.2f}**"
    (st.success if res["ratio_ok"] else st.error)(("✅ PASS - " if res["ratio_ok"] else "❌ FAIL - ") + msg)

    st.markdown("#### Baffle design")
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Baffle spacing B = Ds/" + f"{p['baffle_div']:g}", f"{res['B']:.1f} mm")
    b2.metric("Shell cross-flow area As", f"{res['As']*1e4:.2f} cm²", help=f"{res['As']:.5f} m²")
    b3.metric("Baffle cut", f"{p['baffle_cut_pct']:.0f} % Ds", help=f"{res['cut_height']:.1f} mm")
    b4.metric("Number of baffles", f"{res['n_baffles']}")
    st.info(f"**Baffle orientation:** {res['orientation']}")
    st.latex(r"A_s=\frac{(p_t-d_o)\,B\,D_s}{p_t}\qquad p_t=1.25\,d_o")

    st.markdown("#### Tube side")
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Tube ID", f"{res['di']:.1f} mm")
    t2.metric("Tubes per pass", f"{res['tubes_per_pass']:.1f}")
    t3.metric("Tube-side flow area", f"{res['tube_flow_area']*1e4:.2f} cm²", help=f"{res['tube_flow_area']:.5f} m²")
    t4.metric("Mean tube-side T", f"{res['tube_mean_T']:.1f} °C")
    st.caption("Scope: this tool covers duty, area, tube count, bundle/shell sizing and flow areas "
               "(the report's scope). Film coefficients, pressure drops and the U-check come next.")

# ------------------------------------------------------------------ TAB 3 ----
with tab3:
    st.plotly_chart(fig_profile(res))
    st.caption("Ideal counter-current profile (constant U and Cp, Ft = 1). "
               "Multi-component latent steps (e.g. E-1 cyclohexane) are not resolved.")
    st.plotly_chart(fig_cases(p["title"]))
    st.plotly_chart(fig_sweep(p, res))
    st.caption("Mass = shell + tubes + 2 tube sheets, +15 % allowance for heads/flanges/baffles; "
               "indicative only, for comparing alternatives.")

# ------------------------------------------------------------------ TAB 4 ----
with tab4:
    st.plotly_chart(fig_side_view(p, res))
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.plotly_chart(fig_cross_section(p, res))
    with col_b:
        st.markdown("#### ASCII layout")
        st.code(ascii_layout(p, res), language=None)
