import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Arc, FancyArrowPatch
import matplotlib.patheffects as pe

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Analytical Solvers
# ─────────────────────────────────────────────────────────────────────────────

def _trapz(y, x):
    try:
        return np.trapezoid(y, x=x)
    except AttributeError:
        return np.trapz(y, x=x)


def calculate_unit_H(w_top, h_channel, theta_deg):
    """
    Numerically solves the Castigliano Strain Energy integral for a half-channel
    subjected to a unit Brazier load (q = 1 N/mm).
    """
    n = 500
    theta_rad = np.radians(theta_deg)

    x1 = np.linspace(0, w_top / 2, n)
    y1 = np.full(n, h_channel)

    x2 = np.linspace(w_top / 2, w_top / 2 + h_channel * np.tan(theta_rad), n)
    y2 = np.linspace(h_channel, 0, n)

    x = np.concatenate([x1, x2])
    y = np.concatenate([y1, y2])

    ds = np.zeros_like(x)
    ds[1:] = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    s = np.cumsum(ds)

    A = _trapz(np.ones_like(x), s)
    B = _trapz(y, s)
    C = _trapz(y ** 2, s)
    Dx = _trapz(x1, s[:n])
    Exy = _trapz(x1 * y1, s[:n])

    matrix = np.array([[A, -B], [-B, C]])
    constants = np.array([-Dx, Exy])
    solution = np.linalg.solve(matrix, constants)
    return solution[1]


def calculate_longitudinal_cmax(H, t, w1, w2, r_top_out, r_bot_out, theta_deg):
    """
    Calculates the neutral axis and c_max for a corrugated bipolar plate via
    1-D centre-line integration.
    """
    theta = np.radians(theta_deg)
    h = H - t
    R_cl_top = r_top_out - t / 2
    R_cl_bot = r_bot_out - t / 2
    phi = np.pi / 2 - theta

    dy_top = R_cl_top * (1 - np.cos(phi))
    dy_bot = R_cl_bot * (1 - np.cos(phi))

    L_flat_top = w1 - 2 * R_cl_top * np.tan(phi / 2)
    L_flat_bot = w2 - 2 * R_cl_bot * np.tan(phi / 2)
    L_arc_top = R_cl_top * phi
    L_arc_bot = R_cl_bot * phi
    L_web = (h - dy_top - dy_bot) / np.cos(theta)

    C_flat_top = h
    C_flat_bot = 0.0
    C_arc_top = h - R_cl_top + R_cl_top * np.sin(phi) / phi
    C_arc_bot = R_cl_bot - R_cl_bot * np.sin(phi) / phi
    C_web = (dy_bot + (h - dy_top)) / 2

    num = (L_flat_top * C_flat_top + L_flat_bot * C_flat_bot
           + 2 * L_arc_top * C_arc_top + 2 * L_arc_bot * C_arc_bot
           + 2 * L_web * C_web)
    den = L_flat_top + L_flat_bot + 2 * L_arc_top + 2 * L_arc_bot + 2 * L_web

    NA_cl = num / den
    NA_bot = NA_cl + t / 2
    c_max = max(NA_bot, H - NA_bot)
    return c_max, NA_bot


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Geometry Preview (matplotlib)
# ─────────────────────────────────────────────────────────────────────────────

def _fillet_arc_points(cx, cy, r, a_start_deg, a_end_deg, n=30):
    """Return (x, y) arrays for an arc centred at (cx,cy)."""
    angles = np.linspace(np.radians(a_start_deg), np.radians(a_end_deg), n)
    return cx + r * np.cos(angles), cy + r * np.sin(angles)


def build_longitudinal_profile(H, t, w1, w2, r_top_out, r_bot_out, theta_deg):
    theta = np.radians(theta_deg)
    phi = np.pi / 2 - theta
    r_t = r_top_out
    r_b = r_bot_out

    half_w1 = w1 / 2
    half_w2 = w2 / 2

    tan_half_phi = np.tan(phi / 2)
    arc_t_cx = half_w1 - r_t * tan_half_phi + r_t * np.sin(theta)
    arc_t_cy = H - r_t

    arc_b_cx = half_w2 - r_b * tan_half_phi + r_b * np.sin(theta)
    arc_b_cy = r_b

    web_tx = arc_t_cx - r_t * np.sin(theta)
    web_ty = arc_t_cy - r_t * np.cos(theta)

    web_bx = arc_b_cx - r_b * np.sin(theta)
    web_by = arc_b_cy + r_b * np.cos(theta)

    def one_side(sign):
        xs, ys = [], []
        xs.append(0.0);  ys.append(H)
        xs.append(sign * (half_w1 - r_t * tan_half_phi)); ys.append(H)
        if sign > 0:
            ax, ay = _fillet_arc_points(sign * arc_t_cx, arc_t_cy,
                                        r_t, 90, 90 - np.degrees(phi))
        else:
            ax, ay = _fillet_arc_points(sign * arc_t_cx, arc_t_cy,
                                        r_t, 90, 90 + np.degrees(phi))
        xs.extend(ax.tolist()); ys.extend(ay.tolist())
        xs.append(sign * web_bx); ys.append(web_by)
        bot_start = np.degrees(np.arctan2(web_by - arc_b_cy, sign * web_bx - sign * arc_b_cx))
        bot_end = 90.0
        if sign > 0:
            ax2, ay2 = _fillet_arc_points(sign * arc_b_cx, arc_b_cy,
                                          r_b, bot_start, bot_end)
        else:
            ax2, ay2 = _fillet_arc_points(sign * arc_b_cx, arc_b_cy,
                                          r_b, 180 - bot_start, 90)
        xs.extend(ax2.tolist()); ys.extend(ay2.tolist())
        xs.append(sign * half_w2); ys.append(0.0)
        return xs, ys

    xl, yl = one_side(-1)
    xr, yr = one_side(+1)

    xr_rev = list(reversed(xl))
    yr_rev = list(reversed(yl))
    full_x = xr_rev + xr
    full_y = yr_rev + yr
    return np.array(full_x), np.array(full_y)


def build_transverse_profile(w_top, h_channel, theta_deg, t):
    theta = np.radians(theta_deg)
    hw1 = w_top / 2
    offset = h_channel * np.tan(theta)

    def half_right():
        xs = [0, hw1, hw1 + offset]
        ys = [h_channel, h_channel, 0]
        return xs, ys

    hx, hy = half_right()
    xs = [-x for x in reversed(hx)] + hx
    ys = list(reversed(hy)) + hy
    return np.array(xs), np.array(ys)


def draw_geometry_preview(pattern_type, t,
                           H_total=None, w1=None, w2=None,
                           r_top=None, r_bot=None,
                           theta_deg=15.0,
                           h_channel=None, w_top=None,
                           NA_bot=None, c_max=None):
    DARK = "#1a1a2e"
    LIGHT_FILL = "#e8f4f8"
    PROFILE_COLOR = "#2563eb"
    NA_COLOR = "#dc2626"
    DIM_COLOR = "#6b7280"
    LABEL_COLOR = "#111827"
    TICK_COLOR = "#9ca3af"

    fig, ax = plt.subplots(figsize=(7, 4.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f9fafb")
    ax.tick_params(colors=TICK_COLOR, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#e5e7eb")
        spine.set_linewidth(0.6)

    def annotate_dim(x0, y0, x1, y1, label, orient="v", offset=0.05):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="<->", color=DIM_COLOR,
                                   lw=0.8, mutation_scale=8))
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        if orient == "v":
            ax.text(mx + offset, my, label, fontsize=7, color=DIM_COLOR,
                    va="center", ha="left")
        else:
            ax.text(mx, my + offset, label, fontsize=7, color=DIM_COLOR,
                    va="bottom", ha="center")

    if pattern_type == "Plain Foil":
        L = 3.0
        rect = plt.Rectangle((-L / 2, 0), L, t,
                              facecolor=LIGHT_FILL, edgecolor=PROFILE_COLOR, lw=1.5)
        ax.add_patch(rect)
        ax.set_xlim(-L / 2 - 0.3, L / 2 + 0.6)
        ax.set_ylim(-t * 1.5, t * 4)
        ax.axhline(t / 2, color=NA_COLOR, lw=1.2, ls="--", label="Neutral axis")
        annotate_dim(L / 2 + 0.1, 0, L / 2 + 0.1, t,
                     f"t = {t:.3f} mm", orient="v", offset=0.05)
        ax.text(0, t / 2, "Neutral axis  (centre)",
                fontsize=7.5, color=NA_COLOR, ha="center", va="bottom")
        ax.set_xlabel("Width (mm)", fontsize=8, color=DIM_COLOR)
        ax.set_ylabel("Height (mm)", fontsize=8, color=DIM_COLOR)
        ax.set_title("Cross-section — Plain foil", fontsize=9,
                     color=LABEL_COLOR, fontweight="bold", pad=8)

    elif pattern_type == "Longitudinal Channels":
        ox, oy = build_longitudinal_profile(H_total, t, w1, w2, r_top, r_bot, theta_deg)
        ax.fill(ox, oy, facecolor=LIGHT_FILL, edgecolor=PROFILE_COLOR, lw=1.5, zorder=2)
        xmin, xmax = ox.min(), ox.max()
        ax.set_xlim(xmin - 0.15, xmax + 0.55)
        ax.set_ylim(-0.05, H_total * 1.45)
        if NA_bot is not None:
            ax.axhline(NA_bot, color=NA_COLOR, lw=1.2, ls="--",
                       xmin=0.05, xmax=0.75, zorder=3)
            ax.text(xmin - 0.05, NA_bot, "NA", fontsize=7,
                    color=NA_COLOR, va="center", ha="right", zorder=4)
        if NA_bot is not None and c_max is not None:
            annotate_dim(xmax + 0.08, NA_bot, xmax + 0.08, NA_bot + c_max,
                         f"c_max\n{c_max:.3f} mm", orient="v", offset=0.04)
        annotate_dim(xmax + 0.35, 0, xmax + 0.35, H_total,
                     f"H = {H_total:.3f} mm", orient="v", offset=0.04)
        annotate_dim(-w1 / 2, H_total + 0.06, w1 / 2, H_total + 0.06,
                     f"w₁={w1:.2f}", orient="h", offset=0.04)
        ax.set_xlabel("Width (mm)", fontsize=8, color=DIM_COLOR)
        ax.set_ylabel("Height (mm)", fontsize=8, color=DIM_COLOR)
        ax.set_title("Cross-section — Longitudinal channels", fontsize=9,
                     color=LABEL_COLOR, fontweight="bold", pad=8)

    elif pattern_type == "Transverse Channels":
        px, py = build_transverse_profile(w_top, h_channel, theta_deg, t)
        ax.fill(px, py, facecolor=LIGHT_FILL, edgecolor=PROFILE_COLOR, lw=1.5, zorder=2)
        period = max(px) - min(px)
        for shift in [-period, period]:
            ax.plot(px + shift, py, color=PROFILE_COLOR, lw=1.0,
                    ls="-", alpha=0.35, zorder=1)
        xmin, xmax = px.min(), px.max()
        ax.set_xlim(xmin - period * 0.6, xmax + period * 0.6 + 0.5)
        ax.set_ylim(-h_channel * 0.25, h_channel * 1.7)
        annotate_dim(xmax + 0.08, 0, xmax + 0.08, h_channel,
                     f"h = {h_channel:.3f} mm", orient="v", offset=0.04)
        annotate_dim(-w_top / 2, h_channel + h_channel * 0.15,
                     w_top / 2, h_channel + h_channel * 0.15,
                     f"w₁={w_top:.2f} mm", orient="h", offset=h_channel * 0.06)
        theta_rad = np.radians(theta_deg)
        web_x0 = w_top / 2
        web_x1 = w_top / 2 + h_channel * np.tan(theta_rad)
        mid_wx = (web_x0 + web_x1) / 2 + 0.05
        mid_wy = h_channel / 2
        ax.text(mid_wx, mid_wy, f"θ={theta_deg:.0f}°",
                fontsize=7, color=DIM_COLOR, va="center", ha="left",
                rotation=-theta_deg)
        na_y = t / 2
        ax.axhline(na_y, color=NA_COLOR, lw=1.1, ls="--",
                   xmin=0.1, xmax=0.55, zorder=3)
        ax.text(xmin - period * 0.55, na_y,
                f"Local NA (flat, t/2={t/2:.3f})",
                fontsize=6.5, color=NA_COLOR, va="bottom", ha="left")
        ax.set_xlabel("Width (mm)", fontsize=8, color=DIM_COLOR)
        ax.set_ylabel("Height (mm)", fontsize=8, color=DIM_COLOR)
        ax.set_title("Cross-section — Transverse channels", fontsize=9,
                     color=LABEL_COLOR, fontweight="bold", pad=8)

    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, color="#e5e7eb", lw=0.4, zorder=0)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Stress Calculators (for "Given Diameter" mode)
# ─────────────────────────────────────────────────────────────────────────────

def compute_stress_components(pattern_type, D_roll,
                               E, nu, sigma_tension, t,
                               c_max=None,
                               w_top=None, h_channel=None, theta_deg=None):
    """
    Returns a dict with individual stress components and von Mises stress.
    All values in MPa.
    """
    R = D_roll / 2  # radius

    if pattern_type == "Plain Foil":
        sigma_bend = (E * t) / D_roll
        sigma_axial = sigma_tension
        sigma_transverse = 0.0
        # Von Mises for uniaxial + bending (same axis): sum directly
        sigma_vm = sigma_bend + sigma_axial
        return {
            "sigma_bend": sigma_bend,
            "sigma_axial": sigma_axial,
            "sigma_transverse": sigma_transverse,
            "sigma_total": sigma_vm,
            "sigma_vm": sigma_vm,
            "components": [
                ("Bending stress", sigma_bend,
                 r"\sigma_{bend} = \frac{E \cdot t}{D}"),
                ("Web tension", sigma_axial,
                 r"\sigma_{tension}"),
            ],
        }

    elif pattern_type == "Longitudinal Channels":
        sigma_bend = (E * 2 * c_max) / D_roll
        sigma_axial = sigma_tension
        sigma_transverse = 0.0
        sigma_vm = sigma_bend + sigma_axial
        return {
            "sigma_bend": sigma_bend,
            "sigma_axial": sigma_axial,
            "sigma_transverse": sigma_transverse,
            "sigma_total": sigma_vm,
            "sigma_vm": sigma_vm,
            "components": [
                ("Profile bending stress", sigma_bend,
                 r"\sigma_{bend} = \frac{E \cdot 2 \cdot c_{max}}{D}"),
                ("Web tension", sigma_axial,
                 r"\sigma_{tension}"),
            ],
        }

    elif pattern_type == "Transverse Channels":
        H_tilde = calculate_unit_H(w_top, h_channel, theta_deg)
        # Radial pressure from tension
        q = (sigma_tension * t) / R
        H_reaction = H_tilde * q  # N/mm

        sigma_bend_ps = (E / (1 - nu**2)) * (t / D_roll)   # plane-strain bending
        sigma_membrane = H_reaction / t                      # Brazier membrane
        sigma_axial = sigma_tension

        # Von Mises: principal stresses σ1 (longitudinal) and σ2 (transverse)
        # σ1 = bending + tension (longitudinal)
        # σ2 = membrane (transverse, compressive → negative contribution)
        sigma_1 = sigma_bend_ps + sigma_axial
        sigma_2 = -sigma_membrane  # compressive transverse stress
        sigma_vm = np.sqrt(sigma_1**2 - sigma_1 * sigma_2 + sigma_2**2)

        return {
            "sigma_bend": sigma_bend_ps,
            "sigma_axial": sigma_axial,
            "sigma_transverse": sigma_membrane,
            "sigma_1": sigma_1,
            "sigma_2": sigma_2,
            "sigma_total": sigma_bend_ps + sigma_membrane + sigma_axial,
            "sigma_vm": sigma_vm,
            "H_tilde": H_tilde,
            "H_reaction": H_reaction,
            "components": [
                ("Plane-strain bending", sigma_bend_ps,
                 r"\sigma_{bend} = \frac{E}{1-\nu^2} \cdot \frac{t}{D}"),
                ("Brazier membrane (transverse)", sigma_membrane,
                 r"\sigma_{mem} = \frac{H}{t}"),
                ("Web tension", sigma_axial,
                 r"\sigma_{tension}"),
            ],
        }


def draw_stress_bar(components, sigma_vm, sigma_yield):
    """Draw a stacked horizontal bar chart of stress contributions."""
    COLORS = ["#2563eb", "#16a34a", "#d97706", "#dc2626"]
    fig, ax = plt.subplots(figsize=(7, 2.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f9fafb")

    left = 0.0
    for i, (name, val, _) in enumerate(components):
        ax.barh(0, abs(val), left=left, height=0.5,
                color=COLORS[i % len(COLORS)], label=f"{name}: {abs(val):.1f} MPa",
                edgecolor="white", linewidth=0.5)
        if abs(val) > 5:
            ax.text(left + abs(val) / 2, 0, f"{abs(val):.1f}",
                    ha="center", va="center", fontsize=7, color="white", fontweight="bold")
        left += abs(val)

    # Yield line
    if sigma_yield:
        ax.axvline(sigma_yield, color="#dc2626", lw=1.5, ls="--",
                   label=f"σ_yield = {sigma_yield:.0f} MPa")

    ax.set_yticks([])
    ax.set_xlabel("Stress (MPa)", fontsize=8, color="#6b7280")
    ax.set_title("Stress component breakdown", fontsize=9,
                 color="#111827", fontweight="bold")
    ax.legend(loc="upper right", fontsize=6.5, framealpha=0.9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#e5e7eb")
        spine.set_linewidth(0.6)
    ax.tick_params(colors="#9ca3af", labelsize=8)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4.  UI
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="BPP Roll Diameter Calculator", layout="wide")

# ── Custom CSS for mode toggle button ────────────────────────────────────────
st.markdown("""
<style>
div[data-testid="stHorizontalBlock"] .stButton button {
    font-size: 0.82rem;
}
.mode-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}
</style>
""", unsafe_allow_html=True)

# ── Mode toggle (session state) ───────────────────────────────────────────────
if "calc_mode" not in st.session_state:
    st.session_state.calc_mode = "dmin"   # "dmin" or "stress"

col_title, col_toggle = st.columns([3, 1])
with col_title:
    st.title("Analytical $D_{min}$ Calculator")

with col_toggle:
    st.write("")  # vertical spacer
    if st.session_state.calc_mode == "dmin":
        mode_label = "🔄 Switch to: Given D → Max Stress"
        next_mode = "stress"
    else:
        mode_label = "🔄 Switch to: Given Yield → Min Diameter"
        next_mode = "dmin"

    if st.button(mode_label, use_container_width=True):
        st.session_state.calc_mode = next_mode
        st.rerun()

# Mode banner
if st.session_state.calc_mode == "dmin":
    st.markdown(
        "**Mode: Minimum Roll Diameter** — Enter material yield strength; "
        "the tool calculates the minimum roller diameter to prevent plastic deformation."
    )
else:
    st.markdown(
        "**Mode: Stress Analysis** — Enter the known roller diameter; "
        "the tool calculates all stress components and the resulting **von Mises stress**."
    )

st.markdown("---")

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.header("1. Material Properties")
E = st.sidebar.number_input("Young's Modulus (MPa)", value=200000.0, step=1000.0)
nu = st.sidebar.number_input("Poisson's Ratio", value=0.30, step=0.01)
sigma_yield = st.sidebar.number_input("Yield Strength (MPa)", value=350.0, step=10.0)

st.sidebar.header("2. Process Parameters")
sigma_tension = st.sidebar.number_input("Web Tension (MPa)", value=15.0, step=1.0)

# Only show D_roll input in stress mode
if st.session_state.calc_mode == "stress":
    st.sidebar.header("3. Roll Diameter")
    D_roll = st.sidebar.number_input("Roll Diameter (mm)", value=100.0, step=5.0, min_value=1.0)

st.sidebar.header("4. Foil Geometry" if st.session_state.calc_mode == "stress" else "3. Foil Geometry")
t = st.sidebar.number_input("Base Foil Thickness (mm)", value=0.100,
                             step=0.001, format="%.3f")
pattern_type = st.sidebar.selectbox(
    "Engraving Pattern",
    ["Plain Foil", "Longitudinal Channels", "Transverse Channels"],
)

# Safe defaults
w_top = w_bot = 0.50
h_channel = 0.46
theta_deg = 15.0
c_max = 0.380
H_total = 0.460
r_top = r_bot = 0.200
NA_bot = None

if pattern_type == "Longitudinal Channels":
    st.sidebar.markdown("*Longitudinal Specific Inputs:*")
    H_total = st.sidebar.number_input("Total Outer Height ($H$) [mm]",
                                      value=0.460, step=0.01)
    w_top = st.sidebar.number_input("Top Flat Width ($w_1$) [mm]",
                                    value=0.597, step=0.01)
    w_bot = st.sidebar.number_input("Bottom Flat Width ($w_2$) [mm]",
                                    value=0.558, step=0.01)
    r_top = st.sidebar.number_input("Top Outer Radius [mm]", value=0.200, step=0.01)
    r_bot = st.sidebar.number_input("Bottom Outer Radius [mm]", value=0.200, step=0.01)
    theta_deg = st.sidebar.number_input("Web Angle ($\\theta$) [degrees]",
                                        value=15.0, step=1.0)
    c_max, NA_bot = calculate_longitudinal_cmax(
        H_total, t, w_top, w_bot, r_top, r_bot, theta_deg
    )
    st.sidebar.markdown("---")
    st.sidebar.info(
        f"**Calculated Metrics:**\n\n"
        f"Neutral Axis: **{NA_bot:.3f} mm**\n\n"
        f"Outer Fiber ($c_{{max}}$): **{c_max:.3f} mm**"
    )

elif pattern_type == "Transverse Channels":
    st.sidebar.markdown("*Transverse Specific Inputs:*")
    w_top = st.sidebar.number_input("Top Flat Width ($w_1$) [mm]",
                                    value=0.500, step=0.01)
    w_bot = st.sidebar.number_input("Bottom Flat Width ($w_2$) [mm]",
                                    value=0.550, step=0.01)
    h_channel = st.sidebar.number_input("Channel Height ($h$) [mm]",
                                        value=0.460, step=0.01)
    theta_deg = st.sidebar.number_input("Web Angle ($\\theta$) [degrees]",
                                        value=15.0, step=1.0)

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_calc, tab_geom, tab_theory = st.tabs(
    ["🧮 Calculator", "📐 Geometry Preview", "📚 Theory & Rationale"]
)

# ═════════════════════════════════════════════════════════════════════════════
# Tab 1: Calculator
# ═════════════════════════════════════════════════════════════════════════════
with tab_calc:

    # ── MODE A: Given yield → D_min ──────────────────────────────────────────
    if st.session_state.calc_mode == "dmin":
        st.header("Results — Minimum Roll Diameter")

        if sigma_tension >= sigma_yield:
            st.error(
                "⚠️ **Yield Condition Reached:** The uniform web tension exceeds the "
                "material yield strength. Plastic deformation occurs prior to curvature engagement."
            )
        else:
            D_min = None
            formula_latex = ""

            if pattern_type == "Plain Foil":
                D_min = (E * t) / (sigma_yield - sigma_tension)
                formula_latex = r"D_{min} = \frac{E \cdot t}{\sigma_{yield} - \sigma_{tension}}"

            elif pattern_type == "Longitudinal Channels":
                D_min = (E * 2 * c_max) / (sigma_yield - sigma_tension)
                formula_latex = r"D_{min} = \frac{E \cdot 2 \cdot c_{max}}{\sigma_{yield} - \sigma_{tension}}"

            elif pattern_type == "Transverse Channels":
                H_tilde = calculate_unit_H(w_top, h_channel, theta_deg)
                numerator = (E * t / (1 - nu ** 2)) + (2 * H_tilde * sigma_tension)
                denominator = sigma_yield - sigma_tension

                if denominator <= 0:
                    st.error(
                        "⚠️ **Yield Condition Reached:** The combination of global tension "
                        "and local membrane stresses exceeds the yield threshold."
                    )
                else:
                    D_min = numerator / denominator
                    actual_q = (sigma_tension * t) / (D_min / 2)
                    actual_H = H_tilde * actual_q
                    st.info(
                        f"**Castigliano Integration Output:** The induced radial compressive "
                        f"pressure generates a transverse horizontal reaction force ($H$) of "
                        f"**{actual_H:.3f} N/mm**."
                    )
                    formula_latex = (
                        r"D_{min} = \frac{\frac{E \cdot t}{1-\nu^2} + "
                        r"2 \cdot \tilde{H} \cdot \sigma_{tension}}{\sigma_{yield} - \sigma_{tension}}"
                    )

            if D_min is not None:
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric(label="Minimum Roll Diameter", value=f"{D_min:.2f} mm")
                with col2:
                    st.markdown("**Governing Formula:**")
                    st.latex(formula_latex)

    # ── MODE B: Given D → Stress ─────────────────────────────────────────────
    else:
        st.header("Results — Von Mises Stress Analysis")

        result = compute_stress_components(
            pattern_type=pattern_type,
            D_roll=D_roll,
            E=E,
            nu=nu,
            sigma_tension=sigma_tension,
            t=t,
            c_max=(c_max if pattern_type == "Longitudinal Channels" else None),
            w_top=(w_top if pattern_type == "Transverse Channels" else None),
            h_channel=(h_channel if pattern_type == "Transverse Channels" else None),
            theta_deg=(theta_deg if pattern_type == "Transverse Channels" else None),
        )

        sigma_vm = result["sigma_vm"]
        safety_factor = sigma_yield / sigma_vm if sigma_vm > 0 else float("inf")
        yielded = sigma_vm >= sigma_yield

        # ── Key metrics row
        col_vm, col_sf, col_status = st.columns(3)
        with col_vm:
            st.metric("Von Mises Stress (σ_vm)", f"{sigma_vm:.2f} MPa")
        with col_sf:
            st.metric("Safety Factor", f"{safety_factor:.3f}",
                      delta=f"{'⚠ Below 1.0' if safety_factor < 1 else 'Above 1.0'}",
                      delta_color="inverse")
        with col_status:
            if yielded:
                st.error("🔴 **YIELDED** — Plastic deformation expected at this diameter.")
            else:
                st.success(f"🟢 **ELASTIC** — Material remains below yield ({sigma_yield:.0f} MPa).")

        st.divider()

        # ── Component breakdown
        st.subheader("Stress Components")
        col_bars, col_formulas = st.columns([2, 1])

        with col_bars:
            bar_fig = draw_stress_bar(result["components"], sigma_vm, sigma_yield)
            st.pyplot(bar_fig, use_container_width=True)
            plt.close(bar_fig)

        with col_formulas:
            for name, val, latex in result["components"]:
                st.markdown(f"**{name}:** `{abs(val):.2f} MPa`")
                st.latex(latex + f" = {abs(val):.2f} \\text{{ MPa}}")

            if pattern_type == "Transverse Channels":
                st.divider()
                st.markdown("**Von Mises (biaxial):**")
                st.latex(
                    r"\sigma_{vm} = \sqrt{\sigma_1^2 - \sigma_1 \sigma_2 + \sigma_2^2}"
                )
                st.caption(
                    f"σ₁ = {result['sigma_1']:.2f} MPa (longitudinal)  \n"
                    f"σ₂ = {result['sigma_2']:.2f} MPa (transverse, compressive)"
                )
            else:
                st.divider()
                st.markdown("**Von Mises (uniaxial superposition):**")
                st.latex(r"\sigma_{vm} = \sigma_{bend} + \sigma_{tension}")

        # ── Transverse-specific Brazier info
        if pattern_type == "Transverse Channels":
            st.info(
                f"**Castigliano Integration:** Geometric stiffness constant "
                f"$\\tilde{{H}}$ = **{result['H_tilde']:.4f}** → "
                f"Horizontal reaction force $H$ = **{result['H_reaction']:.4f} N/mm**"
            )

        # ── Sensitivity: stress vs diameter curve
        st.subheader("Stress vs. Roll Diameter")
        D_range = np.linspace(max(10, D_roll * 0.2), D_roll * 3, 200)
        vm_curve = []
        for D_i in D_range:
            r_i = compute_stress_components(
                pattern_type=pattern_type,
                D_roll=D_i, E=E, nu=nu,
                sigma_tension=sigma_tension, t=t,
                c_max=(c_max if pattern_type == "Longitudinal Channels" else None),
                w_top=(w_top if pattern_type == "Transverse Channels" else None),
                h_channel=(h_channel if pattern_type == "Transverse Channels" else None),
                theta_deg=(theta_deg if pattern_type == "Transverse Channels" else None),
            )
            vm_curve.append(r_i["sigma_vm"])

        fig_s, ax_s = plt.subplots(figsize=(7, 3.2))
        fig_s.patch.set_facecolor("white")
        ax_s.set_facecolor("#f9fafb")
        ax_s.plot(D_range, vm_curve, color="#2563eb", lw=2, label="σ_vm")
        ax_s.axhline(sigma_yield, color="#dc2626", lw=1.4, ls="--",
                     label=f"σ_yield = {sigma_yield:.0f} MPa")
        ax_s.axvline(D_roll, color="#d97706", lw=1.2, ls=":",
                     label=f"Current D = {D_roll:.1f} mm")
        ax_s.scatter([D_roll], [sigma_vm], color="#d97706", zorder=5, s=60)
        ax_s.set_xlabel("Roll Diameter (mm)", fontsize=8, color="#6b7280")
        ax_s.set_ylabel("Von Mises Stress (MPa)", fontsize=8, color="#6b7280")
        ax_s.set_title("Von Mises stress vs. roll diameter", fontsize=9,
                       color="#111827", fontweight="bold")
        ax_s.legend(fontsize=7.5)
        ax_s.grid(True, color="#e5e7eb", lw=0.4)
        for spine in ax_s.spines.values():
            spine.set_edgecolor("#e5e7eb")
            spine.set_linewidth(0.6)
        ax_s.tick_params(colors="#9ca3af", labelsize=8)
        fig_s.tight_layout()
        st.pyplot(fig_s, use_container_width=True)
        plt.close(fig_s)

# ═════════════════════════════════════════════════════════════════════════════
# Tab 2: Geometry Preview
# ═════════════════════════════════════════════════════════════════════════════
with tab_geom:
    st.header("Cross-section geometry preview")
    st.markdown(
        "The profile below is generated analytically from your sidebar inputs — "
        "no CAD file required. It updates live as you change any parameter."
    )

    col_fig, col_info = st.columns([2, 1])

    with col_fig:
        fig = draw_geometry_preview(
            pattern_type=pattern_type,
            t=t,
            H_total=H_total,
            w1=w_top,
            w2=w_bot,
            r_top=r_top,
            r_bot=r_bot,
            theta_deg=theta_deg,
            h_channel=h_channel,
            w_top=w_top,
            NA_bot=NA_bot,
            c_max=(c_max if pattern_type == "Longitudinal Channels" else None),
        )
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_info:
        st.markdown("**Dimension summary**")

        if pattern_type == "Plain Foil":
            st.metric("Thickness t", f"{t:.3f} mm")
            st.metric("Neutral axis", f"{t/2:.3f} mm from bottom")

        elif pattern_type == "Longitudinal Channels":
            st.metric("Total height H", f"{H_total:.3f} mm")
            st.metric("Top flat w₁", f"{w_top:.3f} mm")
            st.metric("Bottom flat w₂", f"{w_bot:.3f} mm")
            st.metric("Web angle θ", f"{theta_deg:.1f}°")
            st.metric("Top fillet r", f"{r_top:.3f} mm")
            st.metric("Bottom fillet r", f"{r_bot:.3f} mm")
            if NA_bot is not None:
                st.divider()
                st.metric("Neutral axis from bottom", f"{NA_bot:.3f} mm")
                st.metric("c_max (outer fiber)", f"{c_max:.3f} mm")
                pct = (NA_bot / H_total) * 100
                st.caption(
                    f"NA sits at {pct:.1f}% of total height — "
                    + ("shifted toward bottom" if pct < 50 else "shifted toward top")
                )

        elif pattern_type == "Transverse Channels":
            st.metric("Channel height h", f"{h_channel:.3f} mm")
            st.metric("Top flat w₁", f"{w_top:.3f} mm")
            st.metric("Web angle θ", f"{theta_deg:.1f}°")
            st.metric("Wall thickness t", f"{t:.3f} mm")
            st.divider()
            st.caption(
                "For transverse channels the bending neutral axis acts on "
                "the local flat panel (at t/2 from the bottom surface), "
                "not on the full corrugated profile."
            )

        st.markdown("---")
        st.caption(
            "Red dashed line = neutral axis.  "
            "Blue outline = outer surface profile.  "
            "Adjacent periods shown at reduced opacity for context."
        )

# ═════════════════════════════════════════════════════════════════════════════
# Tab 3: Theory
# ═════════════════════════════════════════════════════════════════════════════
with tab_theory:
    st.header("Mechanics & Rationale")
    st.markdown(
        "This analytical tool utilises the **Principle of Superposition** to determine "
        "the minimum required roller diameter to prevent plastic deformation. The "
        "mechanical behaviour of the foil varies significantly depending on the "
        "orientation of the flow field channels relative to the manufacturing transport direction."
    )

    st.subheader("Calculation Modes")
    st.markdown(
        "The tool supports two complementary calculation directions:\n\n"
        "- **Given Yield → D_min:** Finds the minimum roller diameter such that "
        "the total stress (bending + tension) does not exceed the material yield strength.\n"
        "- **Given D → Max Stress:** For a known roller diameter, computes all individual "
        "stress components and combines them into an equivalent **von Mises stress**, "
        "which is then compared against yield to determine the safety factor."
    )

    st.subheader("1. Plain Foil")
    st.markdown(
        "**Theory:** Modelled using classic Euler-Bernoulli beam theory. "
        "Maximum bending stress is determined entirely by material thickness and roller curvature.\n\n"
        "**Stress components:** Bending stress + global tensile stress."
    )
    st.latex(r"\sigma_{total} = \frac{E \cdot t}{D} + \sigma_{tension}")

    st.subheader("2. Longitudinal Channels (parallel to transport direction)")
    st.markdown(
        "**Theory:** The foil conforms to the curvature along its entire cross-section. "
        "The corrugated profile increases the area moment of inertia; the base thickness "
        "$t$ is replaced by the maximum distance from the cross-sectional neutral axis "
        "to the outermost fibre ($c_{max}$). The neutral axis is computed dynamically "
        "via 1-D centreline length-weighted integration.\n\n"
        "**Stress components:** Amplified profile bending stress + global tensile stress."
    )
    st.latex(r"\sigma_{total} = \frac{E \cdot 2 \cdot c_{max}}{D} + \sigma_{tension}")

    st.subheader("3. Transverse Channels (perpendicular to transport direction)")
    st.markdown(
        "**Theory:** Deformation mechanics transition from 1-D beam bending to 3-D plate "
        "kinematics. Three superimposed stress components govern:\n\n"
        "- **Plane-strain bending stress:** Web sections prevent lateral contraction "
        "(Poisson effect), raising bending stiffness by $1/(1-\\nu^2)$.\n"
        "- **Membrane tensile stress (Brazier effect):** Longitudinal tension generates "
        "radial compressive pressure, deflecting flat sections and inducing a horizontal "
        "reaction force $H$ computed via Castigliano numerical integration.\n"
        "- **Global tensile stress:** Uniform machine tension.\n\n"
        "In **Stress Analysis mode**, the longitudinal and transverse stresses are "
        "treated as principal stresses and combined via the von Mises criterion:"
    )
    st.latex(
        r"\sigma_{vm} = \sqrt{\sigma_1^2 - \sigma_1 \sigma_2 + \sigma_2^2}"
    )
    st.latex(
        r"\sigma_1 = \frac{E}{1-\nu^2} \cdot \frac{t}{D} + \sigma_{tension}, \quad "
        r"\sigma_2 = -\frac{H}{t}"
    )

    st.markdown("---")
    st.header("Variable nomenclature")
    st.markdown(
        "**Material & process:**\n"
        "- $E$ — Young's Modulus\n"
        "- $\\nu$ — Poisson's Ratio\n"
        "- $\\sigma_{yield}$ — yield strength\n"
        "- $\\sigma_{tension}$ — continuous web tension (force / cross-sectional area)\n"
        "- $\\sigma_{vm}$ — von Mises equivalent stress\n\n"
        "**Geometry:**\n"
        "- $t$ — base foil thickness\n"
        "- $H$ — total outer height of the corrugated profile (longitudinal)\n"
        "- $r_{top}, r_{bot}$ — outer fillet radii (longitudinal)\n"
        "- $c_{max}$ — distance from neutral axis to outermost fibre (longitudinal)\n"
        "- $w_1$ — top flat width (centre-to-centre of fillets)\n"
        "- $w_2$ — bottom flat width (centre-to-centre of fillets)\n"
        "- $h$ — centreline height (transverse)\n"
        "- $\\theta$ — web draft angle from vertical\n"
        "- $H$ — horizontal reaction force, N/mm (transverse)\n"
        "- $\\tilde{H}$ — geometric stiffness constant for $H$ (unit-load Castigliano result)"
    )
