import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Analytical Solvers
# ─────────────────────────────────────────────────────────────────────────────

def calculate_longitudinal_cmax(H, t, w1, w2, r_top_out, r_bot_out, theta_deg):
    """Calculate neutral axis and c_max for longitudinal channels."""
    theta  = np.radians(theta_deg)
    h      = H - t
    R_cl_top = r_top_out - t / 2
    R_cl_bot = r_bot_out - t / 2
    phi    = np.pi / 2 - theta
    dy_top = R_cl_top * (1 - np.cos(phi))
    dy_bot = R_cl_bot * (1 - np.cos(phi))
    L_flat_top = w1 - 2 * R_cl_top * np.tan(phi / 2)
    L_flat_bot = w2 - 2 * R_cl_bot * np.tan(phi / 2)
    L_arc_top  = R_cl_top * phi
    L_arc_bot  = R_cl_bot * phi
    L_web      = (h - dy_top - dy_bot) / np.cos(theta)
    C_flat_top = h
    C_flat_bot = 0.0
    C_arc_top  = h - R_cl_top + R_cl_top * np.sin(phi) / phi if phi > 0 else h
    C_arc_bot  = R_cl_bot - R_cl_bot * np.sin(phi) / phi if phi > 0 else 0
    C_web      = (dy_bot + (h - dy_top)) / 2
    num = (L_flat_top * C_flat_top + L_flat_bot * C_flat_bot
           + 2 * L_arc_top * C_arc_top + 2 * L_arc_bot * C_arc_bot
           + 2 * L_web * C_web)
    den = L_flat_top + L_flat_bot + 2*L_arc_top + 2*L_arc_bot + 2*L_web
    NA_cl  = num / den
    NA_bot = NA_cl + t / 2
    c_max  = max(NA_bot, H - NA_bot)
    return c_max, NA_bot


def calculate_transverse_alpha(H, t, r_top_out, r_bot_out, theta_deg):
    """
    Calculate the corrugation amplification factor α for transverse channels.

    Based on the derivation:
        α = h / (4 · L_eff)

    where:
        h = H - t (centerline corrugation height)
        L_eff = L_arc_bot + L_web + L_arc_top (effective web path length)

    This factor accounts for:
        - Wrapping kinematics (chord rotation ψ ∝ h)
        - Frame response (slope-deflection: moment ∝ 1/L_eff)
        - Contact constraint (roll enforces curvature in flat sections)

    Returns
    -------
    alpha : float
        Amplification factor
    L_eff : float
        Effective web path length (mm)
    L_web : float
        Straight inclined web length (mm)
    L_arc_top : float
        Top arc centerline length (mm)
    L_arc_bot : float
        Bottom arc centerline length (mm)
    h : float
        Centerline height (mm)
    """
    theta = np.radians(theta_deg)
    h = H - t  # centerline height

    R_cl_top = max(r_top_out - t / 2, 0.0)
    R_cl_bot = max(r_bot_out - t / 2, 0.0)

    # Arc subtended angle (from vertical to end of fillet)
    phi = np.pi / 2 - theta

    # Vertical height consumed by each arc
    Dy_top = R_cl_top * (1 - np.sin(theta))
    Dy_bot = R_cl_bot * (1 - np.sin(theta))

    # Straight inclined web length
    L_web_num = h - Dy_top - Dy_bot
    if L_web_num < 0:
        L_web_num = 0.0
    L_web = L_web_num / np.cos(theta) if np.cos(theta) > 1e-10 else L_web_num

    # Arc centerline lengths
    L_arc_top = R_cl_top * phi if phi > 0 else 0.0
    L_arc_bot = R_cl_bot * phi if phi > 0 else 0.0

    # Effective web path length
    L_eff = L_arc_bot + L_web + L_arc_top

    # Safety: avoid division by zero
    if L_eff < 1e-10:
        L_eff = h  # fallback to sharp-corner approximation

    # Amplification factor
    alpha = h / (4.0 * L_eff)

    return alpha, L_eff, L_web, L_arc_top, L_arc_bot, h


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Geometry builders — correct closed-wall cross-section polygon
# ─────────────────────────────────────────────────────────────────────────────

def _arc(cx, cy, r, a0, a1, n=40):
    """Points along a circular arc from a0 to a1 (radians, CCW)."""
    if a1 < a0:
        a1 += 2 * np.pi
    ang = np.linspace(a0, a1, n)
    return cx + r * np.cos(ang), cy + r * np.sin(ang)


def _compute_tangent_normal(bot_cx, bot_cy, r_bot, top_cx, top_cy, r_top):
    """
    Compute the outward normal direction for the RIGHT-side external tangent
    between two fillet circles.
    """
    dx = top_cx - bot_cx
    dy = top_cy - bot_cy
    d = np.sqrt(dx**2 + dy**2)
    dr = r_bot - r_top

    if d < 1e-10:
        return 0.0

    cos_val = np.clip(dr / d, -1.0, 1.0)
    gamma = np.arccos(cos_val)
    alpha = np.arctan2(dy, dx)

    phi1 = alpha + gamma
    phi2 = alpha - gamma

    if np.cos(phi1) >= np.cos(phi2):
        return phi1
    else:
        return phi2


def _profile_wall(H, t, w1, w2, r_top_out, r_bot_out, theta_deg):
    """
    Returns outer and inner profile polygons for ONE corrugation period.
    """
    r_top_out = min(r_top_out, H * 0.48)
    r_bot_out = min(r_bot_out, H * 0.48)

    bot_cx = w2 / 2
    bot_cy = r_bot_out
    top_cx = w1 / 2
    top_cy = H - r_top_out

    if top_cy <= bot_cy:
        available = H * 0.95
        scale = available / (r_top_out + r_bot_out)
        r_top_out *= scale
        r_bot_out *= scale
        bot_cy = r_bot_out
        top_cy = H - r_top_out

    phi_n = _compute_tangent_normal(bot_cx, bot_cy, r_bot_out,
                                    top_cx, top_cy, r_top_out)

    wnx = np.cos(phi_n)
    wny = np.sin(phi_n)

    bx_o = bot_cx + r_bot_out * wnx
    by_o = bot_cy + r_bot_out * wny
    tx_o = top_cx + r_top_out * wnx
    ty_o = top_cy + r_top_out * wny

    bot_wa_o = phi_n
    top_wa_o = phi_n

    def right_outer():
        xs, ys = [0.0, bot_cx], [0.0, 0.0]
        a0, a1 = -np.pi / 2, bot_wa_o
        if a1 < a0:
            a1 += 2 * np.pi
        ax, ay = _arc(bot_cx, bot_cy, r_bot_out, a0, a1)
        xs.extend(ax); ys.extend(ay)
        xs.append(tx_o); ys.append(ty_o)
        a0, a1 = top_wa_o, np.pi / 2
        if a1 < a0:
            a1 += 2 * np.pi
        ax, ay = _arc(top_cx, top_cy, r_top_out, a0, a1)
        xs.extend(ax); ys.extend(ay)
        xs.append(0.0); ys.append(H)
        return np.array(xs), np.array(ys)

    rx_o, ry_o = right_outer()
    ox = np.concatenate([-rx_o[::-1], rx_o[1:]])
    oy = np.concatenate([ ry_o[::-1], ry_o[1:]])

    r_bot_in = max(r_bot_out - t, 0.005)
    r_top_in = max(r_top_out - t, 0.005)

    bx_i = bot_cx + r_bot_in * wnx
    by_i = bot_cy + r_bot_in * wny
    tx_i = top_cx + r_top_in * wnx
    ty_i = top_cy + r_top_in * wny

    bot_wa_i = phi_n
    top_wa_i = phi_n

    def right_inner():
        xs, ys = [0.0, bot_cx], [t, t]
        a0, a1 = -np.pi / 2, bot_wa_i
        if a1 < a0:
            a1 += 2 * np.pi
        ax, ay = _arc(bot_cx, bot_cy, r_bot_in, a0, a1)
        xs.extend(ax); ys.extend(ay)
        xs.append(tx_i); ys.append(ty_i)
        a0, a1 = top_wa_i, np.pi / 2
        if a1 < a0:
            a1 += 2 * np.pi
        ax, ay = _arc(top_cx, top_cy, r_top_in, a0, a1)
        xs.extend(ax); ys.extend(ay)
        xs.append(0.0); ys.append(H - t)
        return np.array(xs), np.array(ys)

    rx_i, ry_i = right_inner()
    ix = np.concatenate([-rx_i[::-1], rx_i[1:]])
    iy = np.concatenate([ ry_i[::-1], ry_i[1:]])

    wall_x = np.concatenate([ox, ix[::-1], [ox[0]]])
    wall_y = np.concatenate([oy, iy[::-1], [oy[0]]])
    return ox, oy, ix, iy, wall_x, wall_y


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Geometry Preview figure
# ─────────────────────────────────────────────────────────────────────────────

_AX_BG = "#F9FAFB"


def draw_geometry_preview(pattern_type, t,
                           H_total=None, w1=None, w2=None,
                           r_top=None, r_bot=None,
                           theta_deg=15.0,
                           NA_bot=None, c_max=None):

    FILL   = "#DBEAFE"
    STROKE = "#1D4ED8"
    GHOST  = "#93C5FD"
    NA_C   = "#DC2626"
    DIM_C  = "#6B7280"
    LABEL_C= "#111827"

    fig, ax = plt.subplots(figsize=(7, 4.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(_AX_BG)
    ax.tick_params(colors="#9CA3AF", labelsize=7.5)
    for sp in ax.spines.values():
        sp.set_edgecolor("#E5E7EB"); sp.set_linewidth(0.5)

    def dim_arrow(x0, y0, x1, y1, label, orient="v", off=0.04, fs=7):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="<->", color=DIM_C,
                                   lw=0.8, mutation_scale=7))
        mx, my = (x0+x1)/2, (y0+y1)/2
        kw = dict(fontsize=fs, color=DIM_C)
        if orient == "v":
            ax.text(mx+off, my, label, va="center", ha="left", **kw)
        else:
            ax.text(mx, my+off, label, va="bottom", ha="center", **kw)

    # ── PLAIN FOIL ──────────────────────────────────────────────────────────
    if pattern_type == "Plain Foil":
        L = 3.0
        ax.fill([-L/2, L/2, L/2, -L/2], [0, 0, t, t],
                facecolor=FILL, edgecolor=STROKE, lw=1.5)
        ax.axhline(t/2, color=NA_C, lw=1.2, ls="--")
        ax.text(0, t/2 + t*0.2, "Neutral axis", fontsize=7,
                color=NA_C, ha="center", va="bottom")
        dim_arrow(L/2+0.1, 0, L/2+0.1, t, f"t = {t:.3f} mm", "v", off=0.06)
        ax.set_xlim(-L/2 - 0.2, L/2 + 0.55)
        ax.set_ylim(-t*2, t*5)
        ax.set_title("Cross-section — Plain foil", fontsize=9,
                     color=LABEL_C, fontweight="bold", pad=8)

    # ── LONGITUDINAL CHANNELS ────────────────────────────────────────────────
    elif pattern_type == "Longitudinal Channels":
        try:
            ox, oy, ix, iy, _, _ = _profile_wall(
                H_total, t, w1, w2, r_top, r_bot, theta_deg)
            period = ox.max() - ox.min()

            for shift in [-period, period]:
                ax.fill(ox + shift, oy, facecolor=FILL, edgecolor=GHOST,
                        lw=0.7, alpha=0.35, zorder=1)

            ax.fill(ox, oy, facecolor=FILL, edgecolor=STROKE, lw=1.6, zorder=2)
            ax.fill(ix, iy, facecolor=_AX_BG, edgecolor=STROKE, lw=1.0, zorder=3)

            xmin, xmax = ox.min(), ox.max()

            if NA_bot is not None:
                ax.hlines(NA_bot, xmin - 0.1, xmax + 0.05,
                          color=NA_C, lw=1.3, ls="--", zorder=4)
                ax.text(xmin - 0.12, NA_bot, "NA", fontsize=6.5,
                        color=NA_C, va="center", ha="right")

            if NA_bot is not None and c_max is not None:
                y_far = (NA_bot + c_max) if (H_total - NA_bot) > NA_bot else (NA_bot - c_max)
                dim_arrow(xmax + 0.10, NA_bot, xmax + 0.10, y_far,
                          f"c_max\n{c_max:.3f} mm", "v", off=0.03)

            dim_arrow(xmax + 0.36, 0, xmax + 0.36, H_total,
                      f"H = {H_total:.3f} mm", "v", off=0.03)
            dim_arrow(-w1/2, H_total + 0.07, w1/2, H_total + 0.07,
                      f"w₁ = {w1:.2f} mm", "h", off=H_total*0.09)
            dim_arrow(-w2/2, -0.07, w2/2, -0.07,
                      f"w₂ = {w2:.2f} mm", "h", off=-H_total*0.12)

            ax.set_xlim(xmin - period*0.55, xmax + 0.68)
            ax.set_ylim(-H_total*0.42, H_total*1.6)

        except Exception as e:
            ax.text(0.5, 0.5, f"Geometry error:\n{e}",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=8, color="red")

        ax.set_title("Cross-section — Longitudinal channels", fontsize=9,
                     color=LABEL_C, fontweight="bold", pad=8)

    # ── TRANSVERSE CHANNELS ──────────────────────────────────────────────────
    elif pattern_type == "Transverse Channels":
        try:
            ox, oy, ix, iy, _, _ = _profile_wall(
                H_total, t, w1, w2, r_top, r_bot, theta_deg)
            period = ox.max() - ox.min()

            for shift in [-period, period]:
                ax.fill(ox + shift, oy, facecolor=FILL, edgecolor=GHOST,
                        lw=0.7, alpha=0.35, zorder=1)

            ax.fill(ox, oy, facecolor=FILL, edgecolor=STROKE, lw=1.6, zorder=2)
            ax.fill(ix, iy, facecolor=_AX_BG, edgecolor=STROKE, lw=1.0, zorder=3)

            xmin, xmax = ox.min(), ox.max()

            # For transverse channels, NA is at t/2 from the bottom (local flat bending)
            ax.hlines(t/2, xmin - 0.1, xmax + 0.05,
                      color=NA_C, lw=1.1, ls="--", zorder=4)
            ax.text(xmin - 0.12, t/2, f"NA (t/2)", fontsize=6,
                    color=NA_C, va="center", ha="right")

            dim_arrow(xmax + 0.36, 0, xmax + 0.36, H_total,
                      f"H = {H_total:.3f} mm", "v", off=0.03)
            dim_arrow(-w1/2, H_total + 0.07, w1/2, H_total + 0.07,
                      f"w₁ = {w1:.2f} mm", "h", off=H_total*0.09)
            dim_arrow(-w2/2, -0.07, w2/2, -0.07,
                      f"w₂ = {w2:.2f} mm", "h", off=-H_total*0.12)

            ax.set_xlim(xmin - period*0.55, xmax + 0.68)
            ax.set_ylim(-H_total*0.42, H_total*1.6)

        except Exception as e:
            ax.text(0.5, 0.5, f"Geometry error:\n{e}",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=8, color="red")

        ax.set_title("Cross-section — Transverse channels (sideways)", fontsize=9,
                     color=LABEL_C, fontweight="bold", pad=8)

    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, color="#E5E7EB", lw=0.35, zorder=0)
    ax.set_xlabel("Width (mm)", fontsize=8, color=DIM_C)
    ax.set_ylabel("Height (mm)", fontsize=8, color=DIM_C)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Stress Analysis helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_stress_components(pattern_type, D_roll, E, nu, sigma_tension, t,
                               c_max=None,
                               H_total=None, r_top_out=None, r_bot_out=None,
                               theta_deg=None):
    """Compute all stress components for a given configuration."""

    if pattern_type == "Plain Foil":
        sb = (E * t) / D_roll
        sv = sb + sigma_tension
        return {"sigma_bend": sb, "sigma_axial": sigma_tension,
                "sigma_total": sv, "sigma_vm": sv,
                "components": [
                    ("Bending stress", sb, r"\sigma_{bend}=\frac{E\cdot t}{D}"),
                    ("Web tension", sigma_tension, r"\sigma_{tension}"),
                ]}

    elif pattern_type == "Longitudinal Channels":
        sb = (E * 2 * c_max) / D_roll
        sv = sb + sigma_tension
        return {"sigma_bend": sb, "sigma_axial": sigma_tension,
                "sigma_total": sv, "sigma_vm": sv,
                "components": [
                    ("Profile bending", sb, r"\sigma_{bend}=\frac{E\cdot2c_{max}}{D}"),
                    ("Web tension", sigma_tension, r"\sigma_{tension}"),
                ]}

    elif pattern_type == "Transverse Channels":
        # New model: σ = E·t/(D·(1-ν²)) · (1 + α) + σ_tension
        alpha, L_eff, L_web, L_arc_top, L_arc_bot, h = \
            calculate_transverse_alpha(H_total, t, r_top_out, r_bot_out, theta_deg)

        # Base plane-strain bending stress
        sigma_ps_base = (E * t) / (D_roll * (1 - nu**2))

        # Hinge amplification stress
        sigma_hinge = sigma_ps_base * alpha

        # Total von Mises stress
        sv = sigma_ps_base + sigma_hinge + sigma_tension

        return {"sigma_ps_base": sigma_ps_base,
                "sigma_hinge": sigma_hinge,
                "sigma_axial": sigma_tension,
                "sigma_total": sv,
                "sigma_vm": sv,
                "alpha": alpha,
                "L_eff": L_eff,
                "L_web": L_web,
                "L_arc_top": L_arc_top,
                "L_arc_bot": L_arc_bot,
                "h": h,
                "components": [
                    ("Plane-strain bending", sigma_ps_base,
                     r"\sigma_{ps}=\frac{E\cdot t}{D\cdot(1-\nu^2)}"),
                    ("Hinge amplification", sigma_hinge,
                     r"\sigma_{hinge}=\sigma_{ps}\cdot\alpha"),
                    ("Web tension", sigma_tension, r"\sigma_{tension}"),
                ]}


def draw_stress_bar(components, sigma_vm, sigma_yield):
    COLORS = ["#2563EB", "#7C3AED", "#16A34A", "#D97706", "#DC2626"]
    fig, ax = plt.subplots(figsize=(7, 2.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#F9FAFB")
    left = 0.0
    for i, (name, val, _) in enumerate(components):
        ax.barh(0, abs(val), left=left, height=0.5,
                color=COLORS[i % len(COLORS)], label=f"{name}: {abs(val):.1f} MPa",
                edgecolor="white", linewidth=0.5)
        if abs(val) > 5:
            ax.text(left+abs(val)/2, 0, f"{abs(val):.1f}",
                    ha="center", va="center", fontsize=7, color="white", fontweight="bold")
        left += abs(val)
    if sigma_yield:
        ax.axvline(sigma_yield, color="#DC2626", lw=1.5, ls="--",
                   label=f"σ_yield = {sigma_yield:.0f} MPa")
    ax.set_yticks([])
    ax.set_xlabel("Stress (MPa)", fontsize=8, color="#6B7280")
    ax.set_title("Stress component breakdown", fontsize=9,
                 color="#111827", fontweight="bold")
    ax.legend(loc="upper right", fontsize=6.5, framealpha=0.9)
    for sp in ax.spines.values():
        sp.set_edgecolor("#E5E7EB"); sp.set_linewidth(0.6)
    ax.tick_params(colors="#9CA3AF", labelsize=8)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5.  UI
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="BPP Roll Diameter Calculator", layout="wide")

if "calc_mode" not in st.session_state:
    st.session_state.calc_mode = "dmin"

col_title, col_toggle = st.columns([3, 1])
with col_title:
    st.title("Analytical $D_{min}$ Calculator")
with col_toggle:
    st.write("")
    btn_label = ("🔄 Switch to: Given D → Max Stress"
                 if st.session_state.calc_mode == "dmin"
                 else "🔄 Switch to: Given Yield → Min Diameter")
    if st.button(btn_label, use_container_width=True):
        st.session_state.calc_mode = ("stress" if st.session_state.calc_mode == "dmin"
                                       else "dmin")
        st.rerun()

if st.session_state.calc_mode == "dmin":
    st.markdown("**Mode: Minimum Roll Diameter** — enter yield strength; tool calculates "
                "the minimum roller diameter to prevent plastic deformation.")
else:
    st.markdown("**Mode: Stress Analysis** — enter known roller diameter; tool calculates "
                "all stress components and the von Mises stress.")
st.markdown("---")

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.header("1. Material Properties")
E           = st.sidebar.number_input("Young's Modulus (MPa)", value=200000.0, step=1000.0)
nu          = st.sidebar.number_input("Poisson's Ratio", value=0.30, step=0.01)
sigma_yield = st.sidebar.number_input("Yield Strength (MPa)", value=350.0, step=10.0)

st.sidebar.header("2. Process Parameters")
sigma_tension = st.sidebar.number_input("Web Tension (MPa)", value=15.0, step=1.0)

if st.session_state.calc_mode == "stress":
    st.sidebar.header("3. Roll Diameter")
    D_roll = st.sidebar.number_input("Roll Diameter (mm)", value=150.0, step=5.0, min_value=1.0)

hdr = "4." if st.session_state.calc_mode == "stress" else "3."
st.sidebar.header(f"{hdr} Foil Geometry")
t = st.sidebar.number_input("Base Foil Thickness (mm)", value=0.100, step=0.001, format="%.3f")
pattern_type = st.sidebar.selectbox(
    "Engraving Pattern",
    ["Plain Foil", "Longitudinal Channels", "Transverse Channels"],
)

# Safe defaults
w1 = w2 = 0.50
H_total = 0.460
r_top = r_bot = 0.100
theta_deg = 15.0
c_max = 0.380
NA_bot = None

if pattern_type == "Longitudinal Channels":
    st.sidebar.markdown("*Longitudinal Specific Inputs:*")
    H_total   = st.sidebar.number_input("Total Outer Height ($H$) [mm]", value=0.460, step=0.01, format="%.3f")
    w1        = st.sidebar.number_input("Top Flat Width ($w_1$) [mm]", value=0.597, step=0.01, format="%.3f")
    w2        = st.sidebar.number_input("Bottom Flat Width ($w_2$) [mm]", value=0.558, step=0.01, format="%.3f")
    r_top     = st.sidebar.number_input("Top Outer Radius [mm]", value=0.200, step=0.01, format="%.3f")
    r_bot     = st.sidebar.number_input("Bottom Outer Radius [mm]", value=0.200, step=0.01, format="%.3f")
    theta_deg = st.sidebar.number_input("Web Angle ($\\theta$) [degrees]", value=15.0, step=1.0)
    c_max, NA_bot = calculate_longitudinal_cmax(H_total, t, w1, w2, r_top, r_bot, theta_deg)
    st.sidebar.markdown("---")
    st.sidebar.info(
        f"**Calculated Metrics:**\n\n"
        f"Neutral Axis: **{NA_bot:.3f} mm**\n\n"
        f"Outer Fiber ($c_{{max}}$): **{c_max:.3f} mm**"
    )

elif pattern_type == "Transverse Channels":
    st.sidebar.markdown("*Transverse (Sideways) Specific Inputs:*")
    H_total   = st.sidebar.number_input("Total Outer Height ($H$) [mm]", value=0.460, step=0.01, format="%.3f")
    w1        = st.sidebar.number_input("Top Flat Width ($w_1$) [mm]", value=0.280, step=0.01, format="%.3f")
    w2        = st.sidebar.number_input("Bottom Flat Width ($w_2$) [mm]", value=0.500, step=0.01, format="%.3f")
    r_top     = st.sidebar.number_input("Top Outer Radius ($r_{top,out}$) [mm]", value=0.100, step=0.01, format="%.3f")
    r_bot     = st.sidebar.number_input("Bottom Outer Radius ($r_{bot,out}$) [mm]", value=0.100, step=0.01, format="%.3f")
    theta_deg = st.sidebar.number_input("Web Angle ($\\theta$) [degrees]", value=15.0, step=1.0)

    # Calculate alpha and show intermediate results
    alpha, L_eff, L_web, L_arc_top, L_arc_bot, h = \
        calculate_transverse_alpha(H_total, t, r_top, r_bot, theta_deg)

    st.sidebar.markdown("---")
    st.sidebar.info(
        f"**Calculated Metrics:**\n\n"
        f"Centerline height $h$: **{h:.4f} mm**\n\n"
        f"$L_{{eff}}$: **{L_eff:.4f} mm**\n\n"
        f"  ├ $L_{{arc,bot}}$: {L_arc_bot:.4f} mm\n\n"
        f"  ├ $L_{{web}}$: {L_web:.4f} mm\n\n"
        f"  └ $L_{{arc,top}}$: {L_arc_top:.4f} mm\n\n"
        f"**α = h/(4·L_eff):** **{alpha:.4f}**\n\n"
        f"**Amplification (1+α):** **{1+alpha:.4f}**"
    )


# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_calc, tab_geom, tab_theory = st.tabs(
    ["🧮 Calculator", "📐 Geometry Preview", "📚 Theory & Rationale"]
)

# ═════════════════════════════════════════════════════════════════════════════
# Tab 1: Calculator
# ═════════════════════════════════════════════════════════════════════════════
with tab_calc:

    if st.session_state.calc_mode == "dmin":
        st.header("Results — Minimum Roll Diameter")
        if sigma_tension >= sigma_yield:
            st.error("⚠️ **Yield Condition Reached:** Web tension exceeds yield strength.")
        else:
            D_min = None
            formula_latex = ""
            if pattern_type == "Plain Foil":
                D_min = (E * t) / (sigma_yield - sigma_tension)
                formula_latex = r"D_{min}=\frac{E\cdot t}{\sigma_{yield}-\sigma_{tension}}"

            elif pattern_type == "Longitudinal Channels":
                D_min = (E * 2 * c_max) / (sigma_yield - sigma_tension)
                formula_latex = r"D_{min}=\frac{E\cdot2\cdot c_{max}}{\sigma_{yield}-\sigma_{tension}}"

            elif pattern_type == "Transverse Channels":
                alpha_val, _, _, _, _, _ = calculate_transverse_alpha(
                    H_total, t, r_top, r_bot, theta_deg)
                den = (1 - nu**2) * (sigma_yield - sigma_tension)
                if den <= 0:
                    st.error("⚠️ Combined stresses exceed yield.")
                else:
                    D_min = (E * t * (1 + alpha_val)) / den
                    formula_latex = (
                        r"D_{min}=\frac{E\cdot t\cdot(1+\alpha)}"
                        r"{(1-\nu^2)\cdot(\sigma_{yield}-\sigma_{tension})}"
                    )

            if D_min is not None:
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.metric("Minimum Roll Diameter", f"{D_min:.2f} mm")
                with c2:
                    st.markdown("**Governing Formula:**")
                    st.latex(formula_latex)

                if pattern_type == "Transverse Channels":
                    st.markdown("---")
                    st.markdown("**Intermediate values:**")
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("α (amplification factor)", f"{alpha_val:.4f}")
                    with col_b:
                        st.metric("L_eff (web path)", f"{L_eff:.4f} mm")
                    with col_c:
                        st.metric("h (centerline height)", f"{h:.4f} mm")

                    # Verification stress at D_min
                    sigma_check = E * t * (1 + alpha_val) / (D_min * (1 - nu**2)) + sigma_tension
                    st.caption(f"Verification: σ at D_min = {sigma_check:.2f} MPa "
                              f"(should equal σ_yield = {sigma_yield:.0f} MPa)")

    else:
        st.header("Results — Von Mises Stress Analysis")
        result = compute_stress_components(
            pattern_type=pattern_type, D_roll=D_roll, E=E, nu=nu,
            sigma_tension=sigma_tension, t=t,
            c_max=(c_max if pattern_type == "Longitudinal Channels" else None),
            H_total=(H_total if pattern_type == "Transverse Channels" else None),
            r_top_out=(r_top if pattern_type == "Transverse Channels" else None),
            r_bot_out=(r_bot if pattern_type == "Transverse Channels" else None),
            theta_deg=(theta_deg if pattern_type == "Transverse Channels" else None),
        )
        sv = result["sigma_vm"]
        sf = sigma_yield / sv if sv > 0 else float("inf")

        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Von Mises Stress", f"{sv:.2f} MPa")
        with c2: st.metric("Safety Factor", f"{sf:.3f}",
                            delta=("⚠ Below 1.0" if sf < 1 else "Above 1.0"),
                            delta_color="inverse")
        with c3:
            if sv >= sigma_yield: st.error("🔴 **YIELDED**")
            else: st.success(f"🟢 **ELASTIC** (below {sigma_yield:.0f} MPa)")

        st.divider()
        st.subheader("Stress Components")
        cb, cf = st.columns([2, 1])
        with cb:
            bfig = draw_stress_bar(result["components"], sv, sigma_yield)
            st.pyplot(bfig, use_container_width=True); plt.close(bfig)
        with cf:
            for name, val, latex in result["components"]:
                st.markdown(f"**{name}:** `{abs(val):.2f} MPa`")
                st.latex(latex + f"={abs(val):.2f}\\text{{ MPa}}")

            st.divider()
            if pattern_type == "Transverse Channels":
                st.markdown("**Combined formula:**")
                st.latex(r"\sigma_{vm}=\frac{E\cdot t}{D\cdot(1-\nu^2)}\cdot(1+\alpha)+\sigma_{tension}")
                st.caption(
                    f"α = {result['alpha']:.4f} · "
                    f"L_eff = {result['L_eff']:.4f} mm · "
                    f"h = {result['h']:.4f} mm"
                )
            elif pattern_type == "Longitudinal Channels":
                st.latex(r"\sigma_{vm}=\frac{E\cdot 2c_{max}}{D}+\sigma_{tension}")
            else:
                st.latex(r"\sigma_{vm}=\frac{E\cdot t}{D}+\sigma_{tension}")

        # ── Transverse channels: show detailed breakdown ─────────────────────
        if pattern_type == "Transverse Channels":
            st.divider()
            st.subheader("Detailed Transverse Channel Analysis")

            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.markdown("**Geometric Parameters**")
                data_geom = {
                    "Parameter": ["H (outer height)", "t (thickness)", "h = H−t",
                                  "R_cl,top", "R_cl,bot", "θ (web angle)",
                                  "L_web", "L_arc,top", "L_arc,bot", "L_eff"],
                    "Value": [f"{H_total:.4f} mm", f"{t:.4f} mm", f"{result['h']:.4f} mm",
                              f"{max(r_top - t/2, 0):.4f} mm", f"{max(r_bot - t/2, 0):.4f} mm",
                              f"{theta_deg:.1f}°",
                              f"{result['L_web']:.4f} mm", f"{result['L_arc_top']:.4f} mm",
                              f"{result['L_arc_bot']:.4f} mm", f"{result['L_eff']:.4f} mm"],
                }
                st.table(data_geom)

            with col_d2:
                st.markdown("**Stress Buildup**")
                sigma_plain = E * t / D_roll
                sigma_ps = result["sigma_ps_base"]
                sigma_h = result["sigma_hinge"]
                data_stress = {
                    "Step": [
                        "1. Plain foil (E·t/D)",
                        "2. + Poisson correction (×1/(1−ν²))",
                        "3. + Hinge amplification (×α)",
                        "4. + Web tension",
                        "**Total (von Mises)**"
                    ],
                    "Stress (MPa)": [
                        f"{sigma_plain:.2f}",
                        f"{sigma_ps:.2f}",
                        f"+{sigma_h:.2f}",
                        f"+{sigma_tension:.2f}",
                        f"**{sv:.2f}**"
                    ],
                    "Cumulative (MPa)": [
                        f"{sigma_plain:.2f}",
                        f"{sigma_ps:.2f}",
                        f"{sigma_ps + sigma_h:.2f}",
                        f"{sv:.2f}",
                        f"{sv:.2f}"
                    ],
                }
                st.table(data_stress)

        # ── Stress vs. Roll Diameter curve ───────────────────────────────────
        st.subheader("Stress vs. Roll Diameter")
        D_range = np.linspace(max(10, D_roll*0.2), D_roll*3, 200)
        vm_curve = [
            compute_stress_components(
                pattern_type=pattern_type, D_roll=Di, E=E, nu=nu,
                sigma_tension=sigma_tension, t=t,
                c_max=(c_max if pattern_type == "Longitudinal Channels" else None),
                H_total=(H_total if pattern_type == "Transverse Channels" else None),
                r_top_out=(r_top if pattern_type == "Transverse Channels" else None),
                r_bot_out=(r_bot if pattern_type == "Transverse Channels" else None),
                theta_deg=(theta_deg if pattern_type == "Transverse Channels" else None),
            )["sigma_vm"]
            for Di in D_range
        ]
        fs, asc = plt.subplots(figsize=(7, 3.2))
        fs.patch.set_facecolor("white"); asc.set_facecolor("#F9FAFB")
        asc.plot(D_range, vm_curve, color="#2563EB", lw=2, label="Von Mises stress")

        # Also plot plain foil reference
        plain_curve = [E * t / Di + sigma_tension for Di in D_range]
        asc.plot(D_range, plain_curve, color="#9CA3AF", lw=1, ls=":",
                 label="Plain foil reference")

        asc.axhline(sigma_yield, color="#DC2626", lw=1.4, ls="--",
                    label=f"σ_yield = {sigma_yield:.0f} MPa")
        asc.axvline(D_roll, color="#D97706", lw=1.2, ls=":",
                    label=f"Current D = {D_roll:.1f} mm")
        asc.scatter([D_roll], [sv], color="#D97706", zorder=5, s=60)
        asc.set_xlabel("Roll Diameter (mm)", fontsize=8, color="#6B7280")
        asc.set_ylabel("Von Mises Stress (MPa)", fontsize=8, color="#6B7280")
        asc.set_title("Von Mises stress vs. roll diameter", fontsize=9,
                      color="#111827", fontweight="bold")
        asc.legend(fontsize=7.5)
        asc.grid(True, color="#E5E7EB", lw=0.4)
        for sp in asc.spines.values(): sp.set_edgecolor("#E5E7EB"); sp.set_linewidth(0.6)
        asc.tick_params(colors="#9CA3AF", labelsize=8)
        fs.tight_layout()
        st.pyplot(fs, use_container_width=True); plt.close(fs)


# ═════════════════════════════════════════════════════════════════════════════
# Tab 2: Geometry Preview
# ═════════════════════════════════════════════════════════════════════════════
with tab_geom:
    st.header("Cross-section geometry preview")
    st.markdown(
        "Generated analytically from sidebar inputs. "
        "Adjacent periods shown at reduced opacity for context."
    )

    col_fig, col_info = st.columns([2, 1])

    with col_fig:
        fig = draw_geometry_preview(
            pattern_type=pattern_type, t=t,
            H_total=H_total, w1=w1, w2=w2,
            r_top=r_top, r_bot=r_bot, theta_deg=theta_deg,
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
            st.metric("Top flat w₁", f"{w1:.3f} mm")
            st.metric("Bottom flat w₂", f"{w2:.3f} mm")
            st.metric("Web angle θ", f"{theta_deg:.1f}°")
            st.metric("Top fillet radius", f"{r_top:.3f} mm")
            st.metric("Bottom fillet radius", f"{r_bot:.3f} mm")
            if NA_bot is not None:
                st.divider()
                st.metric("Neutral axis from bottom", f"{NA_bot:.3f} mm")
                st.metric("c_max (outer fibre)", f"{c_max:.3f} mm")
                pct = (NA_bot / H_total) * 100
                st.caption(
                    f"NA at {pct:.1f}% of H — "
                    + ("shifted toward bottom" if pct < 50 else "shifted toward top")
                )

        elif pattern_type == "Transverse Channels":
            st.metric("Total height H", f"{H_total:.3f} mm")
            st.metric("Wall thickness t", f"{t:.3f} mm")
            st.metric("Top flat w₁", f"{w1:.3f} mm")
            st.metric("Bottom flat w₂", f"{w2:.3f} mm")
            st.metric("Web angle θ", f"{theta_deg:.1f}°")
            st.metric("Top fillet radius", f"{r_top:.3f} mm")
            st.metric("Bottom fillet radius", f"{r_bot:.3f} mm")
            st.divider()
            st.metric("α (amplification factor)", f"{alpha:.4f}")
            st.metric("L_eff (web path)", f"{L_eff:.4f} mm")
            st.caption(
                "For transverse channels the governing neutral axis is the local "
                "flat-panel NA at t/2 from the bottom surface. The corrugation "
                "amplifies the local bending stress by factor (1+α)."
            )

        st.markdown("---")
        st.caption(
            "Red dashed = neutral axis · "
            "Blue fill = foil wall material · "
            "Lighter border = adjacent corrugation periods"
        )


# ═════════════════════════════════════════════════════════════════════════════
# Tab 3: Theory
# ═════════════════════════════════════════════════════════════════════════════
with tab_theory:
    st.header("Mechanics & Rationale")
    st.markdown(
        "This tool uses the **Principle of Superposition** to determine the minimum "
        "roller diameter preventing plastic deformation. Behaviour varies significantly "
        "with channel orientation relative to the transport direction."
    )

    st.subheader("Calculation modes")
    st.markdown(
        "- **Given Yield → D_min:** finds the minimum diameter such that total stress "
        "does not exceed yield strength.\n"
        "- **Given D → Max Stress:** for a known diameter, computes all stress components "
        "and combines them via the **von Mises** criterion."
    )

    st.subheader("1. Plain Foil")
    st.markdown("Classic Euler-Bernoulli beam theory. Bending + global tension.")
    st.latex(r"\sigma_{total}=\frac{E\cdot t}{D}+\sigma_{tension}")

    st.subheader("2. Longitudinal Channels")
    st.markdown(
        "Foil bends through the full corrugated cross-section. Base thickness $t$ "
        "replaced by $c_{max}$ — outer fibre distance from NA — computed via 1-D "
        "centreline length-weighted integration."
    )
    st.latex(r"\sigma_{total}=\frac{E\cdot2c_{max}}{D}+\sigma_{tension}")

    st.subheader("3. Transverse (Sideways) Channels")
    st.markdown(
        """
        When channels run **parallel to the roll axis** (perpendicular to the bending direction),
        the corrugation cross-section **deforms** during wrapping. The stress is governed by
        **local bending of the flat sections** (thickness $t$), amplified by two effects:
        """
    )

    st.markdown("#### Factor 1: Plane-Strain Constraint (Poisson Effect)")
    st.markdown(
        "The corrugation webs act as stiffeners in the transverse direction, preventing "
        "free Poisson contraction during bending. This creates a plane-strain-like condition:"
    )
    st.latex(r"\sigma_{ps} = \frac{E \cdot t}{D \cdot (1-\nu^2)}")

    st.markdown("#### Factor 2: Hinge Amplification (α)")
    st.markdown(
        """
        The stiff web–arc junctions (hinges) partially resist the cross-section distortion
        during wrapping. This creates **chord rotation** in the web, which transmits additional
        moment to the flat sections, increasing local curvature beyond $\\kappa_{roll}$.

        The amplification factor α is derived from:
        1. **Wrapping kinematics** — chord rotation ψ ∝ h/R
        2. **Slope-deflection frame analysis** — moment ∝ EI/L_eff
        3. **Beam-on-rigid-foundation contact** — curvature concentration

        Combined result:
        """
    )
    st.latex(r"\alpha = \frac{h}{4 \cdot L_{eff}} = \frac{H - t}{4 \cdot (L_{arc,bot} + L_{web} + L_{arc,top})}")

    st.markdown("#### Complete Formula")
    st.latex(r"\boxed{\sigma_{max} = \frac{E \cdot t}{D \cdot (1-\nu^2)} \cdot (1 + \alpha) + \sigma_{tension}}")

    st.markdown("#### Minimum Diameter (inverted)")
    st.latex(r"D_{min} = \frac{E \cdot t \cdot (1+\alpha)}{(1-\nu^2) \cdot (\sigma_{yield} - \sigma_{tension})}")

    st.markdown("#### Geometric Sub-Formulas")
    st.latex(r"h = H - t")
    st.latex(r"R_{cl,top} = r_{top,out} - t/2, \quad R_{cl,bot} = r_{bot,out} - t/2")
    st.latex(r"\Delta y_{top} = R_{cl,top}(1-\sin\theta), \quad \Delta y_{bot} = R_{cl,bot}(1-\sin\theta)")
    st.latex(r"L_{web} = \frac{h - \Delta y_{top} - \Delta y_{bot}}{\cos\theta}")
    st.latex(r"L_{arc,top} = R_{cl,top}\left(\frac{\pi}{2}-\theta\right), \quad L_{arc,bot} = R_{cl,bot}\left(\frac{\pi}{2}-\theta\right)")
    st.latex(r"L_{eff} = L_{arc,bot} + L_{web} + L_{arc,top}")

    st.markdown("#### Physical Interpretation of α = h/(4·L_eff)")
    st.markdown(
        """
        | Component | Physical meaning |
        |-----------|-----------------|
        | **h** (numerator) | Corrugation height = radial lever arm driving chord rotation |
        | **L_eff** (denominator) | Web flexibility path = longer/softer web reduces transmitted moment |
        | **Factor 1/4** | Combined coefficient from: geometry projection (sin θ), moment distribution in portal frame, and contact zone averaging |
        """
    )

    st.markdown("#### Validation Against FEA (Slides 26–27)")
    st.markdown(
        """
        | | Profile 1 (Slide 26) | Profile 2 (Slide 27) |
        |---|:---:|:---:|
        | H / t / θ | 0.460 / 0.100 / 15° | 0.460 / 0.100 / 15° |
        | w₁ / w₂ | 0.280 / 0.500 | 0.500 / 0.280 |
        | r_top / r_bot | 0.100 / 0.100 | 0.050 / 0.050 |
        | L_eff | 0.427 mm | 0.373 mm |
        | **α** | **0.211** | **0.241** |
        | **σ_predicted** | **192.4 MPa** | **196.9 MPa** |
        | **σ_FEA** | **192 MPa** | **197 MPa** |
        | **Error** | **0.2%** | **0.05%** |
        """
    )

    st.markdown("#### Limiting Behaviors")
    st.markdown(
        """
        | Condition | Result | Physical meaning |
        |-----------|--------|-----------------|
        | h → 0 (no corrugation) | α → 0 | Reduces to plain foil |
        | L_eff → ∞ (very flexible web) | α → 0 | No coupling between flats |
        | R_cl → 0 (sharp corners) | α = cos θ / 4 | Maximum amplification |
        | θ → 0° (vertical web) | α = 1/4 (maximum) | Stiffest web geometry |
        """
    )

    st.markdown("---")
    st.header("Variable Nomenclature")
    st.markdown(
        """
        **Material & process:** $E$ Young's modulus · $\\nu$ Poisson ratio ·
        $\\sigma_{yield}$ yield strength · $\\sigma_{tension}$ web tension ·
        $\\sigma_{vm}$ von Mises stress

        **Geometry:** $t$ foil thickness · $H$ total outer height ·
        $h$ centerline height ($H-t$) ·
        $r_{top,out}/r_{bot,out}$ outer fillet radii ·
        $R_{cl}$ centerline fillet radius ($r_{out} - t/2$) ·
        $c_{max}$ outer-fibre distance (longitudinal only) ·
        $w_1$ top flat width (center-to-center) ·
        $w_2$ bottom flat width (center-to-center) ·
        $\\theta$ web angle from vertical ·
        $L_{eff}$ effective web path length ·
        $\\alpha$ corrugation amplification factor
        """
    )

    st.markdown("---")
    st.header("Derivation Sources")
    st.markdown(
        """
        The transverse channel formula is derived from established structural mechanics:

        | Formula | Source | Role |
        |---------|--------|------|
        | Euler-Bernoulli bending | Timoshenko, *Strength of Materials* | Base stress E·t/D |
        | Plane-strain stiffening | Timoshenko & Goodier, *Theory of Elasticity* | Factor 1/(1−ν²) |
        | Wrapping chord rotation | Shell kinematics (Ventsel & Krauthammer) | Driving displacement ψ ∝ h/R |
        | Slope-deflection equations | Hibbeler, *Structural Analysis*, Ch. 11 | Frame moment distribution → 1/L_eff |
        | Beam on rigid foundation | Johnson, *Contact Mechanics*, Ch. 5 | Contact constraint → curvature amplification |
        | Combined coefficient 1/4 | Portal frame moment distribution (Norris & Wilbur) | Validated against FEA |
        """
    )
