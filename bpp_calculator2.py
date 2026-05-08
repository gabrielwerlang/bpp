import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Analytical Solvers (unchanged)
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
    """Calculate the corrugation amplification factor α for transverse channels."""
    theta = np.radians(theta_deg)
    h = H - t
    R_cl_top = max(r_top_out - t / 2, 0.0)
    R_cl_bot = max(r_bot_out - t / 2, 0.0)
    phi = np.pi / 2 - theta
    Dy_top = R_cl_top * (1 - np.sin(theta))
    Dy_bot = R_cl_bot * (1 - np.sin(theta))
    L_web_num = h - Dy_top - Dy_bot
    if L_web_num < 0:
        L_web_num = 0.0
    L_web = L_web_num / np.cos(theta) if np.cos(theta) > 1e-10 else L_web_num
    L_arc_top = R_cl_top * phi if phi > 0 else 0.0
    L_arc_bot = R_cl_bot * phi if phi > 0 else 0.0
    L_eff = L_arc_bot + L_web + L_arc_top
    if L_eff < 1e-10:
        L_eff = h
    alpha = h / (4.0 * L_eff)
    return alpha, L_eff, L_web, L_arc_top, L_arc_bot, h


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Geometry builder — continuous corrugated thin-wall profile
# ─────────────────────────────────────────────────────────────────────────────

def _build_corrugation(H, t, w1, w2, r_top_out, r_bot_out, theta_deg,
                       n_periods=3, pts_per_arc=20):
    """
    Build continuous outer and inner profiles for a corrugated thin-wall section.

    Profile structure (one period, left to right):
      bottom_flat → arc1(up) → web_up → arc2(flat) → top_flat → arc3(down) → web_down → arc4(flat)

    The outer profile traces the bottom (y=0) and top (y=H) surfaces.
    The inner profile traces y=t and y=H-t with reduced fillet radii.
    Arc centers are shared so the wall thickness is approximately uniform.

    Returns
    -------
    outer_x, outer_y : ndarray
    inner_x, inner_y : ndarray
    info : dict with period width and central-period flat positions
    """
    theta = np.radians(theta_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # Constrain radii to feasible range
    r_top_out = min(max(r_top_out, 0.001), H * 0.45)
    r_bot_out = min(max(r_bot_out, 0.001), H * 0.45)
    r_top_in = max(r_top_out - t, 0.005)
    r_bot_in = max(r_bot_out - t, 0.005)

    # Vertical arc extents (outer)
    dy_bot = r_bot_out * (1 - sin_t)
    dy_top = r_top_out * (1 - sin_t)
    web_vert = H - dy_bot - dy_top

    if web_vert < 0:
        scale = (H * 0.9) / (dy_bot + dy_top) if (dy_bot + dy_top) > 0 else 1.0
        r_top_out *= scale
        r_bot_out *= scale
        r_top_in = max(r_top_out - t, 0.005)
        r_bot_in = max(r_bot_out - t, 0.005)
        dy_bot = r_bot_out * (1 - sin_t)
        dy_top = r_top_out * (1 - sin_t)
        web_vert = H - dy_bot - dy_top

    dx_web = web_vert * np.tan(theta) if cos_t > 1e-10 else 0.0

    # Period width
    period = w2 + 2 * (r_bot_out * cos_t + dx_web + r_top_out * cos_t) + w1

    # Arc center y-coordinates (shared by inner & outer)
    cy_bot = r_bot_out
    cy_top = H - r_top_out

    def trace(r_bot, r_top, y_bot, y_top, x0):
        """Trace one surface (outer or inner) over n_periods."""
        xs, ys = [], []
        x = x0
        for _ in range(n_periods):
            # 1. Bottom flat
            xs.append(x); ys.append(y_bot)
            x += w2
            xs.append(x); ys.append(y_bot)

            # 2. Arc 1: CCW from -π/2 to -θ (bottom → up-web)
            cx1 = x
            for a in np.linspace(-np.pi / 2, -theta, pts_per_arc + 1)[1:]:
                xs.append(cx1 + r_bot * np.cos(a))
                ys.append(cy_bot + r_bot * np.sin(a))

            # 3. Web up — straight line to arc 2 start
            cx2 = cx1 + r_bot_out * cos_t + dx_web + r_top_out * cos_t
            xs.append(cx2 + r_top * np.cos(np.pi - theta))
            ys.append(cy_top + r_top * np.sin(np.pi - theta))

            # 4. Arc 2: (π−θ) → π/2 (web → top flat)
            for a in np.linspace(np.pi - theta, np.pi / 2, pts_per_arc + 1)[1:]:
                xs.append(cx2 + r_top * np.cos(a))
                ys.append(cy_top + r_top * np.sin(a))
            x = cx2

            # 5. Top flat
            xs.append(x); ys.append(y_top)
            x += w1
            xs.append(x); ys.append(y_top)

            # 6. Arc 3: π/2 → θ (top flat → down-web)
            cx3 = x
            for a in np.linspace(np.pi / 2, theta, pts_per_arc + 1)[1:]:
                xs.append(cx3 + r_top * np.cos(a))
                ys.append(cy_top + r_top * np.sin(a))

            # 7. Web down — straight line to arc 4 start
            cx4 = cx3 + r_top_out * cos_t + dx_web + r_bot_out * cos_t
            xs.append(cx4 + r_bot * np.cos(np.pi + theta))
            ys.append(cy_bot + r_bot * np.sin(np.pi + theta))

            # 8. Arc 4: CCW (π+θ) → 3π/2 (down-web → bottom flat)
            for a in np.linspace(np.pi + theta, 3 * np.pi / 2, pts_per_arc + 1)[1:]:
                xs.append(cx4 + r_bot * np.cos(a))
                ys.append(cy_bot + r_bot * np.sin(a))
            x = cx4

        # Final point to close the last flat
        xs.append(x); ys.append(y_bot)
        return np.array(xs), np.array(ys)

    # Center the span so that the middle period is around x ≈ 0
    x_start = -period * n_periods / 2

    outer_x, outer_y = trace(r_bot_out, r_top_out, 0.0, H, x_start)
    inner_x, inner_y = trace(r_bot_in, r_top_in, t, H - t, x_start)

    # Central period element positions (period index 1 for n_periods=3)
    x_cp = x_start + period
    info = {
        'period': period,
        'x_bf_left': x_cp,
        'x_bf_right': x_cp + w2,
        'x_tf_left': x_cp + w2 + r_bot_out * cos_t + dx_web + r_top_out * cos_t,
        'x_tf_right': x_cp + w2 + r_bot_out * cos_t + dx_web + r_top_out * cos_t + w1,
    }
    return outer_x, outer_y, inner_x, inner_y, info


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Geometry Preview figure
# ─────────────────────────────────────────────────────────────────────────────

_AX_BG = "#F9FAFB"


def draw_geometry_preview(pattern_type, t,
                          H_total=None, w1=None, w2=None,
                          r_top=None, r_bot=None,
                          theta_deg=15.0,
                          NA_bot=None, c_max=None):

    FILL    = "#DBEAFE"
    STROKE  = "#1D4ED8"
    NA_C    = "#DC2626"
    DIM_C   = "#6B7280"
    LABEL_C = "#111827"

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
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        kw = dict(fontsize=fs, color=DIM_C)
        if orient == "v":
            ax.text(mx + off, my, label, va="center", ha="left", **kw)
        else:
            ax.text(mx, my + off, label, va="bottom", ha="center", **kw)

    # ── PLAIN FOIL ──────────────────────────────────────────────────────────
    if pattern_type == "Plain Foil":
        L = 3.0
        ax.fill([-L / 2, L / 2, L / 2, -L / 2], [0, 0, t, t],
                facecolor=FILL, edgecolor=STROKE, lw=1.5)
        ax.axhline(t / 2, color=NA_C, lw=1.2, ls="--")
        ax.text(0, t / 2 + t * 0.2, "Neutral axis", fontsize=7,
                color=NA_C, ha="center", va="bottom")
        dim_arrow(L / 2 + 0.1, 0, L / 2 + 0.1, t,
                  f"t = {t:.3f} mm", "v", off=0.06)
        ax.set_xlim(-L / 2 - 0.2, L / 2 + 0.55)
        ax.set_ylim(-t * 2, t * 5)
        ax.set_title("Cross-section — Plain foil", fontsize=9,
                     color=LABEL_C, fontweight="bold", pad=8)

    # ── LONGITUDINAL / TRANSVERSE CHANNELS ───────────────────────────────────
    elif pattern_type in ("Longitudinal Channels", "Transverse Channels"):
        try:
            ox, oy, ix, iy, info = _build_corrugation(
                H_total, t, w1, w2, r_top, r_bot, theta_deg, n_periods=3)

            # --- Draw filled wall (material between outer & inner) ---
            wall_x = np.concatenate([ox, ix[::-1]])
            wall_y = np.concatenate([oy, iy[::-1]])
            ax.fill(wall_x, wall_y, facecolor=FILL, edgecolor='none', zorder=2)
            ax.plot(ox, oy, color=STROKE, lw=1.5, zorder=3)
            ax.plot(ix, iy, color=STROKE, lw=0.8, zorder=3)

            period = info['period']
            x_bf_l = info['x_bf_left']
            x_bf_r = info['x_bf_right']
            x_tf_l = info['x_tf_left']
            x_tf_r = info['x_tf_right']

            # --- Neutral axis ---
            if pattern_type == "Longitudinal Channels" and NA_bot is not None:
                ax.axhline(NA_bot, color=NA_C, lw=1.2, ls="--", zorder=4)
                ax.text(ox.min() - period * 0.02, NA_bot, "NA", fontsize=7,
                        color=NA_C, va="center", ha="right")
            elif pattern_type == "Transverse Channels":
                ax.axhline(t / 2, color=NA_C, lw=1.1, ls="--", zorder=4)
                ax.text(ox.min() - period * 0.02, t / 2, "NA (t/2)",
                        fontsize=6, color=NA_C, va="center", ha="right")

            # --- Dimension arrows ---
            x_dim_right = ox.max() + period * 0.05

            # H (total height)
            dim_arrow(x_dim_right, 0, x_dim_right, H_total,
                      f"H = {H_total:.3f} mm", "v", off=period * 0.02)

            # w1 (top flat)
            y_w1 = H_total + H_total * 0.08
            dim_arrow(x_tf_l, y_w1, x_tf_r, y_w1,
                      f"w₁ = {w1:.2f} mm", "h", off=H_total * 0.06)

            # w2 (bottom flat)
            y_w2 = -H_total * 0.08
            dim_arrow(x_bf_l, y_w2, x_bf_r, y_w2,
                      f"w₂ = {w2:.2f} mm", "h", off=-H_total * 0.10)

            # c_max (longitudinal only)
            if (pattern_type == "Longitudinal Channels"
                    and NA_bot is not None and c_max is not None):
                x_c = x_dim_right + period * 0.12
                y_far = (NA_bot + c_max) if (H_total - NA_bot) > NA_bot \
                    else (NA_bot - c_max)
                dim_arrow(x_c, NA_bot, x_c, y_far,
                          f"c_max\n{c_max:.3f} mm", "v", off=period * 0.02)

            # Axis limits
            margin_x = period * 0.12
            ax.set_xlim(ox.min() - margin_x, ox.max() + margin_x + period * 0.25)
            ax.set_ylim(-H_total * 0.35, H_total * 1.55)

        except Exception as e:
            ax.text(0.5, 0.5, f"Geometry error:\n{e}",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=8, color="red")

        title_suffix = ("Longitudinal channels" if pattern_type == "Longitudinal Channels"
                        else "Transverse channels (sideways)")
        ax.set_title(f"Cross-section — {title_suffix}", fontsize=9,
                     color=LABEL_C, fontweight="bold", pad=8)

    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, color="#E5E7EB", lw=0.35, zorder=0)
    ax.set_xlabel("Width (mm)", fontsize=8, color=DIM_C)
    ax.set_ylabel("Height (mm)", fontsize=8, color=DIM_C)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Stress Analysis helpers (unchanged)
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
        alpha, L_eff, L_web, L_arc_top, L_arc_bot, h = \
            calculate_transverse_alpha(H_total, t, r_top_out, r_bot_out, theta_deg)
        sigma_ps_base = (E * t) / (D_roll * (1 - nu**2))
        sigma_hinge = sigma_ps_base * alpha
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
            ax.text(left + abs(val) / 2, 0, f"{abs(val):.1f}",
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

        st.subheader("Stress vs. Roll Diameter")
        D_range = np.linspace(max(10, D_roll * 0.2), D_roll * 3, 200)
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
        "Shows 3 corrugation periods as a continuous thin wall."
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
            "Dark blue lines = outer/inner surfaces"
        )


# ═════════════════════════════════════════════════════════════════════════════
# Tab 3: Theory (unchanged)
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
    st.latex(r"\sigma_{ps} = \frac{E \cdot t}{D \cdot (1-\nu^2)}")

    st.markdown("#### Factor 2: Hinge Amplification (α)")
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
        $w_1$ top flat width ·
        $w_2$ bottom flat width ·
        $\\theta$ web angle from vertical ·
        $L_{eff}$ effective web path length ·
        $\\alpha$ corrugation amplification factor
        """
    )
