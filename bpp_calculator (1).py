import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Analytical Solvers
# ─────────────────────────────────────────────────────────────────────────────

def _trapz(y, x):
    try:
        return np.trapezoid(y, x=x)
    except AttributeError:
        return np.trapz(y, x=x)


def calculate_unit_H(w_top, h_channel, theta_deg):
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
    A  = _trapz(np.ones_like(x), s)
    B  = _trapz(y, s)
    C  = _trapz(y ** 2, s)
    Dx = _trapz(x1, s[:n])
    Exy= _trapz(x1 * y1, s[:n])
    solution = np.linalg.solve(np.array([[A,-B],[-B,C]]), np.array([-Dx, Exy]))
    return solution[1]


def calculate_longitudinal_cmax(H, t, w1, w2, r_top_out, r_bot_out, theta_deg):
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
    C_arc_top  = h - R_cl_top + R_cl_top * np.sin(phi) / phi
    C_arc_bot  = R_cl_bot - R_cl_bot * np.sin(phi) / phi
    C_web      = (dy_bot + (h - dy_top)) / 2
    num = (L_flat_top * C_flat_top + L_flat_bot * C_flat_bot
           + 2 * L_arc_top * C_arc_top + 2 * L_arc_bot * C_arc_bot
           + 2 * L_web * C_web)
    den = L_flat_top + L_flat_bot + 2*L_arc_top + 2*L_arc_bot + 2*L_web
    NA_cl  = num / den
    NA_bot = NA_cl + t / 2
    c_max  = max(NA_bot, H - NA_bot)
    return c_max, NA_bot


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Geometry builders — correct closed-wall cross-section polygon
# ─────────────────────────────────────────────────────────────────────────────

def _arc(cx, cy, r, a0, a1, n=40):
    """Points along a circular arc from a0 to a1 (radians, both inclusive)."""
    if a1 < a0:
        a1 += 2 * np.pi
    ang = np.linspace(a0, a1, n)
    return cx + r * np.cos(ang), cy + r * np.sin(ang)


def _profile_wall(H, t, w1, w2, r_top_out, r_bot_out, theta_deg):
    """
    Closed polygon of ONE corrugation period showing the foil wall material.

    Coordinate convention
    ---------------------
    Y = 0  : outer bottom surface (valley floor)
    Y = H  : outer top surface (peak top)
    w1     : top flat width — distance between the two top arc centres
    w2     : bottom flat width — distance between the two bottom arc centres
    theta  : web angle from vertical (deg); positive = web leans outward at bottom

    Returns
    -------
    ox, oy   : outer profile (CCW, one period)
    ix, iy   : inner profile (CCW, one period)
    wall_x, wall_y : closed wall polygon (fill this to show material)
    """
    theta  = np.radians(theta_deg)
    cos_t  = np.cos(theta)
    sin_t  = np.sin(theta)

    # Outer arc centres
    bot_cx = w2 / 2;   bot_cy = r_bot_out
    top_cx = w1 / 2;   top_cy = H - r_top_out

    # Web outward normal (pointing away from channel interior, rightward for right web)
    # Web direction (upward): (-sin_t, cos_t)
    # Outward normal (90° CW from web direction): (cos_t, sin_t)
    wnx = cos_t;  wny = sin_t

    # Tangency points (outer surface ↔ web)
    bx_o = bot_cx + r_bot_out * wnx;  by_o = bot_cy + r_bot_out * wny
    tx_o = top_cx + r_top_out * wnx;  ty_o = top_cy + r_top_out * wny

    bot_wa_o = np.arctan2(by_o - bot_cy, bx_o - bot_cx)
    top_wa_o = np.arctan2(ty_o - top_cy, tx_o - top_cx)

    def right_outer():
        xs, ys = [0.0, bot_cx], [0.0, 0.0]           # bottom centre → flat tangency
        a0, a1 = -np.pi / 2, bot_wa_o                  # bot arc: flat → web
        if a1 < a0: a1 += 2 * np.pi
        ax, ay = _arc(bot_cx, bot_cy, r_bot_out, a0, a1)
        xs.extend(ax); ys.extend(ay)
        xs.append(tx_o); ys.append(ty_o)               # web straight line
        a0, a1 = top_wa_o, np.pi / 2                   # top arc: web → flat
        if a1 < a0: a1 += 2 * np.pi
        ax, ay = _arc(top_cx, top_cy, r_top_out, a0, a1)
        xs.extend(ax); ys.extend(ay)
        xs.append(0.0); ys.append(H)                   # flat → top centre
        return np.array(xs), np.array(ys)

    rx_o, ry_o = right_outer()
    # Mirror: full outer = left half (x negated, reversed) + right half
    ox = np.concatenate([-rx_o[::-1], rx_o[1:]])
    oy = np.concatenate([ ry_o[::-1], ry_o[1:]])

    # ── Inner surface ─────────────────────────────────────────────────────────
    # Same arc centres; radii shrunk by t.  Arc tangency with flat surfaces
    # still lands at the correct y = t (bottom) and y = H-t (top) because
    # distance from centre to those surfaces equals r_out − t = r_in.
    r_bot_in = max(r_bot_out - t, 1e-4)
    r_top_in = max(r_top_out - t, 1e-4)

    bx_i = bot_cx + r_bot_in * wnx;  by_i = bot_cy + r_bot_in * wny
    tx_i = top_cx + r_top_in * wnx;  ty_i = top_cy + r_top_in * wny

    bot_wa_i = np.arctan2(by_i - bot_cy, bx_i - bot_cx)
    top_wa_i = np.arctan2(ty_i - top_cy, tx_i - top_cx)

    def right_inner():
        xs, ys = [0.0, bot_cx], [t, t]
        a0, a1 = -np.pi / 2, bot_wa_i
        if a1 < a0: a1 += 2 * np.pi
        ax, ay = _arc(bot_cx, bot_cy, r_bot_in, a0, a1)
        xs.extend(ax); ys.extend(ay)
        xs.append(tx_i); ys.append(ty_i)
        a0, a1 = top_wa_i, np.pi / 2
        if a1 < a0: a1 += 2 * np.pi
        ax, ay = _arc(top_cx, top_cy, r_top_in, a0, a1)
        xs.extend(ax); ys.extend(ay)
        xs.append(0.0); ys.append(H - t)
        return np.array(xs), np.array(ys)

    rx_i, ry_i = right_inner()
    ix = np.concatenate([-rx_i[::-1], rx_i[1:]])
    iy = np.concatenate([ ry_i[::-1], ry_i[1:]])

    # Closed wall polygon: outer CCW then inner CW (reversed)
    wall_x = np.concatenate([ox, ix[::-1], [ox[0]]])
    wall_y = np.concatenate([oy, iy[::-1], [oy[0]]])
    return ox, oy, ix, iy, wall_x, wall_y


def _transverse_wall(w_top, h_channel, theta_deg, t):
    """
    Transverse channel cross-section.
    Same corrugated profile shape; w2 derived from theta so the angled
    web connects peak (w1=w_top) to valley at the correct angle.
    """
    theta    = np.radians(theta_deg)
    half_bot = w_top / 2 + h_channel * np.tan(theta)
    w2_trans = 2 * half_bot
    r_fillet = min(w_top * 0.15, h_channel * 0.25, max(t * 1.2, 0.02))
    r_fillet = max(r_fillet, t * 0.5)
    H_outer  = h_channel + t
    return _profile_wall(H_outer, t, w_top, w2_trans, r_fillet, r_fillet, theta_deg)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Geometry Preview figure
# ─────────────────────────────────────────────────────────────────────────────

def draw_geometry_preview(pattern_type, t,
                           H_total=None, w1=None, w2=None,
                           r_top=None, r_bot=None,
                           theta_deg=15.0,
                           h_channel=None, w_top=None,
                           NA_bot=None, c_max=None):

    FILL   = "#DBEAFE"
    STROKE = "#1D4ED8"
    GHOST  = "#93C5FD"
    NA_C   = "#DC2626"
    DIM_C  = "#6B7280"
    LABEL_C= "#111827"

    fig, ax = plt.subplots(figsize=(7, 4.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#F9FAFB")
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
            ox, oy, ix, iy, wx, wy = _profile_wall(
                H_total, t, w1, w2, r_top, r_bot, theta_deg)
            period = ox.max() - ox.min()

            for shift in [-period, period]:
                ax.fill(wx+shift, wy, facecolor=FILL, edgecolor=GHOST,
                        lw=0.7, alpha=0.35, zorder=1)
            ax.fill(wx, wy, facecolor=FILL, edgecolor=STROKE, lw=1.6, zorder=2)

            xmin, xmax = ox.min(), ox.max()

            if NA_bot is not None:
                ax.hlines(NA_bot, xmin-0.1, xmax+0.05,
                          color=NA_C, lw=1.3, ls="--", zorder=4)
                ax.text(xmin-0.12, NA_bot, "NA", fontsize=6.5,
                        color=NA_C, va="center", ha="right")

            if NA_bot is not None and c_max is not None:
                y_far = (NA_bot + c_max) if (H_total - NA_bot) > NA_bot else (NA_bot - c_max)
                dim_arrow(xmax+0.10, NA_bot, xmax+0.10, y_far,
                          f"c_max\n{c_max:.3f} mm", "v", off=0.03)

            dim_arrow(xmax+0.36, 0, xmax+0.36, H_total,
                      f"H = {H_total:.3f} mm", "v", off=0.03)
            dim_arrow(-w1/2, H_total+0.07, w1/2, H_total+0.07,
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
            ox, oy, ix, iy, wx, wy = _transverse_wall(
                w_top, h_channel, theta_deg, t)
            period = ox.max() - ox.min()
            H_outer = h_channel + t

            for shift in [-period, period]:
                ax.fill(wx+shift, wy, facecolor=FILL, edgecolor=GHOST,
                        lw=0.7, alpha=0.35, zorder=1)
            ax.fill(wx, wy, facecolor=FILL, edgecolor=STROKE, lw=1.6, zorder=2)

            xmin, xmax = ox.min(), ox.max()

            ax.hlines(t/2, xmin-0.1, xmax+0.05,
                      color=NA_C, lw=1.1, ls="--", zorder=3)
            ax.text(xmin-0.12, t/2, f"NA\n(flat, t/2\n={t/2:.3f})",
                    fontsize=6, color=NA_C, va="center", ha="right")

            dim_arrow(xmax+0.10, 0, xmax+0.10, H_outer,
                      f"H = {H_outer:.3f}\nmm (h+t)", "v", off=0.03)
            dim_arrow(-w_top/2, H_outer+0.07, w_top/2, H_outer+0.07,
                      f"w₁ = {w_top:.2f} mm", "h", off=H_outer*0.09)

            theta_r = np.radians(theta_deg)
            web_mid_x = w_top/2 + (h_channel/2)*np.tan(theta_r) + 0.04
            ax.text(web_mid_x, H_outer/2, f"θ = {theta_deg:.0f}°",
                    fontsize=7, color=DIM_C, va="center", ha="left",
                    rotation=-theta_deg)

            margin = period * 0.6
            ax.set_xlim(xmin-margin, xmax+margin+0.48)
            ax.set_ylim(-H_outer*0.32, H_outer*1.68)

        except Exception as e:
            ax.text(0.5, 0.5, f"Geometry error:\n{e}",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=8, color="red")

        ax.set_title("Cross-section — Transverse channels", fontsize=9,
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
                               c_max=None, w_top=None, h_channel=None, theta_deg=None):
    R = D_roll / 2
    if pattern_type == "Plain Foil":
        sb = (E * t) / D_roll
        sv = sb + sigma_tension
        return {"sigma_bend": sb, "sigma_axial": sigma_tension,
                "sigma_transverse": 0.0, "sigma_total": sv, "sigma_vm": sv,
                "components": [
                    ("Bending stress", sb, r"\sigma_{bend}=\frac{E\cdot t}{D}"),
                    ("Web tension", sigma_tension, r"\sigma_{tension}"),
                ]}
    elif pattern_type == "Longitudinal Channels":
        sb = (E * 2 * c_max) / D_roll
        sv = sb + sigma_tension
        return {"sigma_bend": sb, "sigma_axial": sigma_tension,
                "sigma_transverse": 0.0, "sigma_total": sv, "sigma_vm": sv,
                "components": [
                    ("Profile bending", sb, r"\sigma_{bend}=\frac{E\cdot2c_{max}}{D}"),
                    ("Web tension", sigma_tension, r"\sigma_{tension}"),
                ]}
    elif pattern_type == "Transverse Channels":
        H_tilde  = calculate_unit_H(w_top, h_channel, theta_deg)
        q        = (sigma_tension * t) / R
        H_react  = H_tilde * q
        sb_ps    = (E / (1 - nu**2)) * (t / D_roll)
        sigma_mem= H_react / t
        s1 = sb_ps + sigma_tension
        s2 = -sigma_mem
        sv = np.sqrt(s1**2 - s1*s2 + s2**2)
        return {"sigma_bend": sb_ps, "sigma_axial": sigma_tension,
                "sigma_transverse": sigma_mem, "sigma_1": s1, "sigma_2": s2,
                "sigma_total": sb_ps + sigma_mem + sigma_tension, "sigma_vm": sv,
                "H_tilde": H_tilde, "H_reaction": H_react,
                "components": [
                    ("Plane-strain bending", sb_ps,
                     r"\sigma_{bend}=\frac{E}{1-\nu^2}\cdot\frac{t}{D}"),
                    ("Brazier membrane", sigma_mem, r"\sigma_{mem}=\frac{H}{t}"),
                    ("Web tension", sigma_tension, r"\sigma_{tension}"),
                ]}


def draw_stress_bar(components, sigma_vm, sigma_yield):
    COLORS = ["#2563EB", "#16A34A", "#D97706", "#DC2626"]
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
    D_roll = st.sidebar.number_input("Roll Diameter (mm)", value=100.0, step=5.0, min_value=1.0)

hdr = "4." if st.session_state.calc_mode == "stress" else "3."
st.sidebar.header(f"{hdr} Foil Geometry")
t = st.sidebar.number_input("Base Foil Thickness (mm)", value=0.100, step=0.001, format="%.3f")
pattern_type = st.sidebar.selectbox(
    "Engraving Pattern",
    ["Plain Foil", "Longitudinal Channels", "Transverse Channels"],
)

# safe defaults
w_top = w_bot = 0.50
h_channel = 0.46
theta_deg = 15.0
c_max = 0.380
H_total = 0.460
r_top = r_bot = 0.200
NA_bot = None

if pattern_type == "Longitudinal Channels":
    st.sidebar.markdown("*Longitudinal Specific Inputs:*")
    H_total   = st.sidebar.number_input("Total Outer Height ($H$) [mm]", value=0.460, step=0.01)
    w_top     = st.sidebar.number_input("Top Flat Width ($w_1$) [mm]", value=0.597, step=0.01)
    w_bot     = st.sidebar.number_input("Bottom Flat Width ($w_2$) [mm]", value=0.558, step=0.01)
    r_top     = st.sidebar.number_input("Top Outer Radius [mm]", value=0.200, step=0.01)
    r_bot     = st.sidebar.number_input("Bottom Outer Radius [mm]", value=0.200, step=0.01)
    theta_deg = st.sidebar.number_input("Web Angle ($\\theta$) [degrees]", value=15.0, step=1.0)
    c_max, NA_bot = calculate_longitudinal_cmax(H_total, t, w_top, w_bot, r_top, r_bot, theta_deg)
    st.sidebar.markdown("---")
    st.sidebar.info(
        f"**Calculated Metrics:**\n\n"
        f"Neutral Axis: **{NA_bot:.3f} mm**\n\n"
        f"Outer Fiber ($c_{{max}}$): **{c_max:.3f} mm**"
    )

elif pattern_type == "Transverse Channels":
    st.sidebar.markdown("*Transverse Specific Inputs:*")
    w_top     = st.sidebar.number_input("Top Flat Width ($w_1$) [mm]", value=0.500, step=0.01)
    w_bot     = st.sidebar.number_input("Bottom Flat Width ($w_2$) [mm]", value=0.550, step=0.01)
    h_channel = st.sidebar.number_input("Channel Height ($h$) [mm]", value=0.460, step=0.01)
    theta_deg = st.sidebar.number_input("Web Angle ($\\theta$) [degrees]", value=15.0, step=1.0)

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
                H_tilde = calculate_unit_H(w_top, h_channel, theta_deg)
                num = (E * t / (1 - nu**2)) + (2 * H_tilde * sigma_tension)
                den = sigma_yield - sigma_tension
                if den <= 0:
                    st.error("⚠️ Combined stresses exceed yield.")
                else:
                    D_min = num / den
                    actual_q = (sigma_tension * t) / (D_min / 2)
                    actual_H = H_tilde * actual_q
                    st.info(f"**Castigliano:** Horizontal reaction $H$ = **{actual_H:.3f} N/mm**.")
                    formula_latex = (
                        r"D_{min}=\frac{\frac{E\cdot t}{1-\nu^2}+"
                        r"2\tilde{H}\sigma_{tension}}{\sigma_{yield}-\sigma_{tension}}"
                    )
            if D_min is not None:
                c1, c2 = st.columns([1, 2])
                with c1: st.metric("Minimum Roll Diameter", f"{D_min:.2f} mm")
                with c2:
                    st.markdown("**Governing Formula:**")
                    st.latex(formula_latex)

    else:
        st.header("Results — Von Mises Stress Analysis")
        result = compute_stress_components(
            pattern_type=pattern_type, D_roll=D_roll, E=E, nu=nu,
            sigma_tension=sigma_tension, t=t,
            c_max=(c_max if pattern_type == "Longitudinal Channels" else None),
            w_top=(w_top if pattern_type == "Transverse Channels" else None),
            h_channel=(h_channel if pattern_type == "Transverse Channels" else None),
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
            if pattern_type == "Transverse Channels":
                st.divider()
                st.markdown("**Von Mises (biaxial):**")
                st.latex(r"\sigma_{vm}=\sqrt{\sigma_1^2-\sigma_1\sigma_2+\sigma_2^2}")
                st.caption(
                    f"σ₁ = {result['sigma_1']:.2f} MPa (longitudinal)  \n"
                    f"σ₂ = {result['sigma_2']:.2f} MPa (transverse, compressive)"
                )
            else:
                st.divider()
                st.latex(r"\sigma_{vm}=\sigma_{bend}+\sigma_{tension}")

        if pattern_type == "Transverse Channels":
            st.info(
                f"**Castigliano:** $\\tilde{{H}}$ = {result['H_tilde']:.4f} → "
                f"$H$ = {result['H_reaction']:.4f} N/mm"
            )

        st.subheader("Stress vs. Roll Diameter")
        D_range = np.linspace(max(10, D_roll*0.2), D_roll*3, 200)
        vm_curve = [
            compute_stress_components(
                pattern_type=pattern_type, D_roll=Di, E=E, nu=nu,
                sigma_tension=sigma_tension, t=t,
                c_max=(c_max if pattern_type == "Longitudinal Channels" else None),
                w_top=(w_top if pattern_type == "Transverse Channels" else None),
                h_channel=(h_channel if pattern_type == "Transverse Channels" else None),
                theta_deg=(theta_deg if pattern_type == "Transverse Channels" else None),
            )["sigma_vm"]
            for Di in D_range
        ]
        fs, asc = plt.subplots(figsize=(7, 3.2))
        fs.patch.set_facecolor("white"); asc.set_facecolor("#F9FAFB")
        asc.plot(D_range, vm_curve, color="#2563EB", lw=2)
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
        "Generated analytically from sidebar inputs — no CAD required. "
        "Updates live as you change any parameter. "
        "Adjacent periods shown at reduced opacity for context."
    )

    col_fig, col_info = st.columns([2, 1])

    with col_fig:
        fig = draw_geometry_preview(
            pattern_type=pattern_type, t=t,
            H_total=H_total, w1=w_top, w2=w_bot,
            r_top=r_top, r_bot=r_bot, theta_deg=theta_deg,
            h_channel=h_channel, w_top=w_top,
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
            st.metric("Channel depth h", f"{h_channel:.3f} mm")
            st.metric("Wall thickness t", f"{t:.3f} mm")
            st.metric("Outer height H = h+t", f"{h_channel+t:.3f} mm")
            st.metric("Top flat w₁", f"{w_top:.3f} mm")
            st.metric("Web angle θ", f"{theta_deg:.1f}°")
            st.divider()
            st.caption(
                "For transverse channels the governing neutral axis is the local "
                "flat-panel NA at t/2 from the bottom surface, not the full "
                "corrugated profile centroid."
            )

        st.markdown("---")
        st.caption(
            "Red dashed = neutral axis.  "
            "Blue fill = foil wall material.  "
            "Lighter border = adjacent corrugation periods."
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

    st.subheader("3. Transverse Channels")
    st.markdown(
        "3-D plate kinematics. Three superimposed components:\n\n"
        "- **Plane-strain bending:** webs prevent lateral Poisson contraction → "
        "stiffness ×$1/(1-\\nu^2)$.\n"
        "- **Brazier membrane tension:** radial pressure from web tension → "
        "horizontal reaction $H$ (Castigliano integration).\n"
        "- **Global web tension.**\n\n"
        "In Stress Analysis mode, principal stresses combined via von Mises:"
    )
    st.latex(r"\sigma_{vm}=\sqrt{\sigma_1^2-\sigma_1\sigma_2+\sigma_2^2}")
    st.latex(
        r"\sigma_1=\frac{E}{1-\nu^2}\frac{t}{D}+\sigma_{tension},\quad"
        r"\sigma_2=-\frac{H}{t}"
    )

    st.markdown("---")
    st.header("Variable nomenclature")
    st.markdown(
        "**Material & process:** $E$ Young's modulus · $\\nu$ Poisson ratio · "
        "$\\sigma_{yield}$ yield strength · $\\sigma_{tension}$ web tension · "
        "$\\sigma_{vm}$ von Mises stress.\n\n"
        "**Geometry:** $t$ foil thickness · $H$ total outer height · "
        "$r_{top}/r_{bot}$ outer fillet radii · $c_{max}$ outer-fibre distance · "
        "$w_1$ top flat width · $w_2$ bottom flat width · $h$ channel depth (transverse) · "
        "$\\theta$ web angle from vertical · $H$ reaction force (N/mm) · "
        "$\\tilde{H}$ Castigliano geometric stiffness constant."
    )
