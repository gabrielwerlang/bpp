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
    """
    Returns (outer_x, outer_y) for ONE full corrugation period of a
    longitudinal channel cross-section (two half-channels, mirrored).
    The profile is drawn at the OUTER surface.
    Y = 0 at the bottom outer surface.
    """
    theta = np.radians(theta_deg)
    phi = np.pi / 2 - theta          # arc sweep angle
    r_t = r_top_out
    r_b = r_bot_out

    # half-pitch at top and bottom
    half_w1 = w1 / 2
    half_w2 = w2 / 2

    # Top-fillet arc centre positions
    # The flat top runs from -half_w1 to +half_w1.
    # The arc tangency point on the flat is at x = ±(half_w1 - r_t*tan(phi/2))
    tan_half_phi = np.tan(phi / 2)
    # Arc centre (top right)
    arc_t_cx = half_w1 - r_t * tan_half_phi + r_t * np.sin(theta)
    arc_t_cy = H - r_t

    # Bottom-fillet arc centre (bottom right)
    arc_b_cx = half_w2 - r_b * tan_half_phi + r_b * np.sin(theta)
    arc_b_cy = r_b

    # Web end points
    # Top of web (tangency with top arc)
    web_tx = arc_t_cx - r_t * np.sin(theta)
    web_ty = arc_t_cy - r_t * np.cos(theta)

    # Bottom of web (tangency with bottom arc)
    web_bx = arc_b_cx - r_b * np.sin(theta)
    web_by = arc_b_cy + r_b * np.cos(theta)

    def one_side(sign):
        xs, ys = [], []
        # flat top
        xs.append(0.0);  ys.append(H)
        xs.append(sign * (half_w1 - r_t * tan_half_phi)); ys.append(H)
        # top arc
        if sign > 0:
            ax, ay = _fillet_arc_points(sign * arc_t_cx, arc_t_cy,
                                        r_t, 90, 90 - np.degrees(phi))
        else:
            ax, ay = _fillet_arc_points(sign * arc_t_cx, arc_t_cy,
                                        r_t, 90, 90 + np.degrees(phi))
        xs.extend(ax.tolist()); ys.extend(ay.tolist())
        # web
        xs.append(sign * web_bx); ys.append(web_by)
        # bottom arc
        bot_start = np.degrees(np.arctan2(web_by - arc_b_cy, sign * web_bx - sign * arc_b_cx))
        bot_end = 90.0
        if sign > 0:
            ax2, ay2 = _fillet_arc_points(sign * arc_b_cx, arc_b_cy,
                                          r_b, bot_start, bot_end)
        else:
            ax2, ay2 = _fillet_arc_points(sign * arc_b_cx, arc_b_cy,
                                          r_b, 180 - bot_start, 90)
        xs.extend(ax2.tolist()); ys.extend(ay2.tolist())
        # flat bottom
        xs.append(sign * half_w2); ys.append(0.0)
        return xs, ys

    xl, yl = one_side(-1)
    xr, yr = one_side(+1)

    # mirror: left side reversed + right side
    xr_rev = list(reversed(xl))
    yr_rev = list(reversed(yl))    # Ensure the profile is closed by connecting the two sides
    full_x = xr_rev + xr
    full_y = yr_rev + yr
    return np.array(full_x), np.array(full_y)


def build_transverse_profile(w_top, h_channel, theta_deg, t):
    """
    Returns (x, y) arrays for ONE full corrugation period of a transverse
    channel cross-section (outer surface only, simplified trapezoidal shape).
    """
    theta = np.radians(theta_deg)
    hw1 = w_top / 2
    offset = h_channel * np.tan(theta)

    # One full period: bottom-left → web up-left → top-left → top-right → web down-right → bottom-right
    # We mirror to get a full symmetric period.
    def half_right():
        xs = [0, hw1, hw1 + offset]
        ys = [h_channel, h_channel, 0]
        return xs, ys

    hx, hy = half_right()
    # mirror
    xs = [-x for x in reversed(hx)] + hx
    ys = list(reversed(hy)) + hy
    return np.array(xs), np.array(ys)


def draw_geometry_preview(pattern_type, t,
                           # longitudinal
                           H_total=None, w1=None, w2=None,
                           r_top=None, r_bot=None,
                           # shared
                           theta_deg=15.0,
                           # transverse
                           h_channel=None, w_top=None,
                           # NA info
                           NA_bot=None, c_max=None):
    """
    Draws a cross-section preview of the selected channel geometry.
    Returns a matplotlib Figure.
    """
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
        """Draw a simple dimension line with text."""
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

    # ── PLAIN FOIL ──────────────────────────────────────────────────────────
    if pattern_type == "Plain Foil":
        L = 3.0
        rect = plt.Rectangle((-L / 2, 0), L, t,
                              facecolor=LIGHT_FILL, edgecolor=PROFILE_COLOR, lw=1.5)
        ax.add_patch(rect)
        ax.set_xlim(-L / 2 - 0.3, L / 2 + 0.6)
        ax.set_ylim(-t * 1.5, t * 4)

        # NA line
        ax.axhline(t / 2, color=NA_COLOR, lw=1.2, ls="--", label="Neutral axis")

        # dimension: thickness
        annotate_dim(L / 2 + 0.1, 0, L / 2 + 0.1, t,
                     f"t = {t:.3f} mm", orient="v", offset=0.05)

        ax.text(0, t / 2, "Neutral axis  (centre)",
                fontsize=7.5, color=NA_COLOR, ha="center", va="bottom")
        ax.set_xlabel("Width (mm)", fontsize=8, color=DIM_COLOR)
        ax.set_ylabel("Height (mm)", fontsize=8, color=DIM_COLOR)
        ax.set_title("Cross-section — Plain foil", fontsize=9,
                     color=LABEL_COLOR, fontweight="bold", pad=8)

    # ── LONGITUDINAL CHANNELS ───────────────────────────────────────────────
    elif pattern_type == "Longitudinal Channels":
        ox, oy = build_longitudinal_profile(H_total, t, w1, w2, r_top, r_bot, theta_deg)

        # Offset inner wall (approximate: shrink by t in y)
        ax.fill(ox, oy, facecolor=LIGHT_FILL, edgecolor=PROFILE_COLOR, lw=1.5, zorder=2)

        xmin, xmax = ox.min(), ox.max()
        ax.set_xlim(xmin - 0.15, xmax + 0.55)
        ax.set_ylim(-0.05, H_total * 1.45)

        # Neutral axis
        if NA_bot is not None:
            ax.axhline(NA_bot, color=NA_COLOR, lw=1.2, ls="--",
                       xmin=0.05, xmax=0.75, zorder=3)
            ax.text(xmin - 0.05, NA_bot, "NA", fontsize=7,
                    color=NA_COLOR, va="center", ha="right", zorder=4)

        # c_max bracket
        if NA_bot is not None and c_max is not None:
            top_pt = max(NA_bot, H_total - NA_bot)
            bot_pt = min(NA_bot, H_total - NA_bot)
            annotate_dim(xmax + 0.08, NA_bot, xmax + 0.08, NA_bot + c_max,
                         f"c_max\n{c_max:.3f} mm", orient="v", offset=0.04)

        # H dimension
        annotate_dim(xmax + 0.35, 0, xmax + 0.35, H_total,
                     f"H = {H_total:.3f} mm", orient="v", offset=0.04)

        # w1 dimension
        annotate_dim(-w1 / 2, H_total + 0.06, w1 / 2, H_total + 0.06,
                     f"w₁={w1:.2f}", orient="h", offset=0.04)

        ax.set_xlabel("Width (mm)", fontsize=8, color=DIM_COLOR)
        ax.set_ylabel("Height (mm)", fontsize=8, color=DIM_COLOR)
        ax.set_title("Cross-section — Longitudinal channels", fontsize=9,
                     color=LABEL_COLOR, fontweight="bold", pad=8)

    # ── TRANSVERSE CHANNELS ─────────────────────────────────────────────────
    elif pattern_type == "Transverse Channels":
        px, py = build_transverse_profile(w_top, h_channel, theta_deg, t)
        ax.fill(px, py, facecolor=LIGHT_FILL, edgecolor=PROFILE_COLOR, lw=1.5, zorder=2)

        # Repeat profile for context (one period each side)
        period = max(px) - min(px)
        for shift in [-period, period]:
            ax.plot(px + shift, py, color=PROFILE_COLOR, lw=1.0,
                    ls="-", alpha=0.35, zorder=1)

        xmin, xmax = px.min(), px.max()
        ax.set_xlim(xmin - period * 0.6, xmax + period * 0.6 + 0.5)
        ax.set_ylim(-h_channel * 0.25, h_channel * 1.7)

        # height dimension
        annotate_dim(xmax + 0.08, 0, xmax + 0.08, h_channel,
                     f"h = {h_channel:.3f} mm", orient="v", offset=0.04)

        # w_top dimension
        annotate_dim(-w_top / 2, h_channel + h_channel * 0.15,
                     w_top / 2, h_channel + h_channel * 0.15,
                     f"w₁={w_top:.2f} mm", orient="h", offset=h_channel * 0.06)

        # web angle annotation
        theta_rad = np.radians(theta_deg)
        web_x0 = w_top / 2
        web_x1 = w_top / 2 + h_channel * np.tan(theta_rad)
        mid_wx = (web_x0 + web_x1) / 2 + 0.05
        mid_wy = h_channel / 2
        ax.text(mid_wx, mid_wy, f"θ={theta_deg:.0f}°",
                fontsize=7, color=DIM_COLOR, va="center", ha="left",
                rotation=-theta_deg)

        # NA line (plane-strain bending acts on flat plate = t/2 from bottom)
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
# 3.  UI
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="BPP Roll Diameter Calculator", layout="wide")
st.title("Analytical $D_{min}$ Calculator")
st.markdown(
    "Calculate the minimum roller diameter required to prevent plastic "
    "deformation during roll-to-roll manufacturing."
)
st.markdown("---")

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.header("1. Material Properties")
E = st.sidebar.number_input("Young's Modulus (MPa)", value=200000.0, step=1000.0)
nu = st.sidebar.number_input("Poisson's Ratio", value=0.30, step=0.01)
sigma_yield = st.sidebar.number_input("Yield Strength (MPa)", value=350.0, step=10.0)

st.sidebar.header("2. Process Parameters")
sigma_tension = st.sidebar.number_input("Web Tension (MPa)", value=15.0, step=1.0)

st.sidebar.header("3. Foil Geometry")
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

# ── Tab 1: Calculator ────────────────────────────────────────────────────────
with tab_calc:
    st.header("Results")

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

# ── Tab 2: Geometry Preview ──────────────────────────────────────────────────
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

# ── Tab 3: Theory ────────────────────────────────────────────────────────────
with tab_theory:
    st.header("Mechanics & Rationale")
    st.markdown(
        "This analytical tool utilises the **Principle of Superposition** to determine "
        "the minimum required roller diameter to prevent plastic deformation. The "
        "mechanical behaviour of the foil varies significantly depending on the "
        "orientation of the flow field channels relative to the manufacturing transport direction."
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
        "- **Global tensile stress:** Uniform machine tension."
    )
    st.latex(
        r"\sigma_{total} = \left[\frac{E}{1-\nu^2} \cdot \frac{t}{D}\right]"
        r"+ \left[\frac{H}{t}\right] + \sigma_{tension}"
    )

    st.markdown("---")
    st.header("Variable nomenclature")
    st.markdown(
        "**Material & process:**\n"
        "- $E$ — Young's Modulus\n"
        "- $\\nu$ — Poisson's Ratio\n"
        "- $\\sigma_{yield}$ — yield strength\n"
        "- $\\sigma_{tension}$ — continuous web tension (force / cross-sectional area)\n\n"
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
