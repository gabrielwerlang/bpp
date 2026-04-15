import streamlit as st
import numpy as np

# --- 1. Analytical Solvers ---

def calculate_unit_H(w_top, h_channel, theta_deg):
    """
    Numerically solves the Castigliano Strain Energy integral for a half-channel 
    subjected to a unit Brazier load (q = 1 N/mm).
    """
    # --- FIX: NumPy 2.0 Compatibility ---
    try:
        trapz_func = np.trapezoid
    except AttributeError:
        trapz_func = np.trapz
        
    n_points = 500
    
    # Segment 1: Top Flat (from center cut to top corner)
    x1 = np.linspace(0, w_top/2, n_points)
    y1 = np.full(n_points, h_channel)
    
    # Segment 2: Angled Web (from top corner down to bottom corner)
    # --- FIX: Incorporate Draft Angle ---
    theta_rad = np.radians(theta_deg)
    x2 = np.linspace(w_top/2, w_top/2 + h_channel * np.tan(theta_rad), n_points)
    y2 = np.linspace(h_channel, 0, n_points)
    
    # Combine geometry
    x = np.concatenate([x1, x2])
    y = np.concatenate([y1, y2])
    
    # Segment lengths (ds) for numerical integration
    ds = np.zeros_like(x)
    ds[1:] = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    s = np.cumsum(ds)
    
    # Integration Matrices
    A = trapz_func(np.ones_like(x), x=s)
    B = trapz_func(y, x=s)
    C = trapz_func(y**2, x=s)
    
    q = 1.0
    Dx = trapz_func(q * x1, x=s[:n_points]) 
    Exy = trapz_func(q * x1 * y1, x=s[:n_points])
    
    matrix = np.array([[A, -B], [-B, C]])
    constants = np.array([-Dx, Exy])
    
    solution = np.linalg.solve(matrix, constants)
    return solution[1] # H_tilde

def calculate_longitudinal_cmax(H, t, w1, w2, r_top_out, r_bot_out, theta_deg):
    """
    Calculates the neutral axis and maximum outer fiber distance (c_max) 
    for a corrugated bipolar plate using 1D centerline integration.
    """
    # Convert web angle to radians
    theta = np.radians(theta_deg)
    
    # Geometry Constants (Centerline base)
    h = H - t
    R_cl_top = r_top_out - (t / 2.0)
    R_cl_bot = r_bot_out - (t / 2.0)
    
    # Swept angle of the arcs (from horizontal to the angled web)
    phi = (np.pi / 2.0) - theta
    
    # Vertical drop/rise of the arcs
    dy_top = R_cl_top * (1.0 - np.cos(phi))
    dy_bot = R_cl_bot * (1.0 - np.cos(phi))
    
    # Component Lengths
    L_flat_top = w1 - 2.0 * R_cl_top * np.tan(phi / 2.0)
    L_flat_bot = w2 - 2.0 * R_cl_bot * np.tan(phi / 2.0)
    L_arc_top = R_cl_top * phi
    L_arc_bot = R_cl_bot * phi
    
    # Web length (straight inclined section)
    vertical_web_dist = h - dy_top - dy_bot
    L_web = vertical_web_dist / np.cos(theta)
    
    # Component Centroids (Y-coordinates from bottom centerline)
    C_flat_top = h
    C_flat_bot = 0.0
    C_arc_top = h - R_cl_top + (R_cl_top * np.sin(phi) / phi)
    C_arc_bot = R_cl_bot - (R_cl_bot * np.sin(phi) / phi)
    C_web = (dy_bot + (h - dy_top)) / 2.0
    
    # Neutral Axis Calculation (Weighted Average)
    numerator = (L_flat_top * C_flat_top) + (L_flat_bot * C_flat_bot) + \
                (2.0 * L_arc_top * C_arc_top) + (2.0 * L_arc_bot * C_arc_bot) + \
                (2.0 * L_web * C_web)
                
    denominator = L_flat_top + L_flat_bot + (2.0 * L_arc_top) + (2.0 * L_arc_bot) + (2.0 * L_web)
    
    NA_centerline = numerator / denominator
    
    # Convert to absolute coordinates and find c_max
    NA_bottom = NA_centerline + (t / 2.0)
    distance_to_bottom = NA_bottom
    distance_to_top = H - NA_bottom
    
    c_max = max(distance_to_bottom, distance_to_top)
    
    return c_max, NA_bottom

# --- 2. User Interface & Dashboard ---
st.set_page_config(page_title="BPP Roll Diameter Calculator", layout="wide")
st.title("Analytical $D_{min}$ Calculator")
st.markdown("Calculate the minimum roller diameter required to prevent plastic deformation during roll-to-roll manufacturing.")
st.markdown("---")

# Sidebar
st.sidebar.header("1. Material Properties")
E = st.sidebar.number_input("Young's Modulus (MPa)", value=200000.0, step=1000.0)
nu = st.sidebar.number_input("Poisson's Ratio", value=0.30, step=0.01)
sigma_yield = st.sidebar.number_input("Yield Strength (MPa)", value=350.0, step=10.0)

st.sidebar.header("2. Process Parameters")
sigma_tension = st.sidebar.number_input("Web Tension (MPa)", value=15.0, step=1.0)

st.sidebar.header("3. Foil Geometry")
t = st.sidebar.number_input("Base Foil Thickness (mm)", value=0.100, step=0.001, format="%.3f")
pattern_type = st.sidebar.selectbox("Engraving Pattern", ["Plain Foil", "Longitudinal Channels", "Transverse Channels"])

# Default fallback values to prevent undefined errors
w_top, w_bot, h_channel, theta_deg, c_max = 0.50, 0.50, 0.46, 15.0, 0.380

if pattern_type == "Longitudinal Channels":
    st.sidebar.markdown("*Longitudinal Specific Inputs:*")
    H_total = st.sidebar.number_input("Total Outer Height ($H$) [mm]", value=0.460, step=0.01)
    w_top = st.sidebar.number_input("Top Flat Width ($w_1$) [mm]", value=0.597, step=0.01)
    w_bot = st.sidebar.number_input("Bottom Flat Width ($w_2$) [mm]", value=0.558, step=0.01)
    r_top = st.sidebar.number_input("Top Outer Radius [mm]", value=0.200, step=0.01)
    r_bot = st.sidebar.number_input("Bottom Outer Radius [mm]", value=0.200, step=0.01)
    theta_deg = st.sidebar.number_input("Web Angle ($\theta$) [degrees]", value=15.0, step=1.0)
    
    # Calculate c_max dynamically based on 1D centerline kinematics
    c_max, NA_bot = calculate_longitudinal_cmax(H_total, t, w_top, w_bot, r_top, r_bot, theta_deg)
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"**Calculated Metrics:**\n\nNeutral Axis: **{NA_bot:.3f} mm**\nOuter Fiber ($c_{{max}}$): **{c_max:.3f} mm**")

elif pattern_type == "Transverse Channels":
    st.sidebar.markdown("*Transverse Specific Inputs:*")
    w_top = st.sidebar.number_input("Top Flat Width ($w_1$) [mm]", value=0.500, step=0.01)
    w_bot = st.sidebar.number_input("Bottom Flat Width ($w_2$) [mm]", value=0.550, step=0.01)
    h_channel = st.sidebar.number_input("Channel Height ($h$) [mm]", value=0.460, step=0.01)
    theta_deg = st.sidebar.number_input("Web Angle ($\theta$) [degrees]", value=15.0, step=1.0)

# --- 3. Main Content Area (Tabs) ---
tab_calc, tab_theory = st.tabs(["🧮 Calculator", "📚 Theory & Rationale"])

with tab_calc:
    st.header("Results")

    if sigma_tension >= sigma_yield:
        st.error("⚠️ **Yield Condition Reached:** The uniform web tension exceeds the material yield strength. Plastic deformation occurs prior to curvature engagement.")
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
            H_tilde = calculate_unit_H(w_top, h_channel, theta_deg) # <-- Added theta_deg            
            numerator = (E * t / (1 - nu**2)) + (2 * H_tilde * sigma_tension)
            denominator = sigma_yield - sigma_tension
            
            if denominator <= 0:
                st.error("⚠️ **Yield Condition Reached:** The combination of global tension and local membrane stresses exceeds the yield threshold.")
            else:
                D_min = numerator / denominator
                actual_q = (sigma_tension * t) / (D_min / 2)
                actual_H = H_tilde * actual_q
                st.info(f"**Castigliano Integration Output:** The induced radial compressive pressure generates a transverse horizontal reaction force ($H$) of **{actual_H:.3f} N/mm**.")
                formula_latex = r"D_{min} = \frac{\frac{E \cdot t}{1-\nu^2} + 2 \cdot \tilde{H} \cdot \sigma_{tension}}{\sigma_{yield} - \sigma_{tension}}"

        if D_min is not None:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric(label=f"Minimum Roll Diameter", value=f"{D_min:.2f} mm")
            with col2:
                st.markdown("**Governing Formula:**")
                st.latex(formula_latex)

with tab_theory:
    st.header("Mechanics & Rationale")
    st.markdown("""
    This analytical tool utilizes the **Principle of Superposition** to determine the minimum required roller diameter to prevent plastic deformation. The mechanical behavior of the foil varies significantly depending on the orientation of the flow field channels relative to the manufacturing transport direction.
    """)

    st.subheader("1. Plain Foil")
    st.markdown("""
    **Theory:** The deformation is modeled using classic Euler-Bernoulli beam theory. The maximum bending stress is determined entirely by the material thickness and the curvature of the roller.

    **Stress Components:**
    * Bending Stress
    * Global Tensile Stress
    """)
    st.latex(r"\sigma_{total} = \frac{E \cdot t}{D} + \sigma_{tension}")

    st.subheader("2. Longitudinal Channels (Parallel to Transport Direction)")
    st.markdown("""
    **Theory:** When the flow channels are oriented parallel to the machine direction, the foil conforms to the curvature along its entire cross-section. The geometric profile increases the area moment of inertia. Consequently, the base thickness ($t$) is replaced by the maximum distance from the cross-sectional neutral axis to the outermost fiber ($c_{max}$). The neutral axis is computed dynamically via a 1D centerline length-weighted integration of the channel components.

    **Stress Components:**
    * Amplified Profile Bending Stress
    * Global Tensile Stress
    """)
    st.latex(r"\sigma_{total} = \frac{E \cdot 2 \cdot c_{max}}{D} + \sigma_{tension}")

    st.subheader("3. Transverse Channels (Perpendicular to Transport Direction)")
    st.markdown("""
    **Theory:** For channels oriented perpendicularly to the transport direction, the deformation mechanics transition from 1D beam bending to 3D plate kinematics. The standard neutral-axis distance ($c_{max}$) becomes inapplicable. Instead, local structural stiffness, transverse kinematic restrictions, and radial compressive pressures govern the foil's mechanical behavior. The total stress is calculated through the superposition of three distinct components:

    **Stress Components:**
    * **Plane Strain Bending Stress:** The vertical web sections act as rigid structural constraints, preventing the flat valleys from contracting laterally (Poisson effect) as they conform to the cylindrical roller. This kinematic restriction increases the local bending stiffness by a factor of $1/(1-\nu^2)$.
    * **Membrane Tensile Stress (Brazier Effect):** The longitudinal tension of the foil wrapping around the roller generates a radial compressive pressure. This pressure induces a downward deflection on the upper flat sections of the channels, forcing the vertical webs to deflect laterally. The structural stiffness of the profile corners resists this lateral deflection, generating a horizontal reaction force ($H$). This reaction force subjects the bottom flat section to uniform membrane tension. The magnitude of $H$ is computed dynamically using a numerical Castigliano Strain Energy integration matrix.
    * **Global Tensile Stress:** The baseline uniform continuous machine tension applied by the roll-to-roll processing equipment.
    """)
    st.latex(r"\sigma_{total} = \left[ \frac{E}{1-\nu^2} \cdot \frac{t}{D} \right] + \left[ \frac{H}{t} \right] + \sigma_{tension}")

    st.markdown("---")
    st.header("Variable Nomenclature")
    
    st.markdown("""
    **Material & Process Parameters:**
    * **$E$**: Young's Modulus of the foil material.
    * **$\nu$**: Poisson's Ratio of the foil material.
    * **$\sigma_{yield}$**: The yield strength threshold of the foil material.
    * **$\sigma_{tension}$**: The continuous web tension applied by the manufacturing equipment (Pulling Force / Cross-Sectional Area).
    """)

    st.markdown("""
    **Geometric Parameters:**
    * **$t$**: The thickness of the unformed foil material.
    * **$H$**: (Longitudinal channels) The total vertical outer height of the profile.
    * **$r_{top}, r_{bot}$**: (Longitudinal channels) The outer radii of the upper and lower channel fillets.
    * **$c_{max}$**: (Longitudinal channels) The distance from the calculated geometric neutral axis of the corrugated profile to the furthest outer fiber.
    * **$w_1$**: The width of the upper flat peak of the channel geometry (center-to-center distance).
    * **$w_2$**: The width of the lower flat valley of the channel geometry (center-to-center distance).
    * **$h$**: (Transverse channels) The total vertical centerline height of the profile.
    * **$\theta$**: The draft angle of the web wall, measured relative to the vertical axis.
    * **$H$**: (Transverse channels) The horizontal reaction force (N/mm) generated by the channel geometry resisting the radial compressive pressure.
    * **$\tilde{H}$**: (Transverse channels) The geometric stiffness constant for $H$, solved numerically via Castigliano's theorem for a unit-load of 1 N/mm.
    """)