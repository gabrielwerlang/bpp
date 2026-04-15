import streamlit as st
import numpy as np

# --- 1. Analytical Frame Solver (Castigliano's Theorem) ---
def calculate_unit_H(w_top, h_channel):
    """
    Numerically solves the Castigliano Strain Energy integral for a half-channel 
    subjected to a unit Brazier load (q = 1 N/mm).
    """
    n_points = 500
    
    # Segment 1: Top Flat (from center cut to corner)
    x1 = np.linspace(0, w_top/2, n_points)
    y1 = np.full(n_points, h_channel)
    
    # Segment 2: Vertical Web (from top corner down to bottom corner)
    x2 = np.full(n_points, w_top/2)
    y2 = np.linspace(h_channel, 0, n_points)
    
    # Combine geometry
    x = np.concatenate([x1, x2])
    y = np.concatenate([y1, y2])
    
    # Segment lengths (ds) for numerical integration
    ds = np.zeros_like(x)
    ds[1:] = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    
    A = np.trapz(np.ones_like(x), x=np.cumsum(ds))
    B = np.trapz(y, x=np.cumsum(ds))
    C = np.trapz(y**2, x=np.cumsum(ds))
    
    q = 1.0
    Dx = np.trapz(q * x1, x=np.cumsum(ds[:n_points])) 
    Exy = np.trapz(q * x1 * y1, x=np.cumsum(ds[:n_points]))
    
    matrix = np.array([[A, -B], [-B, C]])
    constants = np.array([-Dx, Exy])
    
    solution = np.linalg.solve(matrix, constants)
    return solution[1] # H_tilde

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
    c_max = st.sidebar.number_input("Distance to Outer Fiber, $c_{max}$ (mm)", value=0.380, step=0.001, format="%.3f")

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
            H_tilde = calculate_unit_H(w_top, h_channel)
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
    **Theory:** When the flow channels are oriented parallel to the machine direction, the foil conforms to the curvature along its entire cross-section. The geometric profile increases the area moment of inertia. Consequently, the base thickness ($t$) is replaced by the maximum distance from the cross-sectional neutral axis to the outermost fiber ($c_{max}$).

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
    * **$c_{max}$**: (Longitudinal channels) The distance from the geometric neutral axis of the corrugated profile to the furthest outer fiber.
    * **$w_1$**: (Transverse channels) The width of the upper flat peak of the channel geometry.
    * **$w_2$**: (Transverse channels) The width of the lower flat valley of the channel geometry.
    * **$h$**: (Transverse channels) The total vertical height of the profile.
    * **$\theta$**: (Transverse channels) The draft angle of the web wall, measured relative to the vertical axis.
    * **$H$**: The horizontal reaction force (N/mm) generated by the channel geometry resisting the radial compressive pressure.
    * **$\tilde{H}$**: The geometric stiffness constant for $H$, solved numerically via Castigliano's theorem for a unit-load of 1 N/mm.
    """)