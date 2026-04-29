import numpy as np
import streamlit as st

data = np.load('model_weights.npz')

# Pull them back out
weights = data['weights']
bias = data['bias']


def set_seamless_2x2_bg(gif1, gif2, gif3, gif4):
    st.markdown(
        f"""
        <style>
        /* 1. RESET STREAMLIT MARGINS & PADDING */
        [data-testid="stAppViewBlockContainer"] {{
            padding: 0rem !important;
            max-width: 100vw !important;
        }}
        
        .stMainView, .stAppHeader, .stHeader {{
            background-color: transparent !important;
        }}

        /* 2. THE SEAMLESS GRID CONTAINER */
        .quad-container {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            display: grid;
            grid-template-columns: 1fr 1fr;
            grid-template-rows: 1fr 1fr;
            gap: 0 !important;
            z-index: -1;
            overflow: hidden;
        }}

        /* 3. ENSURE GIFS FILL THE QUADRANT SEAMLESSLY */
        .quad-gif {{
            width: 100%;
            height: 100%;
            margin: 0;
            padding: 0;
            display: block; /* Removes tiny inline bottom gaps */
            object-fit: cover;
            transform: scale(1.02); /* Slightly zoom in to hide any seams */
        }}

        /* 4. READABILITY OVERLAY */
        /* This tint ensures your ANN inputs don't get lost in the noise */
        .stApp {{
            background: rgba(0, 0, 0, 0.4) !important;
        }}
        
        /* Center the UI content slightly so it's not touching the edges */
        .ui-content-wrapper {{
            padding: 50px;
        }}
        </style>

        <div class="quad-container">
            <img src="{gif1}" class="quad-gif">
            <img src="{gif2}" class="quad-gif">
            <img src="{gif3}" class="quad-gif">
            <img src="{gif4}" class="quad-gif">
        </div>
        """,
        unsafe_allow_html=True
    )

# --- Your GIF Links ---
# Use Direct Links (right-click > copy image address)
g3 = "https://media1.tenor.com/m/L5DjRE5OcJ0AAAAC/f16-unity.gif"
g2 = "https://media1.tenor.com/m/cRVgIJrKr3QAAAAC/bruh-missile.gif"
g1 = "https://media1.tenor.com/m/AOIRgNl5CF8AAAAd/a10-aviation.gif"
g4 = "https://media1.tenor.com/m/5WuiQehCWfEAAAAC/military-air-force.gif"

set_seamless_2x2_bg(g1, g2, g3, g4)
# --- STEP 4: INTERACTIVE TARGET IDENTIFICATION ---
def identify_target(s, r, a, h):
    print("\n--- AEGIS RADAR INPUT ---")
    try:
        new_contact = np.array([s, r, a, h])
        
        # The ANN Prediction
        score = np.dot(new_contact, weights) + bias
        if score > 0:
            print(">>> WARNING: HOSTILE DETECTED. ENGAGE TARGET. <<<")
        else:
            print(">>> STATUS: FRIENDLY. CLEAR FOR APPROACH. <<<")
    except ValueError:
        print("Invalid data format.")

st.markdown(
    """
    <style>
    .glow-title {
        color: white;
        text-align: center;
        font-size: 1px;
        font-weight: bold;
        text-shadow: 0 0 10px #00b7ff, 0 0 20px #00b7ff;
        margin-bottom: 10px;
    }
    </style>
    <p class="glow-title">AEGIS Target Identification System</p>
    """,
    unsafe_allow_html=True
)
# Run the prediction
st.markdown(":orange[**Input Speed (Mach 0-5)**]")
s = st.number_input("Input Speed (Mach 0-5)", label_visibility="collapsed")
st.markdown(":orange[**Input Radar Size (0.1-1.0)**]")
r = st.number_input("Input Radar Size (0.1-1.0)", label_visibility="collapsed")
st.markdown(":orange[**Input Altitude (k-ft)**]")
a = st.number_input("Input Altitude (k-ft)", label_visibility="collapsed")
st.markdown(":orange[**Input Heat Signature (0-10)**]")
h = st.number_input("Input Heat Signature (0-10)", label_visibility="collapsed")

if st.button("Identify Target"):
    try:
        new_contact = np.array([s, r, a, h])
        
        # The ANN Prediction
        score = np.dot(new_contact, weights) + bias
        if score > 0:
            st.markdown(":red[>>> WARNING: HOSTILE DETECTED. ENGAGE TARGET. <<<]")
        else:
            st.markdown(":green[>>> STATUS: FRIENDLY. CLEAR FOR APPROACH. <<<]")
    except ValueError:
        st.write("Invalid data format.")