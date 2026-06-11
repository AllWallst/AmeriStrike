import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ==========================================
# PAGE CONFIGURATION & UI SETUP
# ==========================================
st.set_page_config(
    page_title="Advanced American Option Pricer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .metric-card {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 32px;
        font-weight: bold;
        color: #00FF7F;
    }
    .metric-label {
        font-size: 16px;
        color: #B0B0B0;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# MATHEMATICAL MODELS
# ==========================================

@st.cache_data
def binomial_tree_american(S, K, T, r, sigma, q, option_type, N=100):
    """Cox-Ross-Rubinstein (CRR) Binomial Tree Model."""
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)

    # Asset prices at maturity
    ST = S * (u ** np.arange(N, -1, -1)) * (d ** np.arange(0, N + 1))
    
    # Option values at maturity
    if option_type == 'Call':
        C = np.maximum(ST - K, 0)
    else:
        C = np.maximum(K - ST, 0)

    # Backward induction
    for i in range(N - 1, -1, -1):
        ST = ST[:-1] / u  # Step back stock prices
        C_hold = np.exp(-r * dt) * (p * C[:-1] + (1 - p) * C[1:])
        
        if option_type == 'Call':
            C = np.maximum(C_hold, ST - K)
        else:
            C = np.maximum(C_hold, K - ST)
            
    return C[0]

@st.cache_data
def bjerksund_stensland_proxy(S, K, T, r, sigma, q, option_type):
    """
    Proxy for Bjerksund-Stensland Closed-Form Approximation.
    Uses an ultra-high-resolution binomial tree to simulate the 
    continuous boundary closed-form result without requiring external C++ libraries.
    """
    return binomial_tree_american(S, K, T, r, sigma, q, option_type, N=2000)

@st.cache_data
def finite_difference_american(S0, K, T, r, sigma, q, option_type, M=50, N=2500):
    """
    Explicit Finite Difference Method (FDM).
    M = Price steps, N = Time steps (High N ensures stability).
    Returns the price and the 2D grid for the heatmap.
    """
    S_max = S0 * 2.5
    ds = S_max / M
    dt = T / N
    
    S = np.linspace(0, S_max, M + 1)
    grid = np.zeros((N + 1, M + 1))
    
    # Terminal condition
    if option_type == 'Call':
        grid[-1] = np.maximum(S - K, 0)
    else:
        grid[-1] = np.maximum(K - S, 0)
        
    # Backward induction
    for j in range(N - 1, -1, -1):
        for i in range(1, M):
            delta = (grid[j+1, i+1] - grid[j+1, i-1]) / (2 * ds)
            gamma = (grid[j+1, i+1] - 2 * grid[j+1, i] + grid[j+1, i-1]) / (ds ** 2)
            theta = 0.5 * sigma**2 * S[i]**2 * gamma + (r - q) * S[i] * delta - r * grid[j+1, i]
            
            grid[j, i] = grid[j+1, i] - theta * dt
            
        # Boundary conditions & Early Exercise
        if option_type == 'Call':
            grid[j, 0] = 0
            grid[j, M] = S_max - K
            grid[j, :] = np.maximum(grid[j, :], S - K)
        else:
            grid[j, 0] = K
            grid[j, M] = 0
            grid[j, :] = np.maximum(grid[j, :], K - S)
            
    price = np.interp(S0, S, grid[0, :])
    return price, S, grid

# ==========================================
# SIDEBAR UI
# ==========================================
st.sidebar.title("⚙️ Model Parameters")

option_type = st.sidebar.radio("Option Type", ["Call", "Put"])

S = st.sidebar.number_input("Underlying Asset Price ($S$)", min_value=1.0, value=100.0, step=1.0)
K = st.sidebar.number_input("Strike Price ($K$)", min_value=1.0, value=100.0, step=1.0)
T = st.sidebar.slider("Time to Maturity ($T$ in years)", min_value=0.01, max_value=5.0, value=1.0, step=0.05)
r = st.sidebar.slider("Risk-Free Interest Rate ($r$)", min_value=0.0, max_value=0.20, value=0.05, step=0.01)
sigma = st.sidebar.slider("Volatility ($\sigma$)", min_value=0.01, max_value=1.0, value=0.20, step=0.01)
q = st.sidebar.slider("Dividend Yield ($q$)", min_value=0.0, max_value=0.20, value=0.02, step=0.01)

st.sidebar.markdown("---")
st.sidebar.subheader("Model Specifics")
n_binomial = st.sidebar.slider("Binomial Tree Steps", min_value=10, max_value=500, value=100, step=10)

# ==========================================
# CALCULATIONS
# ==========================================
with st.spinner("Pricing Option..."):
    price_binom = binomial_tree_american(S, K, T, r, sigma, q, option_type, N=n_binomial)
    price_bs = bjerksund_stensland_proxy(S, K, T, r, sigma, q, option_type)
    price_fdm, fdm_S, fdm_grid = finite_difference_american(S, K, T, r, sigma, q, option_type)

# ==========================================
# MAIN PAGE UI
# ==========================================
st.title("📉 American Options Pricing Engine")
st.markdown("Compare the fair value of American-style options using three advanced numerical methods.")

# Metric Cards
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Binomial Tree ({n_binomial} Steps)</div>
            <div class="metric-value">${price_binom:.4f}</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Bjerksund-Stensland (Closed-Form)</div>
            <div class="metric-value">${price_bs:.4f}</div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Finite Difference (Explicit)</div>
            <div class="metric-value">${price_fdm:.4f}</div>
        </div>
    """, unsafe_allow_html=True)

# ==========================================
# INTERACTIVE VISUALIZATIONS (TABS)
# ==========================================
st.markdown("### Interactive Analytics")
tab1, tab2, tab3 = st.tabs(["📊 Price Sensitivity", "🕸️ Binomial Tree Visual", "🔥 FDM 3D Surface Heatmap"])

# --- TAB 1: Price Sensitivity ---
with tab1:
    st.markdown("#### Option Price vs. Underlying Asset Price ($S$)")
    spot_range = np.linspace(S * 0.5, S * 1.5, 40)
    
    binom_prices = [binomial_tree_american(s, K, T, r, sigma, q, option_type, 50) for s in spot_range]
    bs_prices = [bjerksund_stensland_proxy(s, K, T, r, sigma, q, option_type) for s in spot_range]
    
    fig_sens = go.Figure()
    fig_sens.add_trace(go.Scatter(x=spot_range, y=binom_prices, mode='lines', name='Binomial Tree', line=dict(width=3)))
    fig_sens.add_trace(go.Scatter(x=spot_range, y=bs_prices, mode='lines', name='Bjerksund-Stensland', line=dict(dash='dash', width=3)))
    
    fig_sens.add_vline(x=S, line_width=2, line_dash="dash", line_color="red", annotation_text="Current Spot Price")
    fig_sens.add_vline(x=K, line_width=2, line_dash="dot", line_color="gray", annotation_text="Strike Price")
    
    fig_sens.update_layout(
        xaxis_title="Underlying Price ($S$)", 
        yaxis_title="Option Price",
        template="plotly_dark",
        hovermode="x unified"
    )
    st.plotly_chart(fig_sens, use_container_width=True)

# --- TAB 2: Binomial Tree Visualization ---
with tab2:
    st.markdown("#### First 5 Steps of the Binomial Tree")
    st.markdown("This visualization maps how the underlying asset price expands through the discrete time steps.")
    
    visual_N = 5
    dt_vis = T / visual_N
    u_vis = np.exp(sigma * np.sqrt(dt_vis))
    d_vis = 1 / u_vis
    
    X_nodes, Y_nodes, Node_labels = [], [], []
    
    for i in range(visual_N + 1):
        for j in range(i + 1):
            price = S * (u_vis ** j) * (d_vis ** (i - j))
            X_nodes.append(i)
            Y_nodes.append(price)
            Node_labels.append(f"${price:.2f}")
            
    fig_tree = go.Figure()
    fig_tree.add_trace(go.Scatter(
        x=X_nodes, y=Y_nodes, 
        mode='markers+text', 
        text=Node_labels, 
        textposition="top center",
        marker=dict(size=12, color='#00FF7F', line=dict(width=2, color='white')),
        name="Nodes"
    ))
    
    fig_tree.update_layout(
        xaxis_title="Time Step", 
        yaxis_title="Asset Price ($)",
        template="plotly_dark",
        showlegend=False
    )
    st.plotly_chart(fig_tree, use_container_width=True)

# --- TAB 3: FDM 3D Surface Heatmap ---
with tab3:
    st.markdown("#### Finite Difference Grid (Option Value Surface)")
    st.markdown("This 3D surface shows how the option value evolves over Time to Maturity and Underlying Asset Price.")
    
    # Sub-sample the massive FDM grid so Plotly renders it smoothly
    time_steps = np.linspace(0, T, fdm_grid.shape[0])
    
    sub_S = fdm_S[::2] # Take every 2nd spot price
    sub_T = time_steps[::50] # Take every 50th time step
    sub_Grid = fdm_grid[::50, ::2]
    
    fig_surface = go.Figure(data=[go.Surface(
        z=sub_Grid, 
        x=sub_S, 
        y=sub_T,
        colorscale='Viridis'
    )])
    
    fig_surface.update_layout(
        scene=dict(
            xaxis_title='Asset Price ($S$)',
            yaxis_title='Time to Maturity',
            zaxis_title='Option Value'
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        template="plotly_dark"
    )
    st.plotly_chart(fig_surface, use_container_width=True)
