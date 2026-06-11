import streamlit as st
import numpy as np
import plotly.graph_objects as go
import yfinance as yf

# ==========================================
# PAGE CONFIGURATION & UI SETUP
# ==========================================
st.set_page_config(
    page_title="DerivLab: American Options",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    .metric-value { font-size: 32px; font-weight: bold; color: #00FF7F; }
    .metric-label { font-size: 16px; color: #B0B0B0; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# STATE INITIALIZATION
# ==========================================
# We use session state so the yfinance button can dynamically update the sliders
default_vals = {'S': 150.0, 'K': 150.0, 'T': 1.0, 'r': 0.05, 'sigma': 0.25, 'q': 0.01}
for key, val in default_vals.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ==========================================
# YFINANCE AUTO-FILL LOGIC
# ==========================================
def fetch_live_data():
    ticker_symbol = st.session_state.ticker_input.strip().upper()
    if not ticker_symbol:
        return
        
    try:
        # Fetch Stock Data
        stock = yf.Ticker(ticker_symbol)
        hist = stock.history(period="1y")
        
        if hist.empty:
            st.sidebar.error("Ticker not found.")
            return
            
        # 1. Spot Price
        current_price = hist['Close'].iloc[-1]
        
        # 2. Historical Volatility (Annualized)
        returns = np.log(hist['Close'] / hist['Close'].shift(1))
        historical_vol = returns.std() * np.sqrt(252)
        
        # 3. Dividend Yield
        div_yield = stock.info.get('dividendYield', 0.0)
        if div_yield is None: div_yield = 0.0
        
        # 4. Risk-Free Rate (Using 13-week Treasury Bill ^IRX)
        irx = yf.Ticker("^IRX")
        irx_hist = irx.history(period="5d")
        rf_rate = irx_hist['Close'].iloc[-1] / 100.0 if not irx_hist.empty else 0.05
        
        # Update session state values
        st.session_state['S'] = float(current_price)
        st.session_state['K'] = float(round(current_price)) # Set Strike ATM
        st.session_state['sigma'] = float(historical_vol)
        st.session_state['q'] = float(div_yield)
        st.session_state['r'] = float(rf_rate)
        
    except Exception as e:
        st.sidebar.error(f"Error fetching data: {e}")

# ==========================================
# MATHEMATICAL MODELS
# ==========================================
@st.cache_data
def binomial_tree_american(S, K, T, r, sigma, q, option_type, N=100):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)

    ST = S * (u ** np.arange(N, -1, -1)) * (d ** np.arange(0, N + 1))
    
    if option_type == 'Call': C = np.maximum(ST - K, 0)
    else: C = np.maximum(K - ST, 0)

    for i in range(N - 1, -1, -1):
        ST = ST[:-1] / u 
        C_hold = np.exp(-r * dt) * (p * C[:-1] + (1 - p) * C[1:])
        if option_type == 'Call': C = np.maximum(C_hold, ST - K)
        else: C = np.maximum(C_hold, K - ST)
            
    return C[0]

@st.cache_data
def bjerksund_stensland_proxy(S, K, T, r, sigma, q, option_type):
    # Proxy using an ultra-high resolution Binomial Tree to simulate the closed-form math limit
    return binomial_tree_american(S, K, T, r, sigma, q, option_type, N=1500)

@st.cache_data
def finite_difference_american(S0, K, T, r, sigma, q, option_type, M=50, N=2500):
    S_max = S0 * 2.5
    ds = S_max / M
    dt = T / N
    
    S = np.linspace(0, S_max, M + 1)
    grid = np.zeros((N + 1, M + 1))
    
    if option_type == 'Call': grid[-1] = np.maximum(S - K, 0)
    else: grid[-1] = np.maximum(K - S, 0)
        
    for j in range(N - 1, -1, -1):
        for i in range(1, M):
            delta = (grid[j+1, i+1] - grid[j+1, i-1]) / (2 * ds)
            gamma = (grid[j+1, i+1] - 2 * grid[j+1, i] + grid[j+1, i-1]) / (ds ** 2)
            theta = 0.5 * sigma**2 * S[i]**2 * gamma + (r - q) * S[i] * delta - r * grid[j+1, i]
            grid[j, i] = grid[j+1, i] - theta * dt
            
        if option_type == 'Call':
            grid[j, 0] = 0; grid[j, M] = S_max - K
            grid[j, :] = np.maximum(grid[j, :], S - K)
        else:
            grid[j, 0] = K; grid[j, M] = 0
            grid[j, :] = np.maximum(grid[j, :], K - S)
            
    price = np.interp(S0, S, grid[0, :])
    return price, S, grid

# ==========================================
# SIDEBAR UI
# ==========================================
st.sidebar.title("📥 Live Data Auto-Fill")
st.sidebar.text_input("Enter Ticker (e.g., AAPL)", key="ticker_input")
st.sidebar.button("Fetch Real-Time Data", on_click=fetch_live_data, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.title("⚙️ Model Parameters")

option_type = st.sidebar.radio("Option Type", ["Call", "Put"], horizontal=True)

S = st.sidebar.number_input("Underlying Asset Price ($S$)", min_value=1.0, step=1.0, key='S')
K = st.sidebar.number_input("Strike Price ($K$)", min_value=1.0, step=1.0, key='K')
T = st.sidebar.slider("Time to Maturity ($T$ in years)", min_value=0.01, max_value=5.0, step=0.05, key='T')
sigma = st.sidebar.slider("Volatility ($\sigma$)", min_value=0.01, max_value=3.0, step=0.01, key='sigma')
r = st.sidebar.slider("Risk-Free Rate ($r$)", min_value=0.0, max_value=0.20, step=0.01, key='r')
q = st.sidebar.slider("Dividend Yield ($q$)", min_value=0.0, max_value=0.20, step=0.01, key='q')

st.sidebar.markdown("---")
n_binomial = st.sidebar.slider("Binomial Tree Steps", min_value=10, max_value=500, value=150, step=10)

# ==========================================
# CALCULATIONS
# ==========================================
with st.spinner("Calculating American Option Boundaries..."):
    price_binom = binomial_tree_american(S, K, T, r, sigma, q, option_type, N=n_binomial)
    price_bs = bjerksund_stensland_proxy(S, K, T, r, sigma, q, option_type)
    price_fdm, fdm_S, fdm_grid = finite_difference_american(S, K, T, r, sigma, q, option_type)

# ==========================================
# MAIN PAGE UI
# ==========================================
st.title("📉 DerivLab: American Options Engine")
st.markdown("Valuate American-style options accounting for early-exercise premiums using advanced numerical boundaries.")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Binomial Tree ({n_binomial} Steps)</div><div class="metric-value">${price_binom:.4f}</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Bjerksund-Stensland (Proxy)</div><div class="metric-value">${price_bs:.4f}</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Finite Difference (Explicit)</div><div class="metric-value">${price_fdm:.4f}</div></div>', unsafe_allow_html=True)

# ==========================================
# INTERACTIVE VISUALIZATIONS (TABS)
# ==========================================
tab1, tab2, tab3 = st.tabs(["📊 Price Sensitivity", "🕸️ Binomial Tree Visual", "🔥 FDM 3D Surface Heatmap"])

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
    
    fig_sens.update_layout(xaxis_title="Underlying Price ($S$)", yaxis_title="Option Price", template="plotly_dark", hovermode="x unified")
    st.plotly_chart(fig_sens, use_container_width=True)

with tab2:
    st.markdown("#### First 5 Steps of the Binomial Tree")
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
        x=X_nodes, y=Y_nodes, mode='markers+text', text=Node_labels, textposition="top center",
        marker=dict(size=12, color='#00FF7F', line=dict(width=2, color='white'))
    ))
    fig_tree.update_layout(xaxis_title="Time Step", yaxis_title="Asset Price ($)", template="plotly_dark", showlegend=False)
    st.plotly_chart(fig_tree, use_container_width=True)

with tab3:
    st.markdown("#### Finite Difference Grid (Option Value Surface)")
    time_steps = np.linspace(0, T, fdm_grid.shape[0])
    sub_S = fdm_S[::2] 
    sub_T = time_steps[::50] 
    sub_Grid = fdm_grid[::50, ::2]
    
    fig_surface = go.Figure(data=[go.Surface(z=sub_Grid, x=sub_S, y=sub_T, colorscale='Viridis')])
    fig_surface.update_layout(
        scene=dict(xaxis_title='Asset Price ($S$)', yaxis_title='Time to Maturity', zaxis_title='Option Value'),
        margin=dict(l=0, r=0, b=0, t=40), template="plotly_dark"
    )
    st.plotly_chart(fig_surface, use_container_width=True)
