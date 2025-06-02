import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from crypto_data_pipeline import main_pipeline, load_data
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Set page config
st.set_page_config(
    page_title="Crypto Asset Recommendation Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🚀"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        margin: 1rem 0;
    }
    
    .success-metric {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    
    .warning-card {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .stSelectbox > div > div {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    .top-asset-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4facfe;
        margin: 0.5rem 0;
        backdrop-filter: blur(10px);
    }
    
    .chart-container {
        background: rgba(255, 255, 255, 0.02);
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🚀 Crypto Asset Recommendation Dashboard</h1>', unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.markdown("### ⚙️ Dashboard Configuration")
    
    # Theme switcher with better styling
    theme = st.selectbox("🎨 Select Theme", options=["Dark", "Light"], index=0)
    
    # File uploader with better styling
    st.markdown("### 📤 Data Upload")
    uploaded_file = st.file_uploader(
        "Upload your Excel data file",
        type=["xlsx"],
        help="Upload your crypto data in Excel format (.xlsx)"
    )
    
    if uploaded_file:
        st.success("✅ File uploaded successfully!")

@st.cache_data(show_spinner=True)
def load_and_process_data(file):
    """Load and process the uploaded data file"""
    df = load_data(file)
    return df

# Main content area
if uploaded_file is not None:
    try:
        # Loading section with progress
        with st.spinner("🔄 Processing your data..."):
            df = load_and_process_data(uploaded_file)

        with st.spinner("🧠 Running AI recommendation pipeline..."):
            df_rekom, df_eval = main_pipeline(df, mode='full')
            
        st.balloons()
        st.success("🎉 Pipeline executed successfully!")

        # Key metrics section
        if 'Accuracy %' in df_eval.columns:
            total_correct = df_eval['Correct Decisions'].sum()
            total_decisions = df_eval['Total Decisions'].sum()
            overall_accuracy = (total_correct / total_decisions) * 100 if total_decisions > 0 else 0
            mean_accuracy = df_eval['Accuracy %'].mean()
            
            st.markdown("## 📊 Performance Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Overall Accuracy</h3>
                    <h1>{overall_accuracy:.2f}%</h1>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card success-metric">
                    <h3>Average Accuracy</h3>
                    <h1>{mean_accuracy:.2f}%</h1>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Total Decisions</h3>
                    <h1>{total_decisions:,}</h1>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card success-metric">
                    <h3>Correct Predictions</h3>
                    <h1>{total_correct:,}</h1>
                </div>
                """, unsafe_allow_html=True)

        # Top recommendations section
        if 'Accuracy %' in df_eval.columns:
            st.markdown("## 🏆 Top 5 Recommended Assets")
            top5_results = df_eval.sort_values(by='Accuracy %', ascending=False).head(5)
            
            for idx, row in top5_results.iterrows():
                col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
                
                with col1:
                    st.markdown(f"**{row['Ticker']}**")
                with col2:
                    st.markdown(f"**Accuracy:** {row['Accuracy %']:.2f}%")
                with col3:
                    st.markdown(f"**Decisions:** {row['Total Decisions']}")
                with col4:
                    recommendation_color = "🟢" if row['Last Recommendation'] == "BUY" else "🔴" if row['Last Recommendation'] == "SELL" else "🟡"
                    st.markdown(f"**Signal:** {recommendation_color} {row['Last Recommendation']}")
                
                st.markdown("---")

        # All recommendations table
        st.markdown("## 📈 All Asset Recommendations")
        
        # Add filtering options
        col1, col2 = st.columns([1, 1])
        with col1:
            signal_filter = st.selectbox("Filter by Signal", ["All", "BUY", "SELL", "HOLD"])
        with col2:
            sort_by = st.selectbox("Sort by", ["Ticker", "Accuracy %", "Total Decisions"])
        
        # Apply filters
        display_df = df_eval.copy()
        if signal_filter != "All":
            display_df = display_df[display_df['Last Recommendation'] == signal_filter]
        
        if sort_by == "Accuracy %" and 'Accuracy %' in display_df.columns:
            display_df = display_df.sort_values('Accuracy %', ascending=False)
        elif sort_by == "Total Decisions":
            display_df = display_df.sort_values('Total Decisions', ascending=False)
        else:
            display_df = display_df.sort_values('Ticker')
        
        # Style the dataframe
        def highlight_recommendation(val):
            if val == "BUY":
                return 'background-color: #00ff00; color: black; font-weight: bold'
            elif val == "SELL":
                return 'background-color: #ff0000; color: white; font-weight: bold'
            elif val == "HOLD":
                return 'background-color: #ffff00; color: black; font-weight: bold'
            return ''
        
        styled_df = display_df[['Ticker', 'Accuracy %', 'Total Decisions', 'Correct Decisions', 'Last Recommendation']].style.applymap(
            highlight_recommendation, subset=['Last Recommendation']
        )
        
        st.dataframe(styled_df, use_container_width=True)

        # Individual coin analysis
        st.markdown("## 🔍 Individual Asset Analysis")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            selected_coin = st.selectbox(
                "📊 Select Asset for Analysis", 
                options=sorted(df_rekom['Ticker'].unique()),
                help="Choose an asset to view detailed charts and analysis"
            )
        
        df_coin = df_rekom[df_rekom['Ticker'] == selected_coin].sort_values('Date')

        if not df_coin.empty:
            # Create interactive charts using Plotly
            st.markdown(f"### 📈 Technical Analysis for {selected_coin}")
            
            # Price chart with SMA
            fig_price = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Price & Moving Average', 'Volume'),
                vertical_spacing=0.1,
                row_width=[0.7, 0.3]
            )
            
            # Price and SMA
            fig_price.add_trace(
                go.Scatter(x=df_coin['Date'], y=df_coin['Close'], 
                          name='Close Price', line=dict(color='#00f2fe', width=2)),
                row=1, col=1
            )
            
            if 'SMA_7' in df_coin.columns:
                fig_price.add_trace(
                    go.Scatter(x=df_coin['Date'], y=df_coin['SMA_7'], 
                              name='SMA 7', line=dict(color='#fa709a', width=2)),
                    row=1, col=1
                )
            
            # Volume
            if 'Volume' in df_coin.columns:
                fig_price.add_trace(
                    go.Bar(x=df_coin['Date'], y=df_coin['Volume'], 
                           name='Volume', marker_color='rgba(102, 126, 234, 0.6)'),
                    row=2, col=1
                )
            
            fig_price.update_layout(
                title=f"Price Analysis - {selected_coin}",
                height=600,
                showlegend=True,
                template="plotly_dark" if theme == "Dark" else "plotly_white"
            )
            
            st.plotly_chart(fig_price, use_container_width=True)
            
            # Technical indicators chart
            if any(col in df_coin.columns for col in ['RSI_14', 'MACD', 'OBV']):
                st.markdown(f"### 🔧 Technical Indicators for {selected_coin}")
                
                indicators_fig = make_subplots(
                    rows=3, cols=1,
                    subplot_titles=('RSI', 'MACD', 'OBV'),
                    vertical_spacing=0.08
                )
                
                if 'RSI_14' in df_coin.columns:
                    indicators_fig.add_trace(
                        go.Scatter(x=df_coin['Date'], y=df_coin['RSI_14'], 
                                  name='RSI', line=dict(color='#4facfe')),
                        row=1, col=1
                    )
                    # Add RSI levels
                    indicators_fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
                    indicators_fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
                
                if 'MACD' in df_coin.columns:
                    indicators_fig.add_trace(
                        go.Scatter(x=df_coin['Date'], y=df_coin['MACD'], 
                                  name='MACD', line=dict(color='#667eea')),
                        row=2, col=1
                    )
                
                if 'OBV' in df_coin.columns:
                    indicators_fig.add_trace(
                        go.Scatter(x=df_coin['Date'], y=df_coin['OBV'], 
                                  name='OBV', line=dict(color='#764ba2')),
                        row=3, col=1
                    )
                
                indicators_fig.update_layout(
                    height=800,
                    showlegend=True,
                    template="plotly_dark" if theme == "Dark" else "plotly_white"
                )
                
                st.plotly_chart(indicators_fig, use_container_width=True)
            
            # Market data
            if 'MarketCap' in df_coin.columns and 'tvl' in df_coin.columns:
                st.markdown(f"### 💰 Market Data for {selected_coin}")
                
                market_fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('Market Cap', 'Total Value Locked (TVL)'),
                    specs=[[{"secondary_y": False}, {"secondary_y": False}]]
                )
                
                market_fig.add_trace(
                    go.Scatter(x=df_coin['Date'], y=df_coin['MarketCap'], 
                              name='Market Cap', line=dict(color='#00f2fe')),
                    row=1, col=1
                )
                
                market_fig.add_trace(
                    go.Scatter(x=df_coin['Date'], y=df_coin['tvl'], 
                              name='TVL', line=dict(color='#fa709a')),
                    row=1, col=2
                )
                
                market_fig.update_layout(
                    height=400,
                    showlegend=True,
                    template="plotly_dark" if theme == "Dark" else "plotly_white"
                )
                
                st.plotly_chart(market_fig, use_container_width=True)

        else:
            st.warning("⚠️ No data available for the selected asset.")

    except Exception as e:
        st.error(f"❌ Error processing data: {str(e)}")
        st.markdown("""
        <div class="warning-card">
            <h4>Troubleshooting Tips:</h4>
            <ul>
                <li>Ensure your Excel file has the correct format</li>
                <li>Check that all required columns are present</li>
                <li>Verify the crypto_data_pipeline module is properly installed</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

else:
    # Welcome screen
    st.markdown("""
    <div style="text-align: center; padding: 3rem;">
        <h2>Welcome to Crypto Asset Recommendation Dashboard! 🚀</h2>
        <p style="font-size: 1.2rem; color: #667eea;">
            Upload your crypto data file to get started with AI-powered investment recommendations.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        ### 📋 Features:
        - **Performance Analytics** - Track prediction accuracy
        - **Interactive Charts** - Explore technical indicators
        - **Portfolio Optimization** - Top asset recommendations
        """)
        
        st.markdown("""
        ### 🔧 How to use:
        1. Upload your Excel file (.xlsx) using the sidebar
        2. Wait for the pipeline to process your data
        3. Explore recommendations and analytics
        4. Analyze individual assets with interactive charts
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #667eea; padding: 1rem;">
    <p> Kelompok 5 | Crypto Asset Recommendation System</p>
</div>
""", unsafe_allow_html=True)
