import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go

# ==========================================
# 1. 页面配置与样式
# ==========================================
st.set_page_config(
    page_title="✈️ 智能机票定价系统",
    page_icon="✈️",
    layout="centered"
)

# 自定义 CSS 样式
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        border: none;
        padding: 10px;
        font-size: 16px;
    }
    .price-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin-top: 20px;
        border-left: 5px solid #FF4B4B;
    }
    .price-text {
        font-size: 40px;
        color: #FF4B4B;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. 加载模型与资源
# ==========================================
@st.cache_resource
def load_models():
    try:
        # 这里假设你已经运行了保存代码，生成了这三个文件
        model_economy = joblib.load('model_economy.pkl')
        model_business = joblib.load('model_business.pkl')
        feature_cols = joblib.load('feature_cols.pkl')
        return model_economy, model_business, feature_cols
    except FileNotFoundError:
        return None, None, None

model_economy, model_business, feature_cols = load_models()

# ==========================================
# 3. 核心逻辑函数
# ==========================================

def predict_price(input_data, model_e, model_b, f_cols):
    """
    单点预测函数：修复了特征列不匹配的问题
    """
    # 1. 特征工程
    df_input = pd.DataFrame([input_data])
    df_input['Is_Last_Minute'] = (df_input['Days Before Journey Date'] < 7).astype(int)
    df_input['Is_Early_Bird'] = (df_input['Days Before Journey Date'] > 45).astype(int)
    
    # One-Hot 编码
    X_processed = pd.get_dummies(df_input[f_cols], drop_first=True)
    
    # 2. 列对齐 (关键步骤)
    target_model = model_e if input_data['Class'] == 'Economy' else model_b
    model_features = target_model.get_booster().feature_names
    
    # 补全缺失列并填0
    for col in model_features:
        if col not in X_processed.columns:
            X_processed[col] = 0
            
    # 统一列顺序
    X_processed = X_processed[model_features]
    
    # 3. 预测
    price = target_model.predict(X_processed)[0]
    return price

def simulate_price_trend(input_data, model_e, model_b, f_cols):
    """
    模拟未来 60 天的价格走势
    """
    days_range = list(range(0, 61)) # 0 到 60 天
    prices = []
    
    for day in days_range:
        sim_input = input_data.copy()
        sim_input['Days Before Journey Date'] = day
        
        # 重新计算衍生特征
        sim_input['Is_Last_Minute'] = 1 if day < 7 else 0
        sim_input['Is_Early_Bird'] = 1 if day > 45 else 0
        
        # 预测逻辑复用
        df_sim = pd.DataFrame([sim_input])
        X_sim = pd.get_dummies(df_sim[f_cols], drop_first=True)
        
        target_model = model_e if input_data['Class'] == 'Economy' else model_b
        model_features = target_model.get_booster().feature_names
        
        for col in model_features:
            if col not in X_sim.columns:
                X_sim[col] = 0
        X_sim = X_sim[model_features]
        
        price = target_model.predict(X_sim)[0]
        prices.append(price)
        
    return days_range, prices

def create_interactive_chart(days, prices, current_day):
    """
    使用 Plotly 绘制交互式折线图
    """
    fig = go.Figure()
    
    # 添加价格曲线
    fig.add_trace(go.Scatter(
        x=days, 
        y=prices, 
        mode='lines', 
        name='预测价格',
        line=dict(color='#FF4B4B', width=3),
        hovertemplate='提前 %{x} 天<br>价格: ₹%{y:,.0f}<extra></extra>'
    ))
    
    # 添加当前选中天数的垂直线
    fig.add_vline(
        x=current_day, 
        line_width=2, 
        line_dash="dash", 
        line_color="blue",
        annotation_text="当前选择",
        annotation_position="top right"
    )

    # 添加“最后一刻”区域背景 (0-7天)
    fig.add_vrect(
        x0=0, x1=7, 
        fillcolor="red", 
        opacity=0.1, 
        layer="below", 
        line_width=0,
        annotation_text="最后一刻 (高价)", 
        annotation_position="top left"
    )

    # 布局美化
    fig.update_layout(
        title="📈 未来 60 天价格趋势预测",
        xaxis_title="提前预订天数",
        yaxis_title="票价 (卢比)",
        hovermode="x unified",
        template="plotly_white",
        height=450
    )
    
    return fig

# ==========================================
# 4. 界面布局
# ==========================================
st.title("✈️ 智能机票定价预测系统")
st.markdown("基于 **XGBoost 双模型架构** (经济舱/商务舱独立预测)")

if model_economy is None:
    st.error("❌ **错误：未找到模型文件**。<br>请确保 `model_economy.pkl`, `model_business.pkl`, `feature_cols.pkl` 在当前目录下。", unsafe_allow_html=True)
    st.stop()

# --- 侧边栏：输入参数 ---
with st.sidebar:
    st.header("🎛️ 航班信息配置")
    
    # 舱位选择
    cabin_class = st.radio("选择舱位等级", ["Economy", "Business"], index=0)
    
    st.divider()
    
    # 航空公司与目的地 (根据实际数据调整选项)
    airline = st.selectbox("航空公司", ["Vistara", "Air India", "Indigo", "SpiceJet", "AirAsia", "GO FIRST", "AkasaAir", "AllianceAir"])
    destination = st.selectbox("目的地", ["Bangalore", "Mumbai", "Delhi", "Chennai", "Hyderabad", "Kolkata", "London"])
    
    st.divider()
    
    # 时间相关
    days_before = st.slider("提前预订天数", 0, 60, 10)
    duration = st.number_input("飞行时长 (小时)", min_value=0.5, max_value=20.0, value=2.5, step=0.5)
    stops = st.selectbox("中转次数", [0, 1, 2, 3])

# --- 主区域：展示与预测 ---
col1, col2 = st.columns(2)
with col1:
    st.metric("目的地", destination)
with col2:
    st.metric("航空公司", airline)

# 构建输入字典
user_input = {
    'Class': cabin_class,
    'Airline': airline,
    'Destination': destination,
    'Days Before Journey Date': days_before,
    'Duration (Hours)': duration,
    'Number Of Stops': stops
}

# 预测按钮
if st.button("🔮 开始预测票价"):
    with st.spinner('正在分析市场数据...'):
        try:
            # 1. 计算当前选择的价格
            predicted_price = predict_price(user_input, model_economy, model_business, feature_cols)
            
            # 2. 计算未来 60 天的趋势数据
            trend_days, trend_prices = simulate_price_trend(user_input, model_economy, model_business, feature_cols)
            
            # --- 显示结果 ---
            st.markdown(f"""
                <div class="price-box">
                    <p style="font-size: 18px; color: #555;">预测建议价格</p>
                    <div class="price-text">₹{predicted_price:,.0f}</div>
                    <p style="color: #888; font-size: 14px;">基于 {cabin_class} 模型计算</p>
                </div>
                """, unsafe_allow_html=True)

            # --- 显示图表 ---
            st.divider()
            st.subheader("📊 价格波动分析")
            
            # 调用绘图函数
            chart = create_interactive_chart(trend_days, trend_prices, user_input['Days Before Journey Date'])
            st.plotly_chart(chart, use_container_width=True)
            
            # --- 智能建议 ---
            min_price = min(trend_prices)
            best_day = trend_days[np.argmin(trend_prices)]
            
            st.info(f"💡 **省钱小贴士**：根据模型预测，如果您能提前 **{best_day}** 天预订，可能会获得最低价格 **₹{min_price:,.0f}**。")

        except Exception as e:
            st.error(f"预测出错: {e}")

# 底部说明
st.markdown("---")
st.caption("注：本预测基于历史航班数据训练。实际价格可能受燃油费、节假日等动态因素影响。")