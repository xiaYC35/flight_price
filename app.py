import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ==========================================
# 1. 页面配置与样式
# ==========================================
st.set_page_config(
    page_title="✈️ 智能机票定价系统",
    page_icon="✈️",
    layout="centered"
)

# 自定义 CSS 样式 (让界面更好看)
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
    }
    .price-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin-top: 20px;
    }
    .price-text {
        font-size: 40px;
        color: #FF4B4B;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. 加载模型 (缓存以提高速度)
# ==========================================
@st.cache_resource
def load_models():
    try:
        model_economy = joblib.load('model_economy.pkl')
        model_business = joblib.load('model_business.pkl')
        feature_cols = joblib.load('feature_cols.pkl')
        return model_economy, model_business, feature_cols
    except:
        return None, None, None

model_economy, model_business, feature_cols = load_models()

# ==========================================
# 3. 预测逻辑 (修复版)
# ==========================================
def predict_price(input_data, model_e, model_b, f_cols):
    # 1. 特征工程
    df_input = pd.DataFrame([input_data])
    df_input['Is_Last_Minute'] = (df_input['Days Before Journey Date'] < 7).astype(int)
    df_input['Is_Early_Bird'] = (df_input['Days Before Journey Date'] > 45).astype(int)
    
    # One-Hot 编码
    X_processed = pd.get_dummies(df_input[f_cols], drop_first=True)
    
    # 2. 列对齐 (关键步骤！)
    target_model = model_e if input_data['Class'] == 'Economy' else model_b
    model_features = target_model.get_booster().feature_names
    
    # 补全缺失列
    for col in model_features:
        if col not in X_processed.columns:
            X_processed[col] = 0
            
    # 统一列顺序
    X_processed = X_processed[model_features]
    
    # 3. 预测
    price = target_model.predict(X_processed)[0]
    return price

# ==========================================
# 4. 界面布局
# ==========================================
st.title("✈️ 智能机票定价预测系统")
st.markdown("基于 **XGBoost 双模型架构** (经济舱/商务舱独立预测)")

if model_economy is None:
    st.error("❌ 错误：未找到模型文件。请确保 `model_economy.pkl` 等文件在当前目录下。")
else:
    # --- 侧边栏：输入参数 ---
    with st.sidebar:
        st.header("🎛️ 航班信息配置")
        
        # 舱位选择 (这是路由的关键)
        cabin_class = st.radio("选择舱位等级", ["Economy", "Business"], index=0)
        
        st.divider()
        
        # 航空公司与目的地
        airline = st.selectbox("航空公司", ["Vistara", "Air India", "Indigo", "SpiceJet", "AirAsia", "GO FIRST", "AkasaAir", "AllianceAir"])
        destination = st.selectbox("目的地", ["Bangalore", "Mumbai", "Delhi", "Chennai", "Hyderabad", "Kolkata", "London"]) # 根据实际数据调整
        
        st.divider()
        
        # 时间相关
        days_before = st.slider("提前预订天数", 0, 60, 10)
        duration = st.number_input("飞行时长 (小时)", min_value=0.5, max_value=20.0, value=2.5, step=0.5)
        stops = st.selectbox("中转次数", [0, 1, 2, 3])

    # --- 主区域：展示与预测 ---
    
    # 构建输入字典
    user_input = {
        'Class': cabin_class,
        'Airline': airline,
        'Destination': destination,
        'Days Before Journey Date': days_before,
        'Duration (Hours)': duration,
        'Number Of Stops': stops
    }
    
    # 显示当前配置卡片
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"🛫 **航线**: {destination}")
    with col2:
        st.info(f"✈️ **航司**: {airline}")

    # 预测按钮
    if st.button("🔮 开始预测票价"):
        with st.spinner('正在分析市场数据...'):
            try:
                # 调用预测函数
                predicted_price = predict_price(user_input, model_economy, model_business, feature_cols)
                
                # 结果显示
                st.markdown(f"""
                    <div class="price-box">
                        <p style="font-size: 18px; color: #555;">预测建议价格</p>
                        <div class="price-text">₹{predicted_price:,.0f}</div>
                        <p style="color: #888; font-size: 14px;">基于 {cabin_class} 模型计算</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # 简单的业务建议
                st.divider()
                st.subheader("💡 价格分析")
                if days_before < 7:
                    st.warning("⚠️ **临近出发**：当前处于“最后一刻”购票窗口，价格通常较高。")
                elif days_before > 45:
                    st.success("✅ **早鸟优惠**：提前预订通常能锁定更优惠的价格。")
                else:
                    st.info("ℹ️ 当前处于常规预订窗口。")
                    
            except Exception as e:
                st.error(f"预测出错: {e}")

    # 底部说明
    st.markdown("---")
    st.caption("注：本预测基于历史航班数据训练。实际价格可能受燃油费、节假日等动态因素影响。")