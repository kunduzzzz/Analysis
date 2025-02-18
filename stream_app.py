import streamlit as st
import pandas as pd
import pickle
import os

# 修改模型路径
model_path = os.path.join(os.getcwd(), 'random_forest_model.pkl')

# 加载模型
with open(model_path, 'rb') as file:
    model = pickle.load(file)

st.set_page_config(page_title="卵巢高反应预测计算器", layout="wide")
st.title("卵巢高反应风险预测工具")

# 侧边栏输入新增BMI
with st.sidebar:
    st.header("患者参数输入")
    st.subheader("卵巢储备指标")    
    amh = st.slider("AMH (ng/mL)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
    afc = st.slider("AFC (个)", min_value=0, max_value=40, value=15, step=1)
    fsh = st.slider("基础FSH (IU/L)", min_value=1.0, max_value=20.0, value=8.0, step=0.1)
    
    st.subheader("基础特征")
    age = st.slider("年龄 (years old)", min_value=18, max_value=50, value=30)
    bmi = st.slider("BMI (kg/m²)", min_value=15.0, max_value=40.0, value=22.0, step=0.1)  # 新增BMI输入

# 调整输入数据顺序（关键！必须与训练数据顺序一致）
input_data = pd.DataFrame({
    'AMH': [amh],
    'AFC': [afc],
    'FSH': [fsh],
    'age': [age],
    'bmi': [bmi]  # 新增BMI
})[["AMH", "AFC", "FSH", "age", "bmi"]]  # 确保列顺序与训练数据一致

# 预测与解释
col1, col2 = st.columns([1, 2])

with col1:
    if st.button("开始风险评估"):
        # 概率预测
        prob = model.predict_proba(input_data)[0][1]
        risk_level = "高风险" if prob >= 0.6 else "中风险" if prob >= 0.3 else "低风险"

        # 临床解读
        st.subheader("评估结果")

        # 使用 st.metric 显示预测概率和风险等级
        st.metric(label="预测概率", value=f"{prob:.2%}")
        st.metric(label="风险等级", value=risk_level)

        st.markdown(f"""
        **临床建议**:
        - {">5% Gn剂量减少" if risk_level == "高风险" else "常规剂量"}
        - {"建议使用拮抗剂方案" if risk_level == "高风险" else "可考虑长方案"}
        - {"建议冷冻全胚" if risk_level == "高风险" else "可考虑鲜胚移植"}
        """)


# 注意事项
st.markdown("---")
st.warning("""
**使用限制**:
1. 适用于未接受过卵巢手术的患者
2. 多囊卵巢患者需结合超声评估
3. 最终决策需结合临床判断
""")
