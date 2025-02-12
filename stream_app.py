import streamlit as st
import pandas as pd
import pickle
import shap

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'xgboost_model.pkl')
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# 侧边栏输入
with st.sidebar:
    st.header("患者参数输入")
    st.subheader("卵巢储备指标")
    amh = st.slider("AMH (ng/mL)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
    afc = st.slider("AFC (个)", min_value=0, max_value=40, value=15, step=1)
    fsh = st.slider("基础FSH (IU/L)", min_value=1.0, max_value=20.0, value=8.0, step=0.1)

    st.subheader("基础特征")
    age = st.slider("年龄 (years old)", min_value=18, max_value=50, value=30)

# 检查输入值是否为有效的数字（即确保没有 NaN）
if amh is None or afc is None or fsh is None or age is None:
    st.error("输入值不能为空，请重新输入")
else:
    # 创建输入数据框
    input_data = pd.DataFrame({
        'AMH': [amh],
        'AFC': [afc],
        'FSH': [fsh],
        'age': [age],
    })

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

    with col2:
        if 'prob' in locals():
            st.subheader("风险因素解析")

            # SHAP解释
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_data)

            # 可视化设置
            st.set_option('deprecation.showPyplotGlobalUse', False)  # 隐藏警告
            shap.summary_plot(shap_values, input_data, plot_type="bar", show=False)
            st.pyplot()

            # 特征解释文本
            st.markdown("""
            **特征说明**:
            - 正值增加风险，负值降低风险
            - AMH/AFC是主要预测因子，BMI呈U型影响
            """)

# 注意事项
st.markdown("---")
st.warning("""
**使用限制**:
1. 适用于未接受过卵巢手术的患者
2. 多囊卵巢患者需结合超声评估
3. 最终决策需结合临床判断
""")
