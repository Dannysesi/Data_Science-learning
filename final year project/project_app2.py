import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor

st.write("""
# Boston House Price Prediction App
This app predicts the **Boston House Price**!
""")
st.write('---')

# Loads the Boston House Price Dataset
boston = pd.read_csv('Data.csv')
X = boston.drop('MEDV', axis=1).values
y = boston['MEDV'].values

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')
with st.sidebar:
    def user_input_features():
    #     CRIM = st.sidebar.slider('CRIM', X.CRIM.min(), X.CRIM.max(), X.CRIM.mean())
    #     ZN = st.sidebar.slider('ZN', X.ZN.min(), X.ZN.max(), X.ZN.mean())
    #     INDUS = st.sidebar.slider('INDUS', X.INDUS.min(), X.INDUS.max(), X.INDUS.mean())
    #     CHAS = st.sidebar.slider('CHAS', X.CHAS.min(), X.CHAS.max(), X.CHAS.mean())
    #     NOX = st.sidebar.slider('NOX', X.NOX.min(), X.NOX.max(), X.NOX.mean())
    #     RM = st.sidebar.slider('RM', X.RM.min(), X.RM.max(), X.RM.mean())
    #     AGE = st.sidebar.slider('AGE', X.AGE.min(), X.AGE.max(), X.AGE.mean())
    #     DIS = st.sidebar.slider('DIS', X.DIS.min(), X.DIS.max(), X.DIS.mean())
    #     RAD = st.sidebar.slider('RAD', X.RAD.min(), X.RAD.max(), X.RAD.mean())
    #     TAX = st.sidebar.slider('TAX', X.TAX.min(), X.TAX.max(), X.TAX.mean())
    #     PTRATIO = st.sidebar.slider('PTRATIO', X.PTRATIO.min(), X.PTRATIO.max(), X.PTRATIO.mean())
    #     B = st.sidebar.slider('B', X.B.min(), X.B.max(), X.B.mean())
    #     LSTAT = st.sidebar.slider('LSTAT', X.LSTAT.min(), X.LSTAT.max(), X.LSTAT.mean())
        crim = st.slider('CRIM', min_value=0.0, max_value=100.0, value=0.0, step=1.0)
        zn = st.slider('ZN', min_value=0.0, max_value=100.0, value=0.0, step=1.0)
        indus = st.slider('INDUS', min_value=0.0, max_value=30.0, value=0.0, step=1.0)
        chas = st.selectbox('CHAS', [0, 1])
        nox = st.slider('NOX', min_value=0.0, max_value=1.0, value=0.0, step=0.01)
        rm = st.slider('RM', min_value=0.0, max_value=10.0, value=0.0, step=0.5)
        age = st.slider('AGE', min_value=0.0, max_value=100.0, value=0.0, step=0.1)
        dis = st.slider('DIS', min_value=0.0, max_value=15.0, value=0.0, step=0.01)
        rad = st.slider('RAD', min_value=0.0, max_value=30.0, value=0.0, step=1.0)
        tax = st.slider('TAX', min_value=0.0, max_value=750.0, value=0.0, step=1.0)
        ptratio = st.slider('PTRATIO', min_value=0.0, max_value=25.0, value=0.0, step=1.0)
        b = st.slider('B', min_value=0.0, max_value=400.0, value=0.0, step=1.0)
        lstat = st.slider('LSTAT', min_value=0.0, max_value=40.0, value=0.0, step=0.5)
        data = {'CRIM': crim,
                'ZN': zn,
                'INDUS': indus,
                'CHAS': chas,
                'NOX': nox,
                'RM': rm,
                'AGE': age,
                'DIS': dis,
                'RAD': rad,
                'TAX': tax,
                'PTRATIO': ptratio,
                'B': b,
                'LSTAT': lstat}
        features = pd.DataFrame(data, index=[0])
        return features


df = user_input_features()

# Main Panel

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')

# Build Regression Model
# X = pd.DataFrame(boston.data, columns=boston.feature_names)
# Y = pd.DataFrame(boston.data, columns=["MEDV"])


model = RandomForestRegressor()
model.fit(X, y)
# Apply Model to Make Prediction
prediction = model.predict(df)

st.header('Prediction of MEDV')
st.write(prediction)
st.write('---')

# Explaining the model's predictions using SHAP values
# https://github.com/slundberg/shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

st.header('Feature Importance')
plt.title('Feature importance based on SHAP values')
shap.summary_plot(shap_values, X)
st.pyplot(bbox_inches='tight')
st.write('---')

plt.title('Feature importance based on SHAP values (Bar)')
shap.summary_plot(shap_values, X, plot_type="bar")
st.pyplot(bbox_inches='tight')