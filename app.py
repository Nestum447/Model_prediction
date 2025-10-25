import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# --- Configuración ---
st.set_page_config(page_title="Dashboard Comparativo de Modelos", layout="wide")
st.title("📊 Dashboard Profesional: Comparación de Modelos de Predicción de Precios de Casas")
st.markdown("""
Este dashboard compara **Regresión Lineal**, **Árbol de Decisión** y **Random Forest**  
para predecir el **precio de una casa** según **metros cuadrados** y **habitaciones**.  
Incluye visualización **3D interactiva**, gráficos comparativos y tabla de datos.
""")

# --- Datos de entrenamiento ---
X = np.array([
    [60, 1],
    [80, 2],
    [100, 2],
    [120, 3],
    [150, 4],
    [200, 5]
])
y = np.array([70000, 95000, 105000, 130000, 170000, 220000])
df = pd.DataFrame({
    "Metros cuadrados": X[:,0],
    "Habitaciones": X[:,1],
    "Precio ($)": y
})

# --- Entrenar modelos ---
linreg = LinearRegression().fit(X, y)
tree = DecisionTreeRegressor(max_depth=3, random_state=42).fit(X, y)
rf = RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42).fit(X, y)

# --- Entrada del usuario ---
st.sidebar.header("🔧 Ajusta la casa")
metros = st.sidebar.slider("Metros cuadrados", 40, 250, 100, step=10)
habitaciones = st.sidebar.slider("Número de habitaciones", 1, 6, 3)
entrada = np.array([[metros, habitaciones]])

# --- Predicciones ---
pred_lin = linreg.predict(entrada)[0]
pred_tree = tree.predict(entrada)[0]
pred_rf = rf.predict(entrada)[0]

st.subheader("💡 Predicciones del modelo")
predicciones = pd.DataFrame({
    "Modelo": ["Regresión Lineal", "Árbol de Decisión", "Random Forest"],
    "Precio Predicho ($)": [pred_lin, pred_tree, pred_rf]
})
st.dataframe(predicciones.style.format({"Precio Predicho ($)": "${:,.2f}"}))

# --- Gráfico 3D interactivo ---
x_range = np.linspace(50, 220, 30)
y_range = np.linspace(1, 5, 30)
x_surf, y_surf = np.meshgrid(x_range, y_range)
grid = np.c_[x_surf.ravel(), y_surf.ravel()]

z_lin = linreg.predict(grid).reshape(x_surf.shape)
z_tree = tree.predict(grid).reshape(x_surf.shape)
z_rf = rf.predict(grid).reshape(x_surf.shape)

fig = go.Figure()

# Superficies
fig.add_trace(go.Surface(x=x_surf, y=y_surf, z=z_lin, colorscale='Viridis', opacity=0.5, name='Regresión Lineal', showscale=False))
fig.add_trace(go.Surface(x=x_surf, y=y_surf, z=z_tree, colorscale='Plasma', opacity=0.5, name='Árbol de Decisión', showscale=False))
fig.add_trace(go.Surface(x=x_surf, y=y_surf, z=z_rf, colorscale='Cividis', opacity=0.5, name='Random Forest', showscale=False))

# Datos reales
fig.add_trace(go.Scatter3d(x=X[:,0], y=X[:,1], z=y, mode='markers', marker=dict(size=6, color='blue'), name='Datos reales'))

# Predicciones usuario
fig.add_trace(go.Scatter3d(
    x=[metros]*3, y=[habitaciones]*3, z=[pred_lin, pred_tree, pred_rf],
    mode='markers',
    marker=dict(size=8, color=['green','red','orange']),
    name='Predicción usuario'
))

fig.update_layout(
    scene=dict(xaxis_title='Metros cuadrados', yaxis_title='Habitaciones', zaxis_title='Precio ($)'),
    title="📈 Comparación de Modelos 3D Interactivo",
    legend=dict(x=0, y=1),
    margin=dict(l=0, r=0, b=0, t=50)
)

st.plotly_chart(fig, use_container_width=True)

# --- Gráfico comparativo de predicciones ---
st.subheader("📊 Comparación de predicciones")
bar_fig = go.Figure(data=[
    go.Bar(name='Precio Predicho', x=predicciones["Modelo"], y=predicciones["Precio Predicho ($)"], marker_color=['green','red','orange'])
])
bar_fig.update_layout(title_text="Predicciones por Modelo", yaxis_title="Precio ($)")
st.plotly_chart(bar_fig, use_container_width=True)

# --- Tabla de datos de entrenamiento ---
st.subheader("📋 Datos de Entrenamiento")
st.dataframe(df.style.format({"Precio ($)": "${:,.2f}"}))

# --- Coeficientes regresión lineal ---
st.markdown("---")
st.subheader("📐 Coeficientes de la Regresión Lineal")
st.write({
    "Peso - Metros cuadrados": f"{linreg.coef_[0]:,.2f}",
    "Peso - Habitaciones": f"{linreg.coef_[1]:,.2f}",
    "Intersección (bias)": f"{linreg.intercept_:,.2f}"
})
st.caption("💡 El plano de regresión lineal muestra cómo el modelo relaciona las variables con el precio.")
