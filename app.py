import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Dashboard Comparación Modelos 3D", layout="wide")
st.title("📊 Dashboard Interactivo: Comparación de Modelos de Aprendizaje Supervisado")
st.markdown("""
Comparación de tres algoritmos para predecir **precios de casas** según tamaño y número de habitaciones:  
- **Regresión Lineal**  
- **Árbol de Decisión**  
- **Random Forest**  
Con visualización **3D interactiva** usando Plotly.
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

# --- Entrenamiento de modelos ---
linreg = LinearRegression().fit(X, y)
tree = DecisionTreeRegressor(max_depth=3, random_state=42).fit(X, y)
rf = RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42).fit(X, y)

# --- Sliders de entrada ---
st.sidebar.header("🔧 Ajusta las características de la casa")
metros = st.sidebar.slider("Metros cuadrados", 40, 250, 100, step=10)
habitaciones = st.sidebar.slider("Número de habitaciones", 1, 6, 3)

entrada = np.array([[metros, habitaciones]])

# --- Predicciones ---
pred_lin = linreg.predict(entrada)[0]
pred_tree = tree.predict(entrada)[0]
pred_rf = rf.predict(entrada)[0]

st.subheader("💡 Predicciones del modelo")
st.write(f"🏷️ **Regresión Lineal:** ${pred_lin:,.2f}")
st.write(f"🌳 **Árbol de Decisión:** ${pred_tree:,.2f}")
st.write(f"🌲 **Random Forest:** ${pred_rf:,.2f}")

# --- Crear malla para superficies ---
x_range = np.linspace(50, 220, 30)
y_range = np.linspace(1, 5, 30)
x_surf, y_surf = np.meshgrid(x_range, y_range)
grid = np.c_[x_surf.ravel(), y_surf.ravel()]

z_lin = linreg.predict(grid).reshape(x_surf.shape)
z_tree = tree.predict(grid).reshape(x_surf.shape)
z_rf = rf.predict(grid).reshape(x_surf.shape)

# --- Crear figura Plotly ---
fig = go.Figure()

# Superficie de Regresión Lineal
fig.add_trace(go.Surface(
    x=x_surf, y=y_surf, z=z_lin,
    colorscale='Viridis', opacity=0.5,
    name='Regresión Lineal', showscale=False
))

# Superficie de Árbol de Decisión
fig.add_trace(go.Surface(
    x=x_surf, y=y_surf, z=z_tree,
    colorscale='Plasma', opacity=0.5,
    name='Árbol de Decisión', showscale=False
))

# Superficie de Random Forest
fig.add_trace(go.Surface(
    x=x_surf, y=y_surf, z=z_rf,
    colorscale='Cividis', opacity=0.5,
    name='Random Forest', showscale=False
))

# Puntos de entrenamiento
fig.add_trace(go.Scatter3d(
    x=X[:,0], y=X[:,1], z=y,
    mode='markers',
    marker=dict(size=6, color='blue'),
    name='Datos reales'
))

# Predicciones del usuario
fig.add_trace(go.Scatter3d(
    x=[metros]*3, y=[habitaciones]*3, z=[pred_lin, pred_tree, pred_rf],
    mode='markers',
    marker=dict(size=8, color=['green','red','orange']),
    name='Predicción usuario'
))

# Configuración de layout
fig.update_layout(
    scene=dict(
        xaxis_title='Metros cuadrados',
        yaxis_title='Habitaciones',
        zaxis_title='Precio ($)'
    ),
    title="📈 Comparación de Modelos - Interactivo",
    legend=dict(x=0, y=1)
)

st.plotly_chart(fig, use_container_width=True)

# --- Coeficientes del modelo lineal ---
st.markdown("---")
st.subheader("📊 Coeficientes de la Regresión Lineal")
st.write({
    "Peso - Metros cuadrados": f"{linreg.coef_[0]:,.2f}",
    "Peso - Habitaciones": f"{linreg.coef_[1]:,.2f}",
    "Intersección (bias)": f"{linreg.intercept_:,.2f}"
})
st.caption("💡 El plano de regresión lineal representa cómo el modelo relaciona las variables con el precio.")
