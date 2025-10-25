import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Comparación de Modelos", layout="centered")

st.title("🏠 Comparación de Modelos de Aprendizaje Supervisado")
st.markdown("""
Comparación de tres algoritmos para predecir el **precio de una casa** según su **tamaño** y **habitaciones**:
- **Regresión Lineal**  
- **Árbol de Decisión**  
- **Random Forest**
""")

# --- Datos de entrenamiento ---
X = np.array([
    [60, 1],
    [80, 2],
    [100, 2],
    [120, 3],
    [150, 4],
    [200, 5],
])
y = np.array([70000, 95000, 105000, 130000, 170000, 220000])

# --- Entrenamiento de modelos ---
linreg = LinearRegression()
linreg.fit(X, y)

tree = DecisionTreeRegressor(max_depth=3, random_state=42)
tree.fit(X, y)

rf = RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42)
rf.fit(X, y)

# --- Entrada del usuario ---
st.sidebar.header("🔧 Ajusta las características de la casa")
metros = st.sidebar.slider("Metros cuadrados", 40, 250, 100, step=10)
habitaciones = st.sidebar.slider("Número de habitaciones", 1, 6, 3)

entrada = np.array([[metros, habitaciones]])

# --- Predicciones ---
pred_lin = linreg.predict(entrada)[0]
pred_tree = tree.predict(entrada)[0]
pred_rf = rf.predict(entrada)[0]

st.subheader("💡 Predicciones del modelo")
st.write(f"🏷️ Regresión Lineal: ${pred_lin:,.2f}")
st.write(f"🌳 Árbol de Decisión: ${pred_tree:,.2f}")
st.write(f"🌲 Random Forest: ${pred_rf:,.2f}")

# --- Visualización ---
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Puntos reales
ax.scatter(X[:,0], X[:,1], y, color='blue', s=60, label='Datos reales')

# Crear malla para plano y superficies
x_surf, y_surf = np.meshgrid(np.linspace(50, 220, 20), np.linspace(1, 5, 20))
grid = np.c_[x_surf.ravel(), y_surf.ravel()]

# Predicciones de cada modelo
z_lin = linreg.predict(grid).reshape(x_surf.shape)
z_tree = tree.predict(grid).reshape(x_surf.shape)
z_rf = rf.predict(grid).reshape(x_surf.shape)

# Graficar superficies
ax.plot_surface(x_surf, y_surf, z_lin, alpha=0.4, cmap='viridis', edgecolor='none', label='Regresión Lineal')
ax.plot_surface(x_surf, y_surf, z_tree, alpha=0.3, cmap='plasma', edgecolor='none')
ax.plot_surface(x_surf, y_surf, z_rf, alpha=0.3, cmap='cividis', edgecolor='none')

# Predicción del usuario
ax.scatter(metros, habitaciones, pred_lin, color='green', s=100, label='Predicción LinReg')
ax.scatter(metros, habitaciones, pred_tree, color='red', s=100, label='Predicción Tree')
ax.scatter(metros, habitaciones, pred_rf, color='orange', s=100, label='Predicción RF')

# Etiquetas
ax.set_xlabel("Metros cuadrados")
ax.set_ylabel("Habitaciones")
ax.set_zlabel("Precio ($)")
ax.set_title("Comparación de Modelos")
ax.view_init(30, 30)

st.pyplot(fig)

# --- Coeficientes del modelo lineal ---
st.markdown("---")
st.subheader("📊 Coeficientes de la Regresión Lineal")
st.write({
    "Peso - Metros cuadrados": f"{linreg.coef_[0]:,.2f}",
    "Peso - Habitaciones": f"{linreg.coef_[1]:,.2f}",
    "Intersección (bias)": f"{linreg.intercept_:,.2f}"
})
