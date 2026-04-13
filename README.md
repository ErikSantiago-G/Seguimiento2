# Proyecto de Análisis: AI Job Market 2025 🤖📊

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ErikSantiago-G/Seguimiento2)

Bienvenido al proyecto analítico del mercado laboral global de Inteligencia artificial, donde exploramos tendencias salariales, localizaciones empresariales y factores que influyen en la remuneración en roles tecnológicos para el 2025.

Este proyecto consolida los descubrimientos de diversos scripts de exploración de datos en:
1. Una atractiva **Landing Page HTML (`index.html`)** que enumera los pasos clave del análisis de datos.
2. Un **Dashboard Interactivo en Streamlit (`app.py`)** con visualizaciones, mapas geopolíticos y estimaciones en tiempo real basadas en Machine Learning.

## 🚀 Instalación y Ejecución

Para iniciar el proyecto localmente, asegúrate de tener Python instalado y sigue estos pasos:

1. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ejecutar el Dashboard:**
   ```bash
   streamlit run app.py
   ```

## 🗺️ Estructura del Aplicativo (app.py)

Nuestra aplicación evita el código spaghetti estructurando sus vistas y funciones a través de métodos modulares:
- `load_data()`: Descarga dinámica del dataset empleando el caché de `kagglehub`.
- `train_predictive_model()`: Construcción automatizada y segmentada de un modelo **Random Forest** usando Scikit-Learn.
- `show_kpis()`: Renderización estructurada de columnas métricas mediante `st.columns()` y `st.metric`.
- `show_map()`: Utilización de **Plotly Express** para visualizar salarios globales a través de un choropleth map dinámico.
- Navegación global controlada por la barra lateral (`st.sidebar`).

## 🖼️ Interfaz Web HTML (`index.html`)

Para la visualización del Landing se utilizó una combinación de **Glassmorphism**, selectores CSS modernos de tipografía (`Inter` y `Outfit`) y un Dark mode, prescindiendo de librerías CSS pesadas adicionales como Tailwind, según especificaciones, utilizando en su lugar un limpio `styles.css`.

---
*Desarrollado y estructurado con herramientas de data science y buenas prácticas de programación.*
