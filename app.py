import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import kagglehub
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

# =========================================================================
# Configuración y Carga de Datos
# =========================================================================

st.set_page_config(page_title="AI Job Market 2025", page_icon="🤖", layout="wide")

@st.cache_data
def load_data():
    """Descarga y carga el dataset de Kaggle de manera eficiente."""
    try:
        path = kagglehub.dataset_download("bismasajjad/global-ai-job-market-and-salary-trends-2025")
        csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
        
        main_file = "ai_job_dataset1.csv"
        # Si el archvo principal no esta, cogemos el primero disponible
        if main_file not in csv_files and len(csv_files) > 0:
            main_file = csv_files[0]
            
        file_path = os.path.join(path, main_file)
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Error al cargar el dataset: {e}")
        return pd.DataFrame()

@st.cache_resource
def train_predictive_model(df):
    """Entrena un modelo Random Forest Regressor simple."""
    df_model = df.copy()
    
    # Seleccionar features relevantes para el modelo
    cat_features = ['experience_level', 'remote_ratio', 'company_location']
    
    # Convertir categorical a mumerico para el RF
    le_dict = {}
    for col in cat_features:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col].astype(str))
        le_dict[col] = le
        
    X = df_model[cat_features]
    y = df_model['salary_usd']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    metrics = {
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    }
    
    return rf, le_dict, metrics

# =========================================================================
# Componentes UI de la Aplicación
# =========================================================================

def show_kpis(df):
    """Muestra tarjetas con KPIs clave."""
    st.subheader("📊 Fichas de Rendimiento (KPIs)")
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Total de Trabajos", f"{len(df):,}")
    col2.metric("Salario Promedio (USD)", f"${df['salary_usd'].mean():,.0f}")
    col3.metric("Países (Empresas)", f"{df['company_location'].nunique()}")
    col4.metric("Títulos Únicos", f"{df['job_title'].nunique()}")

def show_data_exploration(df):
    """Componente para mostrar estadisticas principales."""
    st.markdown("---")
    st.subheader("📈 Exploración Transversal de Datos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Distribución por Nivel de Experiencia**")
        exp_counts = df['experience_level'].value_counts().reset_index()
        exp_counts.columns = ['Nivel', 'Cantidad']
        fig_exp = px.pie(exp_counts, values='Cantidad', names='Nivel', hole=0.4, 
                         color_discrete_sequence=px.colors.sequential.Teal)
        st.plotly_chart(fig_exp, use_container_width=True)
        
    with col2:
        st.markdown("**Salario Promedio por Nivel de Experiencia**")
        salary_exp = df.groupby('experience_level')['salary_usd'].mean().reset_index()
        fig_sal = px.bar(salary_exp, x='experience_level', y='salary_usd', 
                         color='experience_level', 
                         color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_sal, use_container_width=True)

    st.markdown("**Tabla de Frecuencias: Títulos de Trabajo Top 10**")
    top_jobs = df['job_title'].value_counts().head(10).reset_index()
    top_jobs.columns = ['Título del Puesto', 'Frecuencia']
    st.dataframe(top_jobs, use_container_width=True)

def show_map(df):
    """Genera Mapa Geográfico Interactivo."""
    st.markdown("---")
    st.subheader("🗺️ Mapa Global de Salarios Promedio (Empresas)")
    
    # Agrupamos por locacion
    map_data = df.groupby('company_location')['salary_usd'].mean().reset_index()
    
    fig = px.choropleth(map_data, 
                        locations="company_location", 
                        locationmode='country names', # Asume que company_location son nombres de paises
                        color="salary_usd",
                        hover_name="company_location",
                        color_continuous_scale=px.colors.sequential.Plasma,
                        title="Promedio Salarial en USD por País")
    
    # Configurar fondo y altura
    fig.update_layout(height=500, margin={"r":0,"t":40,"l":0,"b":0})
    st.plotly_chart(fig, use_container_width=True)

def show_predictive_analysis(df):
    """Muestra la sección predictiva en tiempo real."""
    st.markdown("---")
    st.subheader("🔮 Análisis Predictivo en Tiempo Real")
    st.write("Estima el **Salario en USD** usando un modelo `RandomForestRegressor` entrenado con el dataset actual.")
    
    rf, le_dict, metrics = train_predictive_model(df)
    
    st.markdown(f"**Métricas del Modelo:** R²: `{metrics['r2']:.2f}` | MAE: `${metrics['mae']:,.0f}`")
    
    st.write("### Introduce los datos de entrada")
    
    col1, col2, col3 = st.columns(3)
    
    # inputs
    with col1:
        exp = st.selectbox("Nivel de Experiencia", df['experience_level'].unique())
    with col2:
        remote = st.selectbox("Modalidad (Remote Ratio %)", df['remote_ratio'].unique())
    with col3:
        loc = st.selectbox("País de Empresa", sorted(df['company_location'].unique()))
    
    if st.button("Predecir Salario 🚀"):
        # Preparar data
        try:
            exp_enc = le_dict['experience_level'].transform([exp])[0]
            rmt_enc = le_dict['remote_ratio'].transform([str(remote)])[0]
            loc_enc = le_dict['company_location'].transform([loc])[0]
            
            input_df = pd.DataFrame([{
                'experience_level': exp_enc,
                'remote_ratio': rmt_enc,
                'company_location': loc_enc
            }])
            
            prediction = rf.predict(input_df)[0]
            st.success(f"### Salario Estimado: ${prediction:,.2f} USD")
        except Exception as e:
            st.error(f"Ocurrió un error al predecir: {e}")

# =========================================================================
# Lógica Principal (Main)
# =========================================================================

def main():
    st.title("🤖 Dashboard: AI Job Market Trends 2025")
    
    # Cargar datos
    df = load_data()
    if df.empty:
        st.stop()

    # Panel de Navegación Lateral (Filtros)
    st.sidebar.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=100)
    st.sidebar.header("🕹️ Navegación y Filtros")
    
    # Copia para filtrar
    filtered_df = df.copy()

    # Filtros interactivos
    st.sidebar.subheader("Filtros Globales")
    
    if 'employment_type' in df.columns:
        emp_type = st.sidebar.multiselect("Tipo de Empleo", options=df['employment_type'].unique(), default=df['employment_type'].unique())
        filtered_df = filtered_df[filtered_df['employment_type'].isin(emp_type)]

    if 'company_size' in df.columns:
        comp_size = st.sidebar.multiselect("Tamaño de Empresa", options=df['company_size'].unique(), default=df['company_size'].unique())
        filtered_df = filtered_df[filtered_df['company_size'].isin(comp_size)]

    # Mostramos los componentes modulares pasándole filtered_df excepto prediccion
    show_kpis(filtered_df)
    show_data_exploration(filtered_df)
    show_map(filtered_df)
    show_predictive_analysis(df) # Predictivo siempre con el dataset base para asegurar features estables

if __name__ == "__main__":
    main()
