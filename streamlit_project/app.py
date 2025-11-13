import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- Carga de Datos ---
# Asegúrate de que este archivo "base_da_2_con_faltantes.csv" esté
# en la raíz de tu repositorio de GitHub, junto a este app.py
path = "base_da_2_con_faltantes.csv"

try:
    df = pd.read_csv(path, sep=None, engine='python')

    st.write('\n------> Dimensiones del dataset: ', df.shape)

    st.write('\n\n------> Información del dataset:')
    # df.info() no se muestra bien en st.write, usamos st.text o un dataframe
    # Para mostrar info() de pandas, es mejor capturarla
    import io
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)


    st.write('\n\n------> Descripción del dataset <------')

    st.write('\n--- Describe (numéricas) ---')
    st.dataframe(df.describe(include=[np.number]).T)

    st.write('\n--- Describe (objetos) ---')
    st.dataframe(df.describe(include=['object']).T)

    st.write('\n\n------> Información completa de las 5 primeras filas del dataset:')
    st.dataframe(df.head())


    #-------------------------------------------
    #     Detección de valores nulos
    #-------------------------------------------

    # Conteo y porcentaje de nulos por columna
    st.write('\n\n------> Detección de valores nulos (Original) <------')
    nulos = df.isnull().sum().sort_values(ascending=False)
    porc_nulos = (df.isnull().mean() * 100).round(2).sort_values(ascending=False)
    df_nulos = pd.concat([nulos, porc_nulos], axis=1, keys=['nulos','%_nulos'])
    st.dataframe(df_nulos.head(60))


    #--------------------------------------------------
    #       Limpieza y tratamiento de datos Nulos
    #--------------------------------------------------

    st.write('\n\n------> Limpieza y tratamiento de datos Nulos <------')

    # Creamos copia para preservar original
    df_clean = df.copy()

    # Definimos numéricas y de categorías
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df_clean.select_dtypes(include=['object','category']).columns.tolist()

    # 1) Categóricas: sustituir nulos por 'Desconocido'
    cat_con_nulos = [c for c in cat_cols if df_clean[c].isnull().sum()>0]
    st.write('\n--> Columnas categóricas con nulos: ', cat_con_nulos)
    for c in cat_con_nulos:
        df_clean[c] = df_clean[c].fillna('Desconocido')

    # 2) Numéricas: sustituir por mediana global
    num_con_nulos = [c for c in numeric_cols if df_clean[c].isnull().sum()>0]
    st.write('\n--> Columnas numéricas con nulos:', num_con_nulos)
    for c in num_con_nulos:
        med = df_clean[c].median()
        df_clean[c] = df_clean[c].fillna(med)

    # 3) Ejemplo: si existe columna con 'prov' en su nombre, usar mediana por provincia
    prov_candidates = [c for c in df_clean.columns if 'prov' in c.lower() or 'provincia' in c.lower()]
    if prov_candidates and num_con_nulos:
        prov_col = prov_candidates[0]
        # Asegurémonos de tomar una columna numérica que *tenía* nulos
        if num_con_nulos:
            target_num = num_con_nulos[0]
            st.write(f'\n--> Imputando {target_num} por mediana según {prov_col} (si hay nulos)')
            # Usamos df (original) para el groupby.transform y aplicamos a df_clean
            # Nota: Esto puede ser complejo si la propia 'prov_col' tenía nulos
            try:
                df_clean[target_num] = df.groupby(prov_col)[target_num].transform(lambda x: x.fillna(x.median()))
            except Exception as e:
                st.write(f"No se pudo imputar por grupo (quizás {prov_col} tenía nulos): {e}")


    # 4) Reemplazo por moda/0 como ejemplos para columnas seleccionadas
    if cat_cols:
        example_cat = cat_cols[0]
        moda = df_clean[example_cat].mode()[0]
        df_clean[example_cat] = df_clean[example_cat].fillna(moda)
        st.write(f'Reemplazado ejemplo categórica {example_cat} nulos por moda: {moda}')
    if numeric_cols:
        example_num = numeric_cols[0]
        df_clean[example_num] = df_clean[example_num].fillna(0)
        st.write(f'Reemplazado ejemplo numérica {example_num} nulos por 0')


    st.write('\n\n--> Nulos restantes por columna (después):')
    st.dataframe(df_clean.isnull().sum().sort_values(ascending=False).head(60))


    # Conteo y porcentaje de nulos en clean
    st.write('\n\n------> Dataset limpio sin valores nulos (Resumen) <------')
    nulos_clean = df_clean.isnull().sum().sort_values(ascending=False)
    porc_nulos_clean = (df_clean.isnull().mean() * 100).round(2).sort_values(ascending=False)
    df_nulos_clean = pd.concat([nulos_clean, porc_nulos_clean], axis=1, keys=['nulos','%_nulos'])
    st.dataframe(df_nulos_clean.head(60))


    st.write('\n\n------> Análisis Exploratorio de Datos (EDA) <------')

    st.write(f'Número de columnas numéricas: {len(numeric_cols)}')
    st.write(f'Número de columnas categóricas: {len(cat_cols)}')

    # Histogramas — hasta 6 primeras numéricas
    st.subheader('Distribución de variables numéricas')
    for col in numeric_cols[:6]:
        fig, ax = plt.subplots()
        df_clean[col].dropna().hist(bins=30, ax=ax)
        ax.set_title(f'Distribución de {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Frecuencia')
        st.pyplot(fig)
        plt.close(fig) # Cierra la figura para liberar memoria

    # Barras — primeras 6 categóricas con <=30 niveles
    st.subheader('Conteo de niveles - variables categóricas')
    count = 0
    for c in cat_cols:
        if df_clean[c].nunique(dropna=True) <= 30:
            fig, ax = plt.subplots()
            df_clean[c].value_counts(dropna=False).head(30).plot(kind='bar', ax=ax)
            ax.set_title(f'Conteo niveles: {c}')
            plt.xticks(rotation=45)
            st.pyplot(fig)
            plt.close(fig) # Cierra la figura
            count += 1
        if count >= 6:
            break

    # Scatter: elegir dos numéricas
    st.subheader('Relación entre variables numéricas')
    if len(numeric_cols) >= 2:
        # Selecciona las dos primeras columnas numéricas por defecto
        x_col_default = numeric_cols[0]
        y_col_default = numeric_cols[1]
        
        # Opcional: Dejar que el usuario elija
        x_col = st.selectbox('Elige variable X:', numeric_cols, index=0)
        y_col = st.selectbox('Elige variable Y:', numeric_cols, index=1)

        fig, ax = plt.subplots()
        ax.scatter(df_clean[x_col], df_clean[y_col], alpha=0.5)
        ax.set_title(f'Scatter: {x_col} vs {y_col}')
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        st.pyplot(fig)
        plt.close(fig) # Cierra la figura
    else:
        st.write('No hay suficientes columnas numéricas para scatter.')


    st.write('# Transformaciones y creación de nuevas columnas útiles')

    st.write('\n\n------> Variables numéricas - Resumen <------')

    # Variables numéricas resumen (suma/mean)
    df_clean['numeric_sum'] = df_clean.select_dtypes(include=[np.number]).sum(axis=1)
    df_clean['numeric_mean'] = df_clean.select_dtypes(include=[np.number]).mean(axis=1)

    # Display the head of the dataframe to see all columns including the new ones
    st.dataframe(df_clean.head())

except FileNotFoundError:
    st.error(f"Error: No se pudo encontrar el archivo '{path}'.")
    st.error("Por favor, asegúrate de que el archivo CSV esté en el repositorio de GitHub, en la misma carpeta que app.py.")
except Exception as e:
    st.error(f"Ocurrió un error al cargar o procesar el archivo: {e}")