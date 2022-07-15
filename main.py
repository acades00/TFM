# Antonio Cadenas Sarmiento

import streamlit as st
import pandas as pd
from models import TextDistance
from models import FastText
from models import Plugin

# Creación del marco de la interfaz
st.set_page_config(
    page_title="Product Matching",
    page_icon="💻",
)
st.set_option('deprecation.showPyplotGlobalUse', False)

def inicializar():
    # Lectura de los conjuntos de datos
    global validation_df, test_df, train_df
    #Inicializar la base completa o reducida
    validation_df = pd.read_csv('val_red.csv')
    train_df = pd.read_csv('ent_red.csv')
    test_df = pd.read_csv('test_red.csv')

    #Unión de las columnas para su comparación
    train_df = train_df.assign(all_right=train_df.right_title.astype(str) + ',' + train_df.right_description.astype(
        str) + ',' + train_df.right_brand.astype(
        str) + ',' + train_df.right_specTableContent.astype(str)
                               )
    train_df = train_df.assign(all_left=train_df.left_title.astype(str) + ',' + train_df.left_description.astype(
        str) + ',' + train_df.left_brand.astype(
        str) + ',' + train_df.left_specTableContent.astype(str)
                               )

inicializar()


st.title("DETERMINAR PUNTO DE OPERACIÓN BLOCKING")
with st.form(key="punto_blocking"):
    opcion_modelo = st.selectbox(
        'Seleccione el modelo a utilizar PARA BLOCKING',
        ('Jaccard', 'Jaro Winkler', 'Hamming', 'FastText', 'Metodo Externo'))
    boton_fase1 = st.form_submit_button(label="COMENZAR EJECUCIÓN")


if boton_fase1:
    params = {
        "metodo": opcion_modelo,
        "param1": 0.1,
        "operacion": "blocking",
        "dataframe": train_df
    }
    if params["metodo"] == "FastText":
        metodo = FastText(**params)
        fig = metodo.blocking()
        st.pyplot(fig)
    elif params["metodo"] == "Metodo Externo":
        metodo = Plugin(**params)
        x = metodo.entrenar()
        fig = metodo.calcular()
        st.pyplot(fig)
    else:
        metodo = TextDistance(**params)
        fig = metodo.blocking()
        st.pyplot(fig)


st.title("FASE BLOCKING")
with st.form(key="block"):
    opcion_modelo = st.selectbox(
        'Seleccione el modelo a utilizar PARA BLOCKING',
        ('Jaccard', 'Jaro Winkler', 'Hamming', 'FastText', 'Metodo Externo'))
    parametro = st.number_input(
        'Introduzca el parámetro para realizar el BLOCKING',
        min_value=0.0,
        max_value=1.0
    )

    boton_blocking = st.form_submit_button(label="COMENZAR EJECUCIÓN")

if boton_blocking:
    params = {
        "metodo": opcion_modelo,
        "param1": parametro,
        "operacion": "calcular",
        "dataframe": train_df
    }
    if params["metodo"] == "FastText":
        metodo = FastText(**params)
    elif params["metodo"] == "Metodo Externo":
        metodo = Plugin(**params)
        x = metodo.entrenar()
        resultados = metodo.predecir(x)
        st.code(resultados[0][['left_title', 'right_title']].head(5))
        st.code(resultados[1])
        st.code(resultados[2])
    else:
        metodo = TextDistance(**params)
        resultados = metodo.predecir()
        st.code(resultados[0][['left_title', 'right_title']].head(5))
        st.code(resultados[1])
        st.code(resultados[2])

    archivo = resultados[0].to_csv(index=False)
    st.download_button(
        label="Descargar DataFrame",
        data=archivo,
        file_name='resultados_blocking.csv',
        mime='text/csv',
    )

st.title("DETERMINAR PUNTO DE OPERACIÓN MATCHING")
with st.form(key="punto_matching"):
    opcion_modelo2 = st.selectbox(
        'Seleccione el modelo a utilizar PARA MATCHING',
        ('Jaccard', 'Jaro Winkler', 'Hamming', 'FastText', 'Metodo Externo'))
    boton_fase2 = st.form_submit_button(label="COMENZAR EJECUCIÓN")


if boton_fase2:
    params = {
        "metodo": opcion_modelo2,
        "param1": 0.1,
        "operacion": "matching",
        "dataframe": train_df
    }
    if params["metodo"] == "FastText":
        metodo = FastText(**params)
    elif params["metodo"] == "Metodo Externo":
        metodo = Plugin(**params)
    else:
        metodo = TextDistance(**params)
        fig = metodo.blocking()
        st.pyplot(fig)

st.title("FASE MATCHING")
with st.form(key="match"):
    opcion_modelo2 = st.selectbox(
        'Seleccione el modelo a utilizar PARA MATCHING',
        ('Jaccard', 'Jaro Winkler', 'Hamming', 'FastText', 'Metodo Externo'))
    parametro = st.number_input(
        'Introduzca el parámetro para realizar el MATCHING',
        min_value=0.0,
        max_value=1.0
    )

    boton_matching = st.form_submit_button(label="COMENZAR EJECUCIÓN")

if boton_matching:
    df = pd.read_csv("resultados_blocking.csv")
    df = df.assign(all_right=train_df.right_title.astype(str) + ',' + train_df.right_description.astype(
        str) + ',' + train_df.right_brand.astype(
        str) + ',' + train_df.right_specTableContent.astype(str)
                               )
    df = df.assign(all_left=train_df.left_title.astype(str) + ',' + train_df.left_description.astype(
        str) + ',' + train_df.left_brand.astype(
        str) + ',' + train_df.left_specTableContent.astype(str)
                                )
    params = {
        "metodo": opcion_modelo2,
        "param1": parametro,
        "operacion": "calcular",
        "dataframe": df
    }
    if params["metodo"] == "FastText":
        metodo = FastText(**params)
        resultados = metodo.predecir()
        st.code(resultados[0][['left_title', 'right_title']].head(5))
        st.code(resultados[1])
        st.code(resultados[2])
    elif params["metodo"] == "Metodo Externo":
        metodo = Plugin(**params)
        x = metodo.entrenar()
        resultados = metodo.predecir(x)
        st.code(resultados[0][['left_title', 'right_title']].head(5))
        st.code(resultados[1])
        st.code(resultados[2])
    else:
        metodo = TextDistance(**params)
        resultados = metodo.predecir()
        st.code(resultados[0][['left_title', 'right_title']].head(5))
        st.code(resultados[1])
        st.code(resultados[2])

    archivo = resultados[0].to_csv(index=False)
    st.download_button(
        label="Descargar DataFrame",
        data=archivo,
        file_name='resultados_matching.csv',
        mime='text/csv',
    )