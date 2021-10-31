import streamlit as st
import pandas as pd
from utils import extract_text, tf_idf_calc, convert_to_csv
import datetime

st.title('TF * IDF Analyser')

languages_df = pd.read_csv('./languages.csv')

records = languages_df.to_dict("records")

info = []

with st.form(key="url_form"):
    language = st.selectbox('Lenguajes', options=records, format_func=lambda record: record["Lenguage"])
    urls = st.text_area("Lista de URLs", height=300)
    submit = st.form_submit_button('Comienza!!')

if submit:
    url_list = urls.split()
    sub_colums = ["Kw", "TF-IDF"]
    data = []

    for url in url_list:
        text = extract_text(url, language['Code'])
        tfdf_data = tf_idf_calc(text)
        data.append(tfdf_data)

    cols = pd.MultiIndex.from_product([url_list, sub_colums])
    df = pd.DataFrame(tfdf_data, columns=cols)
    st.write(df)

    st.download_button(label="Download", data=convert_to_csv(df), file_name=f"TF-IDF-{datetime.date.today()}.csv")

