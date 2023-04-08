from ml_app import run_ml_app
import streamlit as st
import streamlit.components.v1 as stc

from eda_app import run_eda_app

html_temp = """
            <div style="background-color:#8043E8;
                        padding:10px;
                        border-radius:10px">
            <h1 style="color:white;text-align:center;">Resiko Diabetes Stadium  Awal</h1>
            <h2 style="color:white;text-align:center;">Machine Learning App</h2>
            </div>
            """

desc_temp = """
            ### Resiko Diabetes Stadium Awal Detector App
            Dataset berikut berisi data-data tanda dan gejala dari pasien yang / akan mengidap Diabetes stadium awal
            #### Sumber data
                - https://archive.ics.uci.edu/ml/datasets/Early+stage+diabetes+risk+prediction+dataset.
            #### Konten App
                - EDA : Exploratory Data Analysis (Explorasi Analisis Data)
                - ML : Machine Learning Predictor (Prediksi Machine Learning)  
            """

def main():
    stc.html(html_temp)

    menu = ["Home","EDA","ML","About"]
    choice  =st.sidebar.selectbox("Menu",menu)

    if choice == "Home":
        st.subheader("Home")
        st.write(desc_temp)
    elif choice == "EDA":
        run_eda_app()
    elif choice == "ML":
        run_ml_app()
    else:
        st.subheader("About")
        st.write("Web Applikasi ini membantu anda mengenali resiko awal diri anda terhadap Diabetes.")
        st.write("Di hitung berdasarkan data-data yang telah dicantumkan pada bagian 'Home'.")
        st.write("Hubungi dokter atau fasilitas kesehatan terdekat untuk memastikan kondisi anda.")




if __name__ == '__main__':
    main()