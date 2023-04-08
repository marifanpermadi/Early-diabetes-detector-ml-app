import streamlit as st
from keras.models import model_from_json
from keras.models import load_model
from keras import backend as K

#ml pkg
import joblib
import os

#eda pkg
import numpy as np

attrib_info = """
### Attribute Information
    - Age: 1.16-90
    - Sex: 1.Male, 2.Female
    - Polyuria: 1.Yes, 2.No.
    - Polydipsia: 1.Yes, 2.No.
    - Sudden weight loss: 1.Yes, 2.No.
    - Weakness: 1.Yes, 2.No.
    - Polyphagia: 1.Yes, 2.No.
    - Genital thrush: 1.Yes, 2.No.
    - Visual blurring: 1.Yes, 2.No.
    - Itching: 1.Yes, 2.No.
    - Irritability: 1.Yes, 2.No.
    - Delayed healing: 1.Yes, 2.No.
    - Partial paresis: 1.Yes, 2.No.
    - Muscle stiffness: 1.Yes, 2.No.
    - Alopecia: 1.Yes, 2.No.
    - Obesity: 1.Yes, 2.No.
    - Class: 1.Positive, 2.Negative.
"""
label_dict = {"No":0,"Yes":1}
gender_map = {"Female":0,"Male":1}
target_label_map = {"Negative":0,"Positive":1}

def get_lvalue(val):
    for key,value in label_dict.items():
        if val == key:
            return value

def get_gvalue(val,gender_dict):
    for key,value in gender_dict.items():
        if val == key:
            return value

#load ml model
@st.cache(allow_output_mutation=True)
def load_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
    return loaded_model

def run_ml_app():
    st.subheader("Prediksi Machine Learning")

    with st.expander("📜 Informasi Attribute"):
        st.markdown(attrib_info)

    #layout
    col1,col2 = st.columns(2)
    with col1:
        age = st.number_input("🧑 Age",16,90,20)
        gender = st.radio("⚤ Gender",["Female","Male"])
        polyuria = st.radio("📈 Polyuria",["No","Yes"])
        polydipsia = st.radio("📈 Polydipsia",["No","Yes"])
        genital_thrush = st.selectbox("📈 Genital thrush",["No","Yes"])
        weakness = st.radio("📈 Weakness",["No","Yes"])
        polyphagia = st.radio("📈 Polyphagia",["No","Yes"])
        sudden_weight_loss = st.select_slider("⚖️ Sudden weight loss",["No","Yes"])
        

    with col2:
        visual_blurring = st.selectbox("👁️ Visual blurring",["No","Yes"])
        itching = st.radio("📈 Itching",["No","Yes"])
        irritability = st.radio("📈 Irritability",["No","Yes"])
        delayed_healing = st.radio("📈 Delayed healing",["No","Yes"])
        partial_paresis = st.selectbox("📈 Partial paresis",["No","Yes"])
        muscle_stiffness = st.radio("💪 Muscle stiffness",["No","Yes"])
        alopecia = st.radio("📈 Alopecia",["No","Yes"])
        obesity = st.select_slider("📈 Obesity",["No","Yes"])

    with st.expander("📝 Ringkasan data input"):
        result = {
            'age':age,
            'gender':gender,
            'polyuria':polyuria,
            'polydipsia':polydipsia,
            'genital_thrush':genital_thrush,
            'weakness':weakness,
            'polyphagia':polyphagia,
            'sudden_weight_loss':sudden_weight_loss,
            'visual_blurring':visual_blurring,
            'itching':itching,
            'irritability':irritability,
            'delayed_healing':delayed_healing,
            'partial_paresis':partial_paresis,
            'muscle_stiffness':muscle_stiffness,
            'alopecia':alopecia,
            'obesity':obesity
        }
        st.write(result)

        encoded_input = []
        for i in result.values():
            if type(i) == int:
                encoded_input.append(i)
            elif i in ["Female","Male"]:
                res = get_gvalue(i,gender_map)
                encoded_input.append(res)
            else:
                encoded_input.append(get_lvalue(i))
        
    data_input  =np.array(encoded_input).reshape(1,-1)
    # st.write(data_input)
     
    # with st.expander("Hasil Prediksi"):      
        
    #     model = load_model("models/logistic_regression_model.pkl")
    #     prediksi = model.predict(data_input)
    #     pred_prob = model.predict_proba(data_input)
    #     # st.write(prediksi)
    #     # st.write(pred_prob)

    #     if prediksi == 1:
    #         st.warning("Positive Beresiko Diabetes!")
    #         pred_prob_score = {"Negative skor":pred_prob[0][0]*100,
    #                             "Positive skor":pred_prob[0][1]*100}
    #         st.write(pred_prob_score)
    #     else:
    #         st.success("Negative Beresiko Diabetes")
    #         pred_prob_score = {"Negative skor":pred_prob[0][0]*100,
    #                             "Positive skor":pred_prob[0][1]*100}
    #         st.write(pred_prob_score)

    
    submenu  = st.selectbox("⚙️ Pilih Machine Learning Model",["Linear Regression (akurasi: 61/100)",
                            "Logistic Regression (akurasi: 93/100)","Decission Tree (akurasi: 97/100)","Convolutional Neural Network",
                            "K Nearest Neighbors (akurasi: 89/100)","Support Vector Machine (akurasi: 96/100)","Naive Bayes (akurasi: 87/100) by. Ani",
                            "K Nearest Neighbors (akurasi: 94/100) by. Nouva"])
    if st.button("Prediksi"):
        if submenu == "Logistic Regression (akurasi: 93/100)":
            model = load_model("models/logistic_regression_model_diabetes.pkl")
            prediksi = model.predict(data_input)
            pred_prob = model.predict_proba(data_input)
            # st.write(prediksi)
            # st.write(pred_prob)

            if prediksi == 1:
                st.warning("Positive Resiko Diabetes!")
                pred_prob_score = {"Negative skor":pred_prob[0][0]*100,
                                    "Positive skor":pred_prob[0][1]*100}
                st.write(pred_prob_score)
            else:
                st.success("Negative Resiko Diabetes")
                pred_prob_score = {"Negative skor":pred_prob[0][0]*100,
                                    "Positive skor":pred_prob[0][1]*100}
                st.write(pred_prob_score)
        
        elif submenu == "Decission Tree (akurasi: 97/100)":
            model1 = load_model("models/decision_tree_model_diabetes.pkl")
            prediksi = model1.predict(data_input)
            pred_prob2 = model1.predict_proba(data_input)

            if prediksi == 1:
                st.warning("Positive Resiko Diabetes!")
                pred_prob_score2 = {"Negative skor":pred_prob2[0][0],
                                    "Positive skor":pred_prob2[0][1]}
                st.write(pred_prob_score2)
            else:
                st.success("Negative Resiko Diabetes")
                pred_prob_score2 = {"Negative skor":pred_prob2[0][0],
                                    "Positive skor":pred_prob2[0][1]}
                st.write(pred_prob_score2)

        elif submenu == "Linear Regression (akurasi: 61/100)":
            model2 = load_model("models/linear_regression_model.pkl")
            prediksi = model2.predict(data_input)

            if prediksi == 1:
                st.warning("Positive Resiko Diabetes!")
            else:
                st.success("Negative Resiko Diabetes")

        elif submenu == "Convolutional Neural Network":
            st.warning("Fitur yang anda pilih sedang dalam tahap pengembangan. Mohon cek beberapa tahun lagi.")
        
        elif submenu == "K Nearest Neighbors (akurasi: 89/100)":
            model3 = load_model("models/knn_model.pkl")
            prediksi = model3.predict(data_input)
            pred_prob3 = model3.predict_proba(data_input)

            if prediksi == 1:
                st.warning("Positive Resiko Diabetes!")
                pred_prob_score3 = {"Negative skor":pred_prob3[0][0],
                                    "Positive skor":pred_prob3[0][1]}
                st.write(pred_prob_score3)
            else:
                st.success("Negative Resiko Diabetes")
                pred_prob_score3 = {"Negative skor":pred_prob3[0][0],
                                    "Positive skor":pred_prob3[0][1]}
                st.write(pred_prob_score3)
        
        elif submenu == "Support Vector Machine (akurasi: 96/100)":
            model4 = load_model("models/svce_model.pkl")
            prediksi = model4.predict(data_input)
            pred_prob4 = model4.predict_proba(data_input)

            if prediksi == 1:
                st.warning("Positive Resiko Diabetes!")
                pred_prob_score4 = {"Negative skor":pred_prob4[0][0]*100,
                                    "Positive skor":pred_prob4[0][1]*100}
                st.write(pred_prob_score4)
            else:
                st.success("Negative Resiko Diabetes")
                pred_prob_score4 = {"Negative skor":pred_prob4[0][0]*100,
                                    "Positive skor":pred_prob4[0][1]*100}
                st.write(pred_prob_score4)

        elif submenu == "Naive Bayes (akurasi: 87/100) by. Ani":
            model5 = load_model("models/nb_model_ani2.pkl")
            prediksi = model5.predict(data_input)
            pred_prob5 = model5.predict_proba(data_input)

            if prediksi == 1:
                st.warning("Positive Resiko Diabetes!")
                pred_prob_score5 = {"Negative skor":pred_prob5[0][0]*100,
                                    "Positive skor":pred_prob5[0][1]*100}
                st.write(pred_prob_score5)
            else:
                st.success("Negative Resiko Diabetes")
                pred_prob_score5 = {"Negative skor":pred_prob5[0][0]*100,
                                    "Positive skor":pred_prob5[0][1]*100}
                st.write(pred_prob_score5)
        
        elif submenu == "K Nearest Neighbors (akurasi: 94/100) by. Nouva":
            model6 = load_model("models/knn_model_nouva.pkl")
            prediksi = model6.predict(data_input)
            pred_prob6 = model6.predict_proba(data_input)

            if prediksi == 1:
                st.warning("Positive Resiko Diabetes!")
                pred_prob_score6 = {"Negative skor":pred_prob6[0][0]*100,
                                    "Positive skor":pred_prob6[0][1]*100}
                st.write(pred_prob_score6)
            else:
                st.success("Negative Resiko Diabetes")
                pred_prob_score6 = {"Negative skor":pred_prob6[0][0]*100,
                                    "Positive skor":pred_prob6[0][1]*100}
                st.write(pred_prob_score6)
                


