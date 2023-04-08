import streamlit as st

#eda pkg
import pandas as pd
import numpy as np

# datav vsi pkg
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import plotly.express as px
from sklearn.feature_selection import SelectKBest,chi2

#load data
@st.cache
def load_data(data):
    df = pd.read_csv(data)
    return df

def run_eda_app():
    st.subheader("Explorasi Analisis Data")
    #df = pd.read_csv("data/diabetes_data.csv")
    df = load_data("data/diabetes_data.csv")
    df_normal = load_data("data/diabetes_data_normal.csv")
    age_distrib = load_data("data/age_distrib.csv")
       

    submenu  = st.sidebar.selectbox("Submenu",["Deskriptif","Plots"])
    if submenu == "Deskriptif":
        st.dataframe(df)

        with st.expander("Tipe Data"):
            st.dataframe((df.dtypes).astype(str))

        with st.expander("Deskripsi"):
            st.dataframe(df_normal.describe())
            
        with st.expander("Distribusi Kelas"):
            st.dataframe(df['class'].value_counts())

        with st.expander("Distribusi Gender"):
            st.dataframe(df['Gender'].value_counts())

        with st.expander("Distribusi gejala"):
            df_gejala = df.drop(['Age','Gender','class'],axis=1)
            st.dataframe(df_gejala.apply(pd.Series.value_counts))

        with st.expander("Pengaruh Fitur"):
            X = df_normal.drop(['class'], axis=1)
            y = df_normal['class']
            skb = SelectKBest(score_func=chi2,k=10)
            best_feature_fit = skb.fit(X,y)
            feature_scores = pd.DataFrame(best_feature_fit.scores_,columns=['Feature_Scores'])
            feature_column_names = pd.DataFrame(X.columns,columns=['Feature_name'])
            best_feat_df = pd.concat([feature_scores,feature_column_names],axis=1)
            st.dataframe(best_feat_df)
        
        with st.expander("Fitur paling berpengaruh"):
            st.dataframe(best_feat_df.nlargest(10,'Feature_Scores'))


        
        

    elif submenu == "Plots":
        st.subheader("Plots")

        #layout
        col1,col2 = st.columns([2,1])
        with col1:
            #gender dist
            with st.expander("Plot Distribusi Gender"):
                fig = plt.figure()
                sns.countplot(df['Gender'])
                st.pyplot(fig)

                gen_df  =df['Gender'].value_counts().to_frame()
                gen_df  =gen_df.reset_index()
                gen_df.columns = ["Gender","Jumlah"]
                #st.dataframe(gen_df)

                p1 = px.pie(gen_df,names='Gender',values='Jumlah')
                st.plotly_chart(p1,use_container_width=True)

            #class dist
            with st.expander("Plot Distribusi Kelas"):
                fig = plt.figure()
                sns.countplot(df['class'])
                st.pyplot(fig)

                class_df  =df['class'].value_counts().to_frame()
                class_df  =class_df.reset_index()
                class_df.columns = ["Kelas","Jumlah"]

                p2 = px.pie(class_df,names='Kelas',values='Jumlah')
                st.plotly_chart(p2,use_container_width=True)

        
        with col2:
            with st.expander("Distribusi Gender"):
                gen_df  =df['Gender'].value_counts().to_frame()
                gen_df  =gen_df.reset_index()
                gen_df.columns = ["Gender","Jumlah"]
                st.dataframe(gen_df)

            with st.expander("Distribusi Kelas"):
                st.dataframe(df['class'].value_counts())

        #age dist
        with st.expander("Distribusi Umur"):
            st.dataframe(age_distrib)
            p3 = px.bar(age_distrib,x='Umur',y='jumlah')
            st.plotly_chart(p3)

        #outlier detection
        with st.expander("Plot Deteksi Outlier"):
            fig = plt.figure()
            sns.boxplot(df['Age'])
            st.pyplot(fig)

            p4 = px.box(df,x='Age',color='Gender')
            st.plotly_chart(p4)
        
        #correlation
        with st.expander("Plot Korelasi"):
            corr_matrix = df_normal.corr()
            fig = plt.figure(figsize=(20,10))
            sns.heatmap(corr_matrix,annot=True)
            st.pyplot(fig)

            # p5 = px.imshow(corr_matrix)
            # st.plotly_chart(p5)
        
        #relation
        # with st.expander("Plot Relasi"):
        #     fig = sns.pairplot(df_normal)
        #     st.pyplot(fig)

        with st.expander("Relasi Tertinggi"):
            fig = sns.lmplot(x='polyuria',y='class',data=df_normal)
            st.pyplot(fig,use_container_width=True)

            fig2 = sns.lmplot(x='polydipsia',y='class',data=df_normal)
            st.pyplot(fig2,use_container_width=True)

        with st.expander("Relasi Terendah"):    
            fig3 = sns.lmplot(x='alopecia',y='class',data=df_normal)
            st.pyplot(fig3,use_container_width=True)

            fig4 = sns.lmplot(x='gender',y='class',data=df_normal)
            st.pyplot(fig4,use_container_width=True)
