import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def tampilkan():
    # 1. Load data
    def load_data():
        df = pd.read_csv('bank_churn_data.csv', sep='\t')
        return df 

    df = load_data() 

    # 2. Show preview
    st.subheader("Preview Data")
    st.dataframe(df.head())

tampilkan()

# 2. Preprocessing sederhana
st.subheader("💡 Preprocessing dan Pelatihan Model Bank (Churn)")
st.subheader("📊 Visualisasi EDA Interaktif")

tab1, tab2, = st.tabs(["Distribusi Usia", "Gender & Level Edukasi"])

with tab1:
    st.markdown("### Usia vs Churn")
    fig, ax = plt.subplots()
    df_plot = pd.read_csv("bank_churn_data.csv")
    sns.histplot(data=df_plot, x='customer_age', hue='attrition_flag', kde=True, bins=30, palette='coolwarm', ax=ax)
    st.pyplot(fig)

with tab2:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Churn berdasarkan Gender")
        fig, ax = plt.subplots()
        sns.countplot(data=df_plot, x='gender', hue='attrition_flag', palette='Set2', ax=ax)
        st.pyplot(fig)

    with col2:
        st.markdown("### Churn berdasarkan Level Edukasi")
        fig, ax = plt.subplots()
        sns.countplot(data=df_plot, x='education_level', hue='attrition_flag', palette='Set1', ax=ax)
        st.pyplot(fig)
