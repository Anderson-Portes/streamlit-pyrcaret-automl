import streamlit as st
import pandas as pd
import os
from streamlit_pandas_profiling import st_profile_report
import pycaret.classification as classific
import pycaret.regression as regression

st.set_page_config(layout='wide')

if os.path.exists('appsourcedata.csv'):
    df = pd.read_csv('appsourcedata.csv', index_col=None)

with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("AutoStreamML")
    choice = st.radio(
        "Navigation", ["Upload", "Profiling", "Machine Learning", "Download Model"])
    st.info("This project application helps you build and explore your data.")

if choice == 'Upload':
    st.title('Upload your data for modelling')
    file = st.file_uploader('Load your dataset here:')
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv('appsourcedata.csv', index=None)
        st.dataframe(df)
        st.markdown('Shape:')
        st.info(df.shape)
        st.markdown('Describe:')
        st.dataframe(df.describe())
elif choice == 'Profiling':
    st.title('Exploratory data analysis')
    st_profile_report(df.profile_report())
elif choice == 'Machine Learning':
    st.title('Machine learning models')
    type = st.selectbox('Select your data type: ', [
                        'Classification', 'Regression'])
    target = st.selectbox('Select yout target variable: ', df.columns)
    if st.button('Train model'):
        lib = classific if type == 'Classification' else regression
        lib.setup(df, target=target)
        setup_df = lib.pull()
        st.info('This is the ML experiment settings:')
        st.dataframe(setup_df)
        best_model = lib.compare_models()
        compare_df = lib.pull()
        st.info('Best ML models:')
        st.dataframe(compare_df)
        best_model
        lib.save_model(best_model, 'best_model')
elif choice == 'Download Model':
    with open('best_model.pkl', 'rb') as f:
        st.download_button('Download the model file', f, 'trained_model.pkl')
