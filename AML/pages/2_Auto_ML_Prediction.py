from sklearn.pipeline import FunctionTransformer
import streamlit as st

import pandas as pd
from data_preprocessing import preparation_supervised, preparation_unsupervised

# --- Configuration and Setup ---
st.set_page_config(page_title="AutoML Generator", page_icon="⚙️", layout="centered", initial_sidebar_state="expanded")

# --- Title and Description ---
st.title("AutoML Generator")
st.write("Upload your dataset to be make the prediction!")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload your Dataset (CSV or Excel)", type=["csv", "xlsx"])

# --- Data Handling ---
def simple_info(df):
    total_rows = len(df)
    total_columns = len(df.columns)
    total_missing = df.isna().sum().sum()
    return st.write(f"Total Rows: {total_rows}, Total Columns: {total_columns}, Missing Values: {total_missing}")

if uploaded_file is not None:
    try:
        # Attempt to read as CSV first, then try Excel
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            print(f"Error reading CSV: {e}")
            df = pd.read_excel(uploaded_file)

        st.write("Data preview:")
            
        selected_cols = st.session_state['selected_cols']
        if selected_cols is None:
            st.warning("No columns selected. Creating a default column with None values.")
            df_filtered = pd.DataFrame({'default_col': [None] * len(df)})
        else:
            df_filtered = df.copy()
            if st.session_state['target_col'] is not None and  st.session_state['target_col']  not in df_filtered.columns:
                df_filtered[st.session_state['target_col']] = 99999
                df_filtered = df_filtered[selected_cols]

        if st.session_state['target_col'] is not None and st.session_state['task_type'] != "Unsupervised Learning":
            print(st.session_state['target_col'])
            print(df_filtered.columns)
            transformer = FunctionTransformer(preparation_supervised(df_filtered, st.session_state['target_col'], inner = False))
            
            print(df_filtered.columns)
        else:
            transformer = FunctionTransformer(preparation_unsupervised(df_filtered))
        
        
        transformer.transform(df_filtered)
        # st.dataframe(df.head(10))

        simple_info(df_filtered)
        st.write("Data preview after transformation:")
        if st.session_state['task_type'] != "Unsupervised Learning":
            st.dataframe(df_filtered.drop(st.session_state['target_col'], axis= 1).head(10))  # Display first 10 rows
        else:
            st.dataframe(df_filtered.head(10))  # Display first 10 rows

        if st.button("Make prediction"):
            st.write("Show data to predict")
            if st.session_state['task_type'] != "Unsupervised Learning":
                st.dataframe(df_filtered.drop(st.session_state['target_col'], axis= 1))
                data_in = df_filtered.drop(st.session_state['target_col'], axis= 1).copy()
            else :
                data_in = df_filtered.copy()
            predictions = st.session_state['predictor'].make_predictions(data_in)

            data_in['prediction'] = predictions
            if st.session_state['supervised_type']  == 'Classification' and st.session_state['task_type'] != "Unsupervised Learning":
                inv_target_dict = {v: k for k, v in st.session_state['target_dict'].items()}
                data_in['prediction_label'] = data_in['prediction'].map(inv_target_dict)
            st.dataframe(data_in)

    except Exception as e:
        st.write(f"Error reading file: {e}")


