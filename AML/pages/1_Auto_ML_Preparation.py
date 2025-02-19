
import streamlit as st
import pandas as pd
import numpy as np

from aml import AutoMLPipeline, LearningType, TrainDataBuilder
from sklearn.preprocessing import FunctionTransformer
from report_generator import ReportGenerator
from supervised import TargetMetric
from data_preprocessing import preparation_supervised, preparation_unsupervised


# --- Data Handling ---
def simple_info(df):
    total_rows = len(df)
    total_columns = len(df.columns)
    total_missing = df.isna().sum().sum()
    return st.write(f"Total Rows: {total_rows}, Total Columns: {total_columns}, Missing Values: {total_missing}")

# --- Configuration and Setup ---
st.set_page_config(page_title="AutoML Preparation", page_icon="⚙️", initial_sidebar_state="expanded")



# --- Title and Description ---
st.title("AutoML Preparation")
st.write("Upload your dataset and let the magic happen!")


# Initialize data_prep and d_X, d_y in session state
if 'data_prep' not in st.session_state:
    st.session_state['data_prep'] = None
if 'd_X' not in st.session_state:
    st.session_state['d_X'] = None
if 'd_y' not in st.session_state:
    st.session_state['d_y'] = None
if 'target_dict' not in st.session_state:
    st.session_state['target_dict'] = None
if 'task_type' not in st.session_state:
    st.session_state['task_type'] = None
if 'tart_col' not in st.session_state:
    st.session_state['target_col'] = None



# --- File Upload ---
uploaded_file = st.file_uploader("Upload your Dataset (CSV or Excel)", type=["csv", "xlsx"])


if uploaded_file is not None:
    try:
        # Attempt to read as CSV first, then try Excel
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            print(f"Error reading CSV: {e}")
            df = pd.read_excel(uploaded_file)

        st.write("Data preview:")
        st.dataframe(df.head(10))  # Display first 10 rows    

        task_type = st.selectbox("Select Task Type", ["Supervised Learning", "Unsupervised Learning"])
        
    
        if task_type == "Supervised Learning":
            # --- Supervised Learning Type Selection ---
            supervised_type = st.selectbox("Select Supervised Learning Type", ["Classification", "Regression"])
            st.session_state['supervised_type'] = supervised_type
            # --- Column Selection ---
            st.subheader("Column Selection")
            all_cols = list(df.columns)

            # Target Column Selection
            target_col = st.selectbox("Select Target Column", all_cols)

            # Feature Columns (automatically exclude the target)
            feature_cols = [col for col in all_cols if col != target_col]

            # Numeric and Categorical Column Handling (Improved)
            numeric_cols = st.multiselect("Select Numeric Features", feature_cols)
            categorical_cols = st.multiselect("Select Categorical Features", feature_cols)

            if not numeric_cols and not categorical_cols:
                st.warning("Please select at least one numeric or one categorical feature.")
                st.stop()

            # --- Target Metric Selection ---
            st.subheader("Target Metric Selection")

            if supervised_type == "Regression":
                metric_options = [
                    metric.value
                    for metric in TargetMetric
                    if metric
                    in [
                        TargetMetric.RMSE,
                        TargetMetric.MSE,
                        TargetMetric.MAE,
                        TargetMetric.R2,
                    ]
                ]
            else:  # Classification
                metric_options = [
                    metric.value
                    for metric in TargetMetric
                    if metric
                    in [
                        TargetMetric.ACCURACY,
                        TargetMetric.PRECISION,
                        TargetMetric.RECALL,
                        TargetMetric.F1,
                    ]
                ]
            selected_metric = st.selectbox("Select Evaluation Metric", metric_options)

            # Store selected_metric in session state
            st.session_state['selected_metric'] = selected_metric 
            
            #.... Selected Data
            st.write("Selected:")
            selected_cols = numeric_cols + categorical_cols + [target_col]
            
            st.session_state['selected_cols']= selected_cols
            st.session_state['target_col'] = target_col
            df_selected = df[selected_cols]
            st.write("Data preview:")
            st.dataframe(df_selected.head(10))
            st.write("Selected Data size:"  + str(df_selected.shape))
            simple_info(df_selected)

            

            # --- AutoML Training ---
            if st.button("Prepare Data"):
                try:
                    transformer = FunctionTransformer(preparation_supervised(df_selected,target_col))
                    st.session_state['transformer']= transformer
                    transformer.transform(df_selected)

                except Exception as e:
                    st.error(f"An error occurred during data preparation: {e}")
                    st.exception(e)

            if st.button("AML Train") and st.session_state['data_prep'] is not None and st.session_state['d_X'] is not None and st.session_state['d_y'] is not None:
                try:
                    # Convert the selected metric string to a TargetMetric enum
                    target_metric = TargetMetric(st.session_state['selected_metric'])

                    autoML = AutoMLPipeline(LearningType.SUPERVISED, target_metric)
                    autoML.train(TrainDataBuilder.supervised_train(st.session_state['d_X'], st.session_state['d_y']))
                    st.session_state['predictor'] =autoML
                    report_generator = ReportGenerator(LearningType.SUPERVISED, autoML.pipeline, target_metric)

                    st.success("AutoML training completed successfully!")  # Provide feedback

                    # Display results in a new tab
                    tab1, tab2, tab3 = st.tabs(["Training", "Results", "Explainability"])
                    with tab1:
                        st.write("Training completed. Check the 'Results' tab for details.")
                        if supervised_type == "Classification":
                            st.write(st.session_state['targetcat'] )

                    with tab2:
                        st.subheader("AML Training Results")
                        report_generator.generate_report()
                    with tab3:
                        #report_generator.generate_explainability_report()
                        st.write('To be implemented!!')

                    

                except Exception as e:
                    st.error(f"An error occurred during AutoML training: {e}")
                    st.exception(e)

            elif st.session_state['data_prep'] is None:
                st.warning("Please prepare the data first by clicking 'Prepare Data'.")

            elif st.session_state['d_X'] is None or st.session_state['d_y'] is None:
                st.warning("Data preparation did not complete successfully.  Check for errors above.")

        elif task_type == "Unsupervised Learning":
            # --- Supervised Learning Type Selection ---
            unsupervised_type = st.selectbox("Select Supervised Learning Type", ["Clustering", "Dimensionality Reduction", "Apriori"])
            
            if unsupervised_type == "Clustering":
                # --- Column Selection ---
                st.subheader("Column Selection")
                all_cols = list(df.columns)

                # Numeric and Categorical Column Handling (Improved)
                numeric_cols = st.multiselect("Select Numeric Features", all_cols)
                categorical_cols = st.multiselect("Select Categorical Features", all_cols)

                if not numeric_cols and not categorical_cols:
                    st.warning("Please select at least one numeric or one categorical feature.")
                    st.stop()

                #.... Selected Data
                st.write("Selected:")
                selected_cols = numeric_cols + categorical_cols
                df_selected = df[selected_cols]
                st.session_state['cat_feat']= categorical_cols
                st.session_state['selected_cols']= selected_cols
                unsupervised_dim = st.multiselect("Select two dimensions to generate report", df_selected.columns )
                if not unsupervised_dim and len(unsupervised_dim) != 2:
                    st.warning("Please select two numeric features.")
                    st.stop()

                else:
                    st.session_state['dims']= unsupervised_dim

                knum = st.slider("Select number of k (Default is 10):", min_value=1, max_value=10, value=10)
                st.session_state['k_num'] = int(knum)
                
                # --- AutoML Training ---
                if st.button("Prepare Data"):
                    try:
                        st.success("Data preparation completed successfully!")
                        transformer = FunctionTransformer(preparation_unsupervised(df_selected))
                        
                        st.session_state['transformer']= transformer
                        transformer.transform(df_selected)

                    except Exception as e:
                        st.error(f"An error occurred during data preparation: {e}")
                        st.exception(e)

                if st.button("AML Train") and st.session_state['data_prep'] is not None:
                    try:
                        uautoML = AutoMLPipeline(LearningType.UNSUPERVISED)
                        uautoML.train(TrainDataBuilder.unsupervised_train(st.session_state['data_prep'], manual_k=st.session_state['k_num']))
                        st.session_state['predictor'] =uautoML
                        report_generator = ReportGenerator(LearningType.UNSUPERVISED, uautoML.pipeline)
                        
                        st.write("Selected:" + str(st.session_state['dims']))

                        st.success("AutoML training completed successfully!")  # Provide feedback

                        # Display results in a new tab
                        tab1, tab2 = st.tabs(["Training", "Results"])
                        with tab1:
                            st.write("Training completed. Check the 'Results' tab for details.")

                        with tab2:
                            st.subheader("AML Training Results")
                            report_generator.generate_report(target_columns=st.session_state['dims'],  colorBy= "variety")

                    except Exception as e:
                        st.error(f"An error occurred during AutoML training: {e}")
                        st.exception(e)

                elif st.session_state['data_prep'] is None:
                    st.warning("Please prepare the data first by clicking 'Prepare Data'.")

            elif unsupervised_type == "Dimensionality Reduction":
                st.write("To be implemented!!!!")

            elif unsupervised_type == "Apriori":
                st.write("To be implemented!!!!")

    except Exception as e:
        st.error(f"An error occurred: {e}")
