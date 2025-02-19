from data_preparation import DataPreparation, DataClassification, MissingValuesStrategy
import streamlit as st

# --- Data Handling ---
def simple_info(df):
    total_rows = len(df)
    total_columns = len(df.columns)
    total_missing = df.isna().sum().sum()
    return st.write(f"Total Rows: {total_rows}, Total Columns: {total_columns}, Missing Values: {total_missing}")


def preparation_supervised(df, col, inner = True):
    data_prep = DataPreparation(dataframe=df, target_column=col)
    data_prep.remove_outliers()
    data_prep.remove_duplicates(k="first")
    data_prep.handle_missing_values(data_type=DataClassification.CATEGORICAL, strategy=MissingValuesStrategy.MODE, missing_threshold=0.7)
    data_prep.handle_missing_values(data_type=DataClassification.NUMERICAL, strategy=MissingValuesStrategy.MEAN, missing_threshold=0.7)
    data_prep.target_treatment(in_class= inner)
    data_prep.select_best_features() # select best features based on correlation greater than 0.3

    df_cleaned = data_prep.get_dataframe()
    st.dataframe(df_cleaned.head(10))

    st.write("Cleaned Data size:" + str(df_cleaned.shape))
    simple_info(df_cleaned)

    # Store in session state
    st.session_state['data_prep'] = data_prep  
    st.session_state['d_X'], st.session_state['d_y'] = data_prep.get_features_and_target()
    st.session_state['cat_cols'] = data_prep.get_categorical_names()
    st.success("Data preparation completed successfully!")

def preparation_unsupervised(df):
    data_prep = DataPreparation(dataframe=df, target_column=None)
    data_prep.remove_duplicates(k="first")
    data_prep.handle_missing_values(data_type=DataClassification.CATEGORICAL, strategy=MissingValuesStrategy.MODE, missing_threshold=0.7)
    data_prep.handle_missing_values(data_type=DataClassification.NUMERICAL, strategy=MissingValuesStrategy.MEAN, missing_threshold=0.7)
    # data_prep.categorical_treatment(columns=st.session_state['cat_feat'])

    # Encode categorical features
    df_cleaned = data_prep.get_dataframe()
    for col in df_cleaned.select_dtypes(include=['category','object']).columns:
        df_cleaned[col] = df_cleaned[col].astype('category').cat.codes
    st.dataframe(df_cleaned.head(10))
    st.write("Cleaned Data size:" + str(df_cleaned.shape))


    # Store in session state
    st.session_state['data_prep'] = df_cleaned 