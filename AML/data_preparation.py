
from scipy.stats import zscore
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore
from enum import Enum
import numpy as np
import streamlit as st

from utils import DataClassification, MissingValuesStrategy
import  bestvarspk.Variables_selection as  bv



class DataPreparation:
    """
    A versatile data preprocessing class designed to simplify and streamline
    common data cleaning and preparation tasks for machine learning workflows.
    
    Attributes:
        df (pd.DataFrame): The working DataFrame after loading and transformations.
        original_df (pd.DataFrame): A copy of the original DataFrame for comparison.
        target_column (str, optional): The name of the target column for supervised learning.
    """
    
    def __init__(self, dataframe= None ,file_path=None, target_column=None):
        """
        Initializes the DataPreparation class by loading a CSV file into a DataFrame.
        - Reads the CSV file using pandas.
        - Creates a backup copy of the original DataFrame.
        - Stores the target column if provided.
        """
        if dataframe.shape[0] == 0:
            raise ValueError("A file path must be provided.")
        
        self.df = dataframe   # Load data from CSV
        self.original_df = self.df.copy(True)  # Store original data for reference
        self.target_column = target_column
        self.cat_names = None
        
        if self.target_column and self.target_column not in self.df.columns:
            raise ValueError(f"Target column '{target_column}' not found in the dataset.")
    
    def drop_columns(self, columns_to_drop):
        """
        Removes specified columns from the DataFrame.
        - Ensures input is a list.
        - Drops only existing columns to prevent errors.
        """
        if not isinstance(columns_to_drop, list):
            raise ValueError("columns_to_drop should be a list of column names.")
        
        existing_columns = [col for col in columns_to_drop if col in self.df.columns]
        
        if existing_columns:
            self.df.drop(columns=existing_columns, inplace=True)
            print(f"Columns {', '.join(existing_columns)} dropped successfully.")
    
    def handle_missing_values(self, data_type, strategy, fill_value=None, missing_threshold=None):
        """Handles missing values based on data type and strategy."""

        if missing_threshold is not None:
            if not (0 <= missing_threshold <= 1):
                raise ValueError("missing_threshold must be between 0 and 1.")
            missing_percentages = self.df.isnull().mean()
            columns_to_drop = missing_percentages[missing_percentages > missing_threshold].index.tolist()
            self.df.drop(columns=columns_to_drop, inplace=True)
            print(f"Dropped columns with >{missing_threshold*100:.2f}% missing data: {columns_to_drop}")

        if data_type == DataClassification.NUMERICAL:
            if strategy == MissingValuesStrategy.MEAN:
                self.df = self.df.fillna(self.df.select_dtypes(include=np.number).mean())  
            elif strategy == MissingValuesStrategy.MEDIAN:
                self.df = self.df.fillna(self.df.select_dtypes(include=np.number).median()) 
            elif strategy == MissingValuesStrategy.CONSTANT:
                if fill_value is None:
                    raise ValueError("fill_value must be provided for constant strategy.")
                numerical_cols = self.df.select_dtypes(include=np.number).columns
                self.df.loc[:, numerical_cols] = self.df.loc[:, numerical_cols].fillna(fill_value)  
            else:
                raise ValueError(f"Invalid strategy for numerical data: {strategy}")

        elif data_type == DataClassification.CATEGORICAL:
            if strategy == MissingValuesStrategy.MODE:
                for col in self.df.select_dtypes(exclude=np.number).columns:
                    most_frequent = self.df[col].mode()[0]  # Calculate mode ONCE per column
                    self.df[col] = self.df[col].fillna(most_frequent) 
            elif strategy == MissingValuesStrategy.CONSTANT:
                if fill_value is None:
                    raise ValueError("fill_value must be provided for constant strategy.")
                categorical_cols = self.df.select_dtypes(exclude=np.number).columns
                self.df.loc[:, categorical_cols] = self.df.loc[:, categorical_cols].fillna(fill_value) 
            else:
                raise ValueError(f"Invalid strategy for categorical data: {strategy}")
        else:
            raise ValueError("Invalid data_type. Choose from DataClassification.NUMERICAL or DataClassification.CATEGORICAL")

    
    def encode_categorical(self):
        """
        Applies one-hot encoding to categorical columns.
        - Converts categorical variables into numerical format.
        """
        categorical_columns = self.df.select_dtypes(include=['object', 'category']).columns
        if categorical_columns.any():
            self.df = pd.get_dummies(self.df, columns=categorical_columns, drop_first=True)
            print(f"One-hot encoding applied to: {', '.join(categorical_columns)}")
    

    def target_treatment(self, in_class = True):
        """Converts target column to numeric if it's of object type (Pandas version)."""

        if not in_class:
            self.target_column = st.session_state['target_col']

        if pd.api.types.is_object_dtype(self.df[self.target_column]) or pd.api.types.is_bool_dtype(self.df[self.target_column]) and in_class:
            # 1. Create a DataFrame with both original and encoded values:
            mapping_df = pd.DataFrame({
                'original': self.df[self.target_column],
                'encoded': self.df[self.target_column].astype('category').cat.codes
            })

            # 2. Drop duplicates based on the 'original' column to get the correct mapping:
            self.cat_names = mapping_df.drop_duplicates(subset='original').copy()

            # 3. Use the mapping from self.cat_names to update the target column in the main DataFrame
            ziped = dict(zip(self.cat_names['original'], self.cat_names['encoded']))
            self.df[self.target_column] = self.df[self.target_column].map(ziped)

            
            st.write(f"Target column '{self.target_column}' converted to numeric codes: {ziped}")
            st.session_state['targetcat'] = f"Target column '{self.target_column}' converted to numeric codes: {ziped}"
            
            st.session_state['target_dict'] = ziped
            if st is not None:  # If using Streamlit, display in Streamlit
                st.write(f"Target column mapping (ordered):")
                st.write(self.cat_names)

        else:
            st.write(f"Target column '{self.target_column}' is already numeric or another suitable type.")
      

    def categorical_treatment(self, columns : list):
        for column in columns:
            st.write(f"Converting column '{column}' to numeric codes...")
            if pd.api.types.is_object_dtype(self.df[column]) or pd.api.types.is_bool_dtype(self.df[column]):
                # 1. Create a DataFrame with both original and encoded values:

                mapping_df = pd.DataFrame({
                    f'original_({column})': self.df[column],
                    f'encoded_({column})': self.df[column].astype('category').cat.codes
                })

                # 2. Drop duplicates based on the 'original' column to get the correct mapping:
                self.cat_names = mapping_df.drop_duplicates(subset=f'original_({column})').copy()

                # 3. Use the mapping from self.cat_names to update the target column in the main DataFrame
                self.df[column] = self.df[column].map(dict(zip(self.cat_names[f'original_({column})'], self.cat_names[f'encoded_({column})'])))

                st.write(f"Target column '{column}' converted to numeric codes: {dict(zip(self.cat_names[f'original_({column})'], self.cat_names[f'encoded_({column})']))}")
                if st is not None:  # If using Streamlit, display in Streamlit
                    st.write(f"Target column mapping (ordered):")
                    st.write(self.cat_names)

            else:
                st.write(f"Target column '{column}' is already numeric or another suitable type.")

    def get_features_and_target(self):
        """
        Splits the DataFrame into features (X) and target (y).
        - Ensures the target column is specified.
        - Returns separate feature and target DataFrames.
        """
        if not self.target_column:
            raise ValueError("Target column not specified.")
        
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]
        return X, y
 

    def select_best_features(self, threshold=0.3):
        """
        Selects the best features based on correlation with the target column.
        - Keeps features with absolute correlation greater than the specified threshold.
        - Handles both numerical and categorical variables.
        """
        if not self.target_column:
            raise ValueError("Target column not specified.")
        
        # Encode categorical variables temporarily for correlation calculation
        temp_df = pd.get_dummies(self.df, drop_first=True)
        
        # Calculate correlation matrix
        corr_matrix = temp_df.corr()
        
        # Get absolute correlations with the target column
        target_corr = corr_matrix[self.target_column].abs()
        
        # Select features with correlation greater than the threshold
        selected_features = target_corr[target_corr > threshold].index.tolist()

        filtered_columns = [k for col in selected_features for k in self.df.columns if k in col]
        
        
        if len(filtered_columns) > 1:
            # Remove the target column from the list of selected features
            filtered_columns.remove(self.target_column)

            # Keep only the selected features in the DataFrame
            self.df = self.df[filtered_columns + [self.target_column]]
            st.write(f"Selected features based on correlation threshold {threshold}: {filtered_columns}")
        else: 
            st.warning(f"No features found with correlation above {threshold} with the target column. But you can train with all features.")
        
    
    
    def scale_features(self):
        """
        Scales numerical features using MinMaxScaler.
        - Normalizes numerical columns to a range of 0 to 1.
        """
        scaler = MinMaxScaler()
        numeric_columns = self.df.select_dtypes(include=['float64', 'int64']).columns
        self.df[numeric_columns] = scaler.fit_transform(self.df[numeric_columns])
    
    def remove_outliers(self, method='iqr', threshold=1.5):
        """
        Removes outliers using either IQR or Z-score methods.
        - IQR: Removes values outside 1.5 * IQR.
        - Z-score: Removes values with a Z-score > threshold.
        """
        numeric_columns = self.df.select_dtypes(include=['float64', 'int64']).columns
        if method == 'iqr':
            for col in numeric_columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                self.df = self.df[(self.df[col] >= Q1 - threshold * IQR) & (self.df[col] <= Q3 + threshold * IQR)]
        elif method == 'zscore':
            
            z_scores = self.df[numeric_columns].apply(zscore)
            self.df = self.df[(z_scores.abs() <= threshold).all(axis=1)]
    
    def remove_duplicates(self, k='first'):
        """
        Removes duplicate rows from the DataFrame.
        - Keeps the first occurrence of duplicates by default.
        - Supports options to keep the last occurrence or all duplicates.
        """
        initial_rows = len(self.df)
        self.df.drop_duplicates(keep=k, inplace=True)
        print(f"Removed {initial_rows - len(self.df)} duplicate rows.")


    def get_dataframe(self):
        return self.df
    
    def get_categorical_names(self):
        return self.cat_names

    def summary(self):
        """
        Provides a summary of the dataset, including missing values and statistics.
        - Displays total rows, columns, and missing data percentage.
        - Prints column-wise details like unique values and data types.
        """
        total_rows = len(self.df)
        total_columns = len(self.df.columns)
        total_missing = self.df.isna().sum().sum()
        print(f"Total Rows: {total_rows}, Total Columns: {total_columns}, Missing Values: {total_missing}")
        print(self.df.info())
