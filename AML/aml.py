from apriori import AprioriModel
import os, joblib
from enum import Enum
from report_generator import ReportGenerator
from supervised import SupervisedLearningPipeline
from unsupervised import UnsupervisedLearningPipeline
from utils import LearningType
import streamlit as st
import shap

class TrainDataBuilder:
    """
    TrainDataBuilder: A utility class for constructing training data for different types of machine learning models.

    Methods:
        supervised_train(x, y):
            Prepares the training data for supervised learning by returning the features (x) and target variable (y).
            :param x: Features (input data) for training.
            :param y: Target variable for training.
            :return: A dictionary with "x" and "y" as keys and their corresponding values.

        unsupervised_train(x, manual_k=None):
            Prepares the training data for unsupervised learning by returning the features (x) and optional manual K value.
            :param x: Features (input data) for clustering.
            :param manual_k: Optional manually specified number of clusters (K) for unsupervised learning.
            :return: A dictionary with "x" and "manual_k" as keys and their corresponding values.

        apriori_train(x, group_columns, agg_column, aggregation_function="sum", apply_filter=None):
            Prepares the training data for the Apriori algorithm by returning the dataset along with relevant parameters.
            :param x: Input data (Pandas DataFrame) for Apriori.
            :param group_columns: Columns to group by (e.g., user and item).
            :param agg_column: Column to aggregate (e.g., purchase amount).
            :param aggregation_function: Function to aggregate the data (default is 'sum').
            :param apply_filter: Optional filtering condition (Pandas query string).
            :return: A dictionary with keys for "x", "group_columns", "agg_column", "aggregation_function", and "apply_filter".
    """

    @staticmethod
    def supervised_train(x, y):
        return {"x": x, "y": y}

    @staticmethod
    def unsupervised_train(x, manual_k=None):
        return {"x": x, "manual_k": manual_k}

    @staticmethod
    def apriori_train(x, group_columns, agg_column, aggregation_function="sum", apply_filter=None):
        return {
            "x": x,
            "group_columns": group_columns,
            "agg_column": agg_column,
            "aggregation_function": aggregation_function,
            "apply_filter": apply_filter
        }



class AutoMLPipeline:
    """
    AutoMLPipeline: A class that manages the machine learning pipeline based on the specified learning type (Supervised, Unsupervised, or Apriori).
    This class provides functionality to initialize, train, and generate reports for various machine learning models.

    Attributes:
        learning_type: The type of learning for the pipeline, represented as a LearningType enum (e.g., SUPERVISED, UNSUPERVISED, APRIORI).
        target_metric: The target metric to evaluate models, used for supervised learning (e.g., accuracy, R-squared). Default is None for unsupervised learning.
        pipeline: The specific pipeline object initialized based on the learning_type (e.g., SupervisedLearningPipeline, UnsupervisedLearningPipeline, AprioriModel).

    Methods:
        __init__(learning_type, target_metric=None):
            Initializes the AutoML pipeline based on the learning_type and optionally accepts a target_metric for supervised learning.

        train(train_data: dict):
            Trains the model based on the provided training data, which includes numeric and categorical features.
            For supervised learning, it requires the target variable 'y'.
            :param train_data: Dictionary containing training data, including features ('x') and target variable ('y') for supervised learning,
                                and additional parameters like manual_k for unsupervised learning or group_columns and aggregation settings for Apriori.

        report(target_columns=None):
            Generates and prints a report of the model performance, including visualizations and metrics.
            :param target_columns: Optional list of target columns for unsupervised learning reporting (e.g., clustering analysis).
    """

    def __init__(self, learning_type, target_metric=None):
        self.learning_type = learning_type
        self.target_metric = target_metric
        self.pipeline = None
        self.model_path = f"saved_models/{str(learning_type).split('.')[1]}_model.pkl"

        match learning_type:
            case LearningType.SUPERVISED:
                self.pipeline = SupervisedLearningPipeline(target_metric=target_metric)
            case LearningType.UNSUPERVISED:
                self.pipeline = UnsupervisedLearningPipeline()
            case LearningType.APRIORI:
                self.pipeline = AprioriModel()
    
    def train(self, train_data: dict):
        """Train the model and save it."""
        numeric_features = train_data["x"].select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_features = train_data["x"].select_dtypes(include=["object", "category"]).columns.tolist()
        print(f"Numeric Features: {numeric_features}")
        print(f"Numeric Features: {categorical_features}")
        match self.learning_type:
            case LearningType.SUPERVISED:
                st.write("Training supervised model")
                if "y" not in train_data or train_data["y"] is None:
                    raise ValueError("y is required for supervised learning.")
                self.pipeline.train(train_data["x"], train_data["y"], numeric_features, categorical_features)
                st.session_state['x_train'] = train_data["x"]  # Store x_train for SHAP

            case LearningType.UNSUPERVISED:
                st.write("Training unsupervised model")
                self.pipeline.train(train_data["x"], numeric_features, categorical_features, train_data.get("manual_k"))

            case LearningType.APRIORI:
                st.write("Training apriori model")
                self.pipeline.train(train_data["x"], train_data["group_columns"], train_data["agg_column"], train_data["aggregation_function"], train_data["apply_filter"])

        # Save trained model
        self.save_model()


    def save_model(self):
        """Save the trained model to a file."""
        os.makedirs("saved_models", exist_ok=True)
        joblib.dump(self.pipeline, self.model_path)
        print(f"Model saved to {self.model_path}")

    def load_model(self):
        """Load the model from a file."""
        if os.path.exists(self.model_path):
            self.pipeline = joblib.load(self.model_path)
            print(f"Model loaded from {self.model_path}")
        else:
            raise FileNotFoundError("No saved model found. Train the model first.")

    def make_predictions(self, input_data):
        """Make predictions using the trained model."""
       
        print(self.model_path)
        if self.pipeline is None:
            self.load_model()

        print(type(self.pipeline))

        predictions = self.pipeline.best_model.predict(input_data)
        return predictions

    def report(self, target_columns=None):
        """Generate a model report."""
        report_gen = ReportGenerator(self.learning_type, self.pipeline, self.target_metric)
        report_gen.generate_report(target_columns)