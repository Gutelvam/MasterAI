import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import shap

# from aml import LearningType  # Remove this line
from supervised import TargetMetric
import streamlit as st
from enum import Enum
from utils import LearningType  # Import LearningType from enums.py



class ReportGenerator:
    """
    ReportGenerator: A class for generating detailed reports on machine learning models, their performance, and results.

    Attributes:
        learning_type (LearningType): The type of learning (e.g., supervised, unsupervised, or apriori).
        pipeline (Pipeline): The pipeline object containing trained models and data.
        target_metric (TargetMetric, optional): The primary metric for evaluating model performance. Default is None.

    Methods:
        __init__(learning_type, pipeline, target_metric=None):
            Initializes the ReportGenerator with the learning type, pipeline, and target metric.

        generate_report(target_columns=None):
            Generates a comprehensive report of the trained models, including performance metrics, visualizations,
            and detailed insights for supervised, unsupervised, or Apriori-based learning.

        _report_supervised():
            Generates visualizations and statistics for supervised learning models, including real vs. predicted values
            and residuals.

        _report_unsupervised(target_columns):
            Provides clustering insights for unsupervised learning models, including cluster-wise summaries and
            visualizations.

        plot_dashboard(freq_itemsets, rules):
            (For Apriori models only) Visualizes frequent itemsets and association rules using bar plots, scatter plots,
            and a network graph.

        _prepare_apriori_data(x, group_columns, agg_column, aggregation_function):
            (For Apriori models only) Prepares the dataset by grouping, aggregating, and converting it into a binary matrix.

        get_results():
            Retrieves the results of the Apriori algorithm, including frequent itemsets and association rules.

        Usage:
            - The `generate_report` method is the main entry point for creating reports.
            - Supports different report formats based on the learning type, such as:
                - Supervised learning: Metrics and residual analysis.
                - Unsupervised learning: Clustering summaries and visualizations.
                - Apriori models: Frequent itemsets and association rules.
        """


    def __init__(self, learning_type, pipeline, target_metric=None, colorBy = None):
        self.pipeline = pipeline
        self.learning_type = learning_type
        self.target_metric = target_metric
        self.colorBy = colorBy

    def generate_report(self, target_columns=None, colorBy = None):
        if not self.pipeline.results:
            raise ValueError("No results available. Train the models first.")
        if colorBy:
            self.colorBy = colorBy

        # Convert results dictionary to a DataFrame
        results_data = []
        for model_name, result in self.pipeline.results.items():
            results_data.append({
                "Model": model_name,
                "Score": result["score"]  # Explicitly extract the "score"
            })
        results_df = pd.DataFrame(results_data)

        # Determine ascending order based on target metric
        ascending = True if self.target_metric in [TargetMetric.RMSE, TargetMetric.MSE, TargetMetric.MAE] else False

        # Add a Rank column based on scores
        results_df["Rank"] = results_df["Score"].rank(ascending=ascending, method='dense')
        results_df = results_df.sort_values(by="Rank", ascending=True)

        st.write("=== AutoML Report ===")
        st.write(f"Learning Type: {self.learning_type.value}")

        # Handle the case when target_metric is None (for unsupervised learning)
        if self.target_metric is not None:
            st.write(f"Target Metric: {self.target_metric.value}")
        else:
            st.write("Target Metric: Not applicable (Unsupervised Learning)")

        for model_name, result in self.pipeline.results.items():
            st.write(f"\nModel: {model_name}")

            # For Apriori model, display frequent itemsets and rules
            if model_name == "Apriori":
                if "frequent_itemsets" in result:
                    st.write(f"\nFrequent Itemsets:")
                    st.write(result["frequent_itemsets"].head())  # Display top frequent itemsets

                if "rules" in result:
                    st.write(f"\nGenerated Association Rules:")
                    st.write(result["rules"].head())  # Display top association rules

            else:
                # For other models, show score and pipeline steps
                st.write(f"Best Score: {result['score']:.4f}")
                st.write("Pipeline Steps:")
                for step_name, step_obj in result['pipeline'].steps:
                    st.write(f"  - {step_name}: {step_obj}")
            st.write("\n")

        # Enhanced Summary Table
        st.write("\nModel Performance Summary:")
        st.dataframe(results_df)

        # Plot model performance comparison
        fig = plt.figure(figsize=(10, 6))
        sns.barplot(x="Score", y="Model", data=results_df, palette="viridis")
        if(self.target_metric == None):
            plt.title(f"Model Performance Comparison", fontsize=16)
        else:
            plt.title(f"Model Performance Comparison ({self.target_metric.name})", fontsize=16)
        plt.xlabel("Score", fontsize=14)
        plt.ylabel("Model", fontsize=14)
        plt.tight_layout()
        st.pyplot(fig)  # Use st.pyplot to display the plot

        # Details for the best model
        best_model_name = results_df.iloc[0]["Model"]
        best_model_score = results_df.iloc[0]["Score"]
        st.write(f"\nBest Model: {best_model_name} with Score: {best_model_score:.4f}")

        if self.target_metric in [TargetMetric.RMSE, TargetMetric.MSE, TargetMetric.MAE] and self.learning_type == LearningType.SUPERVISED:
                self._report_supervised()
        elif self.target_metric not in [TargetMetric.RMSE, TargetMetric.MSE, TargetMetric.MAE] and self.learning_type == LearningType.SUPERVISED:
                self._report_classification()
        elif self.learning_type == LearningType.UNSUPERVISED:
                self._report_unsupervised(target_columns)
        else:
             pass

        # match self.learning_type:
        #     case LearningType.SUPERVISED :

        #     case LearningType.UNSUPERVISED:
        #         self._report_unsupervised(target_columns)

        #     case LearningType.APRIORI:
        #         pass

    def _report_supervised(self):
        if self.pipeline.x_test is None or self.pipeline.y_test is None:
            raise ValueError("Test data not found. Ensure the train method has been executed.")

        num_models = len(self.pipeline.results)
        fig, axes = plt.subplots(num_models, 2, figsize=(14, 6 * num_models))
        fig.suptitle("Model Comparison: Real vs Predicted and Residuals", fontsize=18)

        for idx, (model_name, result) in enumerate(self.pipeline.results.items()):
            # Get the model pipeline
            model_pipeline = result["pipeline"]

            # Predict using the pipeline
            y_pred = model_pipeline.predict(self.pipeline.x_test)

            # Metrics for the model
            score = result["score"]
            print(f"\nModel: {model_name} | Score: {score:.4f}")

            # Real vs Predicted Plot
            ax1 = axes[idx, 0]
            ax1.scatter(range(len(self.pipeline.y_test)), self.pipeline.y_test, color="blue", label="Real", alpha=0.7)
            ax1.scatter(range(len(self.pipeline.y_test)), y_pred, color="orange", label="Predicted", alpha=0.7)
            ax1.set_title(f"{model_name}: Real vs Predicted", fontsize=14)
            ax1.set_xlabel("Samples", fontsize=12)
            ax1.set_ylabel("Values", fontsize=12)
            ax1.legend()

            # Residual Plot
            residuals = self.pipeline.y_test - y_pred
            ax2 = axes[idx, 1]
            ax2.scatter(range(len(self.pipeline.y_test)), residuals, color="purple", alpha=0.7)
            ax2.axhline(0, color="red", linestyle="--", linewidth=1.5)
            ax2.set_title(f"{model_name}: Residuals", fontsize=14)
            ax2.set_xlabel("Samples", fontsize=12)
            ax2.set_ylabel("Residuals", fontsize=12)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        st.pyplot(fig)  # Display the plot outside the loop


    def _report_classification(self):
        if self.pipeline.x_test is None or self.pipeline.y_test is None:
            raise ValueError("Test data not found. Ensure the train method has been executed.")
        
        num_models = len(self.pipeline.results)
        fig, axes = plt.subplots(num_models, 3, figsize=(18, 6 * num_models))
        fig.suptitle("Model Comparison: Confusion Matrix, ROC Curve, and Observed vs Predicted", fontsize=18)
        
        for idx, (model_name, result) in enumerate(self.pipeline.results.items()):
            model_pipeline = result["pipeline"]
            y_pred = model_pipeline.predict(self.pipeline.x_test)
            y_pred_prob = model_pipeline.predict_proba(self.pipeline.x_test)
            
            # Metrics
            score = result["score"]
            print(f"\nModel: {model_name} | Score: {score:.4f}")
            
            # Confusion Matrix
            cm = confusion_matrix(self.pipeline.y_test, y_pred)
            ax1 = axes[idx, 0]
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax1)
            ax1.set_title(f"{model_name}: Confusion Matrix", fontsize=14)
            ax1.set_xlabel("Predicted Label", fontsize=12)
            ax1.set_ylabel("True Label", fontsize=12)
            
            # Handle ROC Curve for Binary and Multiclass
            ax2 = axes[idx, 1]
            n_classes = len(np.unique(self.pipeline.y_test))
            if n_classes == 2:
                fpr, tpr, _ = roc_curve(self.pipeline.y_test, y_pred_prob[:, 1])
                roc_auc = auc(fpr, tpr)
                ax2.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {roc_auc:.2f}')
            else:
                y_test_bin = label_binarize(self.pipeline.y_test, classes=np.unique(self.pipeline.y_test))
                for i in range(n_classes):
                    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
                    roc_auc = auc(fpr, tpr)
                    ax2.plot(fpr, tpr, lw=2, label=f'Class {i} AUC = {roc_auc:.2f}')
            ax2.plot([0, 1], [0, 1], color='gray', linestyle='--')
            ax2.set_title(f"{model_name}: ROC Curve", fontsize=14)
            ax2.set_xlabel("False Positive Rate", fontsize=12)
            ax2.set_ylabel("True Positive Rate", fontsize=12)
            ax2.legend(loc="lower right")
            
            # Observed vs Predicted
            ax3 = axes[idx, 2]
            ax3.scatter(range(len(self.pipeline.y_test)), self.pipeline.y_test, color="blue", label="Observed", alpha=0.7)
            ax3.scatter(range(len(self.pipeline.y_test)), y_pred, color="orange", label="Predicted", alpha=0.7)
            ax3.set_title(f"{model_name}: Observed vs Predicted", fontsize=14)
            ax3.set_xlabel("Samples", fontsize=12)
            ax3.set_ylabel("Class Label", fontsize=12)
            ax3.legend()
            
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        st.pyplot(fig)

    def _report_unsupervised(self, target_columns):
        st.write("### Unsupervised Learning Report")
        
        for model_name, result in self.pipeline.results.items():
            if model_name == "Apriori":
                st.write("#### Frequent Itemsets:")
                st.dataframe(result["frequent_itemsets"])
                
                st.write("#### Association Rules:")
                st.dataframe(result["rules"])
            else:
                model_pipeline = result["pipeline"]
                labels = model_pipeline.named_steps["model"].labels_
                
                st.write(f"### Model: {model_name}")
                st.write(f"**Silhouette Score:** {result['score']:.4f}")
                
                clustered_data = self.pipeline.x_train.copy()
                clustered_data["Cluster"] = labels
                first_column = target_columns[0]
                last_column = target_columns[1]
                
                cluster_summary = clustered_data.groupby("Cluster")[[first_column, last_column]].agg(["mean", "std", "size"])
                st.write(f"#### Cluster Summary for {first_column} and {last_column}")
                st.dataframe(cluster_summary)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(x=clustered_data[first_column], y=clustered_data[last_column], hue=clustered_data[self.colorBy] if self.colorBy else labels, palette="viridis", s=50, ax=ax)
                ax.set_title(f"{model_name} Clustering Visualization (First vs Last Column)", fontsize=16)
                ax.set_xlabel(first_column, fontsize=14)
                ax.set_ylabel(last_column, fontsize=14)
                st.pyplot(fig)
                
                for cluster_id in clustered_data["Cluster"].unique():
                    st.write(f"#### Cluster {cluster_id} Statistics for {first_column} and {last_column}")
                    cluster_data = clustered_data[clustered_data["Cluster"] == cluster_id]
                    st.dataframe(cluster_data[[first_column, last_column]].describe())
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.histplot(cluster_data[first_column], kde=True, label=first_column, color='blue', ax=ax)
                    sns.histplot(cluster_data[last_column], kde=True, label=last_column, color='red', ax=ax)
                    ax.set_title(f"Feature Distributions for Cluster {cluster_id} ({first_column} & {last_column})", fontsize=16)
                    ax.set_xlabel("Feature Values", fontsize=14)
                    ax.set_ylabel("Frequency", fontsize=14)
                    ax.legend(title="Feature")
                    st.pyplot(fig)

    def generate_explainability_report(self):
        """Generate SHAP explainability report."""
        if self.learning_type == LearningType.SUPERVISED:
            if 'x_train' in st.session_state and st.session_state['x_train'] is not None:
                x_train = st.session_state['x_train']
                shap_values, explainer = self.pipeline.get_feature_importance(x_train)

                if shap_values is not None:
                    st.subheader("Feature Importance (SHAP)")

                    # Bar Plot
                    st.write("Bar Plot:")
                    fig_bar, ax_bar = plt.subplots(figsize=(10, 6))

                    # Handle different SHAP versions and shap_values types
                    try:
                        # Check if shap_values is a list
                        if isinstance(shap_values, list):
                            # Iterate over the list and plot each array separately
                            for i, sv in enumerate(shap_values):
                                # Create Explanation object
                                explanation = shap.Explanation(sv,
                                                                feature_names=x_train.columns)
                                shap.plots.bar(explanation, show=False)
                        else:
                            # Create Explanation object
                            explanation = shap.Explanation(shap_values,
                                                            feature_names=x_train.columns)
                            shap.plots.bar(explanation, show=False)
                    except Exception as e:
                        st.error(f"Error generating SHAP bar plot: {e}")
                        st.exception(e)

                    plt.tight_layout()
                    st.pyplot(fig_bar)

                    # Beeswarm Plot
                    st.write("Beeswarm Plot:")
                    fig_beeswarm, ax_beeswarm = plt.subplots(figsize=(10, 6))

                    # Create Explanation object
                    if not isinstance(shap_values, shap.Explanation):
                        shap_values = shap.Explanation(shap_values,
                                                    feature_names=x_train.columns)
                    shap.plots.beeswarm(shap_values, show=False)
                    plt.tight_layout()
                    st.pyplot(fig_beeswarm)
                else:
                    st.warning("Could not calculate SHAP values.")
            else:
                st.warning("x_train not found in session state. Train the model first.")
    
    