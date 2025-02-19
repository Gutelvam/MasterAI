import streamlit as st
import joblib

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.markdown(
    """
    ## AUTO MACHINE LEARNING !👋

    ## Introduction

    This project undertakes the design and implementation of an automated machine learning (AutoML) framework.  The framework will address the complete machine learning lifecycle, encompassing data preparation, feature selection, model training, and validation, with integrated cross-validation.  A key objective is to establish a baseline for enhancing the speed and efficiency of machine learning development.

    DataSets for showcase:
    - [Iris Dataset](https://www.kaggle.com/uciml/iris)

    ## Goal Formulations

    This project aims to maximize efficiency and versatility in data analysis, preparation, and machine learning through the development of a comprehensive analytical solution.  Automation of key steps, such as data cleaning, transformation, and feature engineering, minimizes manual effort.  A modular pipeline builder supports both supervised and unsupervised learning, allowing users to create custom workflows for a wide range of applications.

    **Key Features**
    1. Facilitated Data Analysis and Preparation:
        - Create an intuitive framework for data cleaning, transformation, and exploration.
        - Automate key steps such as handling missing values, scaling, encoding, and feature engineering to reduce manual effort and improve efficiency.

    2. Abstract Pipeline Builder:
        - Design a modular pipeline builder that supports both supervised and unsupervised learning.
        - Enable users to define and customize pipelines for tasks like regression.

    3. Adaptability Across Problems:
        - Build a solution that is flexible enough to handle a wide range of datasets and problems.
        - Incorporate tools and methods that allow seamless integration of domain-specific requirements.

    4. Scalability and Reusability:
        - Ensure the solution can be reused and scaled for different projects and datasets.
        - Provide clear documentation and templates for quick setup and deployment.
    
    ## Directory Structure

    The project directory is organized as follows:

    ```
    AML/
    ├── AML_app.py
    ├── pages/
    │   ├── 1_Auto_ML_Preparation.py
    │   ├── 2_Auto_ML_Prediction.py
    ├── saved_models/
    ├── utils.py
    ├── aml.py
    ├── apriori.py
    ├── data_preparation.py
    ├── report_generator.py
    ├── supervised.py
    └── unsupervised.py
    ```

    - **utils.py**: Contains utility functions and helper methods used throughout the project for common tasks.

    - **aml.py**: Implements the core functionality of the Auto Machine Learning (AML) framework, including the main classes and methods for building and managing machine learning pipelines.

    - **apriori.py**: Provides the implementation of the Apriori algorithm for association rule learning, enabling the discovery of interesting relationships and patterns within datasets.

    - **data_preparation.py**: Handles data preprocessing tasks such as cleaning, feature scaling, encoding, and imputation, ensuring that the data is ready for machine learning models.
    
    - **data_preprocessing.py**: Handles data_preparation.py methods call.

    - **report_generator.py**: Generates comprehensive reports summarizing the results of data analysis and machine learning experiments, including visualizations and performance metrics.

    - **supervised.py**: Contains classes and methods specific to supervised learning tasks, such as classification and regression, including model training, evaluation, and hyperparameter tuning.

    - **unsupervised.py**: Implements unsupervised learning techniques, such as clustering and dimensionality reduction, providing tools for discovering hidden structures and patterns in the data.


    ## Supported Models
    ### Classification Models

    **GradientBoostingClassifier** – Ensemble learning for high accuracy

    **LogisticRegression** – Simple yet effective linear classification

    **KNeighborsClassifier** – Instance-based learning for pattern recognition

    **RandomForestClassifier** – Robust ensemble method for complex data

    **DecisionTreeClassifier** – Intuitive tree-based decision making

    ### Regression Models

    **LinearRegression** – Classic linear approach for numerical predictions

    **DecisionTreeRegressor** – Non-linear regression with decision trees

    **SVR (Support Vector Regressor)** – Flexible regression with kernel tricks

    **RandomForestRegressor** – Powerful ensemble method for regression tasks

   
    ### Unsupervised Learning

    **K-Means Clustering** – Efficient clustering for pattern discovery

        
    ## Plan and Design

    The approach and design of this project aim to create a modular and extensible framework for end-to-end data analysis and machine learning. The project is structured around several well-defined classes that encapsulate key functionalities, ensuring reusability and ease of integration. The primary design philosophy is to abstract complex tasks into intuitive pipelines, catering to both supervised and unsupervised learning.

    - **Data Preparation** : The DataPreparation class handles tasks such as data cleaning, feature scaling, outlier detection, missing value imputation, and encoding. This ensures that raw data is transformed into a format suitable for downstream machine learning tasks. Additionally, exploratory data analysis capabilities, such as distribution and correlation matrix visualizations, are integrated to aid decision-making during preprocessing.

    - **Pipeline Builder** : The AutoMLPipeline class serves as the core interface for users to build and execute machine learning workflows. Depending on the learning type (supervised, unsupervised, or Apriori-based), specific sub-pipelines like SupervisedLearningPipeline or UnsupervisedLearningPipeline are invoked. This design decouples high-level pipeline management from specific implementations, ensuring modularity.

    - **Customizability** : The pipelines allow users to specify configurations like the choice of imputation strategy, scaling methods, or clustering parameters. Advanced features like hyperparameter tuning and automated evaluation metrics ensure that the solution remains robust across different datasets and problem types.

    ##  Model Saving and Prediction

    In the AML_app.py file, methods have been implemented for saving the best model and generating predictions. Once the optimal model is identified, saving it ensures reusability without retraining, reducing computational costs.

    ### Saving the Best Model  

    The save_model method in the AML class uses the joblib library to save the best model to a file. The model is stored in a 'saved_models' directory with the learning type as part of the filename. This naming convention ensures that different models can be saved and loaded based on the learning type (e.g., supervised, unsupervised, Apriori).

    ### Loading the Model for Prediction

    1. Load the saved model.

    2. Use the model to make predictions on new data

    3. Display the predictions to the user.

    The prediction process is designed to be user-friendly, allowing users to upload new datasets and obtain predictions using the pre-trained model. This functionality enhances the solution's usability and practicality for real-world applications. 

    
    ## To-Do List & Rationale for Enhancements

    **1. Feature Selection & Model Optimization**

    *   **Multi-feature Importance:**
        *   **Current Issue:** The pipeline uses a correlation-based feature selection method, which may not capture non-linear relationships.
        *   **Proposed Enhancement:** Implement additional feature selection methods (e.g., Mutual Information, Recursive Feature Elimination, or SHAP values) to identify the most relevant features, even when correlation fails.
        *   **Rationale:** Ensures robust feature selection, improving model performance and generalizability.

    *   **Automated Pipeline Selection:**
        *   **Current Issue:** Pipeline selection is manual.
        *   **Proposed Enhancement:** Implement an automated approach to evaluate multiple pipeline strategies and select the one with the best performance.
        *   **Rationale:** Increases AutoML system robustness, adaptability, and efficiency.

    **2. Apriori Algorithm Integration**

    *   The Apriori method discovers frequent itemsets and association rules, valuable for market basket analysis and recommendation systems.
    *   **Current Status:** Developed but not integrated into the application.
    *   **Proposed Enhancement:** Integrate the Apriori method into the application for practical use.
    *   **Rationale:** Enables valuable insights and applications based on association rule mining.

    **3. Dimensionality Reduction**

    *   High-dimensional data can negatively impact model performance. Reducing features improves computational efficiency and model generalization.
    *   **PCA (Principal Component Analysis) Improvements:**
        *   **Current Issue:**  Potential loss of information if not carefully tuned.
        *   **Proposed Enhancement:** Allow users to define a threshold of variability for retained components.
        *   **Rationale:** Prevents information loss while improving efficiency.
    *   **MCA (Multiple Correspondence Analysis) for Categorical Data:**
        *   **Current Issue:** Existing methods are not suitable for categorical data.
        *   **Proposed Enhancement:** Implement MCA to reduce dimensions while preserving relationships between categorical features.
        *   **Rationale:** Enables dimensionality reduction for categorical data, improving model performance in relevant scenarios.
    *   **Performance Optimization:**
        *   **Rationale:** Optimized dimensionality reduction significantly enhances performance by addressing issues with redundant or irrelevant features.

    **4. Fully Autonomous AutoML Component**

    *   Automate the entire process, from data preparation to model selection and optimization, making it easier for users without deep ML expertise to deploy high-performing models.
    *   **Key Components:**
        *   Data preparation automation (including missing value handling and transformations).
        *   Integrated dimensionality reduction as a preprocessing step.
        *   Automated feature selection.
        *   Auto-selection of the best ML pipeline, optimized for different problem types.
    *   **Rationale:**  Democratizes access to high-performing ML models and streamlines the model development process.

    ## Conclusion

    This AutoML framework represents a significant advancement in accelerating machine learning development, providing a fast baseline for building and deploying solutions. Its adaptable design, incorporating data preparation, pipeline building, and model training, empowers users to rapidly address a wide range of analytical challenges.

    """
)