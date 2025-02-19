import numpy as np
from scipy.stats import uniform, randint
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from enum import Enum
from sklearn.metrics import make_scorer, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import shap


class TargetMetric(Enum):
    RMSE = "rmse"
    MSE = "mse"
    MAE = "mae"
    R2 = "r2"
    ACCURACY = "accuracy"
    PRECISION = "precision"  
    RECALL = "recall"     
    F1 = "f1"            
    # AUC = "auc"       
    
class SupervisedLearningPipeline:
    """
    SupervisedLearningPipeline: A class for training, selecting, and evaluating supervised learning models.

    Attributes:
        target_metric (TargetMetric): The performance metric to optimize, specified as a TargetMetric enum.
        pipeline (Pipeline or None): The best-trained pipeline with preprocessing and model selection. Initially None.
        best_model (Estimator or None): The best model chosen during training. Initially None.
        results (dict): Stores model training results, with model names as keys and dictionaries containing pipeline and score.

    Methods:
        __init__(target_metric): Initializes the pipeline with a target performance metric.
        train(x, y, numeric_features, categorical_features): Trains and evaluates multiple models, selecting the best one.
        _get_models(): Returns a list of predefined models, their parameters, and respective configurations.
        _build_pipeline(model, numeric_features, categorical_features): Constructs a preprocessing and model pipeline.
        _is_regression(): Checks if the target metric indicates a regression task.
        _get_scorer(): Retrieves the appropriate scorer function for the target metric.
        _is_better_score(score, best_score): Determines if a given score is better than the current best score.
    """

    def __init__(self, target_metric):
        self.target_metric = target_metric
        self.pipeline = None
        self.best_model = None
        self.results = {}
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None



    
    def train(self, x, y, numeric_features, categorical_features):
        models = self._get_models()
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        # --- Debugging ---
        print("Shape of x_train:", self.x_train.shape)
        print("Shape of y_train:", self.y_train.shape)
        print("Data types of x_train:\n", self.x_train.dtypes)
        print("Data types of y_train:\n", self.y_train.dtypes)

        # --- Ensure all columns are numeric before checking for infinite values ---
        # x_train_numeric = self.x_train.apply(pd.to_numeric, errors='ignore')  # Convert to numeric, non-convertible values will be NaN
        # print("Missing values in x_train:\n", x_train_numeric.isnull().sum())
        # print("Infinite values in x_train:\n", np.isinf(x_train_numeric).sum())

        print("Missing values in y_train:\n", self.y_train.isnull().sum())
        print("Infinite values in y_train:\n", np.isinf(self.y_train).sum())
        # --- End Debugging ---

        best_score = -np.inf if self._is_better_score_higher() else np.inf

        for name, model, param_grid in models:
            pipeline = self._build_pipeline(model, numeric_features, categorical_features)

            # --- Debugging ---
            print(f"Training model: {name}")
            # --- End Debugging ---

            search = RandomizedSearchCV(pipeline, param_distributions=param_grid, scoring=self._get_scorer(), cv=5, n_iter=2, random_state=42,error_score='raise') # Reduced n_iter for debugging
            try:
                search.fit(self.x_train, self.y_train)
                self.results[name] = {"pipeline": search.best_estimator_, "score": abs(search.best_score_)}
                if self._is_better_score(search.best_score_, best_score):
                    best_score = search.best_score_
                    self.best_model = search.best_estimator_
            except Exception as e:
                print(f"Error fitting model {name}: {e}")
                continue

    def _get_models(self):
        is_regression = self._is_regression()
        is_classification = self._is_classification()
        if is_regression:
            return [
                ("Random Forest", RandomForestRegressor() if is_regression else RandomForestClassifier(), {
                    "model__n_estimators": randint(10, 200),
                    "model__max_depth": randint(3, 20),
                }),
                ("Linear Regression", LinearRegression() if is_regression else LogisticRegression(), {}),
                ("Decision Tree", DecisionTreeRegressor() if self.target_metric in {TargetMetric.RMSE, TargetMetric.MSE, TargetMetric.MAE, TargetMetric.R2} else DecisionTreeClassifier(), {
                        "model__max_depth": randint(3, 20),
                        "model__min_samples_split": randint(2, 10),
                        "model__min_samples_leaf": randint(1, 5),
                }),
                ("Support Vector Machine", SVR() if self.target_metric in {TargetMetric.RMSE, TargetMetric.MSE, TargetMetric.MAE, TargetMetric.R2} else SVC(probability=True), {
                    "model__C": uniform(0.1, 10),
                    "model__gamma": uniform(0.01, 1),
                }),
            ]
        elif is_classification:
            return [
                    ("Random Forest", RandomForestClassifier(), {
                        "model__n_estimators": randint(10, 200),
                        "model__max_depth": randint(3, 20),
                        "model__min_samples_split": randint(2, 10),  # Added
                        "model__min_samples_leaf": randint(1, 5),      # Added
                        "model__criterion": ["gini", "entropy"],       # Added
                        "model__class_weight": [None, "balanced"], # Added
                    }),
                    ("Logistic Regression", LogisticRegression(max_iter=10000), { # Increased max_iter
                        "model__C": uniform(0.001, 100),       # Wider range for C
                        "model__penalty": ["l1", "l2"],     # Added regularization
                        "model__solver": ["liblinear", "saga"], # Added solvers
                        "model__class_weight": [None, "balanced"], # Added
                    }),
                    ("Decision Tree", DecisionTreeClassifier(), {
                        "model__max_depth": randint(3, 20),
                        "model__min_samples_split": randint(2, 10),
                        "model__min_samples_leaf": randint(1, 5),
                        "model__criterion": ["gini", "entropy"], # Added
                        "model__class_weight": [None, "balanced"], # Added
                    }),
                    # ("Support Vector Machine", SVC(probability=True), {
                    #     "model__C": uniform(0.1, 10),
                    #     "model__gamma": uniform(0.001, 1), # Wider range for gamma
                    #     "model__kernel": ["rbf", "linear", "poly", "sigmoid"], # Added kernels
                    #     "model__class_weight": [None, "balanced"], # Added
                    # }),
                    ("Gradient Boosting", GradientBoostingClassifier(), {
                        "model__n_estimators": randint(10, 200),
                        "model__learning_rate": uniform(0.001, 1),
                        "model__max_depth": randint(3, 20),
                        "model__min_samples_split": randint(2, 10),
                        "model__min_samples_leaf": randint(1, 5),
                    }),
                    ("K-Nearest Neighbors", KNeighborsClassifier(), {
                        "model__n_neighbors": randint(1, 30),
                        "model__weights": ["uniform", "distance"],
                        "model__p": [1, 2], # Manhattan and Euclidean distances
                    }),
                    # ("Gaussian Naive Bayes", GaussianNB(), {}), 
                    # ("Multinomial Naive Bayes", MultinomialNB(), {
                    #     "model__alpha": uniform(0.01, 1.0), # Smoothing parameter
                    # }),
                ]

    def _build_pipeline(self, model, numeric_features, categorical_features):
        return Pipeline([
            ("preprocessor", ColumnTransformer([
                ("num", StandardScaler(), numeric_features),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
            ])),
            ("model", model)
        ])

    def _is_regression(self):
        return self.target_metric in {TargetMetric.RMSE, TargetMetric.MSE, TargetMetric.MAE, TargetMetric.R2}
    
    def _is_classification(self):
        return self.target_metric in {TargetMetric.ACCURACY, TargetMetric.PRECISION, TargetMetric.RECALL, TargetMetric.F1} #, TargetMetric.AUC}
    
    def _get_scorer(self):
        is_regression = self._is_regression()
        if is_regression:
            return {
                TargetMetric.RMSE: "neg_root_mean_squared_error",
                TargetMetric.MSE: "neg_mean_squared_error",
                TargetMetric.MAE: "neg_mean_absolute_error",
                TargetMetric.R2: "r2"
            }[self.target_metric]
        else:
            average = 'weighted'  # Choose an appropriate averaging method
            # if self.target_metric == TargetMetric.AUC:
            #     return make_scorer(roc_auc_score, average=average, multi_class='ovr')
            return {
                TargetMetric.ACCURACY: "accuracy",
                TargetMetric.PRECISION: make_scorer(precision_score, average=average),
                TargetMetric.RECALL: make_scorer(recall_score, average=average),
                TargetMetric.F1: make_scorer(f1_score, average=average),
            }[self.target_metric]
    

    def _is_better_score(self, score, best_score):
        if self._is_regression():
            return abs(score) < abs(best_score) if self.target_metric != TargetMetric.R2 else score > best_score
        return score > best_score

    def _is_better_score_higher(self):
        return self._is_classification() or self.target_metric == TargetMetric.R2
    

    def get_feature_importance(self, x_train):
        """Calculate SHAP values for feature importance."""
        try:
            # Access the actual model from the pipeline
            model = self.best_model.named_steps['model']

            # Determine explainer type based on model
            if isinstance(model, (RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, DecisionTreeClassifier, DecisionTreeRegressor)):
                explainer = shap.TreeExplainer(model)
            elif isinstance(model, (LogisticRegression, LinearRegression)):
                explainer = shap.LinearExplainer(model, x_train)
            else:
                # Fallback to KernelExplainer (slower, model-agnostic)
                explainer = shap.KernelExplainer(model, x_train)

            # Get preprocessor
            preprocessor = self.best_model.named_steps['preprocessor']

            # Transform data
            x_train_transformed = preprocessor.transform(x_train)

            shap_values = explainer.shap_values(x_train_transformed)
            return shap_values, explainer
        except Exception as e:
            print(e)
            return None, None