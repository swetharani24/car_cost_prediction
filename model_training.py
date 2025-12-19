import os
import pickle
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

class ModelTrainer:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.models = {
            "logistic_regression": LogisticRegression(max_iter=1000),
            "random_forest": RandomForestClassifier(),
            "decision_tree": DecisionTreeClassifier(),
            "gaussian_nb": GaussianNB(),
            "svm": SVC()
        }

    def train_and_save_models(self, pipeline_builder, models_dir="models"):
        os.makedirs(models_dir, exist_ok=True)
        trained_models = {}
        for name, model in self.models.items():
            try:
                pipeline = pipeline_builder.build_pipeline(model)
                pipeline.fit(self.X_train, self.y_train)
                trained_models[name] = pipeline
                with open(os.path.join(models_dir, f"{name}.pkl"), "wb") as f:
                    pickle.dump(pipeline, f)
                logging.info(f"{name} trained and saved successfully")
            except Exception as e:
                logging.error(f"Error training {name}: {e}")
        return trained_models
