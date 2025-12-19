import logging
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from data_processing import DataProcessor
from model_training import ModelTrainer
from ml_pipeline import MLPipeline
from logger import setup_logging

setup_logging()
logging.info("Starting Car Classification Project")

# Load and preprocess data
processor = DataProcessor(file_path="data/car_data.csv",   target_column="fuel_type")




processor.load_data()
processor.clean_data()
X, y = processor.feature_selection()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
pipeline_builder = MLPipeline(processor.num_features, processor.cat_features)
trainer = ModelTrainer(X_train, y_train)
trained_models = trainer.train_and_save_models(pipeline_builder)

# Evaluate and save best model
best_model = None
best_accuracy = 0
best_name = None
for name, model in trained_models.items():
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logging.info(f"{name} Accuracy: {acc:.4f}")
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_name = name

# Save best model
os.makedirs("models", exist_ok=True)
best_model_path = os.path.join("models", "best_model.pkl")
with open(best_model_path, "wb") as f:
    pickle.dump(best_model, f)
logging.info(f"Best model: {best_name} saved at {best_model_path} with accuracy {best_accuracy:.4f}")
