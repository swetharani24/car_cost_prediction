import pickle
import pandas as pd
import logging

class CarPredictor:
    def __init__(self):
        self.model_path = "models/best_model.pkl"
        self.model = self.load_model()

        # 🔥 EXACT features used during training
        self.required_features = [
            "car_name",
            "year",
            "selling_price",
            "present_price",
            "kms_driven",
            "seller_type",
            "transmission",
            "owner"
        ]

    def load_model(self):
        try:
            with open(self.model_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logging.error(f"Model loading error: {e}")
            raise

    def predict(self, input_data: dict):
        try:
            # create empty dataframe with ALL features
            df = pd.DataFrame(columns=self.required_features)

            # insert user input
            for key, value in input_data.items():
                df.loc[0, key] = value

            # fill missing columns with 0
            df.fillna(0, inplace=True)

            prediction = self.model.predict(df)[0]
            return prediction

        except Exception as e:
            logging.error(f"Prediction error: {e}")
            return f"Error: {e}"
