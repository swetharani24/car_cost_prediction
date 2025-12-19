import pandas as pd
import numpy as np
import csv
import logging

class DataProcessor:
    def __init__(self, file_path, target_column):
        self.file_path = file_path
        self.target_column = target_column
        self.df = None
        self.num_features = []
        self.cat_features = []

    def load_data(self):
        try:
            self.df = pd.read_csv(self.file_path, header=0)

            # normalize column names
            self.df.columns = self.df.columns.str.strip().str.lower()

            # 🔥 FIX: if CSV loaded as ONE column, split it
            if len(self.df.columns) == 1:
                logging.warning("CSV detected as single column. Fixing automatically...")

                col = self.df.columns[0]
                split_cols = col.split(",")

                # split the single column into multiple columns
                self.df = self.df[col].str.split(",", expand=True)
                self.df.columns = [c.strip().lower() for c in split_cols]

            logging.info(f"Final columns: {self.df.columns.tolist()}")

        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise

    def clean_data(self):
        try:
            # Drop unused columns
            columns_to_drop = ['car_name', 'selling_price']
            self.df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

            # Drop duplicates
            self.df.drop_duplicates(inplace=True)

            # Separate numeric and categorical features
            self.num_features = self.df.select_dtypes(include=np.number).columns.tolist()
            if self.target_column in self.num_features:
                self.num_features.remove(self.target_column)

            self.cat_features = self.df.select_dtypes(include='object').columns.tolist()
            if self.target_column in self.cat_features:
                self.cat_features.remove(self.target_column)

            # Fill missing values
            self.df[self.num_features] = self.df[self.num_features].fillna(0)
            self.df[self.cat_features] = self.df[self.cat_features].fillna("missing")

            logging.info("Data cleaning completed successfully")
        except Exception as e:
            logging.error(f"Error cleaning data: {e}")
            raise

    def feature_selection(self):
        try:
            if self.target_column not in self.df.columns:
                raise ValueError(
                    f"Target column '{self.target_column}' not found.\n"
                    f"Available columns: {self.df.columns.tolist()}"
                )

            X = self.df.drop(columns=[self.target_column])
            y = self.df[self.target_column]

            logging.info("Feature selection successful")
            return X, y

        except Exception as e:
            logging.error(f"Error during feature selection: {e}")
            raise
