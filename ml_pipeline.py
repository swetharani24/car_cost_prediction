from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class MLPipeline:
    def __init__(self, num_features, cat_features):
        self.num_features = num_features
        self.cat_features = cat_features

    def build_pipeline(self, model):
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.num_features),
                ('cat', categorical_transformer, self.cat_features)
            ]
        )

        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', model)])
        return pipeline
