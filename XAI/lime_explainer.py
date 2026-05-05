from lime.lime_tabular import LimeTabularExplainer


class LIMEExplainer:
    def __init__(self, model, preprocessor):
        self.model = model
        self.preprocessor = preprocessor
        self.explainer = None

    def build(self, X_train):
        X_train_transformed = self.preprocessor.transform(X_train)

        feature_names = self.preprocessor.get_feature_names_out()

        self.explainer = LimeTabularExplainer(
            training_data=X_train_transformed,
            feature_names=feature_names,
            class_names=["good", "bad"],
            mode="classification"
        )

    def explain_instance(self, X, index=0, output_path="XAI/output/lime_explanation.html"):
        if self.explainer is None:
            raise ValueError("LIME explainer is not built. Call build(X_train) first.")

        X_transformed = self.preprocessor.transform(X)

        exp = self.explainer.explain_instance(
            X_transformed[index],
            self.model.predict_proba,
            num_features=10
        )

        exp.save_to_file(output_path)

        return output_path