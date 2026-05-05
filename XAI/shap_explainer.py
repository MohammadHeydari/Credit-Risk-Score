import shap
import matplotlib.pyplot as plt


class SHAPExplainer:
    def __init__(self, model, preprocessor):
        self.model = model
        self.preprocessor = preprocessor

        self.explainer = shap.TreeExplainer(self.model)

    def transform(self, X):
        return self.preprocessor.transform(X)

    def get_shap_values(self, X):
        X_transformed = self.transform(X)
        shap_values = self.explainer.shap_values(X_transformed)

        return X_transformed, shap_values

    def summary_plot(self, X, output_path="XAI/output/shap_summary.png"):
        X_transformed, shap_values = self.get_shap_values(X)

        shap.summary_plot(
            shap_values,
            X_transformed,
            show=False
        )

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return output_path

    def explain_instance(self, X, index=0, output_path="XAI/output/shap_force.png"):
        X_transformed, shap_values = self.get_shap_values(X)

        shap.force_plot(
            self.explainer.expected_value,
            shap_values[index],
            X_transformed[index],
            matplotlib=True,
            show=False
        )

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return output_path