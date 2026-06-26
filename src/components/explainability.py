import sys
from typing import List, Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import OneHotEncoder

from src.exception import CustomException
from src.logger import logging


def _extract_feature_names_from_transformer(transformer: Any, columns: List[str]) -> List[str]:
    if hasattr(transformer, "named_steps"):
        # Common pipeline wrappers
        for step in transformer.named_steps.values():
            if isinstance(step, OneHotEncoder):
                categories = step.categories_
                expanded = []
                for name, cats in zip(columns, categories):
                    expanded.extend([f"{name}_{cat}" for cat in cats])
                return expanded

        # For pipelines without an OneHotEncoder, preserve original names.
        return list(columns)

    if isinstance(transformer, OneHotEncoder):
        categories = transformer.categories_
        expanded = []
        for name, cats in zip(columns, categories):
            expanded.extend([f"{name}_{cat}" for cat in cats])
        return expanded

    # Fallback for transformers that do not support get_feature_names_out
    return list(columns)


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    try:
        if hasattr(preprocessor, "get_feature_names_out"):
            try:
                return list(preprocessor.get_feature_names_out())
            except Exception:
                logging.warning(
                    "ColumnTransformer.get_feature_names_out() failed; using fallback feature name extraction"
                )

        feature_names = []
        for name, transformer, columns in preprocessor.transformers_:
            if name == "remainder":
                continue
            feature_names.extend(_extract_feature_names_from_transformer(transformer, list(columns)))

        return feature_names

    except Exception as e:
        logging.error("Failed to get feature names from preprocessor", exc_info=True)
        raise CustomException(e, sys)


def explain_instance(model: Any, preprocessor: ColumnTransformer, raw_features: pd.DataFrame, top_n: int = 10) -> dict:
    try:
        if raw_features is None:
            raise ValueError("raw_features must be a valid pandas DataFrame")

        feature_names = get_feature_names(preprocessor)
        transformed = preprocessor.transform(raw_features)

        if transformed.ndim == 1:
            transformed = transformed.reshape(1, -1)

        if hasattr(model, "coef_"):
            coefficients = model.coef_[0]
            contribution = (coefficients * transformed[0]).tolist()
            explanation = [
                {
                    "feature": feature,
                    "value": float(transformed[0, idx]),
                    "coefficient": float(coefficients[idx]),
                    "contribution": float(contribution[idx])
                }
                for idx, feature in enumerate(feature_names)
            ]
            explanation.sort(key=lambda item: abs(item["contribution"]), reverse=True)
            return {
                "method": "linear-coefficients",
                "top_features": explanation[:top_n]
            }

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_.tolist()
            explanation = [
                {
                    "feature": feature,
                    "importance": float(importances[idx])
                }
                for idx, feature in enumerate(feature_names)
            ]
            explanation.sort(key=lambda item: item["importance"], reverse=True)
            return {
                "method": "feature-importances",
                "top_features": explanation[:top_n]
            }

        raise ValueError("Model does not support coefficient or feature importance explanations")

    except Exception as e:
        logging.error("Error generating instance explanation", exc_info=True)
        raise CustomException(e, sys)


def get_global_feature_importance(model: Any, preprocessor: ColumnTransformer, X_raw: pd.DataFrame, y: pd.Series, top_n: int = 20) -> dict:
    try:
        X_transformed = preprocessor.transform(X_raw)
        result = permutation_importance(
            estimator=model,
            X=X_transformed,
            y=y,
            n_repeats=10,
            random_state=42,
            scoring="f1",
            n_jobs=-1,
        )

        feature_names = get_feature_names(preprocessor)
        importances = [
            {
                "feature": feature,
                "importance_mean": float(result.importances_mean[idx]),
                "importance_std": float(result.importances_std[idx])
            }
            for idx, feature in enumerate(feature_names)
        ]
        importances.sort(key=lambda item: abs(item["importance_mean"]), reverse=True)
        return {
            "method": "permutation-importance",
            "features": importances[:top_n]
        }

    except Exception as e:
        logging.error("Error generating global feature importance", exc_info=True)
        raise CustomException(e, sys)
