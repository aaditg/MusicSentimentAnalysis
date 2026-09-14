from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.model_selection import BaseCrossValidator, GridSearchCV


def _take(data, index):
    if hasattr(data, "iloc"):
        return data.iloc[index]
    return [data[i] for i in index]


@dataclass
class CVResult:
    accuracy_mean: float
    accuracy_std: float
    macro_f1_mean: float
    macro_f1_std: float
    balanced_acc_mean: float
    balanced_acc_std: float
    n_splits: int
    per_class_f1: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "accuracy_mean": round(self.accuracy_mean, 4),
            "accuracy_std": round(self.accuracy_std, 4),
            "macro_f1_mean": round(self.macro_f1_mean, 4),
            "macro_f1_std": round(self.macro_f1_std, 4),
            "balanced_acc_mean": round(self.balanced_acc_mean, 4),
            "balanced_acc_std": round(self.balanced_acc_std, 4),
            "n_splits": self.n_splits,
            "per_class_f1": {k: round(v, 4) for k, v in self.per_class_f1.items()},
        }

    def summary(self) -> str:
        return (
            f"acc={self.accuracy_mean:.3f}+/-{self.accuracy_std:.3f} "
            f"macroF1={self.macro_f1_mean:.3f}+/-{self.macro_f1_std:.3f} "
            f"balAcc={self.balanced_acc_mean:.3f}+/-{self.balanced_acc_std:.3f}"
        )


def cv_classification(
    model,
    X,
    y,
    splitter: BaseCrossValidator,
    groups=None,
) -> CVResult:
    y = pd.Series(y).reset_index(drop=True)
    labels = sorted(y.unique())

    accuracies: list[float] = []
    macro_f1s: list[float] = []
    balanced: list[float] = []
    per_class: dict[str, list[float]] = {label: [] for label in labels}

    for train_index, test_index in splitter.split(np.zeros(len(y)), y, groups):
        estimator = clone(model)
        estimator.fit(_take(X, train_index), y.iloc[train_index])
        predicted = estimator.predict(_take(X, test_index))
        actual = y.iloc[test_index]

        accuracies.append(accuracy_score(actual, predicted))
        macro_f1s.append(f1_score(actual, predicted, average="macro", zero_division=0))
        balanced.append(balanced_accuracy_score(actual, predicted))
        fold_f1 = f1_score(actual, predicted, labels=labels, average=None, zero_division=0)
        for label, score in zip(labels, fold_f1):
            per_class[label].append(score)

    return CVResult(
        accuracy_mean=float(np.mean(accuracies)),
        accuracy_std=float(np.std(accuracies)),
        macro_f1_mean=float(np.mean(macro_f1s)),
        macro_f1_std=float(np.std(macro_f1s)),
        balanced_acc_mean=float(np.mean(balanced)),
        balanced_acc_std=float(np.std(balanced)),
        n_splits=splitter.get_n_splits(),
        per_class_f1={label: float(np.mean(scores)) for label, scores in per_class.items()},
    )


def nested_cv_search(
    estimator,
    param_grid,
    X,
    y,
    outer_cv: BaseCrossValidator,
    inner_cv: BaseCrossValidator,
    groups=None,
    scoring: str = "f1_macro",
) -> tuple[CVResult, list[dict]]:
    y = pd.Series(y).reset_index(drop=True)
    labels = sorted(y.unique())
    groups_arr = np.asarray(groups) if groups is not None else None

    accuracies: list[float] = []
    macro_f1s: list[float] = []
    balanced: list[float] = []
    per_class: dict[str, list[float]] = {label: [] for label in labels}
    chosen: list[dict] = []

    for train_index, test_index in outer_cv.split(np.zeros(len(y)), y, groups):
        search = GridSearchCV(clone(estimator), param_grid, scoring=scoring, cv=inner_cv, n_jobs=-1)
        x_train = _take(X, train_index)
        y_train = y.iloc[train_index]
        if groups_arr is not None:
            search.fit(x_train, y_train, groups=groups_arr[train_index])
        else:
            search.fit(x_train, y_train)

        best = search.best_estimator_
        predicted = best.predict(_take(X, test_index))
        actual = y.iloc[test_index]
        chosen.append(search.best_params_)

        accuracies.append(accuracy_score(actual, predicted))
        macro_f1s.append(f1_score(actual, predicted, average="macro", zero_division=0))
        balanced.append(balanced_accuracy_score(actual, predicted))
        fold_f1 = f1_score(actual, predicted, labels=labels, average=None, zero_division=0)
        for label, score in zip(labels, fold_f1):
            per_class[label].append(score)

    result = CVResult(
        accuracy_mean=float(np.mean(accuracies)),
        accuracy_std=float(np.std(accuracies)),
        macro_f1_mean=float(np.mean(macro_f1s)),
        macro_f1_std=float(np.std(macro_f1s)),
        balanced_acc_mean=float(np.mean(balanced)),
        balanced_acc_std=float(np.std(balanced)),
        n_splits=outer_cv.get_n_splits(),
        per_class_f1={label: float(np.mean(scores)) for label, scores in per_class.items()},
    )
    return result, chosen
