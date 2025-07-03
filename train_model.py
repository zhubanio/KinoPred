import logging
import datetime
import platform
import sys
import warnings
from itertools import combinations

import sklearn
import catboost
import lightgbm
import xgboost
import category_encoders
import mlflow
import mlflow.sklearn
from matplotlib import pyplot as plt
from pathlib import Path
import json
import os
import numpy as np
import pandas as pd
from mlflow.models import infer_signature, ModelSignature
from mlflow.types.schema import ColSpec, Schema
from mlflow.types import DataType
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import ElasticNet, HuberRegressor, RidgeCV, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVR, SVC
from sklearn.ensemble import AdaBoostRegressor, HistGradientBoostingRegressor, StackingRegressor, \
    GradientBoostingClassifier, RandomForestClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import (BaggingRegressor, ExtraTreesRegressor,
                              RandomForestRegressor)
from sklearn.feature_selection import RFE, VarianceThreshold
from sklearn.linear_model import (Lasso, LinearRegression,
                                  Ridge)
from sklearn.metrics import (mean_absolute_error, r2_score,
                             root_mean_squared_error, accuracy_score)
from sklearn.model_selection import (GridSearchCV, KFold, cross_val_predict, cross_val_score)
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer, LabelEncoder
from sklearn.compose import ColumnTransformer
from category_encoders import CatBoostEncoder

from pre_sor import DataTransformer, Pad
from preprocessor import DataPreprocessor


class ClassifierToRegressorWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, classifier=None, min_rating=1, max_rating=10):
        # classifier: Экземпляр модели классификации (например, LogisticRegression())
        # min_rating, max_rating: Минимальное и максимальное значение вашего рейтинга
        if classifier is None:
            raise ValueError("Classifier must be provided.")
        self.classifier = classifier
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.classes_ = np.arange(min_rating, max_rating + 1)  # Все возможные рейтинги
        self.label_encoder = LabelEncoder()

    def fit(self, x, y_):
        y_int = np.round(y_).astype(int)
        y_fit = self.label_encoder.fit_transform(y_int)  # <--- ИСПОЛЬЗУЕМ LABELENCODER
        self.classifier.fit(x, y_fit)
        return self

    def predict(self, x):
        probabilities = self.classifier.predict_proba(x)

        # Используем классы из нашего энкодера, чтобы получить исходные значения (1, 2, 3...)
        class_values = self.label_encoder.classes_

        # Умножаем вероятности на ИСХОДНЫЕ значения классов
        reg_predictions = np.sum(probabilities * class_values, axis=1)

        reg_predictions = np.clip(reg_predictions, self.min_rating, self.max_rating)

        return reg_predictions

    # Метод get_params и set_params наследуются от BaseEstimator
    # Если вы хотите, чтобы параметры внутреннего классификатора были доступны через GridSearchCV,
    # вам нужно проксировать их. Вот пример, как это делается:
    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        if deep and hasattr(self, 'classifier'):
            # Добавляем параметры внутреннего классификатора
            # с префиксом 'classifier__'
            for key, value in self.classifier.get_params(deep=True).items():
                params[f'classifier__{key}'] = value
        return params

    def set_params(self, **params):
        if not params:
            return self
        # Разделяем параметры на те, что для обертки, и те, что для классификатора
        classifier_params = {}
        own_params = {}
        for key, value in params.items():
            if key.startswith('classifier__'):
                classifier_params[key[len('classifier__'):]] = value
            else:
                own_params[key] = value

        # Устанавливаем параметры для внутреннего классификатора
        if classifier_params and hasattr(self, 'classifier'):
            self.classifier.set_params(**classifier_params)

        # Устанавливаем собственные параметры обертки
        return super().set_params(**own_params)


def evaluate_oof_predictions(y_true, y_pred_float):
    """
    Рассчитывает набор метрик для OOF (Out-of-Fold) предсказаний.
    """
    y_pred_rounded = np.clip(np.round(y_pred_float), 1, 10).astype(int)
    y_true_rounded = np.round(y_true).astype(int)
    # y_true = np.round(y_true).astype(int)
    # y_pred = np.round(y_pred_float).astype(int)
    # y_pred_float = y_pred
    acc = accuracy_score(y_true_rounded, y_pred_rounded)
    mae = mean_absolute_error(y_true, y_pred_float)
    rmse = root_mean_squared_error(y_true, y_pred_float)
    r2 = r2_score(y_true, y_pred_float)

    return {"accuracy": acc, "mae": mae, "rmse": rmse, "r2": r2}


def find_best_model(model_name, pipe, param_grid, x, y_, cv):
    """
    Выполняет GridSearchCV для поиска лучших гиперпараметров.
    """
    print(f"--- Подбор параметров для {model_name} ---")
    gs = GridSearchCV(pipe, param_grid, cv=cv, scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=0,
                      error_score='raise')
    # for col in x.columns:
    #     print(col, sum(x[col].isna().astype(int)))
    gs.fit(x, y_)
    best_score_rmse = -gs.best_score_
    best_params = gs.best_params_
    print(f"Лучший RMSE на CV для {model_name}: {best_score_rmse:.4f}")
    return gs.best_estimator_, best_params


def run_experiments(user_id, x_, y_, pipe, grid_, cv_strat, name_, r_state, splits_count):
    global oof_predictions, best_estimators
    # Установка имени эксперимента
    mlflow.set_experiment(experiment_name)

    with (mlflow.start_run(run_name=f"{user_id}_{name_}")):

        # Получаем id забега

        # 3. Основной процесс: Обучение, Предсказания, Метрики, Модель
        print("\n--- Обучение и Логирование Модели ---")
        best_model, best_params = find_best_model(name_, pipe, grid_, x_, y_, cv_strat)
        oof_preds = cross_val_predict(best_model, x_, y_, cv=cv_strat, n_jobs=-1)
        oof_predictions[name_] = oof_preds
        metrics = evaluate_oof_predictions(y_, oof_preds)  # evaluate_oof_predictions уже логирует метрики
        best_estimators[name_] = best_model
        raw_input_example = x_.iloc[[0]]

        # --- ШАГ 1: Вручную создаем схему ВХОДНЫХ данных ---

        input_schema = create_input_schema(raw_input_example)
        # --- ШАГ 2: Создаем схему ВЫХОДНЫХ данных (НОВЫЙ, РАБОЧИЙ СПОСОБ) ---
        sample_prediction = best_model.predict(raw_input_example)
        prediction_signature = infer_signature(sample_prediction)
        output_schema = prediction_signature.inputs

        # --- ШАГ 3: Собираем финальную сигнатуру ---
        final_signature = ModelSignature(inputs=input_schema, outputs=output_schema)

        # --- ШАГ 4: Финальное логирование ---
        fig_y, ax_y = plt.subplots()
        y_.hist(bins=20, ax=ax_y)
        ax_y.set_title("Распределение Целевой Переменной (MYrating)")
        ax_y.set_xlabel("MYrating")
        ax_y.set_ylabel("Частота")
        plt.close(fig_y)

        # График Actual vs. Predicted (для OOF предсказаний)
        fig_pred, ax_pred = plt.subplots()
        ax_pred.scatter(y_, oof_preds, alpha=0.5)
        ax_pred.plot([y_.min(), y_.max()], [y_.min(), y_.max()], 'k--', lw=2)
        ax_pred.set_xlabel("Истинные значения (y)")
        ax_pred.set_ylabel("Предсказанные значения (OOF)")
        ax_pred.set_title(f"Истинные vs Предсказанные (OOF) для {name_}")
        plt.savefig(f"{name_}_actual_vs_predicted_oof.png")
        plt.close(fig_pred)

        # Гистограмма остатков (Residuals)
        residuals = y_ - oof_preds
        fig_res, ax_res = plt.subplots()
        residuals.hist(bins=30, ax=ax_res)
        ax_res.set_title(f"Распределение Остатков для {name_}")
        ax_res.set_xlabel("Ошибка предсказания")
        ax_res.set_ylabel("Частота")
        plt.savefig(f"{name_}_residuals_distribution.png")
        plt.close(fig_res)
        # def cal_to_log():

        mlf_log(name_, r_state, splits_count, user_id, x_, y_, metrics, best_params, best_model, raw_input_example,
                final_signature)


def run_stacking_experiment(oof_pred_s, best_ests, user_id, r_state, splits_count, x_, y_, cv_strat,
                            combination_len):
    # oof_predictions - это словарь с OOF-предсказаниями всех базовых моделей
    if not oof_pred_s:
        print("Ни одна базовая модель не была успешно обучена. Завершение работы.")
        return

    mlflow.set_experiment(experiment_name)
    name_s = 'stacking_ensemble'
    with (mlflow.start_run(run_name=f"{user_id}_{name_s}")):
        oof_df = pd.DataFrame(oof_pred_s)
        best_stacking_rmse = float('inf')
        best_stacking_combo = None

        # Поиск лучшей комбинации моделей
        for combo in combinations(oof_df.columns, min(len(oof_df.columns), combination_len)):
            combo_list = list(combo)
            X_stack = oof_df[combo_list].values
            meta_model = RidgeCV(alphas=np.logspace(-2, 2, 10))
            stack_cv_scores = cross_val_score(meta_model, X_stack, y_, cv=cv_strat,
                                              scoring='neg_root_mean_squared_error', n_jobs=-1)
            current_rmse = -np.mean(stack_cv_scores)
            if current_rmse < best_stacking_rmse:
                best_stacking_rmse = current_rmse
                best_stacking_combo = combo_list

        if best_stacking_combo is None:
            print("Не удалось выбрать комбинацию для стекинга.")
            return
        #
        print(f"Лучшая комбинация для стекинга: {best_stacking_combo} с RMSE = {best_stacking_rmse:.4f}")

        # --- Получение OOF-предсказаний для ЛУЧШЕГО ансамбля ---
        X_stack = oof_df[best_stacking_combo].values
        # best_stacking_combo = ['lin_reg', 'ridge', 'lasso', 'svr', 'knn_regressor', 'hist_gbr', 'adaboost',
        #                        'extra_trees', 'catboost', 'xgboost']
        meta_model = RidgeCV(alphas=np.logspace(-2, 2, 10))
        stack_cv_scores = cross_val_score(meta_model, X_stack, y_, cv=cv_strat, scoring='neg_root_mean_squared_error',
                                          n_jobs=-1)
        current_rmse = -np.mean(stack_cv_scores)
        if current_rmse < best_stacking_rmse:
            best_stacking_rmse = current_rmse
            best_stacking_combo = combo_list

        if best_stacking_combo is None:
            print("Не удалось выбрать комбинацию для стекинга.")

        else:
            print(f"Лучшая комбинация для стекинга: {best_stacking_combo} с RMSE = {best_stacking_rmse:.4f}")
            X_stack_best = oof_df[best_stacking_combo].values
            meta_model_final = RidgeCV(alphas=np.logspace(-2, 2, 10))
            ensemble_oof_preds = cross_val_predict(meta_model_final, X_stack_best, y_, cv=cv_strat, n_jobs=-1)
            # --- Оценка и логирование ---
            ensemble_metrics = evaluate_oof_predictions(y_, ensemble_oof_preds)
            best_params = {"best_stacking_combo": best_stacking_combo}

            # --- Обучение финальной модели для сохранения ---
            final_estimators = [(n, best_ests[n]) for n in best_stacking_combo]
            final_stacking_model = StackingRegressor(
                estimators=final_estimators,
                final_estimator=RidgeCV(),
                cv=cv_strat,
                n_jobs=-1
            )
            final_stacking_model.fit(x_, y_)

            # --- Создание сигнатуры и артефактов для MLflow ---
            raw_input_example = x_.iloc[[0]]
            input_schema = create_input_schema(raw_input_example)
            sample_prediction = final_stacking_model.predict(raw_input_example)
            output_schema = infer_signature(sample_prediction).inputs
            final_signature = ModelSignature(inputs=input_schema, outputs=output_schema)

            # График распределения Y
            fig_y, ax_y = plt.subplots()
            y_.hist(bins=20, ax=ax_y)
            # ... (код для сохранения графика Y) ...

            # --- ИСПРАВЛЕННЫЙ УЧАСТОК ДЛЯ ГРАФИКОВ ---
            # График Actual vs. Predicted (используем предсказания ансамбля)
            fig_pred, ax_pred = plt.subplots()
            ax_pred.scatter(y_, ensemble_oof_preds, alpha=0.5)  # <--- ИСПРАВЛЕНО
            ax_pred.plot([y_.min(), y_.max()], [y_.min(), y_.max()], 'k--', lw=2)
            ax_pred.set_xlabel("Истинные значения (y)")
            ax_pred.set_ylabel("Предсказанные значения (Ensemble OOF)")
            ax_pred.set_title(f"Истинные vs Предсказанные для {name_s}")
            plt.savefig(f"{name_s}_actual_vs_predicted_oof.png")
            plt.close(fig_pred)

            # Гистограмма остатков (используем предсказания ансамбля)
            residuals = y_ - ensemble_oof_preds  # <--- ИСПРАВЛЕНО
            fig_res, ax_res = plt.subplots()
            residuals.hist(bins=30, ax=ax_res)
            ax_res.set_title(f"Распределение Остатков для {name_s}")
            ax_res.set_xlabel("Ошибка предсказания")
            ax_res.set_ylabel("Частота")
            plt.savefig(f"{name_s}_residuals_distribution.png")
            plt.close(fig_res)

            mlf_log(name_s, r_state, splits_count, user_id, x_, y_, ensemble_metrics, best_params,
                    final_stacking_model, raw_input_example, final_signature)


def create_input_schema(raw_input_example):
    input_specs = []
    for col_name, dtype in raw_input_example.dtypes.items():
        dtype_str = str(dtype)
        mlflow_type = dtype_map.get(dtype_str, DataType.string)
        input_specs.append(ColSpec(type=mlflow_type, name=col_name))

    input_scheme = Schema(inputs=input_specs)
    return input_scheme


def mlf_log(name_, r_state, splits_count, user_id, x_, y_, metrics, best_params, best_model, raw_input_example,
            final_signature):
    #  Логирование Общих Параметров и Информации о Данных
    mlflow.log_param("model_type", name_)
    mlflow.log_param("random_seed", r_state)
    mlflow.log_param("cv_n_splits", splits_count)  # Переименовал для ясности
    mlflow.log_param("dataset_id", user_id)
    mlflow.log_param("run_datetime", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    # Информация о размере и структуре исходных данных
    mlflow.log_param("initial_data_n_samples", x_.shape[0])
    mlflow.log_param("initial_data_n_features", x_.shape[1])
    # Логируем имена исходных колонок
    mlflow.log_param("initial_feature_names", json.dumps(x_.columns.tolist()))  # json.dumps для списка

    # 2. Логирование Информации об Окружении
    mlflow.log_param("python_version", sys.version)
    mlflow.log_param("os_platform", platform.platform())

    # Логирование версий ключевых библиотек

    mlflow.log_param("sklearn_version", sklearn.__version__)
    mlflow.log_param("catboost_version", catboost.__version__)
    mlflow.log_param("lightgbm_version", lightgbm.__version__)
    mlflow.log_param("xgboost_version", xgboost.__version__)
    mlflow.log_param("category_encoders_version", category_encoders.__version__)

    mlflow.log_metrics(metrics)
    mlflow.log_params(best_params)

    mlflow.sklearn.log_model(
        sk_model=best_model,
        name=name_,
        input_example=raw_input_example,
        signature=final_signature  # Используем нашу полностью ручную сигнатуру
    )

    mlflow.log_artifact(f"{name_}_residuals_distribution.png", artifact_path="model_diagnostics")
    os.remove(f"{name_}_residuals_distribution.png")
    mlflow.log_artifact(f"{name_}_actual_vs_predicted_oof.png", artifact_path="model_visualizations")
    os.remove(f"{name_}_actual_vs_predicted_oof.png")


mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Замените на свой хост и порт
cu_s = [str(cu).split('\\')[-1].split('.')[0] for cu in Path('raw_movie_files/').glob("*.json")]
u_id = cu_s[0]
data_path = f'raw_movie_files/{u_id}.json'
df = pd.read_json(data_path, encoding="utf-8")
X = df.drop(columns=['MYrating'])
y = df['MYrating']
cat_cols_for_cbe = ["main_actor", "main_director", "main_writer", "ratingMpaa", "main_language"]
random_state = 42
cbe_scaler = ColumnTransformer(
    transformers=[
        ('cat', Pipeline([
            ('encoder', CatBoostEncoder()),
            ('scaler', StandardScaler())
        ]), cat_cols_for_cbe)
    ],
    remainder='passthrough'
)
cbe_scaler.set_output(transform="pandas")
cbe_only = ColumnTransformer(
    transformers=[
        ('cat', CatBoostEncoder(cols=cat_cols_for_cbe), cat_cols_for_cbe)
    ],
    remainder='passthrough'
)
cbe_only.set_output(transform="pandas")
numeric_cols = ["budget", "actors",
                "ageRating", "votes_kp", "votes_imdb", "age", "fees_world",
                "fees_usa", "fees_russia", "rating_kp", "rating_imdb", "rating_critics",
                "audience_count", "rus_audience_count", "spoken_languages_count", "main_language",
                "countries", "genres_count", "countries_count", "writers_count",
                "directors_count", ]
binary_cols = ["short_film", "no_budget", "no_ageRating", "is_on_kp_hd", "no_fees_world", "no_fees_usa",
               "no_fees_russia", "no_rating_kp", "no_rating_imdb", "no_rating_critics", "no_rating_rus_critics",
               "top250", "no_audience", "no_rus_audience", ]
linear_preprocessor = Pipeline(steps=[
    ('pre_transformer', DataTransformer()),
    ("ct1", ColumnTransformer([
        ("scaler", StandardScaler(), numeric_cols),
        ('variance_filter', VarianceThreshold(threshold=0.00), binary_cols),

    ], remainder='passthrough').set_output(transform="pandas")),
    ('pad1', Pad()),
    ("ct2", ColumnTransformer([
        ("power", PowerTransformer(standardize=True), ["votes_rus_critics", "lists_count", "ratingMpaa", "movieLength",
                                                       "rating_rus_critics", "main_director", "main_actor",
                                                       "main_writer", "writers", "votes_critics", "genres",
                                                       "actors_count"]),
    ], remainder='passthrough').set_output(transform="pandas")),
    ('pad2', Pad()),
])
tree_preprocessor = Pipeline(steps=[
    ('pre_transformer', DataTransformer()),
    ("ct1", ColumnTransformer([
        ('variance_filter', VarianceThreshold(threshold=0.00), binary_cols),

    ], remainder='passthrough').set_output(transform="pandas")),
    ('pad1', Pad()),
]) # ______________________ЗАКОНЧИ С ПРЕДПРОЦЕССОРАМИ________________________
pipelines_and_grids = {
    # # 1. Логистическая регрессия (подходит для linear_preprocessor)
    # "clf_to_reg_logreg": (
    #     Pipeline([
    #         ('preprocessor', linear_preprocessor),
    #         ('model', ClassifierToRegressorWrapper(
    #             classifier=LogisticRegression(random_state=random_state, solver='lbfgs',
    #                                           max_iter=20000)
    #         ))
    #     ]),
    #     {
    #         'model__classifier__C': [0.1, 1.0, 10.0],
    #         'model__classifier__penalty': ['l2']
    #     }
    # ),
    #
    # # 2. Случайный лес для классификации (подходит для tree_preprocessor)
    # "clf_to_reg_random_forest": (
    #     Pipeline([
    #         ('preprocessor', tree_preprocessor),
    #         ('model', ClassifierToRegressorWrapper(
    #             classifier=RandomForestClassifier(random_state=random_state, n_jobs=-1)
    #         ))
    #     ]),
    #     {
    #         'model__classifier__n_estimators': [100, 200],
    #         'model__classifier__max_depth': [10, 20],
    #         'model__classifier__min_samples_leaf': [1, 5]
    #     }
    # ),
    #
    # # 3. Градиентный бустинг для классификации (GradientBoostingClassifier)
    # "clf_to_reg_gradient_boosting": (
    #     Pipeline([
    #         ('preprocessor', tree_preprocessor),
    #         ('model', ClassifierToRegressorWrapper(
    #             classifier=GradientBoostingClassifier(random_state=random_state)
    #         ))
    #     ]),
    #     {
    #         'model__classifier__n_estimators': [100, 200],
    #         'model__classifier__learning_rate': [0.05, 0.1],
    #         'model__classifier__max_depth': [3, 5]
    #     }
    # ),
    #
    # # 4. LightGBM для классификации (LGBMClassifier)
    # "clf_to_reg_lgbm": (
    #     Pipeline([
    #         ('preprocessor', tree_preprocessor),
    #         ('model', ClassifierToRegressorWrapper(
    #             classifier=LGBMClassifier(random_state=random_state, verbose=-1, n_jobs=-1)
    #         ))
    #     ]),
    #     {
    #         'model__classifier__n_estimators': [300, 500],
    #         'model__classifier__learning_rate': [0.02, 0.05],
    #         'model__classifier__num_leaves': [20, 31],
    #         'model__classifier__max_depth': [7, -1]  # -1 означает без ограничения
    #     }
    # ),
    #
    # # 5. XGBoost для классификации (XGBClassifier)
    # "clf_to_reg_xgboost": (
    #     Pipeline([
    #         ('preprocessor', tree_preprocessor),
    #         ('model', ClassifierToRegressorWrapper(
    #             classifier=XGBClassifier(random_state=random_state, eval_metric='mlogloss',
    #                                      n_jobs=-1)
    #         ))
    #     ]),
    #     {
    #         'model__classifier__n_estimators': [300, 500],
    #         'model__classifier__learning_rate': [0.02, 0.05],
    #         'model__classifier__max_depth': [6, 8]
    #     }
    # ),
    #
    # # 6. CatBoost для классификации (CatBoostClassifier)
    # "clf_to_reg_catboost": (
    #     Pipeline([
    #         ('preprocessor', tree_preprocessor),
    #         ('model', ClassifierToRegressorWrapper(
    #             classifier=CatBoostClassifier(random_seed=random_state, allow_writing_files=False, verbose=False)
    #         ))
    #     ]),
    #     {
    #         'model__classifier__iterations': [500],
    #         'model__classifier__learning_rate': [0.03],
    #         'model__classifier__depth': [6]
    #     }
    # ),
    #
    # # 7. K-ближайших соседей для классификации (KNeighborsClassifier)
    # # Может быть чувствителен к масштабу, поэтому лучше с linear_preprocessor
    # "clf_to_reg_knn_clf": (
    #     Pipeline([
    #         ('preprocessor', linear_preprocessor),
    #         ('model', ClassifierToRegressorWrapper(
    #             classifier=KNeighborsClassifier()
    #         ))
    #     ]),
    #     {
    #         'model__classifier__n_neighbors': [5, 10, 20],
    #         'model__classifier__weights': ['uniform', 'distance']
    #         # uniform: все точки равны, distance: ближе весят больше
    #     }
    # ),
    #
    # # 8. Дерево решений для классификации (DecisionTreeClassifier)
    # "clf_to_reg_decision_tree_clf": (
    #     Pipeline([
    #         ('preprocessor', tree_preprocessor),
    #         ('model', ClassifierToRegressorWrapper(
    #             classifier=DecisionTreeClassifier(random_state=random_state)
    #         ))
    #     ]),
    #     {
    #         'model__classifier__max_depth': [5, 10, 15],
    #         'model__classifier__min_samples_leaf': [2, 5]
    #     }
    # ),
    #
    # # 9. MLPClassifier (нейронная сеть) - подходит для linear_preprocessor
    # # Может быть медленным и требовать тщательной настройки
    # "clf_to_reg_mlp_clf": (
    #     Pipeline([
    #         ('preprocessor', linear_preprocessor),
    #         ('model', ClassifierToRegressorWrapper(
    #             classifier=MLPClassifier(random_state=random_state, max_iter=2000)  # Увеличить max_iter
    #         ))
    #     ]),
    #     {
    #         'model__classifier__hidden_layer_sizes': [(50,), (100,), (50, 50)],
    #         'model__classifier__activation': ['relu', 'tanh'],
    #         'model__classifier__alpha': [0.0001, 0.001]  # L2 regularization
    #     }
    # ),
    # # 10. SVC (Support Vector Classifier) - очень медленный для больших данных, особенно с RBF ядром
    # # Требует predict_proba=True, что может сделать его еще медленнее.
    # "clf_to_reg_svc": (
    #     Pipeline([
    #         ('preprocessor', linear_preprocessor),
    #         ('model', ClassifierToRegressorWrapper(
    #             classifier=SVC(random_state=random_state, probability=True)  # probability=True критичен
    #         ))
    #     ]),
    #     {
    #         'model__classifier__C': [0.1, 1.0],
    #         'model__classifier__kernel': ['linear', 'rbf']
    #     }
    # ),
    # --- Линейные модели (используют linear_preprocessor) ---
    "lin_reg": (
        Pipeline([('preprocessor', linear_preprocessor), ('feature_selection', RFE(estimator=LinearRegression())),
                  ('model', LinearRegression())]),
        {'feature_selection__n_features_to_select': [20, 40]}),
    "ridge": (
        Pipeline([('preprocessor', linear_preprocessor), ('model', Ridge(max_iter=2000))]),
        {'model__alpha': [1.0, 10.0, 50.0]}
    ),
    "lasso": (
        Pipeline([('preprocessor', linear_preprocessor), ('model', Lasso(max_iter=2000))]),
        {'model__alpha': [0.001, 0.01, 0.1]}
    ),
    "elastic_net": (
        Pipeline([('preprocessor', linear_preprocessor), ('model', ElasticNet(max_iter=2000))]),
        {
            'model__alpha': [0.1, 1.0, 10.0],
            'model__l1_ratio': [0.1, 0.5, 0.9]  # 0 - как Ridge, 1 - как Lasso
        }
    ),
    "svr": (
        Pipeline([('preprocessor', linear_preprocessor), ('model', SVR())]),
        {
            'model__C': [0.1, 1, 10],  # Параметр регуляризации
            'model__kernel': ['linear', 'rbf']  # Линейное и нелинейное (гауссово) ядро
        }
    ),

    "knn_regressor": (
        Pipeline([('preprocessor', linear_preprocessor), ('model', KNeighborsRegressor())]),
        {'model__n_neighbors': [10, 20]}
    ),
    "huber": (
        Pipeline([('preprocessor', linear_preprocessor),
                  ('model', HuberRegressor())]),
        {
            'model__epsilon': [1.35, 1.5, 1.75],
            'model__max_iter': [20000]  # Увеличено
        }
    ),
    # --- Древовидные модели (используют tree_preprocessor) ---
    "decision_tree": (
        Pipeline([('preprocessor', tree_preprocessor), ('model', DecisionTreeRegressor(random_state=random_state))]),
        {'model__max_depth': [5, 10, 20], 'model__min_samples_leaf': [2, 5, 10]}
    ),
    "hist_gbr": (
        Pipeline(
            [('preprocessor', tree_preprocessor), ('model', HistGradientBoostingRegressor(random_state=random_state))]),
        {
            'model__learning_rate': [0.05, 0.1],
            'model__max_leaf_nodes': [20, 31]  # Аналог num_leaves в LGBM
        }
    ),
    "adaboost": (
        Pipeline([('preprocessor', tree_preprocessor),
                  ('model',
                   AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=5), random_state=random_state))]),
        {
            'model__n_estimators': [50, 100],
            'model__learning_rate': [0.1, 0.5, 1.0]
        }
    ),

    "random_forest": (
        Pipeline(
            [('preprocessor', tree_preprocessor),
             ('model', RandomForestRegressor(random_state=random_state, n_jobs=-1))]),
        {'model__n_estimators': [200, 300], 'model__max_depth': [10, 20]}
    ),
    "extra_trees": (
        Pipeline([('preprocessor', tree_preprocessor),
                  ('model', ExtraTreesRegressor(random_state=random_state, n_jobs=-1))]),
        {'model__n_estimators': [200, 300], 'model__max_depth': [10, 20]}
    ),
    "bagging_regressor": (
        Pipeline([('preprocessor', tree_preprocessor),
                  (
                  'model', BaggingRegressor(estimator=DecisionTreeRegressor(), random_state=random_state, n_jobs=-1))]),
        {'model__n_estimators': [50, 100], 'model__max_samples': [0.8, 1.0]}
    ),
    "catboost": (
        Pipeline([('preprocessor', tree_preprocessor),
                  ('model', CatBoostRegressor(random_seed=random_state, allow_writing_files=False, verbose=0))]),
        {'model__depth': [6, 8], 'model__learning_rate': [0.03, 0.05], 'model__iterations': [700]}
    ),
    "lgbm": (
        Pipeline([('preprocessor', tree_preprocessor),
                  # Добавляем verbose=-1, чтобы скрыть предупреждения о разделении
                  ('model', LGBMRegressor(random_state=random_state, verbose=-1))]),
        {
            'model__n_estimators': [500],
            'model__learning_rate': [0.02],
            # --- ДОБАВЛЕННЫЕ ПАРАМЕТРЫ ДЛЯ КОНТРОЛЯ СЛОЖНОСТИ ---
            'model__num_leaves': [31],  # Ограничиваем количество листьев (дефолт: 31)
            'model__max_depth': [7],  # Ограничиваем глубину (-1 означает без ограничений)
            'model__min_child_samples': [10]  # Устанавливаем минимальное число записей в листе (дефолт: 20)
        }
    ),
    "xgboost": (
        Pipeline([('preprocessor', tree_preprocessor),
                  ('model', XGBRegressor(random_state=random_state, n_jobs=-1))]),
        {'model__n_estimators': [500, 800], 'model__learning_rate': [0.02, 0.05]}
    ),
}
os.environ["JOBLIB_TEMP_FOLDER"] = r"C:\joblib_cache"
logging.getLogger("mlflow").setLevel(logging.DEBUG)
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn.pipeline')
dtype_map = {
            'bool': DataType.boolean,
            'int64': DataType.long,
            'float64': DataType.double,
            'object': DataType.string  # <--- КЛЮЧЕВОЙ МОМЕНТ: все сложные объекты считаем строками
        }
n_splits = 5
oof_predictions = {}
best_estimators = {}
experiment_name = "KINOPRED_EXPERIMENT_ALPHA_MALE_4"
cv_strategy = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
comb_len = 10
for name, (pipeline, grid) in pipelines_and_grids.items():
    run_experiments(u_id, X, y, pipeline, grid, cv_strategy, name, r_state=random_state, splits_count=n_splits)
# run_stacking_experiment(oof_predictions, best_estimators, u_id, random_state, n_splits, X, y, cv_strategy, comb_len)
linear_preprocessor.fit(X, y)
X = linear_preprocessor.transform(X)
# for col in X.columns:
#     print(col, sum(X[col].isna().astype(int)))
