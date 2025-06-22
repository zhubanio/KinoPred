import os
import time
from itertools import combinations
import pickle

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from sklearn.ensemble import (BaggingRegressor, ExtraTreesRegressor,
                              RandomForestRegressor, StackingRegressor)
from sklearn.feature_selection import RFE
from sklearn.linear_model import (Lasso, LinearRegression, Ridge, RidgeCV)
from sklearn.metrics import (mean_absolute_error, r2_score,
                             root_mean_squared_error, accuracy_score)
from sklearn.model_selection import (GridSearchCV, KFold, cross_val_predict,
                                     cross_val_score)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from category_encoders import CatBoostEncoder
from preprocessor import DataPreprocessor


def evaluate_oof_predictions(y_true, y_pred_float):
    """
    Рассчитывает набор метрик для OOF (Out-of-Fold) предсказаний.
    y_pred_float - предсказания в виде непрерывных чисел.
    """
    y_pred_rounded = np.clip(np.round(y_pred_float), 1, 10).astype(int)
    y_true_rounded = np.round(y_true).astype(int)

    acc = accuracy_score(y_true_rounded, y_pred_rounded)
    mae = mean_absolute_error(y_true, y_pred_float)
    rmse = root_mean_squared_error(y_true, y_pred_float)
    r2 = r2_score(y_true, y_pred_float)

    return {"acc": acc, "mae": mae, "rmse": rmse, "r2": r2}


def find_best_model(model_name, pipeline, param_grid, x, y, cv):
    """
    Выполняет GridSearchCV для поиска лучших гиперпараметров для пайплайна.
    """
    # print(f"--- Подбор параметров для {model_name} ---")
    start_time = time.time()

    gs = GridSearchCV(pipeline, param_grid, cv=cv, scoring='neg_root_mean_squared_error', n_jobs=-1)
    gs.fit(x, y)

    elapsed_time = time.time() - start_time
    best_score_rmse = -gs.best_score_

    # print(f"Лучшие параметры: {gs.best_params_}")
    # print(f"Лучший RMSE на CV: {best_score_rmse:.4f}")
    # print(f"Время подбора: {elapsed_time:.2f} сек.")

    return gs.best_estimator_, elapsed_time


def train_and_ensemble(file_path: str, output_path: str):
    os.environ["JOBLIB_TEMP_FOLDER"] = r"C:\tmp\cachedir"
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', 100)
    """
    Основная функция, которая загружает данные, обучает множество моделей,
    оценивает их и создает ансамбли.
    """
    try:
        df = pd.read_json(file_path, encoding="utf-8")
    except FileNotFoundError:
        print(f"Ошибка: файл {file_path} не найден.")
        return

    X = df.drop(columns=['MYrating'])
    y = df['MYrating']

    cat_cols_for_cbe = ["main_actor", "main_director", "main_writer", "ratingMpaa", "main_language"]

    linear_preprocessor = Pipeline([
        ('base_prep', DataPreprocessor()),
        ('cbe_scaler', ColumnTransformer(
            transformers=[
                ('cat', Pipeline([
                    ('encoder', CatBoostEncoder()),
                    ('scaler', StandardScaler())
                ]), cat_cols_for_cbe)
            ],
            remainder='passthrough'
        ))
    ])
    tree_preprocessor = Pipeline([
        ('base_prep', DataPreprocessor()),
        ('cbe_only', ColumnTransformer(
            transformers=[
                ('cat', CatBoostEncoder(cols=cat_cols_for_cbe), cat_cols_for_cbe)
            ],
            remainder='passthrough'
        ))
    ])

    pipelines_and_grids = {
        # --- Линейные модели (используют linear_preprocessor) ---
        "lin_reg": (
            Pipeline([('preprocessor', linear_preprocessor), ('feature_selection', RFE(estimator=LinearRegression())),
                      ('model', LinearRegression())]),
            {'preprocessor__cbe_scaler__cat__encoder__cols': [cat_cols_for_cbe],
             'feature_selection__n_features_to_select': [20, 40, 60]}
        ),
        "ridge": (
            Pipeline([('preprocessor', linear_preprocessor), ('model', Ridge(max_iter=2000))]),
            {'preprocessor__cbe_scaler__cat__encoder__cols': [cat_cols_for_cbe], 'model__alpha': [1.0, 10.0, 50.0]}
        ),
        "lasso": (
            Pipeline([('preprocessor', linear_preprocessor), ('model', Lasso(max_iter=2000))]),
            {'preprocessor__cbe_scaler__cat__encoder__cols': [cat_cols_for_cbe], 'model__alpha': [0.001, 0.01, 0.1]}
        ),
        "knn_regressor": (
            Pipeline([('preprocessor', linear_preprocessor), ('model', KNeighborsRegressor())]),
            {'preprocessor__cbe_scaler__cat__encoder__cols': [cat_cols_for_cbe], 'model__n_neighbors': [10, 20]}
        ),

        # --- Древовидные модели (используют tree_preprocessor) ---
        "decision_tree": (
            Pipeline([('preprocessor', tree_preprocessor), ('model', DecisionTreeRegressor(random_state=42))]),
            {'model__max_depth': [5, 10, 20], 'model__min_samples_leaf': [2, 5, 10]}
        ),
        "random_forest": (
            Pipeline(
                [('preprocessor', tree_preprocessor), ('model', RandomForestRegressor(random_state=42, n_jobs=-1))]),
            {'model__n_estimators': [200, 300], 'model__max_depth': [10, 20]}
        ),
        "extra_trees": (
            Pipeline([('preprocessor', tree_preprocessor), ('model', ExtraTreesRegressor(random_state=42, n_jobs=-1))]),
            {'model__n_estimators': [200, 300], 'model__max_depth': [10, 20]}
        ),
        "bagging_regressor": (
            Pipeline([('preprocessor', tree_preprocessor),
                      ('model', BaggingRegressor(estimator=DecisionTreeRegressor(), random_state=42, n_jobs=-1))]),
            {'model__n_estimators': [50, 100], 'model__max_samples': [0.8, 1.0]}
        ),
        "catboost": (
            Pipeline([('preprocessor', tree_preprocessor),
                      ('model', CatBoostRegressor(random_seed=42, allow_writing_files=False))]),
            {'model__depth': [6, 8], 'model__learning_rate': [0.03, 0.05], 'model__iterations': [700]}
        ),
        "lgbm": (
            Pipeline([('preprocessor', tree_preprocessor), ('model', LGBMRegressor(random_state=42))]),
            {'model__n_estimators': [500, 800], 'model__learning_rate': [0.02, 0.05]}
        ),
        "xgboost": (
            Pipeline([('preprocessor', tree_preprocessor),
                      ('model', XGBRegressor(random_state=42, n_jobs=-1))]),
            {'model__n_estimators': [500, 800], 'model__learning_rate': [0.02, 0.05]}
        ),
    }

    cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_predictions = {}
    model_metrics = {}
    best_estimators = {}

    for name, (pipeline, grid) in pipelines_and_grids.items():
        best_model, _ = find_best_model(name, pipeline, grid, X, y, cv_strategy)
        best_estimators[name] = best_model

        # print(f"Получение OOF предсказаний для {name}...")
        oof_preds = cross_val_predict(best_model, X, y, cv=cv_strategy, n_jobs=-1)
        oof_predictions[name] = oof_preds

        metrics = evaluate_oof_predictions(y, oof_preds)
        model_metrics[name] = metrics
        # print(f"Метрики для {name}: RMSE={metrics['rmse']:.4f}, R2={metrics['r2']:.4f}\n")

    oof_df = pd.DataFrame(oof_predictions)
    # print("=" * 50)
    # print("Матрица корреляций OOF-предсказаний базовых моделей:")
    # print(oof_df.corr())
    # print("=" * 50)

    # print("\nПоиск лучшей комбинации для стекинга...")
    best_stacking_rmse = float('inf')
    best_stacking_combo = None

    for k in range(2, min(len(oof_df.columns), 5) + 1):
        for combo in combinations(oof_df.columns, k):
            combo_list = list(combo)
            X_stack = oof_df[combo_list].values

            meta_model = RidgeCV(alphas=np.logspace(-2, 2, 10))

            stack_cv_scores = cross_val_score(meta_model, X_stack, y, cv=cv_strategy,
                                              scoring='neg_root_mean_squared_error', n_jobs=-1)
            current_rmse = -np.mean(stack_cv_scores)

            if current_rmse < best_stacking_rmse:
                best_stacking_rmse = current_rmse
                best_stacking_combo = combo_list

    # print(f"\nЛучшая комбинация для стекинга: {best_stacking_combo} с RMSE = {best_stacking_rmse:.4f}")

    # print("\nОбучение финального ансамбля (Stacking) на всех данных...")

    # if best_stacking_combo is None:
    #     print("Не удалось выбрать комбинацию для стекинга. Проверьте результаты базовых моделей.")
    #     return

    final_estimators = [(name, best_estimators[name]) for name in best_stacking_combo]

    final_stacking_model = StackingRegressor(
        estimators=final_estimators,
        final_estimator=RidgeCV(),
        cv=cv_strategy,
        n_jobs=-1
    )

    final_stacking_model.fit(X, y)
    # print("Финальная модель обучена!")

    with open(output_path, 'wb') as f:
        pickle.dump(final_stacking_model, f)

    # print("\nФинальная ансамблевая модель сохранена в 'final_ensemble_model.pkl'")
    # print("Вы можете загрузить ее и использовать метод .predict(X_new) для новых данных.")
