Продемонстрируем кодом, как логировать все категории данных, о которых мы говорили (параметры, метрики, артефакты, сигнатура модели), используя MLflow. Я включу примеры для различных типов данных и ситуаций.

```python
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import json
import os
import datetime

# --- 0. Предварительная настройка (если используете Tracking Server) ---
# Если вы запустили 'mlflow server --host 127.0.0.1 --port 5000'
# раскомментируйте следующую строку. Иначе логи будут сохраняться локально в папке 'mlruns'.
# mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Устанавливаем имя эксперимента. Все запуски в этом скрипте будут принадлежать этому эксперименту.
mlflow.set_experiment("MLflow_Logging_Demonstration")

# --- 1. Подготовка данных для примера ---
np.random.seed(42)
X = pd.DataFrame(np.random.rand(100, 5), columns=[f'feature_{i}' for i in range(5)])
y = 2 * X['feature_0'] + 3 * X['feature_1'] - 0.5 * X['feature_2'] + np.random.randn(100) * 0.5

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2. Начинаем MLflow Run ---
# Используем 'with' блок для автоматического завершения запуска
with mlflow.start_run(run_name=f"Demo_Run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
    run_id = run.info.run_id
    experiment_id = run.info.experiment_id
    print(f"MLflow Run ID: {run_id}")
    print(f"MLflow Experiment ID: {experiment_id}")

    # ======================================================================
    # 2.1. Логирование Параметров (Parameters)
    # ======================================================================
    print("\n--- Логирование Параметров ---")
    # Логирование отдельных параметров
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_param("random_seed", 42)
    mlflow.log_param("test_split_ratio", 0.2)

    # Логирование словаря параметров
    training_params = {
        "n_samples": X_train.shape[0],
        "n_features": X_train.shape[1],
        "feature_names": X_train.columns.tolist()
    }
    mlflow.log_params(training_params)

    # Параметры, специфичные для модели
    fit_intercept = True
    normalize = False # В старых версиях sklearn, теперь deprecated/removed. Для примера.
    mlflow.log_param("fit_intercept", fit_intercept)
    mlflow.log_param("normalize_data_before_fit", normalize)

    # ======================================================================
    # 2.2. Логирование Модели и Сигнатуры (Model and Signature)
    # ======================================================================
    print("\n--- Логирование Модели и Сигнатуры ---")
    model = LinearRegression(fit_intercept=fit_intercept)
    model.fit(X_train, y_train)

    # Генерируем input_example
    # Важно: input_example должен быть в формате, который ваша модель ожидает на вход.
    # Если модель ожидает DataFrame, передаем DataFrame. Если NumPy array, передаем array.
    # Для sklearn.LinearRegression, обычно ожидается 2D array или DataFrame.
    input_example_data = X_train.head(1) # Берем одну строку из тренировочных данных
    print(f"Input example data type: {type(input_example_data)}")
    print(f"Input example data shape: {input_example_data.shape}")

    # Логируем модель с сигнатурой
    mlflow.sklearn.log_model(
        sk_model=model,
        # 'name' - это путь внутри папки артефактов запуска MLflow, где будет сохранена модель.
        # Это то, что раньше называлось artifact_path.
        name="linear_regression_model",
        input_example=input_example_data, # Передаем пример входных данных для вывода сигнатуры
        registered_model_name=None # Можно зарегистрировать модель в Model Registry, если нужно
    )
    print("Модель успешно залогирована с input_example.")

    # ======================================================================
    # 2.3. Логирование Метрик (Metrics)
    # ======================================================================
    print("\n--- Логирование Метрик ---")
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Логирование отдельных метрик
    mlflow.log_metric("test_rmse", rmse)
    mlflow.log_metric("test_r2_score", r2)
    print(f"Logged Metrics: RMSE={rmse:.4f}, R2={r2:.4f}")

    # Логирование метрик в цикле (например, потери по эпохам)
    # Представим, что это метрики обучения
    for i in range(5):
        mlflow.log_metric("train_loss", 0.5 - i * 0.05, step=i)
        mlflow.log_metric("val_loss", 0.6 - i * 0.04, step=i)
    print("Logged loss metrics over steps.")

    # ======================================================================
    # 2.4. Логирование Артефактов (Artifacts)
    # ======================================================================
    print("\n--- Логирование Артефактов ---")

    # Логирование простого текстового файла
    with open("notes.txt", "w") as f:
        f.write("This is a simple note about the experiment.\n")
        f.write("The linear regression model was chosen for this demo.")
    mlflow.log_artifact("notes.txt", artifact_path="misc_files") # Сохранится в 'misc_files/' внутри артефактов запуска
    print("Logged notes.txt as an artifact.")
    os.remove("notes.txt") # Очищаем временный файл

    # Логирование графика
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    plt.savefig("actual_vs_predicted.png")
    mlflow.log_artifact("actual_vs_predicted.png", artifact_path="plots")
    print("Logged actual_vs_predicted.png as an artifact.")
    plt.close(fig) # Закрываем график, чтобы избежать утечек памяти
    os.remove("actual_vs_predicted.png") # Очищаем временный файл

    # Логирование JSON-файла (как у вас в 'results.json' и 'params.json')
    results_data = {
        "final_rmse": rmse,
        "final_r2": r2,
        "best_features_info": "Top 2 features were most important (for demo purposes)"
    }
    with open("results_summary.json", "w") as f:
        json.dump(results_data, f, indent=4)
    mlflow.log_artifact("results_summary.json", artifact_path="summaries")
    print("Logged results_summary.json as an artifact.")
    os.remove("results_summary.json") # Очищаем временный файл

    # ======================================================================
    # 2.5. Логирование Git-информации (автоматически)
    # ======================================================================
    print("\n--- Git Info ---")
    # MLflow автоматически логирует Git-хеш и другие детали,
    # если ваш проект находится в Git-репозитории.
    # Вам не нужно писать для этого код.
    print("Git information (if available) will be logged automatically by MLflow.")


print(f"\nExperiment finished. View run at: {mlflow.get_tracking_uri()}/#/experiments/{experiment_id}/runs/{run_id}")