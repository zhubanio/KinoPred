import json
from collections import defaultdict
import numpy as np
import pandas as pd
from category_encoders import CatBoostEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 100)
pd.set_option('future.no_silent_downcasting', True)


def add_genres_and_countries_count(d_df):
    d_df["genres_count"] = d_df["genres"].apply(lambda x: len(x) if isinstance(x, list) else 0)
    d_df["countries_count"] = d_df["countries"].apply(lambda x: len(x) if isinstance(x, list) else 0)
    d_df["genres_count"] = np.log1p(d_df["genres_count"])
    d_df["countries_count"] = np.log1p(d_df["countries_count"])
    return d_df


def get_money_value(d_df, field):
    d_df[field] = d_df[field].apply(
        lambda v:
        (v.get("value") if isinstance(v, dict) and (v.get("currency") == "$"
                                                    or v.get("currency") == "€") and v.get("value") != 0 else None)
    )
    return d_df


def adjust_for_euro_safe(d_df: pd.DataFrame, field: str, eur_df):
    """
    Безопасно конвертирует валюту, создавая новые словари вместо изменения старых.
    """

    def convert_row(row):
        cell_value = row[field]
        # Проверяем, что это словарь и валюта - евро
        if isinstance(cell_value, dict) and cell_value.get("currency") == "€":
            # 1. Создаем копию словаря, чтобы не трогать оригинал
            new_dict = cell_value.copy()
            year = row["year"]

            # 2. Определяем курс
            exchange_rate_year = 1999 if year < 1999 else year
            exchange_rate = eur_df.loc[exchange_rate_year, "cs"]

            # 3. Модифицируем КОПИЮ и возвращаем ее
            new_dict["value"] = new_dict.get("value", 0) * exchange_rate
            return new_dict

        # Если конвертация не нужна, возвращаем исходное значение
        return cell_value

    # Применяем функцию к каждой строке и полностью заменяем старый столбец новым
    d_df[field] = d_df.apply(convert_row, axis=1)
    return d_df


def fill_age_and_mpaa(d_df, mpaa_to_age):
    def age_to_mpaa(age):
        if pd.isna(age):
            return None
        if age <= 6:
            return 'g'
        elif age <= 12:
            return 'pg'
        elif age <= 16:
            return 'pg-13'
        else:
            return 'r'

    d_df['ageRating'] = d_df.apply(
        lambda row: mpaa_to_age.get(row['ratingMpaa'].lower()) if pd.isna(row['ageRating']) and isinstance(
            row['ratingMpaa'], str) and row['ratingMpaa'].lower() in mpaa_to_age else row['ageRating'],
        axis=1)

    d_df['ratingMpaa'] = d_df.apply(
        lambda row: age_to_mpaa(row['ageRating']) if pd.isna(row['ratingMpaa']) or row['ratingMpaa'] == '' else row[
            'ratingMpaa'],
        axis=1)
    return d_df


def get_countries_and_genres(d_df):
    d_df["genres"] = d_df["genres"].apply(lambda v: [v0.get("name") for v0 in v] if isinstance(v, list) else None)
    d_df["countries"] = d_df["countries"].apply(lambda v: [v0.get("name") for v0 in v] if isinstance(v, list) else None)
    d_df = add_genres_and_countries_count(d_df)
    d_df["main_genre"] = d_df["genres"].apply(
        lambda v: v[0] if isinstance(v, list) and len(v) >= 1 else None)
    d_df["main_country"] = d_df["countries"].apply(
        lambda v: v[0] if isinstance(v, list) and len(v) >= 1 else None)

    return d_df.copy()


def is_on_kp_hd(d_df):
    def check_kp_hd(watchability):
        if not isinstance(watchability, dict):
            return 0
        items = watchability.get("items", [])
        if not isinstance(items, list):
            return 0
        return int(any(item.get("name") == "Kinopoisk HD" for item in items if isinstance(item, dict)))

    d_df = d_df.copy()
    d_df["is_on_kp_hd"] = d_df["watchability"].apply(check_kp_hd)

    return d_df


def votes(d_df):
    cols = ["votes_kp", "votes_imdb", "votes_critics", "votes_rus_critics"]
    resources = ["kp", "imdb", "filmCritics", "russianFilmCritics"]
    for c in range(len(cols)):
        col = cols[c]
        res = resources[c]
        d_df[col] = d_df["votes"].apply(lambda r: r.get(res) if isinstance(r, dict) else 0)
        d_df[col] = d_df[col].fillna(0)
        d_df[col] = np.log1p(d_df[col])

    return d_df.copy()


def lists(d_df):
    def count_lists(ls):
        if isinstance(ls, list):
            return len(ls)
        return 0

    d_df["lists_count"] = d_df["lists"].apply(count_lists)
    # d_df["lists_count"] = np.log1p(d_df["lists_count"])
    return d_df


def get_ratings(d_df):
    cols = ["rating_kp", "rating_imdb", "rating_critics", "rating_rus_critics"]
    resources = ["kp", "imdb", "filmCritics", "russianFilmCritics"]
    for c in range(len(cols)):
        col = cols[c]
        res = resources[c]
        d_df[col] = d_df["rating"].apply(lambda r: r.get(res) if isinstance(r, dict) else None)
        d_df[col] = d_df[col].replace(0, np.nan)
        d_df[f"no_{col}"] = d_df[col].isna().astype(int)

    return d_df.copy()


def get_audience(d_df):
    def get_audience_count(audience):
        total = 0
        rus = 0
        if isinstance(audience, list):
            for aud in audience:
                cnt = aud.get("count", 0)
                total += cnt
                if aud.get("country") == "Россия":
                    rus += cnt
        return pd.Series({'audience_count': total, 'rus_audience_count': rus})

    audience_features = d_df["audience"].apply(get_audience_count)
    d_df = pd.concat([d_df, audience_features], axis=1)
    d_df["no_audience"] = d_df["audience_count"].isna().astype(int)
    d_df["no_rus_audience"] = d_df["rus_audience_count"].isna().astype(int)
    d_df["audience_count"] = d_df["audience_count"].replace(0, np.nan)
    d_df["rus_audience_count"] = d_df["rus_audience_count"].replace(0, np.nan)
    return d_df


def get_spoken_language(d_df):
    d_df["main_language"] = d_df["spokenLanguages"].apply(lambda v: v[0].get("nameEn") if isinstance(
        v, list) and len(v) >= 1 else None)
    d_df["spoken_languages_count"] = d_df["spokenLanguages"].apply(lambda v: len(v) if isinstance(v, list) else 0)
    d_df["spoken_languages_count"] = np.log1p(d_df["spoken_languages_count"])

    return d_df.copy()


def drop_multicollinear_features(d_df, threshold=0.95):
    # Оставляем только числовые признаки
    numeric_df = d_df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    # if verbose and to_drop:
    #     print(f"Удалено {len(to_drop)} мультиколлинеарных признаков: {to_drop}")

    # Возвращаем исходный DataFrame без мультиколлинеарных числовых признаков
    return d_df.drop(columns=to_drop, errors="ignore")


def save_data_(data, path):
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def adjust_for_inflation_df(d_df: pd.DataFrame, value_col: str, year_col: str, new_col: str,
                            inflation_path: str = 'auxiliary files/inflation_coefficients.json') -> pd.DataFrame:
    try:
        with open(inflation_path, encoding='utf-8') as d_f:
            inflation_table = json.load(d_f)
    except FileNotFoundError:
        print(f"Warning: Inflation file not found at {inflation_path}. Inflation adjustment will be skipped.")
        return d_df

    def adjust_value(value, year):
        if pd.isna(value) or pd.isna(year):
            return None
        coef = inflation_table.get(str(int(year)))
        return value * coef if coef is not None else None

    d_df = d_df.copy()
    d_df[new_col] = d_df[[value_col, year_col]].apply(
        lambda row: adjust_value(row[value_col], row[year_col]),
        axis=1
    )
    return d_df


def _calculate_loo_means(df, entity_col, target_col):
    """Вычисляет средний рейтинг для каждой сущности (актера, сценариста)."""
    actor_values = defaultdict(lambda: {'total_rating': 0, 'total_weight': 0})
    max_n = 5 if entity_col == 'actors' else 2

    for _, row in df.iterrows():
        entities = row[entity_col]
        rating = row[target_col]
        if not isinstance(entities, list) or pd.isna(rating):
            continue
        for i, entity in enumerate(entities[:max_n]):
            weight = 1 / (i + 1)
            actor_values[entity]['total_rating'] += rating * weight
            actor_values[entity]['total_weight'] += weight

    entity_means = {
        entity: data['total_rating'] / data['total_weight']
        for entity, data in actor_values.items() if data['total_weight'] > 0
    }
    return entity_means


def _transform_scaler(data_series, scaler):
    return scaler.transform(data_series.to_frame())


def _fit_scaler(data_series):
    scaler = StandardScaler()
    scaler.fit(data_series.to_frame())
    return scaler


def _apply_loo_means(df, entity_col, entity_means, global_mean):
    """Применяет вычисленные LOO-средние к датафрейму."""
    results = []
    max_n = 5 if entity_col == 'actors' else 2

    for _, row in df.iterrows():
        entities = row[entity_col]
        if not isinstance(entities, list) or not entities:
            results.append(global_mean)
            continue

        weighted_ratings = 0
        total_weights = 0
        for i, entity in enumerate(entities[:max_n]):
            # Используем среднее для сущности из обучающего набора,
            # или глобальное среднее, если сущность новая.
            actor_mean = entity_means.get(entity, global_mean)
            weight = 1 / (i + 1)
            weighted_ratings += actor_mean * weight
            total_weights += weight

        val = weighted_ratings / total_weights if total_weights > 0 else global_mean
        results.append(val)

    return pd.Series(results, index=df.index)


def profession_tr(d_df: pd.DataFrame, profession: str, the_count: int) -> pd.DataFrame:
    for num in range(1, the_count + 1):
        d_df[F"{profession}_{num}"] = d_df[profession].apply(lambda v: v[num - 1] if isinstance(
            v, list) and len(v) >= num else None)
    return d_df


def get_eur_df(eur_path):
    eur_df = pd.read_csv(eur_path)
    eur_cs = eur_df["observation_date;AEXUSEU"].apply(lambda x: float(x.split(";")[1]))
    eur_date = eur_df["observation_date;AEXUSEU"].apply(lambda x: x.split(";")[0])
    eur_df["cs"] = eur_cs
    eur_df["date"] = eur_date
    eur_df = eur_df.drop(["observation_date;AEXUSEU"], axis=1)
    eur_df["date"] = eur_df["date"].apply(lambda x: int(x.split(".")[-1]))
    eur_df.index = eur_df["date"]
    eur_df = eur_df.drop("date", axis=1)
    return eur_df


def collect_actor_statistics(d_df, target_col='MYrating', actors_col='actors', max_n=5):
    """Собирает статистики по актерам из датафрейма."""
    actor_values = defaultdict(list)
    for idx, row in d_df.iterrows():
        actors = row[actors_col]
        rating = row[target_col]
        if not isinstance(actors, list) or pd.isna(rating):
            continue
        for i, actor in enumerate(actors[:max_n]):
            weight = 1 / (i + 1)
            actor_values[actor].append((idx, rating, weight))
    global_mean = d_df[target_col].mean()
    return {'actor_values': actor_values, 'global_mean': global_mean}


# ФУНКЦИЯ 2: ЛОГИКА ДЛЯ TRANSFORM
def create_loo_rating_feature(d_df, actor_stats, actors_col='actors', max_n=5):
    """Создает новую колонку с leave-one-out рейтингом на основе собранных статистик."""
    actor_values = actor_stats['actor_values']
    global_mean = actor_stats['global_mean']
    results = []
    for idx, row in d_df.iterrows():
        actors = row[actors_col]
        if not isinstance(actors, list) or not actors:
            results.append(global_mean)
            continue
        rating_vals, weight_vals = [], []
        for i, actor in enumerate(actors[:max_n]):
            all_ratings_weights = [(k, r, w) for k, r, w in actor_values[actor] if k != idx]
            if all_ratings_weights:
                weighted_sum = sum(r * w for _, r, w in all_ratings_weights)
                total_weight = sum(w for _, _, w in all_ratings_weights)
                actor_mean = weighted_sum / total_weight
            else:
                actor_mean = global_mean
            weight = 1 / (i + 1)
            rating_vals.append(actor_mean * weight)
            weight_vals.append(weight)
        val = sum(rating_vals) / sum(weight_vals) if weight_vals else global_mean
        results.append(val)
    return pd.Series(results, index=d_df.index, name='actors_weighted_rating_loo')


def year_tr(d_df):
    d_df["age"] = 2025 - d_df["year"]
    d_df["age"] = np.log1p(d_df["age"])
    return d_df


def get_budget(eur_df, d_df):
    d_df = adjust_for_euro_safe(d_df, "budget", eur_df)
    d_df = get_money_value(d_df, "budget")
    d_df = adjust_for_inflation_df(d_df, value_col='budget', year_col='year', new_col='budget')
    d_df["no_budget"] = d_df["budget"].isna().astype(int)
    d_df["budget"] = d_df["budget"].replace(0, np.nan)
    return d_df.copy()


def budget_tr(self, x):
    x = get_budget(self.eur_df, x)
    x['budget'] = self.budget_imputer.transform(x[['budget']])
    x["budget"] = np.log1p(x["budget"])
    return x


def persons_fi(self, x, y):
    x = get_persons(x)
    self.actors_stats = collect_actor_statistics(x, target_col='MYrating', actors_col='actors',
                                                 max_n=self.profs_count['actor'])
    self.writers_stats = collect_actor_statistics(x, target_col='MYrating', actors_col='writers',
                                                  max_n=self.profs_count['writer'])
    self.main_actor_cbe.fit(x[["main_actor"]], y)
    self.main_director_cbe.fit(x[["main_director"]], y)
    self.main_writer_cbe.fit(x[["main_writer"]], y)
    return x


def persons_tr(self, x):
    x = get_persons(x)
    x["actors"] = create_loo_rating_feature(x, self.actors_stats,
                                            actors_col='actors',
                                            max_n=self.profs_count['actor'])
    x["writers"] = create_loo_rating_feature(x, self.writers_stats,
                                             actors_col='writers',
                                             max_n=self.profs_count['writer'])
    # x["writers"] = np.log1p(x["writers"])
    x["main_writer"] = self.main_writer_cbe.transform(x[["main_writer"]])
    x["main_actor"] = self.main_actor_cbe.transform(x[["main_actor"]])
    x["main_director"] = self.main_director_cbe.transform(x[["main_director"]])
    # x["main_director"] = np.log1p(x["main_director"])
    return x


def get_persons(d_df):
    # persons
    d_df["actors"] = d_df["persons"].apply(lambda v: [i.get("enName") for i in v if isinstance(i, dict) and
                                                      i.get("enProfession") == "actor"] if isinstance(
        v, list) else None)
    d_df["directors"] = d_df["persons"].apply(lambda v: [i.get("enName") for i in v if isinstance(i, dict) and
                                                         i.get("enProfession") == "director"] if isinstance(
        v, list) else None)
    d_df["writers"] = d_df["persons"].apply(lambda v: [i.get("enName") for i in v if isinstance(i, dict) and
                                                       i.get("enProfession") == "writer"] if isinstance(
        v, list) else None)
    d_df["actors_count"] = d_df["actors"].apply(lambda v: len(v) if isinstance(v, list) else 0)
    d_df["actors_count"] = np.log1p(d_df["actors_count"])
    d_df["directors_count"] = d_df["directors"].apply(lambda v: len(v) if isinstance(v, list) else 0)
    d_df["directors_count"] = np.log1p(d_df["directors_count"])
    d_df["writers_count"] = d_df["writers"].apply(lambda v: len(v) if isinstance(v, list) else 0)
    d_df["writers_count"] = np.log1p(d_df["writers_count"])
    d_df["main_writer"] = d_df["writers"].apply(
        lambda v: v[0] if isinstance(v, list) and len(v) >= 1 else None)
    d_df["main_director"] = d_df["directors"].apply(
        lambda v: v[0] if isinstance(v, list) and len(v) >= 1 else None)
    d_df["main_actor"] = d_df["actors"].apply(
        lambda v: v[0] if isinstance(v, list) and len(v) >= 1 else None)
    return d_df.copy()


def get_fees(self, d_df):
    cols = ["fees_world", "fees_usa", "fees_russia"]
    d_df["fees_world"] = d_df["fees"].apply(lambda x: x.get("world") if isinstance(x, dict) and "world" in x else None)
    d_df["fees_usa"] = d_df["fees"].apply(lambda x: x.get("usa") if isinstance(x, dict) and "usa" in x else None)
    d_df["fees_russia"] = d_df["fees"].apply(lambda x:
                                             x.get("russia") if isinstance(x, dict) and "russia" in x else None)
    d_df["no_fees_world"] = d_df["fees_world"].isna().astype(int)
    d_df["no_fees_usa"] = d_df["fees_usa"].isna().astype(int)
    d_df["no_fees_russia"] = d_df["fees_russia"].isna().astype(int)
    d_df = adjust_for_euro_safe(d_df, "fees_world", self.eur_df)
    for col in cols:
        d_df = get_money_value(d_df, col)

    return d_df


def fees_fi(self, x, y):
    x = get_fees(self, x)
    self.fees_world_imputer.fit(x[["fees_world"]], y)
    self.fees_russia_imputer.fit(x[["fees_russia"]], y)
    self.fees_usa_imputer.fit(x[["fees_usa"]], y)
    return x


def fees_tr(self, x):
    x = get_fees(self, x)
    x["fees_world"] = self.fees_world_imputer.transform(x[["fees_world"]])
    x["fees_usa"] = self.fees_usa_imputer.transform(x[["fees_usa"]])
    x["fees_russia"] = self.fees_russia_imputer.transform(x[["fees_russia"]])
    x["fees_world"] = np.log1p(x["fees_world"])
    x["fees_russia"] = np.log1p(x["fees_russia"])
    x["fees_usa"] = np.log1p(x["fees_usa"])
    return x


def ratings_fi(self, x, y):
    x = get_ratings(x)
    self.rating_kp_imputer.fit(x[["rating_kp"]], y)
    self.rating_imdb_imputer.fit(x[["rating_imdb"]], y)
    self.rating_critics_imputer.fit(x[["rating_critics"]], y)
    self.rating_rus_critics_imputer.fit(x[["rating_rus_critics"]], y)
    return x


def ratings_tr(self, x):
    x = get_ratings(x)
    x["rating_kp"] = self.rating_kp_imputer.transform(x[["rating_kp"]])
    x["rating_imdb"] = self.rating_imdb_imputer.transform(x[["rating_imdb"]])
    x["rating_critics"] = self.rating_critics_imputer.transform(x[["rating_critics"]])
    x["rating_rus_critics"] = self.rating_rus_critics_imputer.transform(
        x[["rating_rus_critics"]])
    return x


def audience_fi(self, x, y):
    x = get_audience(x)
    self.audience_count_imputer.fit(x[["audience_count"]], y)
    self.rus_audience_count_imputer.fit(x[["rus_audience_count"]], y)
    return x


def audience_tr(self, x):
    x = get_audience(x)
    x["audience_count"] = self.audience_count_imputer.transform(x[["audience_count"]])
    x["rus_audience_count"] = self.rus_audience_count_imputer.transform(
        x[["rus_audience_count"]])
    x["audience_count"] = np.log1p(x["audience_count"])
    x["rus_audience_count"] = np.log1p(x["rus_audience_count"])
    return x


def countries_and_genres_fi(self, x, y):
    x = get_countries_and_genres(x)
    self.countries_stats = collect_actor_statistics(x, target_col='MYrating', actors_col='countries',
                                                    max_n=self.max_countries)
    self.genres_stats = collect_actor_statistics(x, target_col='MYrating', actors_col='genres',
                                                 max_n=self.max_genres)
    self.main_genre_cbe.fit(x[["main_genre"]], y)
    self.main_country_cbe.fit(x[["main_country"]], y)
    return x


def countries_and_genres_tr(self, x):
    x = get_countries_and_genres(x)
    x["countries"] = create_loo_rating_feature(x, self.countries_stats,
                                               actors_col='countries', max_n=self.max_countries)
    x["genres"] = create_loo_rating_feature(x, self.genres_stats, actors_col='genres',
                                            max_n=self.max_countries)
    x["main_genre"] = self.main_genre_cbe.transform(x[["main_genre"]])
    x["main_country"] = self.main_country_cbe.transform(x[["main_country"]])
    return x


class DataTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.eur_df = get_eur_df('auxiliary files/AEXUSEU.csv')
        self.mpaa_to_age = {'g': 0, 'pg': 6, 'pg-13': 13, 'r': 17, 'nc-17': 18}
        self.profs_count = {"actor": 5, "writer": 2, "director": 1}
        self.budget_imputer = SimpleImputer(strategy="mean")
        self.main_actor_cbe = CatBoostEncoder(cols=["main_actor"])
        self.main_director_cbe = CatBoostEncoder(cols=["main_director"])
        self.main_writer_cbe = CatBoostEncoder(cols=["main_writer"])
        self.actors_stats = dict()
        self.writers_stats = dict()
        self.age_rating_imputer = SimpleImputer(strategy="most_frequent")
        self.fees_world_imputer = SimpleImputer(strategy="mean")
        self.fees_russia_imputer = SimpleImputer(strategy="mean")
        self.fees_usa_imputer = SimpleImputer(strategy="mean")
        self.rating_mpaa_cbe = CatBoostEncoder(cols=["ratingMpaa"])
        self.movie_length_imputer = SimpleImputer(strategy="mean")
        self.rating_kp_imputer = SimpleImputer(strategy="mean")
        self.rating_imdb_imputer = SimpleImputer(strategy="mean")
        self.rating_critics_imputer = SimpleImputer(strategy="mean")
        self.rating_rus_critics_imputer = SimpleImputer(strategy="mean")
        self.audience_count_imputer = SimpleImputer(strategy="median")
        self.rus_audience_count_imputer = SimpleImputer(strategy="median")
        self.main_language_cbe = CatBoostEncoder(cols=["main_language"])
        self.final_columns = list()
        self.max_countries = 5
        self.max_genres = 5
        self.countries_stats = dict()
        self.genres_stats = dict()
        self.main_genre_cbe = CatBoostEncoder(cols=["main_genre"])
        self.main_country_cbe = CatBoostEncoder(cols=["main_country"])

    def fit(self, x, y=None):
        """
        Метод fit только ОБУЧАЕТ параметры.
        Он не должен возвращать преобразованный DataFrame.
        """
        X_temp = x.copy()
        X_temp["MYrating"] = y
        X_temp = get_budget(self.eur_df, X_temp)
        self.budget_imputer.fit(X_temp[["budget"]], y)
        X_temp = fill_age_and_mpaa(X_temp, self.mpaa_to_age)
        self.age_rating_imputer.fit(X_temp[["ageRating"]], y)
        X_temp = persons_fi(self, X_temp, y)
        X_temp = fees_fi(self, X_temp, y)
        self.rating_mpaa_cbe.fit(X_temp[["ratingMpaa"]], y)
        self.movie_length_imputer.fit(X_temp[["movieLength"]], y)
        X_temp = ratings_fi(self, X_temp, y)
        X_temp = get_spoken_language(X_temp)
        X_temp = audience_fi(self, X_temp, y)
        self.main_language_cbe.fit(X_temp[["main_language"]], y)
        X_temp = countries_and_genres_fi(self, X_temp, y)
        # 3. Метод fit всегда должен возвращать self!
        numeric_cols = ["age", "budget", "no_budget", "no_ageRating", "ageRating", "actors", "writers", "main_director",
                        "main_actor", "main_writer", "genres_count", "countries_count", "is_on_kp_hd", "votes_kp",
                        "votes_imdb", "votes_critics", "votes_rus_critics", "fees_world", "fees_usa", "fees_russia",
                        "no_fees_world", "no_fees_usa", "no_fees_russia", "lists_count", "ratingMpaa", "short_film",
                        "movieLength", "rating_kp", "rating_imdb", "rating_critics", "rating_rus_critics",
                        "no_rating_kp", "no_rating_imdb", "no_rating_critics", "no_rating_rus_critics", "top250",
                        'audience_count', 'rus_audience_count', "no_audience", "no_rus_audience",
                        "spoken_languages_count", "main_language", "actors_count", "writers_count", "directors_count",
                        "countries", "genres"]
        self.final_columns = numeric_cols
        return self

    def transform(self, x, y=None):
        """
        Метод transform только ПРИМЕНЯЕТ преобразования, используя обученные параметры.
        """
        # Создаем копию, чтобы не изменять исходный DataFrame
        X_transformed = x.copy(deep=True)
        X_transformed = year_tr(X_transformed)
        X_transformed = budget_tr(self, X_transformed)
        X_transformed = fill_age_and_mpaa(X_transformed, self.mpaa_to_age)
        X_transformed["no_ageRating"] = X_transformed["ageRating"].isna().astype(int)
        X_transformed["ageRating"] = self.age_rating_imputer.transform(X_transformed[["ageRating"]])
        # X_transformed["ageRating"] = np.log1p(X_transformed["ageRating"])
        X_transformed = persons_tr(self, X_transformed)
        X_transformed = countries_and_genres_tr(self, X_transformed)
        X_transformed = is_on_kp_hd(X_transformed)
        X_transformed = votes(X_transformed)
        X_transformed = fees_tr(self, X_transformed)
        X_transformed = lists(X_transformed)
        X_transformed["ratingMpaa"] = self.rating_mpaa_cbe.transform(X_transformed[["ratingMpaa"]])
        X_transformed["short_film"] = (X_transformed["movieLength"] < 52).astype(int)
        X_transformed["movieLength"] = X_transformed["movieLength"].apply(lambda v: v if v >= 52 else None)
        X_transformed["movieLength"] = self.movie_length_imputer.transform(X_transformed[["movieLength"]])
        X_transformed = ratings_tr(self, X_transformed)
        X_transformed["top250"] = X_transformed["top250"].notna().astype(int)
        X_transformed = audience_tr(self, X_transformed)
        X_transformed = get_spoken_language(X_transformed)
        X_transformed["main_language"] = self.main_language_cbe.transform(X_transformed[["main_language"]])
        return X_transformed[self.final_columns]


class Pad(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self
        pass

    def transform(self, x, y=None):
        x_transformed = x.copy()  # Создаем копию, чтобы не трогать оригинал
        new_cols = []
        for column in x_transformed.columns:
            if "__" in column:
                new_cols.append(column.split("__")[-1])
            else:
                new_cols.append(column)
        x_transformed.columns = new_cols
        return x_transformed
