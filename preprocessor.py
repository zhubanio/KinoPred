import json
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 100)
pd.set_option('future.no_silent_downcasting', True)


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


def get_money_value(d_df, field):
    d_df[field] = d_df[field].apply(
        lambda v:
        (v.get("value") if isinstance(v, dict) and (v.get("currency") == "$"
                                                    or v.get("currency") == "€") and v.get("value") != 0 else None)
    )
    return d_df


def adjust_for_euro(d_df, field: str, eur_df):
    if field in d_df.axes[1]:
        for ind in d_df.index:
            b = d_df.loc[ind, field]
            if isinstance(b, dict) and b.get("currency") == "€":
                if d_df.loc[ind, "year"] < 1999:
                    d_df.loc[ind, field]["value"] = eur_df.loc[1999, "cs"] * b["value"]
                else:
                    d_df.loc[ind, field]["value"] = eur_df.loc[d_df.loc[ind, "year"], "cs"] * b["value"]
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


class DataPreprocessor:
    """
    Класс для предобработки данных о фильмах.
    Обучается на тренировочном наборе (fit) и затем трансформирует
    любые данные по тем же правилам (transform).
    """

    def __init__(self):
        # Атрибуты, которые будут "обучены" в методе fit
        self.scalers = {}
        self.imputation_values = {}
        self.loo_means = {}
        self.all_genres = []
        self.all_countries = []
        self.columns_to_drop = []
        self.final_columns = []
        self.eur_df = None
        self.mpaa_to_age = {'g': 0, 'pg': 6, 'pg-13': 13, 'r': 17, 'nc-17': 18}

    def fit(self, df: pd.DataFrame, y: pd.Series):
        """
        Обучает препроцессор: вычисляет средние, скейлеры, списки колонок и т.д.
        на основе обучающего набора данных.
        """
        try:
            eur_df = pd.read_csv('auxiliary files/AEXUSEU.csv')
            eur_cs = eur_df["observation_date;AEXUSEU"].apply(lambda x: float(x.split(";")[1]))
            eur_date = eur_df["observation_date;AEXUSEU"].apply(lambda x: x.split(";")[0])
            eur_df["cs"] = eur_cs
            eur_df["date"] = eur_date
            eur_df = eur_df.drop(["observation_date;AEXUSEU"], axis=1)
            eur_df["date"] = eur_df["date"].apply(lambda x: int(x.split(".")[-1]))
            eur_df.index = eur_df["date"]
            eur_df = eur_df.drop("date", axis=1)

            self.eur_df = eur_df.copy()
        except FileNotFoundError:
            print("Warning: AEXUSEU.csv not found. Euro conversion will be skipped.")
            self.eur_df = pd.DataFrame(columns=['cs']).rename_axis('date')

        df_for_learning = df.copy()
        df_for_learning['MYrating'] = y

        parsed_genres = df_for_learning["genres"].apply(
            lambda v: [v0.get("name") for v0 in v] if isinstance(v, list) else [])
        parsed_countries = df_for_learning["countries"].apply(
            lambda v: [v0.get("name") for v0 in v] if isinstance(v, list) else [])
        self.all_genres = sorted(set(val for lst in parsed_genres if isinstance(lst, list) for val in lst))
        self.all_countries = sorted(set(val for lst in parsed_countries if isinstance(lst, list) for val in lst))

        actors_for_loo = df_for_learning["persons"].apply(
            lambda v: [i.get("enName") for i in v if
                       isinstance(i, dict) and i.get("enProfession") == "actor"] if isinstance(v, list) else [])
        writers_for_loo = df_for_learning["persons"].apply(
            lambda v: [i.get("enName") for i in v if
                       isinstance(i, dict) and i.get("enProfession") == "writer"] if isinstance(v, list) else [])

        df_for_learning['actors_parsed'] = actors_for_loo
        df_for_learning['writers_parsed'] = writers_for_loo

        self.imputation_values['global_rating_mean'] = df_for_learning['MYrating'].mean()
        self.loo_means['actors'] = _calculate_loo_means(df_for_learning, 'actors_parsed', 'MYrating')
        self.loo_means['writers'] = _calculate_loo_means(df_for_learning, 'writers_parsed', 'MYrating')

        df_transformed = self.transform(df, is_fitting=True)

        numeric_cols = [
            'age', 'budget', 'genres_count', 'countries_count',
            'votes_kp', 'votes_imdb', 'votes_critics', 'votes_rus_critics',
            'fees_world', 'fees_usa', 'fees_russia', 'lists_count', 'movieLength',
            'rating_kp', 'rating_imdb', 'rating_critics', 'rating_rus_critics',
            'audience_count', 'rus_audience_count', 'spoken_languages_count',
            'actors', 'writers'
        ]

        existing_numeric_cols = [col for col in numeric_cols if col in df_transformed.columns]
        df_for_stats = df_transformed[existing_numeric_cols].copy()
        df_for_stats = df_for_stats.apply(pd.to_numeric, errors='coerce')

        for col in existing_numeric_cols:
            mean_val = df_for_stats[col].mean()
            self.imputation_values[f'{col}_mean'] = mean_val
            self.scalers[col] = _fit_scaler(df_for_stats[col].fillna(mean_val))

        one_hot_cols = [col for col in df_transformed.columns if col.startswith('country_') or col.startswith('genre_')]
        indicator_cols = [col for col in df_transformed.columns if
                          col.startswith('no_') or col in ['is_on_kp_hd', 'type_movie', 'top250']]
        cbe_cols = ["main_actor", "main_director", "main_writer", "ratingMpaa", "main_language"]
        candidate_feature_list = list(dict.fromkeys(numeric_cols + one_hot_cols + indicator_cols + cbe_cols))
        existing_candidate_features = [col for col in candidate_feature_list if col in df_transformed.columns]
        df_candidates = df_transformed[existing_candidate_features].copy()

        df_numeric_candidates = df_candidates.select_dtypes(include=np.number)
        corr_matrix = df_numeric_candidates.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        threshold = 0.95
        highly_correlated_pairs = []

        for column in upper.columns:
            correlated_features = upper[column][upper[column] > threshold]
            if not correlated_features.empty:
                for feature, correlation in correlated_features.items():
                    pair = tuple(sorted((column, feature)))
                    highly_correlated_pairs.append((pair[0], pair[1], correlation))

        multicollinear_cols = [column for column in upper.columns if any(upper[column] > 0.95)]
        const_cols = [col for col in df_candidates.columns if df_candidates[col].nunique(dropna=False) == 1]
        almost_const_cols = [col for col in df_candidates.columns if
                             not df_candidates[col].empty and df_candidates[col].value_counts(normalize=True).iloc[
                                 0] > 0.995]
        bin_cols = [col for col in df_candidates.columns if
                    df_candidates[col].dropna().nunique() == 2 and set(df_candidates[col].dropna().unique()) <= {0,
                                                                                                                 1}]
        rare_bin_cols = [col for col in bin_cols if
                         df_candidates[col].mean() < 0.01 or df_candidates[col].mean() > 0.99]
        all_cols_to_drop = set(multicollinear_cols + const_cols + almost_const_cols + rare_bin_cols)
        self.columns_to_drop = list(all_cols_to_drop)
        final_feature_list = existing_numeric_cols

        one_hot_cols = [col for col in df_transformed.columns if col.startswith('country_') or col.startswith('genre_')]
        final_feature_list.extend(one_hot_cols)

        indicator_cols = [col for col in df_transformed.columns if
                          col.startswith('no_') or col in ['is_on_kp_hd', 'type_movie', 'top250']]
        final_feature_list.extend(indicator_cols)

        final_feature_list = list(dict.fromkeys(final_feature_list))
        final_feature_list.extend(cbe_cols)
        self.final_columns = [col for col in final_feature_list if col not in self.columns_to_drop]

        return self

    def transform(self, df: pd.DataFrame, is_fitting: bool = False) -> pd.DataFrame:
        """
        Применяет сохраненные трансформации к новым данным.
        """
        df = df.copy()
        df = df.copy()

        REQUIRED_RAW_COLUMNS = [
            'year', 'persons', 'budget', 'genres', 'countries', 'votes',
            'fees', 'rating', 'watchability', 'type', 'lists', 'movieLength',
            'top250', 'audience', 'spokenLanguages'
        ]
        for col in REQUIRED_RAW_COLUMNS:
            if col not in df.columns:
                print(
                    f"Warning: Required column '{col}' not found in input data. Creating it with default null values.")
                df[col] = None

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

        df['ageRating'] = df.apply(
            lambda r: self.mpaa_to_age.get(str(r['ratingMpaa']).lower()) if pd.isna(r['ageRating']) and pd.notna(
                r['ratingMpaa']) and str(r['ratingMpaa']).lower() in self.mpaa_to_age else r['ageRating'],
            axis=1)

        df['ratingMpaa'] = df.apply(
            lambda r: age_to_mpaa(r['ageRating']) if pd.isna(r['ratingMpaa']) or r['ratingMpaa'] == '' else r[
                'ratingMpaa'],
            axis=1)
        df["no_ageRating"] = df["ageRating"].isna().astype(int)

        # Year -> Age
        df["age"] = 2025 - df["year"]
        df["age"] = np.log1p(df["age"])

        # Persons
        df["directors"] = df["persons"].apply(lambda v: [i.get("enName") for i in v if isinstance(i, dict) and
                                                         i.get("enProfession")
                                                         == "director"] if isinstance(v, list) else None)
        df["actors"] = df["persons"].apply(lambda v: [i.get("enName") for i in v if isinstance(i, dict) and i.get(
            "enProfession") == "actor"] if isinstance(v, list) else [])
        df["writers"] = df["persons"].apply(lambda v: [i.get("enName") for i in v if isinstance(i, dict) and i.get(
            "enProfession") == "writer"] if isinstance(v, list) else [])
        df["main_writer"] = df["writers"].apply(lambda v: v[0] if isinstance(v, list) and len(v) >= 1 else None)
        df["main_director"] = df["directors"].apply(lambda v: v[0] if isinstance(v, list) and len(v) >= 1 else None)
        df["main_actor"] = df["actors"].apply(lambda v: v[0] if isinstance(v, list) and len(v) >= 1 else None)
        global_mean = self.imputation_values.get('global_rating_mean', 7.0)
        df['actors'] = _apply_loo_means(df, 'actors', self.loo_means.get('actors', {}), global_mean)
        df['writers'] = _apply_loo_means(df, 'writers', self.loo_means.get('writers', {}), global_mean)

        # Budget

        df["budget"] = df["budget"].astype(object)
        df = adjust_for_euro(df, "budget", self.eur_df)
        df = get_money_value(df, "budget")
        df = adjust_for_inflation_df(df, value_col='budget', year_col='year', new_col='budget')
        df["no_budget"] = df["budget"].isna().astype(int)
        df[f"budget"] = pd.to_numeric(df[f"budget"], errors='coerce')
        df["budget"] = np.log1p(df["budget"])
        mean_val = self.imputation_values.get(f"budget_mean", 0)
        df[f"budget"] = df[f"budget"].fillna(mean_val)
        # Genres & Countries

        df["genres_count"] = np.log1p(df["genres"].apply(lambda x: len(x)))
        df["countries_count"] = np.log1p(df["countries"].apply(lambda x: len(x)))
        df["genres"] = df["genres"].apply(lambda v: [v0.get("name") for v0 in v] if isinstance(v, list) else [])
        df["countries"] = df["countries"].apply(lambda v: [v0.get("name") for v0 in v] if isinstance(v, list) else [])
        # print(df["genres"])
        # print(df["countries"])
        # Weighted Multi-Hot Encode для Genres & Countries
        for country in self.all_countries:
            df[f'country_{country}'] = 0.0
        for genre in self.all_genres:
            df[f'genre_{genre}'] = 0.0

        for idx, row in df.iterrows():
            # Countries
            weights = [1 / (i + 1) for i in range(len(row['countries']))]
            total_weight = sum(weights)
            if total_weight > 0:
                for country, weight in zip(row['countries'], weights):
                    if country in self.all_countries:
                        df.at[idx, f'country_{country}'] = weight / total_weight
            # Genres
            weights = [1 / (i + 1) for i in range(len(row['genres']))]
            total_weight = sum(weights)
            if total_weight > 0:
                for genre, weight in zip(row['genres'], weights):
                    if genre in self.all_genres:
                        df.at[idx, f'genre_{genre}'] = weight / total_weight

        # Votes
        for res in ["kp", "imdb", "filmCritics", "russianFilmCritics"]:
            df[f'votes_{res}'] = np.log1p(df["votes"].apply(lambda r: r.get(res, 0) if isinstance(r, dict) else 0))
        # Fees
        # Fees
        df["fees"] = df["fees"].fillna({})  # Защита от случая, когда вся колонка None
        for loc in ["world", "usa", "russia"]:
            df[f"fees_{loc}"] = df["fees"].apply(lambda x: x.get(loc) if isinstance(x, dict) and loc in x else None)

            df[f"fees_{loc}"] = df[f"fees_{loc}"].apply(
                lambda v: v.get("value") if isinstance(v, dict) and v.get("currency") in ['$', '€'] and v.get(
                    "value") != 0 else None
            )
            df[f"no_fees_{loc}"] = df[f"fees_{loc}"].isna().astype(int)
            if loc == "world":
                df = adjust_for_euro(df, "fees_world", self.eur_df)
            df[f"fees_{loc}"] = pd.to_numeric(df[f"fees_{loc}"], errors='coerce')
            df[f"fees_{loc}"] = np.log1p(df[f"fees_{loc}"])
            mean_val = self.imputation_values.get(f"fees_{loc}_mean", 0)
            df[f"fees_{loc}"] = df[f"fees_{loc}"].fillna(mean_val)

        # Ratings
        for res in ["kp", "imdb", "filmCritics", "russianFilmCritics"]:
            df[f'rating_{res}'] = (df["rating"].apply(lambda r: r.get(res) if isinstance(r, dict) else None).replace
                                   (0, np.nan))
            df = df.copy()
            df[f"no_rating_{res}"] = df[f'rating_{res}'].isna().astype(int)

        # Other features
        df["is_on_kp_hd"] = df["watchability"].apply(lambda w: int(isinstance(w, dict) and any(
            i.get("name") == "Kinopoisk HD" for i in w.get("items", []) if isinstance(i, dict)))).copy()
        df["type_movie"] = df["type"].apply(lambda v: int(v == "movie")).copy()
        # print(df["type_movie"].describe())
        df["lists_count"] = np.log1p(df["lists"].apply(lambda ls: len(ls) if isinstance(ls, list) else 0)).copy()
        df["movieLength"] = df["movieLength"].fillna(
            self.imputation_values.get('movieLength_mean', df['movieLength'].mean()))
        df["top250"] = df["top250"].notna().astype(int)

        # Audience
        audience_features = df["audience"].apply(
            lambda aud_list: pd.Series({
                'audience_count': sum(a.get("count", 0) for a in aud_list if isinstance(a, dict)),
                'rus_audience_count': sum(
                    a.get("count", 0) for a in aud_list if isinstance(a, dict) and a.get("country") == "Россия")
            }) if isinstance(aud_list, list) else pd.Series({'audience_count': 0, 'rus_audience_count': 0})
        )
        df = pd.concat([df, audience_features], axis=1)
        df['audience_count'] = np.log1p(df['audience_count'])
        df['rus_audience_count'] = np.log1p(df['rus_audience_count'])

        # Spoken Languages
        df["spoken_languages_count"] = np.log1p(
            df["spokenLanguages"].apply(lambda v: len(v) if isinstance(v, list) else 0))

        # ==================== Imputation & Scaling ====================
        for col, scaler in self.scalers.items():
            if col in df.columns:
                mean_val = self.imputation_values.get(f'{col}_mean', df[col].mean())
                df[col] = _transform_scaler(df[col].fillna(mean_val), scaler)
            else:
                print(f"Warning: Column '{col}' expected by scaler not found in dataframe. Skipping.")

        # Если это `transform` после `fit`, то выходим, т.к. колонки еще не нужно удалять/выравнивать
        if is_fitting:
            return df

        # ==================== Final Column Management ====================
        df = df.drop(
            ['alternativeName', 'externalId', 'poster', 'keywordsParsed', 'images', 'backdrop', 'enName', 'color',
             'studioParsed', 'createDate', 'logo', 'premiere', 'shortDescription', 'updatedAt', 'createdAt',
             'releaseYears', 'ticketsOnSale', 'userRatingsParsed', 'isTmdbChecked', 'videos', 'facts', 'seasonsInfo',
             'typeNumber', 'description', 'names', 'imagesInfo', 'isSeries', 'slogan', 'year', 'technology',
             'directors', 'countries', 'totalSeriesLength', 'seriesLength', 'deletedAt', 'top10',
             'productionCompanies', 'similarMovies', 'sequelsAndPrequels', 'status', 'watchabilityParsed', 'type',
             'votes', 'distributors', 'genres', 'persons', 'networks', 'watchability', 'subType', 'lists',
             'collections', 'fees', 'rating', 'audience', 'spokenLanguages', 'audience', "name"],
            axis=1, errors='ignore')
        # Удаление ненужных и мультиколлинеарных колонок
        df = df.drop(columns=self.columns_to_drop, errors='ignore')

        # Выравнивание колонок по образцу из `fit`
        # 1. Добавить недостающие колонки (например, какой-то жанр отсутствовал в X_new)
        missing_cols = set(self.final_columns) - set(df.columns)
        for c in missing_cols:
            df[c] = 0

        # 2. Удалить лишние колонки (появляются из-за обработки сырых полей, которые потом не нужны)
        extra_cols = set(df.columns) - set(self.final_columns)
        df = df.drop(columns=list(extra_cols), errors='ignore')

        # 3. Установить тот же порядок колонок, что и в обучающем наборе
        df = df[self.final_columns]

        return df
