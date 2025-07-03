import copy

import matplotlib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer

from pre_sor import persons, Pad
from pre_sor import DataTransformer
from pre_sor import budget

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 100)
pd.set_option('future.no_silent_downcasting', True)
df = pd.read_json("raw_movie_files/44083.json")
matplotlib.use('TkAgg')

df_budget = budget(copy.deepcopy(df))
df_persons = persons(copy.deepcopy(df))
numeric_cols = ["budget", "actors",
                "ageRating", "votes_kp", "votes_imdb", "age", "fees_world",
                "fees_usa", "fees_russia", "rating_kp", "rating_imdb", "rating_critics",
                "audience_count", "rus_audience_count", "spoken_languages_count", "main_language",
                "countries", "genres_count", "countries_count", "writers_count",
                "directors_count", ]
binary_cols = ["short_film", "no_budget", "no_ageRating", "is_on_kp_hd", "no_fees_world", "no_fees_usa",
               "no_fees_russia", "no_rating_kp", "no_rating_imdb", "no_rating_critics", "no_rating_rus_critics",
               "top250", "no_audience", "no_rus_audience", ]
pipeline = Pipeline(steps=[
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
X = df.drop(columns=["MYrating"])
y = df["MYrating"]
pipeline.fit(X, y)
X = pipeline.transform(X)
print(X.describe().loc[['count']])
# print(X.skew())
# print(X.dtypes)
# print(list(X.columns).count("genres"))
# print(list(X.columns).count("genres_count"))
# print(list(X.columns).count("countries_count"))

# print(X["spoken_languages_count"].describe())
# plt.hist(X["spoken_languages_count"])
# plt.show()
# print(X["main_language"].describe())
# plt.hist(X["main_language"])
# plt.show()
# print(X["rating_critics"].describe())
# plt.hist(X["rating_critics"])
# plt.show()
# print(X["rating_rus_critics"].describe())
for col in X.columns:
    print(col, sum(X[col].isna().astype(int)))