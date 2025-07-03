from pathlib import Path
import os

import pandas as pd

from preprocessor import DataPreprocessor

df = pd.read_json('raw_movie_files/44083.json')
X = df.drop(columns=['MYrating'])
y = df['MYrating']
preprocessor = DataPreprocessor()
# print(X.describe())
preprocessor.fit(X, y)
X = preprocessor.transform(X)
print(X.describe())