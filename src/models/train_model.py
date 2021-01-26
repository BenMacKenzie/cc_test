from catboost import CatBoostClassifier, Pool
import pandas as pd
from pathlib import Path


project_dir = Path(__file__).resolve().parents[2]

data_dir = f"{project_dir}/data/processed"
train_df = pd.read_csv(f"{data_dir}/train.csv")
test_df = pd.read_csv(f"{data_dir}/test.csv")


X_train = train_df.drop(['narrowing-diagnosis'], axis=1)
y_train = train_df['narrowing-diagnosis']

cat_features = ['sex', 'chest-pain-type', 'fasting-blood-sugar', 'resting-ecg', 'exercise-angina',
                'slope', 'colored-vessels', 'thal', 'datetime', 'postalcode']


train_data = Pool(data=X_train, cat_features=cat_features, label=y_train)

X_test = test_df.drop(['narrowing-diagnosis'], axis=1)
y_test = test_df['narrowing-diagnosis']

test_data = Pool(data=X_test, cat_features=cat_features, label=y_test)

model = CatBoostClassifier(iterations=1000, eval_metric="PRAUC", early_stopping_rounds=40)

model.fit(train_data, eval_set=test_data, verbose=False, plot=False)

model.save_model(f"{project_dir}/models/heart.cbm")
