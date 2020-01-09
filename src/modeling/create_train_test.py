import datetime

import pandas as pd
from sklearn.model_selection import train_test_split

from src.database.database_client import DatabaseClient
from src.feature_generation.db_connector import DBConnector


def encode_clients_features(df):
    age_q_1, age_q_99 = df["age"].quantile(.01), df["age"].quantile(.99)
    df["strange_age"] = (df["age"] < age_q_1) | (df["age"] > age_q_99)
    df["strange_age"] = df["strange_age"].astype(int)
    for v in sorted(df["gender"].unique()):
        df[f"gender_{v}"] = (df["gender"] == v)
        df[f"gender_{v}"] = df[f"gender_{v}"].astype(int)
    df.drop("gender", axis=1, inplace=True)
    df['first_redeem_date'] = pd.to_datetime(df['first_redeem_date'])
    df['first_issue_date'] = pd.to_datetime(df['first_issue_date'])
    df['redeem_issue_diff'] = (df['first_redeem_date'] - df['first_issue_date']).dt.total_seconds()
    df['first_issue_dayofyear'] = df['first_issue_date'].dt.dayofyear
    df['first_issue_hour'] = df['first_issue_date'].dt.hour
    df['first_issue_weekday'] = df['first_issue_date'].dt.weekday
    df['first_issue_dayofmonth'] = df['first_issue_date'].dt.day
    df['first_issue_year'] = df['first_issue_date'].dt.year
    df['first_issue_month'] = df['first_issue_date'].dt.month
    df['first_issue_weekofyear'] = df['first_issue_date'].dt.weekofyear
    df['first_issue_week'] = df['first_issue_date'].dt.week
    df['first_issue_quarter'] = df['first_issue_date'].dt.quarter
    df['first_issue_date'] = df['first_issue_date'].astype(int) / 10 ** 9
    df['first_redeem_date'] = df['first_redeem_date'].astype(int) / 10 ** 9
    df['diff'] = df['first_redeem_date'] - df['first_issue_date']
    df.drop(['first_redeem_date', 'first_issue_date'], axis=1, inplace=True)
    return df


if __name__ == "__main__":
    host = "localhost"
    port = 5434
    db_client = DatabaseClient("postgres", "postgres", host, port, "uplift")
    db_connector = DBConnector(db_client)

    train_ = pd.read_csv("../../data/raw/uplift_train.csv", index_col="client_id")
    train_client_ids = train_.index
    train_is_treatment = train_["treatment_flg"]
    train_target = train_["target"]

    test_ = pd.read_csv("../../data/raw/uplift_test.csv", index_col="client_id")
    test_client_ids = test_.index

    print("loading train data from database")
    features_names, train_features = db_connector.generate_purchase_features(
        client_ids=train_client_ids.tolist()
    )
    train_df = pd.DataFrame(train_features, columns=features_names).set_index("client_id")
    assert train_df.shape[0] == train_.shape[0]
    del train_features

    train_df = encode_clients_features(train_df)
    train_df["avg_transaction_hour"] = train_df["avg_transaction_datetime"].apply(lambda x: x.hour)
    avg_hour = int(train_df["last_month_avg_transaction_datetime"].dropna().apply(lambda x: x.hour).mean())
    train_df["last_month_avg_transaction_hour"] = (
        train_df["last_month_avg_transaction_datetime"].fillna(datetime.time(avg_hour, 0, 0)).apply(lambda x: x.hour)
    )
    train_df.drop(["avg_transaction_datetime", "last_month_avg_transaction_datetime"], axis=1, inplace=True)

    print("splitting dataframe")
    train_idxs, valid_idxs = train_test_split(train_client_ids, test_size=0.2, random_state=42)
    X_train = train_df.loc[train_idxs]
    X_valid = train_df.loc[valid_idxs]
    X_train_is_treatment = train_is_treatment.loc[train_idxs]
    X_valid_is_treatment = train_is_treatment.loc[valid_idxs]

    y_train = train_target.loc[train_idxs]
    y_valid = train_target.loc[valid_idxs]

    assert X_train.shape[0] == y_train.shape[0]
    assert X_valid.shape[0] == y_valid.shape[0]
    assert X_train.shape[1] == X_valid.shape[1]

    X_train.to_csv("../../data/processed/two_models/X_train.csv")
    y_train.to_csv("../../data/processed/two_models/y_train.csv", header=False)
    X_train_is_treatment.to_csv("../../data/processed/two_models/X_train_is_treatment.csv", header=False)
    X_valid_is_treatment.to_csv("../../data/processed/two_models/X_valid_is_treatment.csv", header=False)
    X_valid.to_csv("../../data/processed/two_models/X_valid.csv")
    y_valid.to_csv("../../data/processed/two_models/y_valid.csv", header=False)

    del train_df

    print("loading test data from database")
    features_names, test_features = db_connector.generate_purchase_features(
        client_ids=test_client_ids.tolist()
    )
    test_df = pd.DataFrame(test_features, columns=features_names).set_index("client_id")
    assert test_df.shape[0] == test_.shape[0]
    del test_features

    test_df = encode_clients_features(test_df)
    test_df["avg_transaction_hour"] = test_df["avg_transaction_datetime"].apply(lambda x: x.hour)
    test_df["last_month_avg_transaction_hour"] = (
        test_df["last_month_avg_transaction_datetime"].fillna(datetime.time(avg_hour, 0, 0)).apply(lambda x: x.hour)
    )
    test_df.drop(["avg_transaction_datetime", "last_month_avg_transaction_datetime"], axis=1, inplace=True)

    assert X_train.columns.tolist() == X_valid.columns.tolist() == test_df.columns.tolist()

    test_df.to_csv("../../data/processed/two_models/X_test.csv")
