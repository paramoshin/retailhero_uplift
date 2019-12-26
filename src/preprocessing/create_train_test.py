import pandas as pd
from sklearn.model_selection import train_test_split

from src.database.database_client import DatabaseClient
from src.feature_generation.db_connector import DBConnector

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

    features_names, train_features = db_connector.generate_purchase_features(client_ids=train_client_ids.tolist())
    train_df = pd.DataFrame(train_features, columns=features_names).set_index("client_id")

    train_df["avg_transaction_hour"] = train_df["avg_transaction_time"].apply(lambda x: x.hour)
    train_df["avg_transaction_minute"] = train_df["avg_transaction_time"].apply(lambda x: x.minute)
    train_df["avg_transaction_seconds"] = train_df["avg_transaction_time"].apply(lambda x: x.second)
    train_df.drop("avg_transaction_time", axis=1, inplace=True)

    print(train_df.shape)
    print(train_.shape)

    train_idxs, valid_idxs = train_test_split(train_client_ids, test_size=0.3, random_state=42)
    train_control_idxs = train_[(train_.index.isin(train_idxs)) & (train_is_treatment == 0)].index
    train_treatment_idxs = train_[(train_.index.isin(train_idxs)) & (train_is_treatment == 1)].index
    valid_control_idxs = train_[(train_.index.isin(valid_idxs)) & (train_is_treatment == 0)].index
    valid_treatment_idxs = train_[(train_.index.isin(valid_idxs)) & (train_is_treatment == 1)].index

    X_control_train, y_control_train = train_df.loc[train_control_idxs], train_target.loc[train_control_idxs]
    X_treatment_train, y_treatment_train = train_df.loc[train_treatment_idxs], train_target.loc[train_treatment_idxs]
    X_control_valid, y_control_valid = train_df.loc[valid_control_idxs], train_target.loc[valid_control_idxs]
    X_treatment_valid, y_treatment_valid = train_df.loc[valid_treatment_idxs], train_target.loc[valid_treatment_idxs]

    assert X_control_train.shape[0] == y_control_train.shape[0]
    assert X_treatment_train.shape[0] == y_treatment_train.shape[0]
    assert X_control_valid.shape[0] == y_control_valid.shape[0]
    assert X_treatment_valid.shape[0] == y_treatment_valid.shape[0]
    assert (
        X_control_train.shape[1] ==
        X_treatment_train.shape[1] ==
        X_control_valid.shape[1] ==
        X_treatment_valid.shape[1]
    )

    X_control_train.to_csv("../../data/processed/two_models/X_control_train.csv")
    y_control_train.to_csv("../../data/processed/two_models/y_control_train.csv")
    X_treatment_train.to_csv("../../data/processed/two_models/X_treatment_train.csv")
    y_treatment_train.to_csv("../../data/processed/two_models/y_treatment_train.csv")
    X_control_valid.to_csv("../../data/processed/two_models/X_control_valid.csv")
    y_control_valid.to_csv("../../data/processed/two_models/y_control_valid.csv")
    X_treatment_valid.to_csv("../../data/processed/two_models/X_treatment_valid.csv")
    y_treatment_valid.to_csv("../../data/processed/two_models/y_treatment_valid.csv")

    del train_features
    del train_df

    features_names, test_features = db_connector.generate_purchase_features(client_ids=test_client_ids.tolist())
    test_df = pd.DataFrame(test_features, columns=features_names).set_index("client_id")

    test_df["avg_transaction_hour"] = test_df["avg_transaction_time"].apply(lambda x: x.hour)
    test_df["avg_transaction_minute"] = test_df["avg_transaction_time"].apply(lambda x: x.minute)
    test_df["avg_transaction_seconds"] = test_df["avg_transaction_time"].apply(lambda x: x.second)
    test_df.drop("avg_transaction_time", axis=1, inplace=True)

    test_df.to_csv("../../data/processed/two_models/X_test.csv")
