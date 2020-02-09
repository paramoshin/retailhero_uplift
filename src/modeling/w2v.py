import joblib
import pandas as pd
from gensim.models import Word2Vec


if __name__ == "__main__":
    clients = pd.read_csv("../../data/raw/clients.csv")
    purchases = pd.read_csv(
        "../../data/raw/purchases.csv",
        usecols=[
            "client_id",
            "transaction_id",
            "transaction_datetime",
            "store_id",
            "product_id",
            "product_quantity",
            "product_quantity",
        ],
    )
    products = pd.read_csv("../../data/raw/products.csv")

    purchases["products_list"] = purchases["product_id"].apply(lambda x: [x]) * purchases[
        "product_quantity"
    ].fillna(0).astype(int)
    data = purchases.groupby(["client_id", "transaction_id"])[["products_list"]].sum().reset_index()
    data = data.groupby(["client_id"])[["products_list"]].sum()

    model = Word2Vec(
        min_count=1,
        workers=6,
        window=5,
        sg=1,
        hs=0,
        negative=10,
        alpha=0.03,
        min_alpha=0.0007,
        seed=42,
    )
    model.build_vocab(data["products_list"].tolist(), progress_per=200)
    model.train(
        data["products_list"].tolist(), total_examples=model.corpus_count, epochs=10, report_delay=1
    )
    model.init_sims(replace=True)
    print(model)
    X = model[model.wv.vocab]
    print(X.shape)
    joblib.dump(model, "../../models/w2v.pkl")
    joblib.dump(X, "../../data/processed/w2v.npz")
