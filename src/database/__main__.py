import sys
from pathlib import Path
p = str(Path(".").resolve().parent.parent)
sys.path.extend([p])

import pandas as pd
from sqlalchemy import Table, MetaData

from src.database.base import SessionFactory
from src.database.database_client import DatabaseClient
from src.database.models import Clients, Products, Purchases


if __name__ == "__main__":

    host = "localhost"
    port = 5434

    f = SessionFactory(
        dialect="postgresql",
        user="postgres",
        password="postgres",
        host=host,
        port=port,
        database_name="uplift",
    )
    f.init_database(f.engine)
    db_client = DatabaseClient("postgres", "postgres", host, port, "uplift")

    r = db_client.get_first(Clients)
    n_rows = db_client.get_number_of_rows(Clients)
    df = pd.read_csv("../../data/raw/clients.csv")
    df = df.where((pd.notnull(df)), None)
    if not r or df.shape[0] != n_rows:
        print("loading clients")
        if r and df.shape[0] != n_rows:
            db_client.clean_table(Clients)
        rows = df.to_dict(orient='records')
        metadata = MetaData()
        metadata.reflect(f.engine, only=["clients"])
        insert_query = Table("clients", metadata).insert()
        f.engine.execute(insert_query, rows)
        del df
        del rows

    r = db_client.get_first(Products)
    n_rows = db_client.get_number_of_rows(Products)
    df = pd.read_csv("../../data/raw/products.csv")
    df = df.drop([796, 12219, 17818], axis=0)
    df['segment_id'] = df['segment_id'].fillna(-1).astype(int)
    df = df.where((pd.notnull(df)), None)
    if not r or df.shape[0] != n_rows:
        print("loading products")
        if r and df.shape[0] != n_rows:
            db_client.clean_table(Products)
        rows = df.to_dict(orient='records')
        metadata = MetaData()
        metadata.reflect(f.engine, only=["products"])
        insert_query = Table("products", metadata).insert()
        f.engine.execute(insert_query, rows)
        del df
        del rows

    r = db_client.get_first(Purchases)
    n_rows = db_client.get_number_of_rows(Purchases)
    if not r or n_rows != 45780094:
        print("loading purchases")
        if r and n_rows != 45780094:
            db_client.clean_table(Purchases)
        metadata = MetaData()
        metadata.reflect(f.engine, only=["purchases"])
        chunksize = 10 ** 6
        for i, chunk in enumerate(pd.read_csv("../../data/raw/purchases.csv", chunksize=chunksize)):
            print(f"chunk no {i + 1}")
            chunk = chunk.replace({pd.np.nan: None})
            chunk = chunk[~chunk['product_id'].isin({"04d86b4b50", "48cc0e256d", "6a3d708544"})]
            rows = chunk.to_dict(orient='records')
            insert_query = Table("purchases", metadata).insert()
            f.engine.execute(insert_query, rows)
