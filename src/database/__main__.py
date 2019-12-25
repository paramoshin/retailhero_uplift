import pandas as pd
import numpy as np
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

    r = db_client.get_first(Purchases)
    if not r:
        df = pd.read_csv("../../data/raw/purchases.csv")
        df = df.where((pd.notnull(df)), None)
        rows = df.to_dict(orient='records')
        metadata = MetaData()
        metadata.reflect(f.engine, only=["purchases"])
        insert_query = Table("purchases", metadata).insert()
        f.engine.execute(insert_query, rows)
        del df
        del rows

    r = db_client.get_first(Clients)
    if not r:
        df = pd.read_csv("../../data/raw/clients.csv")
        df = df.where((pd.notnull(df)), None)
        rows = df.to_dict(orient='records')
        metadata = MetaData()
        metadata.reflect(f.engine, only=["clients"])
        insert_query = Table("clients", metadata).insert()
        f.engine.execute(insert_query, rows)
        del df
        del rows

    r = db_client.get_first(Products)
    if not r:
        df = pd.read_csv("../../data/raw/products.csv")
        df = df.drop([796, 12219, 17818], axis=0)
        df['segment_id'] = df['segment_id'].fillna(-1).astype(int)
        df = df.where((pd.notnull(df)), None)
        rows = df.to_dict(orient='records')
        metadata = MetaData()
        metadata.reflect(f.engine, only=["products"])
        insert_query = Table("products", metadata).insert()
        f.engine.execute(insert_query, rows)
        del df
        del rows

