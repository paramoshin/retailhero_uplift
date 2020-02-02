import argparse

import pandas as pd
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, default="level_2")
    args = parser.parse_args()

    purchases_client_ids = pd.read_csv("../../data/raw/purchases.csv", usecols=["client_id"])
    purchases_client_ids["count"] = 1
    
    purchases_client_ids = purchases_client_ids.groupby("client_id").sum()

    if args.category == "level_2":
        frames = []
        skip = 0
        i = 0
        for c_id, c in purchases_client_ids.iterrows():
            i += 1
            if i % 5000 == 0:
                print(f"{i} of {purchases_client_ids.shape[0]}")
            frame = pd.read_csv("../../notebooks/purchases_level_2.csv", skiprows=[i for i in range(1, skip + 1)], nrows=c[0])
            data = (
                frame.groupby("client_id")[
                    [x for x in frame.columns if x.startswith("level_")]
                ].sum()
                .reset_index()
                .groupby("client_id")
                .agg([sum, np.mean, np.std])
            )
            data.columns = ['_'.join(col) for col in data.columns]
            frames.append(data)
            skip += c[0]
        df = pd.concat(frames, ignore_index=False)
        df.to_csv("../data/processed/level_2.csv", index=True)
    
    elif args.category == "segment": 
        frames = []
        skip = 0
        i = 0
        for c_id, c in purchases_client_ids.iterrows():
            i += 1
            if i % 5000 == 0:
                print(f"{i} of {purchases_client_ids.shape[0]}")
            frame = pd.read_csv("../../notebooks/purchases_segment.csv", skiprows=[i for i in range(1, skip + 1)], nrows=c[0])
            data = (
                frame.groupby("client_id")[
                    [x for x in frame.columns if x.startswith("segment_id_")]
                ].sum()
                .reset_index()
                .groupby("client_id")
                .agg([sum, np.mean, np.std])
            )
            data.columns = ['_'.join(col) for col in data.columns]
            frames.append(data)
            skip += c[0]
        df = pd.concat(frames, ignore_index=False)
        df.to_csv("segment.csv", index=True)
