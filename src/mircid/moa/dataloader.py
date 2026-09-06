import polars as pl
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

def load_gtex_subs(filepath="gtex_subs_pub.pkl"):
    import pickle
    with open(filepath, "rb") as f:
        gtex_subs = pickle.load(f)
    # print("Data loaded. ", {k: v.shape for k, v in gtex_subs.items()})
    return gtex_subs

def load_drug_pairs(filepath="pair_labels_selected.parquet"):
    drug_pairs = pl.read_parquet(filepath)
    return drug_pairs

# gtex_subs, gdi_l_subs, gdi_r_subs = load_gtex_subs()

def get_dataset(cell_id, drug_pairs, gtex_subs, seed=42):
    gtex_sub = gtex_subs[cell_id]\
        .sort("pert_iname")
    # gdi_l = gdi_l_subs[cell_id]
    # gdi_r = gdi_r_subs[cell_id]
    gtex_sub = gtex_sub.with_columns([
        pl.arange(0, gtex_sub.height).alias("dgi")
    ])
    dgi_l = gtex_sub.select([
        pl.col("pert_iname"),
        pl.col("dgi").alias("dgi_l")
    ]).unique("pert_iname", maintain_order=True, keep="first")
    dgi_r = gtex_sub.select([
        pl.col("pert_iname"),
        pl.col("dgi").alias("dgi_r")
    ]).unique("pert_iname", maintain_order=True, keep="last")

    dgi_l = dict(zip(dgi_l["pert_iname"], dgi_l["dgi_l"]))
    dgi_r = dict(zip(dgi_r["pert_iname"], dgi_r["dgi_r"]))

    drug_list = gtex_sub["pert_iname"].unique().to_list()
    
    drug_pairs = drug_pairs.filter(
        pl.col("drug_1").is_in(drug_list)
    ).filter(pl.col("drug_2").is_in(drug_list))

    gtex_sub = gtex_sub[:,:-4].to_numpy()

    # permute drug_pairs
    np.random.seed(seed)
    re_index = np.random.permutation(drug_pairs.height)
    drug_pairs = drug_pairs[re_index, :]

    def yield_drug_pair_data():
        for row in drug_pairs.iter_rows():
            drug1, drug2, label = row
            data1 = gtex_sub[dgi_l[drug1]:dgi_r[drug1]+1, :]
            data2 = gtex_sub[dgi_l[drug2]:dgi_r[drug2]+1, :]
            assert data1.shape[0] > 0, f"Drug {drug1} has no data!"
            assert data2.shape[0] > 0, f"Drug {drug2} has no data!"
            yield (data1, data2, label)
        
    return yield_drug_pair_data, len(drug_pairs)

if __name__ == "__main__":
    packed_subs = load_gtex_subs()
    drug_pairs = load_drug_pairs()
    head = 100
    cell_id = "A375"
    yield_drug_pair_data, height = get_dataset(cell_id, drug_pairs, packed_subs)
    for data1, data2, label in tqdm(yield_drug_pair_data(), total=height, smoothing=0.5):
        # print(f"Drug 1: {data1.shape}, Drug 2: {data2.shape}, Label: {label}")
        head -= 1
        # assert all float data in data1 and data2
        if head == 0:
            print(f"Drug 1: {data1.shape}, Drug 2: {data2.shape}, Label: {label}")
            for x in data1.flat:
                assert isinstance(x, float) or isinstance(x, np.float32), f"Data1 contains non-float value: {x}"
            for x in data2.flat:
                assert isinstance(x, float) or isinstance(x, np.float32), f"Data2 contains non-float value: {x}"
            break