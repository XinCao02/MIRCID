from __future__ import annotations

import argparse
import itertools
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import multiprocessing as mp

import numpy as np
from tqdm import tqdm

from mircid.moa.dataloader import get_dataset, load_drug_pairs, load_gtex_subs
from mircid.moa.rcsm import GSEAweight1Score, KSScore, XCosScore, ZhangScore
from mircid.paths import DATA_ROOT, WORK_ROOT

ROOT = WORK_ROOT / "moa"

CALLERS = {
    "KS": KSScore,
    "GSEAweight1": GSEAweight1Score,
    "XCos": XCosScore,
    "Zhang": ZhangScore,
}

DEFAULT_TASKS = [
    "HCC515", "VCAP", "A549", "A375", "MCF7", "HA1E",
    "HCC515_miRNA", "HCC515_TF", "HCC515_tot", "VCAP_miRNA",
    "VCAP_TF", "VCAP_tot", "A549_miRNA", "A549_TF", "A549_tot",
    "A375_miRNA", "A375_TF", "A375_tot", "MCF7_miRNA", "MCF7_TF",
    "MCF7_tot", "HA1E_miRNA", "HA1E_TF", "HA1E_tot",
]


def now_for_print() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def test_drug_pair(data1, data2, label, func, top_n: int, permute_num: int):
    labels = []
    scores = []
    for profile1 in data1:
        for profile2 in data2:
            score = func(profile1, profile2, permute_num=permute_num, top_n=top_n)
            labels.append(label)
            scores.append(score)
    return scores, labels


def eval_pair(args):
    data1, data2, label, call, top_n, permute_num = args
    return test_drug_pair(data1, data2, label, call, top_n=top_n, permute_num=permute_num)


def calc_auc_cut(scores, labels, cutoff=0.01, normalized=True):
    from sklearn.metrics import roc_curve, auc

    if not scores:
        raise ValueError("scores is empty")
    if isinstance(scores[0], dict):
        scores = [score["score"] for score in scores]

    fpr, tpr, _ = roc_curve(labels, scores)
    mask = fpr <= cutoff
    fpr_trunc = np.concatenate(([0.0], fpr[mask]))
    tpr_trunc = np.concatenate(([0.0], tpr[mask]))
    auc_cut = auc(fpr_trunc, tpr_trunc)
    return auc_cut / cutoff if normalized else auc_cut


def result_writer(result_file: Path):
    result_file.parent.mkdir(parents=True, exist_ok=True)
    f = open(result_file, "w", encoding="utf-8")
    f.write("Cell_Line\tCall_Name\tAUC_Cutoff\tAUC_Value\n")

    def write_result(cell_id, auc_results, call_name):
        for auc_cut, auc_value in auc_results.items():
            f.write(f"{cell_id}\t{call_name}\t{auc_cut}\t{auc_value:.6f}\n")
        f.flush()

    return write_result


def run_parallel(tasks, head, drug_pairs, gtex_subs, auc_cuts, write_result, top_n, permute_num, max_workers=None):
    if max_workers is None:
        max_workers = os.cpu_count() or 4

    print(f"{now_for_print()} Starting parallel RCSM testing with {max_workers} workers")
    bar = tqdm(total=len(tasks) * len(CALLERS), desc="Overall Progress", position=0)

    for call_name, call_func in CALLERS.items():
        ctx = mp.get_context("spawn")
        for cell_id in tasks:
            yield_drug_pair_data, height = get_dataset(cell_id, drug_pairs, gtex_subs)
            n_to_run = min(head, height)
            pair_iter = itertools.islice(yield_drug_pair_data(), n_to_run)

            test_scores, test_labels = [], []
            args_iter = ((d1, d2, lbl, call_func, top_n, permute_num) for (d1, d2, lbl) in pair_iter)

            futures = []
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
                for args in args_iter:
                    futures.append(ex.submit(eval_pair, args))

                for fut in as_completed(futures):
                    scores, labels = fut.result()
                    test_scores.extend(scores)
                    test_labels.extend(labels)

            auc_results = {}
            for auc_cut in auc_cuts:
                auc_results[auc_cut] = calc_auc_cut(test_scores, test_labels, cutoff=auc_cut, normalized=True)

            write_result(cell_id, auc_results, call_name)
            bar.set_description(f"{now_for_print()} Finished: {cell_id} | {call_name}")
            bar.update(1)
    bar.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RCSM methods on drug-pair benchmark.")
    parser.add_argument("--gtex-subs", default=str(DATA_ROOT / "moa" / "demo" / "gtex_subs_demo.pkl"), help="Path to gtex_subs pickle file.")
    parser.add_argument("--drug-pairs", default=str(DATA_ROOT / "moa" / "demo" / "pair_labels_selected.parquet"), help="Path to drug-pair label parquet.")
    parser.add_argument("--output", default=str(ROOT / "results" / "rcsm_test_results.tsv"), help="Output TSV path.")
    parser.add_argument("--head", type=int, default=50, help="Maximum number of drug pairs per cell line.")
    parser.add_argument("--top-n", type=int, default=2000, help="Number of extreme genes used by RCSM methods.")
    parser.add_argument("--permute-num", type=int, default=10, help="Permutation count used inside each scorer.")
    parser.add_argument("--auc-cutoffs", type=float, nargs="+", default=[0.01], help="AUC cutoffs, e.g. 0.001 0.005 0.01")
    parser.add_argument("--max-workers", type=int, default=None, help="Parallel worker count. Default: os.cpu_count().")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS, help="Cell-line / feature-group task list.")
    return parser.parse_args()


def main():
    args = parse_args()
    gtex_subs = load_gtex_subs(args.gtex_subs)
    drug_pairs = load_drug_pairs(args.drug_pairs)
    write_result = result_writer(Path(args.output))
    run_parallel(
        tasks=args.tasks,
        head=args.head,
        drug_pairs=drug_pairs,
        gtex_subs=gtex_subs,
        auc_cuts=args.auc_cutoffs,
        write_result=write_result,
        top_n=args.top_n,
        permute_num=args.permute_num,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
