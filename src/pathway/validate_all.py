from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from .common import ROOT, load_config, load_processed, read_csv_auto, sha256_file, write_json


def check(rows: list[dict], name: str, condition: bool, detail: str) -> None:
    rows.append({"check": name, "status": "PASS" if condition else "FAIL", "detail": detail})


def main() -> None:
    cfg, data = load_config(), load_processed(); rows=[]
    check(rows,"processed shapes",data["gene"].shape==(565,29045) and data["tfa"].shape==(565,1333) and data["mirna"].shape==(565,414) and data["l1000"].shape==(565,977),str({k:data[k].shape for k in ["gene","tfa","mirna","l1000"]}))
    source_manifest=pd.read_csv(ROOT/"manifests/data/source_files.csv");matches=[]
    for _,r in source_manifest.iterrows(): matches.append(sha256_file(ROOT/"data/source"/r.file)==r.sha256)
    check(rows,"source hashes",all(matches),f"{sum(matches)}/{len(matches)} unchanged")
    splits=pd.read_csv(ROOT/"manifests/splits/frozen_20_splits.csv");sizes=splits.groupby(["split_seed","partition"]).size().unstack();check(rows,"20 frozen disjoint splits",len(sizes)==20 and (sizes.train==395).all() and (sizes.validation==85).all() and (sizes.test==85).all(),str(sizes.drop_duplicates().to_dict("records")))
    check(rows,"split sample uniqueness",splits.groupby("split_seed").sample_id.nunique().eq(565).all(),"565 unique IDs per seed")

    result=ROOT/"paper_exp/figure4_benchmark/results"; mainm=pd.read_csv(result/"main_test_metrics.csv");learn=mainm[~mainm.model.str.startswith("PROGENy")]
    counts=learn.groupby(["model","feature_space","metric"]).size();check(rows,"400 formal learnable runs",len(counts)==5*4*2 and counts.eq(20).all(),f"{len(counts)} model-feature-metric cells; all n=20")
    check(rows,"PROGENy fair baseline",len(mainm[mainm.model=="PROGENy (direction_agnostic_absmax)"])==40,"20 splits × 2 primary metrics; Gene only")
    prediction_files={x:result/f"{x}_predictions.csv.gz" for x in ["rf","svm","mlp","kan","resnet"]}; max_error=0.0;pred_ok=True
    for name,path in prediction_files.items():
        pred=read_csv_auto(path);test=pred[pred.partition=="test"]
        rec=[]
        for (model,space,seed),sub in test.groupby(["model","feature_space","split_seed"]): rec.extend([(model,space,seed,"accuracy",accuracy_score(sub.true_label,sub.pred_label)),(model,space,seed,"macro_f1",f1_score(sub.true_label,sub.pred_label,average="macro",zero_division=0))])
        rec=pd.DataFrame(rec,columns=["model","feature_space","split_seed","metric","recomputed"]);rec.model=rec.model.replace({"RESNET":"ResNet"});stored=mainm.merge(rec,on=["model","feature_space","split_seed","metric"]);err=float(np.max(np.abs(stored.value-stored.recomputed)));max_error=max(max_error,err);pred_ok &= len(test)==20*4*85 and err<1e-12
    check(rows,"prediction-level metric reproduction",pred_ok,f"maximum absolute error={max_error:.3g}")

    resnet_manifests=list((ROOT/"paper_exp/figure4_benchmark/runs/resnet_formal").glob("*/run_manifest.json"));resnet_checks=[]
    for path in resnet_manifests:
        m=json.loads(path.read_text());resnet_checks.append(m.get("checkpoint_reloaded_before_test") and not m.get("test_used_for_tuning") and (path.parent/"best_checkpoint.pt").exists() and (path.parent/"learning_curve.csv").exists())
    check(rows,"ResNet checkpoints and curves",len(resnet_manifests)==80 and all(resnet_checks),f"{sum(resnet_checks)}/{len(resnet_checks)} complete")
    classic_manifests=list((ROOT/"paper_exp/figure4_benchmark/runs").glob("rf__*/run_manifest.json"))+list((ROOT/"paper_exp/figure4_benchmark/runs").glob("svm__*/run_manifest.json"))
    check(rows,"RF/SVM run manifests",len(classic_manifests)==160,f"{len(classic_manifests)}/160 complete")
    deep_dirs=list((ROOT/"paper_exp/figure4_benchmark/runs").glob("mlp__*"))+list((ROOT/"paper_exp/figure4_benchmark/runs").glob("kan__*"))
    deep_ok=[all((d/x).exists() for x in ["run_manifest.json","best_checkpoint.pt","learning_curve.csv"]) for d in deep_dirs]
    check(rows,"MLP/KAN checkpoints and curves",len(deep_dirs)==160 and all(deep_ok),f"{sum(deep_ok)}/{len(deep_ok)} complete")
    tuning=list((ROOT/"paper_exp/figure4_benchmark/runs/resnet_tuning").glob("round_*/manifest.json"));check(rows,"validation-only ResNet tuning",len(tuning)==10 and all(not json.loads(x.read_text()).get("test_loaded_into_training_call",True) for x in tuning),"10 rounds; test never passed")

    emb=ROOT/"additional_exp/5.1_embedding_controls/results";embm=pd.read_csv(emb/"embedding_test_metrics_all_rp_seeds.csv");ec=embm.groupby(["model","feature_space","metric"]).size();check(rows,"5.1 controls complete",ec.eq(20).all() and set(embm.model)=={"SVM","ResNet"},f"{len(ec)} cells; all n=20")
    lin=json.loads((ROOT/"additional_exp/5.1_embedding_controls/artifacts/linear_control_manifest.json").read_text());ae=json.loads((ROOT/"additional_exp/5.1_embedding_controls/artifacts/autoencoder_manifest.json").read_text());check(rows,"5.1 external-only fitting",lin["external_scaler_only"] and ae["external_reference_only"],f"PCA/RP reference n={lin['external_samples']}; AE train n={ae['training_samples']}")
    emb_run_dirs=[d for d in (ROOT/"additional_exp/5.1_embedding_controls/runs/resnet").iterdir() if d.is_dir()]
    emb_artifacts=[all((d/x).exists() for x in ["run_manifest.json","best_checkpoint.pt","learning_curve.csv","predictions.csv.gz"]) for d in emb_run_dirs]
    check(rows,"5.1 ResNet artifacts",len(emb_run_dirs)==140 and all(emb_artifacts),f"{sum(emb_artifacts)}/{len(emb_artifacts)} complete")
    cka=pd.read_csv(ROOT/"additional_exp/5.2_complementarity/results/cka_heldout.csv");cca=pd.read_csv(ROOT/"additional_exp/5.2_complementarity/results/cca_heldout.csv");modes=pd.read_csv(ROOT/"additional_exp/5.2_complementarity/results/cca_modes_heldout.csv");check(rows,"5.2 CKA/CCA complete",len(cka)==640 and len(cca)==320 and len(modes)==3200,f"CKA={len(cka)}, CCA={len(cca)}, modes={len(modes)}")
    original=data["mirna"];errors=[]
    for seed in cfg["split_seeds"]:
        d=np.load(ROOT/f"additional_exp/5.2_complementarity/artifacts/decomposition/split_{seed}.npz");errors.append(float(np.max(np.abs(d['predicted']+d['residual']-original))))
    check(rows,"predicted + residual reconstruction",max(errors)<1e-6,f"maximum float32 error={max(errors):.9g}")
    decomp=pd.read_csv(ROOT/"additional_exp/5.2_complementarity/results/decomposition_test_metrics_all.csv");dc=decomp.groupby(["model","feature_space","metric"]).size();check(rows,"5.2 downstream decomposition complete",dc.eq(20).all() and len(dc)==2*4*2,f"{len(dc)} cells; all n=20")
    decomp_dirs=[d for d in (ROOT/"additional_exp/5.2_complementarity/runs/resnet_decomposition").iterdir() if d.is_dir()]
    decomp_artifacts=[all((d/x).exists() for x in ["run_manifest.json","best_checkpoint.pt","learning_curve.csv","predictions.csv.gz"]) for d in decomp_dirs]
    check(rows,"5.2 ResNet artifacts",len(decomp_dirs)==40 and all(decomp_artifacts),f"{sum(decomp_artifacts)}/{len(decomp_artifacts)} complete")
    rescue=read_csv_auto(ROOT/"paper_exp/figure5_rescue/results/paired_rescue_states.csv.gz");check(rows,"5.4 paired rescue states",len(rescue)==3400 and rescue.groupby(["model","split_seed"]).size().eq(85).all(),"2 models × 20 splits × 85 test samples")
    expected_figs=["figure4_aligned_pathway_benchmark","figure5_aligned_rescue_analysis","figureS_embedding_controls","figureS_feature_complementarity","figureS_resnet_tuning"];fig_ok=all((ROOT/"figures"/f"{x}.{ext}").exists() for x in expected_figs for ext in ["png","pdf","svg","tiff"]);check(rows,"figure exports",fig_ok,"5 figures × PNG/PDF/SVG/TIFF")
    frame=pd.DataFrame(rows);frame.to_csv(ROOT/"validation/final_validation.csv",index=False);status="PASS" if (frame.status=="PASS").all() else "FAIL";write_json(ROOT/"validation/final_validation.json",{"status":status,"checks":rows});md="# Final validation\n\n"+"\n".join(f"- **{r['status']}** — {r['check']}: {r['detail']}" for r in rows)+"\n";(ROOT/"validation/final_validation.md").write_text(md,encoding="utf-8");print(frame.to_string(index=False));print("FINAL",status)
    if status!="PASS":raise SystemExit(1)


if __name__=="__main__":main()
