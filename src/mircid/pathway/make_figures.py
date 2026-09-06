from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image

from .common import ROOT, read_csv_auto


FIG4 = ROOT / "paper_exp" / "figure4_benchmark"
FIG5 = ROOT / "paper_exp" / "figure5_rescue"
EMBED = ROOT / "additional_exp" / "5.1_embedding_controls"
COMP = ROOT / "additional_exp" / "5.2_complementarity"
OUT = ROOT / "figures"

COLORS = {"Gene": "#59A14F", "Gene + TFA": "#F28E2B", "Gene + HubmiR": "#4E79A7", "All features": "#A7A9AC"}
STATE_COLORS = {"rescued": "#4C9F70", "harmed": "#D95F59", "both_correct": "#A8D5A2", "both_wrong": "#8CB6D9", "not_tested": "#E5E7EB"}


def style() -> None:
    mpl.rcParams.update({"font.family": "Arial", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "font.size": 6.5, "axes.titlesize": 7.5, "axes.labelsize": 6.5,
                         "xtick.labelsize": 5.5, "ytick.labelsize": 5.5, "legend.fontsize": 5.5,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none"})


def save_all(fig: plt.Figure, stem: Path, dpi: int = 450) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def draw_boxes(ax: plt.Axes, frame: pd.DataFrame, metric: str) -> None:
    models = ["RF", "SVM", "MLP", "KAN", "ResNet", "PROGENy"]
    features = list(COLORS)
    offsets = {f: x for f, x in zip(features, [-0.27, -0.09, 0.09, 0.27])}
    for mi, model in enumerate(models):
        actual = "PROGENy (direction_agnostic_absmax)" if model == "PROGENy" else model
        for feature in features:
            if model == "PROGENy" and feature != "Gene": continue
            vals = frame[(frame.model == actual) & (frame.feature_space == feature) & (frame.metric == metric)].sort_values("split_seed").value.to_numpy()
            if len(vals) == 0: continue
            pos = mi + offsets[feature]
            bp = ax.boxplot(vals, positions=[pos], widths=0.15, patch_artist=True, showfliers=False,
                            medianprops={"color": "#222222", "linewidth": 1.1},
                            whiskerprops={"color": "#555555", "linewidth": 0.8},
                            capprops={"color": "#555555", "linewidth": 0.8})
            bp["boxes"][0].set(facecolor=COLORS[feature], edgecolor="#444444", alpha=0.82, linewidth=0.7)
            jitter = np.linspace(-0.025, 0.025, len(vals))
            ax.scatter(np.full(len(vals), pos) + jitter, vals, s=7, facecolor="white", edgecolor=COLORS[feature], linewidth=0.5, alpha=0.9, zorder=3)
    ax.set_xticks(range(len(models)), models, rotation=22, ha="right", rotation_mode="anchor")
    ax.set_ylim(0.25, 0.90); ax.set_xlim(-0.55, len(models) - 0.45); ax.grid(axis="y", color="#D9DDE3", linewidth=0.5, alpha=0.8)
    ax.set_title("Accuracy" if metric == "accuracy" else "Macro-F1", fontweight="bold", pad=6)


def figure4() -> None:
    data = pd.read_csv(FIG4 / "results" / "main_test_metrics.csv")
    submitted = np.asarray(Image.open(ROOT / "src" / "legacy_snapshot" / "figure4_submitted.png").convert("RGB"))
    crop = submitted[:, : int(submitted.shape[1] * 0.445)]
    fig = plt.figure(figsize=(7.2, 4.0), constrained_layout=False)
    fig.subplots_adjust(top=0.83, bottom=0.16)
    grid = fig.add_gridspec(1, 2, width_ratios=[0.44, 0.56], wspace=0.035)
    ax_left = fig.add_subplot(grid[0, 0]); ax_left.imshow(crop); ax_left.axis("off"); ax_left.text(0.01, 0.99, "a", transform=ax_left.transAxes, va="top", ha="left", fontsize=8, fontweight="bold")
    right = grid[0, 1].subgridspec(1, 2, wspace=0.15)
    ax_acc = fig.add_subplot(right[0, 0])
    ax_f1 = fig.add_subplot(right[0, 1], sharey=ax_acc)
    draw_boxes(ax_acc, data, "accuracy"); draw_boxes(ax_f1, data, "macro_f1")
    ax_acc.set_ylabel("Held-out test performance"); ax_f1.set_yticklabels([])
    ax_acc.text(-0.15, 1.04, "b", transform=ax_acc.transAxes, fontsize=8, fontweight="bold")
    fig.text(0.72, 0.985, "Pathway classification test-set performance", ha="center", va="top", fontsize=8, fontweight="bold")
    handles = [Patch(facecolor=COLORS[x], edgecolor="#555555", label=x) for x in COLORS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.72, 0.935), ncol=4, frameon=False)
    fig.text(0.985, 0.014, "n=20 paired splits per learnable configuration; PROGENy is a fixed Gene-only baseline", ha="right", fontsize=5, color="#555555")
    save_all(fig, OUT / "figure4_aligned_pathway_benchmark")


def figure5() -> None:
    summary = pd.read_csv(FIG5 / "results" / "pathway_reclassification_summary.csv")
    summary = summary[summary.model == "ResNet"].copy().sort_values("net_reclassification_rate")
    seeds = pd.read_csv(FIG5 / "results" / "seed_reclassification_summary.csv").query("model=='ResNet'")
    states = read_csv_auto(FIG5 / "results" / "paired_rescue_states.csv.gz").query("model=='ResNet'")
    cases = pd.read_csv(FIG5 / "results" / "locked_stable_cases.csv").query("model=='ResNet'")
    fig = plt.figure(figsize=(7.2, 4.4)); gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.25], height_ratios=[1, 1.1], wspace=0.30, hspace=0.42)
    ax = fig.add_subplot(gs[:, 0]); y=np.arange(len(summary)); ax.barh(y, summary.rescue_rate*100, color=STATE_COLORS["rescued"], label="Rescued"); ax.barh(y, -summary.harm_rate*100, color=STATE_COLORS["harmed"], label="Harmed"); ax.axvline(0,color="#333333",lw=0.8); ax.set_yticks(y,summary.true_pathway); ax.set_xlabel("Held-out observations (%)"); ax.set_title("a  Reclassification across all 11 pathways",loc="left",fontweight="bold"); ax.legend(frameon=False,ncol=2,loc="upper center",bbox_to_anchor=(0.5,-0.08)); ax.grid(axis="x",color="#E1E4E8",lw=0.5)
    ax2=fig.add_subplot(gs[0,1]); ax2.scatter(seeds.rescued,seeds.harmed,c=seeds.paired_accuracy_delta,cmap="RdYlGn",vmin=-0.1,vmax=0.1,s=34,edgecolor="white",linewidth=0.6); lim=max(seeds[["rescued","harmed"]].max())+1; ax2.plot([0,lim],[0,lim],ls="--",lw=0.8,color="#777777"); ax2.set(xlabel="Rescued test samples",ylabel="Harmed test samples",xlim=(0,lim),ylim=(0,lim)); ax2.set_title("b  Paired split-level balance",loc="left",fontweight="bold"); ax2.text(0.98,0.05,f"mean Δaccuracy = {seeds.paired_accuracy_delta.mean():+.3f}",transform=ax2.transAxes,ha="right",fontsize=7)
    ax3=fig.add_subplot(gs[1,1]); split_seeds=sorted(states.split_seed.unique()); labels=[]; matrix=[]
    code={"not_tested":0,"both_wrong":1,"harmed":2,"both_correct":3,"rescued":4}
    for _,case in cases.sort_values(["stable_harm","pathway","sample_id"]).iterrows():
        sub=states[states.sample_id==case.sample_id].set_index("split_seed").state.to_dict(); matrix.append([code[sub.get(seed,"not_tested")] for seed in split_seeds]); labels.append(f"{case.pathway} | {case.sample_id.split('.',1)[-1]}")
    if matrix:
        cmap=ListedColormap([STATE_COLORS[x] for x in ["not_tested","both_wrong","harmed","both_correct","rescued"]]); ax3.imshow(np.asarray(matrix),aspect="auto",interpolation="nearest",cmap=cmap,vmin=-0.5,vmax=4.5); ax3.set_yticks(range(len(labels)),labels); ax3.set_xticks(range(len(split_seeds)),split_seeds,rotation=90,rotation_mode="anchor"); ax3.set_xlabel("Frozen split seed");
    ax3.set_title("c  Locked recurrent cases",loc="left",fontweight="bold")
    handles=[Patch(facecolor=STATE_COLORS[x],label=x.replace("_"," ").title()) for x in ["rescued","harmed","both_correct","both_wrong","not_tested"]]; ax3.legend(handles=handles,ncol=3,frameon=False,loc="upper center",bbox_to_anchor=(0.5,-0.28),fontsize=6.5)
    fig.suptitle("Expanded rescue analysis on aligned HubmiR predictions",fontweight="bold",fontsize=8,y=0.995)
    save_all(fig, OUT / "figure5_aligned_rescue_analysis")


def embedding_figure() -> None:
    metrics=pd.read_csv(EMBED/"results"/"embedding_test_metrics_rp_collapsed.csv"); metrics=metrics[metrics.metric=="macro_f1"]
    contrast=pd.read_csv(EMBED/"results"/"embedding_paired_contrasts.csv"); contrast=contrast[contrast.metric=="macro_f1"]
    models=["SVM","ResNet"]; spaces=["Gene","Gene + PCA-414","Gene + AE-414","Gene + RP-414 (5-seed mean)","Gene + HubmiR"]
    palette={spaces[0]:"#59A14F",spaces[1]:"#B8A9D0",spaces[2]:"#9C9C9C",spaces[3]:"#C8C8C8",spaces[4]:"#4E79A7"}
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.1),gridspec_kw={"width_ratios":[1.25,1]},constrained_layout=True)
    ax=axes[0]; positions=[]; labels=[]; p=0
    for model in models:
        for space in spaces:
            vals=metrics[(metrics.model==model)&(metrics.feature_space==space)].sort_values("split_seed").value.to_numpy();
            bp=ax.boxplot(vals,positions=[p],widths=.55,patch_artist=True,showfliers=False,medianprops={"color":"#222"});bp["boxes"][0].set(facecolor=palette[space],edgecolor="#555",alpha=.85);ax.scatter(np.full(len(vals),p)+np.linspace(-.06,.06,len(vals)),vals,s=7,color="#333",alpha=.5);positions.append(p);labels.append(space.replace("Gene + ",""));p+=1
        p+=0.7
    ax.set_xticks(positions,labels,rotation=42,ha="right",rotation_mode="anchor");ax.set_ylabel("Held-out macro-F1");ax.set_title("a  Absolute performance",loc="left",fontweight="bold");ax.grid(axis="y",color="#E1E4E8",lw=.5);ax.text(2,0.98,"SVM",transform=ax.get_xaxis_transform(),ha="center",va="top",fontweight="bold");ax.text(7.7,0.98,"ResNet",transform=ax.get_xaxis_transform(),ha="center",va="top",fontweight="bold")
    ax=axes[1]; order=["Gene + HubmiR − Gene","Gene + HubmiR − Gene + PCA-414","Gene + HubmiR − Gene + AE-414","Gene + HubmiR − Gene + RP-414 (5-seed mean)"]; y=0
    for model,marker in [("SVM","o"),("ResNet","s")]:
        sub=contrast[(contrast.model==model)&contrast.contrast.isin(order)].set_index("contrast").reindex(order)
        ypos=np.arange(len(order))+(0.12 if model=="ResNet" else -0.12);ax.errorbar(sub.mean_delta,ypos,xerr=[sub.mean_delta-sub.ci95_low,sub.ci95_high-sub.mean_delta],fmt=marker,ms=4,capsize=2,label=model,color="#355C7D" if model=="ResNet" else "#777777")
    ax.axvline(0,color="#222",lw=.8);ax.set_yticks(range(len(order)),["vs Gene","vs PCA-414","vs AE-414","vs RP mean"]);ax.set_xlabel("Paired Δmacro-F1 (HubmiR − comparator)");ax.set_title("b  HubmiR-specific contrast",loc="left",fontweight="bold");ax.legend(frameon=False);ax.grid(axis="x",color="#E1E4E8",lw=.5)
    save_all(fig,OUT/"figureS_embedding_controls")


def complementarity_figure() -> None:
    cka=pd.read_csv(COMP/"results"/"cka_summary.csv"); cka=cka[cka.source_space=="Gene-29045"]
    modes=pd.read_csv(COMP/"results"/"cca_modes_heldout.csv"); modes=modes[modes.source_space=="Gene-29045"]
    ridge=pd.read_csv(COMP/"results"/"ridge_predictability_summary.csv"); decomp=pd.read_csv(COMP/"results"/"decomposition_paired_contrasts.csv");decomp=decomp[decomp.metric=="macro_f1"]
    fig,axes=plt.subplots(1,3,figsize=(7.2,2.55),constrained_layout=True)
    ax=axes[0]; targets=["HubmiR","PCA-414","AE-414","RP-414 mean"]; x=np.arange(len(targets));width=.34
    for i,(method,label,color) in enumerate([("debiased_linear","Linear CKA","#4E79A7"),("debiased_rbf","RBF CKA","#F28E2B")]):
        sub=cka[cka.method==method].set_index("target_representation").reindex(targets);ax.bar(x+(i-.5)*width,sub["mean"],width,color=color,label=label,yerr=[sub["mean"]-sub.ci95_low,sub.ci95_high-sub["mean"]],capsize=2)
    ax.set_xticks(x,targets,rotation=35,ha="right",rotation_mode="anchor");ax.set_ylabel("Held-out debiased CKA");ax.set_title("a  Representation geometry",loc="left",fontweight="bold");ax.legend(frameon=False);ax.grid(axis="y",color="#E1E4E8",lw=.5)
    ax=axes[1];
    for target,color,lw in [("HubmiR","#4E79A7",2.0),("PCA-414","#B8A9D0",1.0),("AE-414","#777777",1.0)]:
        sub=modes[modes.target_representation==target].groupby("mode").test_correlation.agg(["mean","sem"]); xvals=sub.index.to_numpy(dtype=float); means=sub["mean"].to_numpy(dtype=float); sems=sub["sem"].to_numpy(dtype=float); ax.plot(xvals,means,marker="o",ms=3,lw=lw,color=color,label=target);ax.fill_between(xvals,means-1.96*sems,means+1.96*sems,color=color,alpha=.12)
    rp=modes[modes.target_representation.str.startswith("RP-414")].groupby(["split_seed","mode"]).test_correlation.mean().groupby("mode").agg(["mean","sem"]);ax.plot(rp.index,rp["mean"],ls="--",color="#999999",label="RP mean");ax.set(xlabel="Canonical mode",ylabel="Held-out correlation",xticks=range(1,11));ax.set_title("b  Regularized CCA",loc="left",fontweight="bold");ax.legend(frameon=False,fontsize=6.5);ax.grid(color="#E1E4E8",lw=.5)
    ax=axes[2]; r2=ridge[ridge.metric=="variance_weighted_r2"].set_index("source_space"); labels=["Gene-29045","HubmiR-input-977"];vals=r2.loc[labels,"mean"];ax.bar([0,1],vals,color=["#59A14F","#7A9CC6"],width=.55,yerr=[vals-r2.loc[labels,"ci95_low"],r2.loc[labels,"ci95_high"]-vals],capsize=2);ax.set_xticks([0,1,2.35,3.05],["Full Gene","977 input","SVM\nresidual","ResNet\nresidual"],rotation=18,rotation_mode="anchor");ax.set_xlim(-.55,3.55);ax.set_ylabel("Held-out variance-weighted R²");ax.set_title("c  Linear predictability / task remainder",loc="left",fontweight="bold");ax.grid(axis="y",color="#E1E4E8",lw=.5);ax.axvline(1.7,color="#C7CBD1",lw=.8)
    ax2=ax.twinx(); primary=decomp[decomp.contrast=="Gene + residual HubmiR − Gene"].set_index("model").reindex(["SVM","ResNet"]);ax2.errorbar([2.35,3.05],primary.mean_delta,yerr=[primary.mean_delta-primary.ci95_low,primary.ci95_high-primary.mean_delta],fmt="D",color="#B34D69",capsize=2,ms=4,label="Residual ΔF1");ax2.axhline(0,color="#B34D69",lw=.6,ls=":");ax2.set_ylabel("Residual paired Δmacro-F1",color="#B34D69");ax2.tick_params(axis="y",colors="#B34D69")
    save_all(fig,OUT/"figureS_feature_complementarity")


def tuning_figure() -> None:
    root=FIG4/"runs"/"resnet_tuning";rows=[]
    for d in sorted(root.glob("round_*")):
        m=json.loads((d/"manifest.json").read_text());s=pd.read_csv(d/"summary.csv");rows.append({"round":m["round_id"],"validation_macro_f1":s.validation_macro_f1.mean(),"validation_sd":s.validation_macro_f1.std(),"train_macro_f1":s.train_macro_f1.mean()})
    d=pd.DataFrame(rows);fig,ax=plt.subplots(figsize=(3.5,2.4),constrained_layout=True);ax.errorbar(d["round"],d.validation_macro_f1,yerr=d.validation_sd,marker="o",capsize=2,label="Validation macro-F1",color="#4E79A7");ax.plot(d["round"],d.train_macro_f1,marker="s",label="Train macro-F1",color="#9C755F");ax.scatter([10],[d.loc[d['round']==10,'validation_macro_f1'].iloc[0]],s=45,facecolors="none",edgecolors="#2E7D32",lw=1.2,label="Frozen round");ax.set(xlabel="Sequential tuning round",ylabel="Mean across seeds 7/14/21",xticks=d["round"]);ax.grid(color="#E1E4E8",lw=.5);ax.legend(frameon=False);ax.set_title("ResNet validation-only tuning",fontweight="bold");save_all(fig,OUT/"figureS_resnet_tuning")


def main() -> None:
    style(); figure4(); figure5(); embedding_figure(); complementarity_figure(); tuning_figure(); print("PASS: all figure source scripts rendered PNG/PDF/SVG/TIFF")


if __name__=="__main__":main()
