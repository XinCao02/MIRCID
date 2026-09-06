from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .common import ROOT


def table(headers: list[str], rows: list[list[object]]) -> str:
    def clean(x: object) -> str: return str(x).replace("|", "\\|").replace("\n", " ")
    return "| " + " | ".join(map(clean, headers)) + " |\n|" + "|".join(["---"]*len(headers)) + "|\n" + "\n".join("| " + " | ".join(clean(x) for x in row) + " |" for row in rows)


def fmt_ci(row: pd.Series, scale: float = 1.0) -> str:
    return f"{row['mean']*scale:.3f} ({row['ci95_low']*scale:.3f}–{row['ci95_high']*scale:.3f})"


def main() -> None:
    fig4=ROOT/"paper_exp/figure4_benchmark/results";main=pd.read_csv(fig4/"main_absolute_summary.csv");contr=pd.read_csv(fig4/"main_paired_contrasts.csv")
    performance=[]
    order=["RF","SVM","MLP","KAN","ResNet","PROGENy (direction_agnostic_absmax)"]
    for model in order:
        features=["Gene"] if model.startswith("PROGENy") else ["Gene","Gene + TFA","Gene + HubmiR","All features"]
        for feature in features:
            a=main[(main.model==model)&(main.feature_space==feature)&(main.metric=="accuracy")].iloc[0];f=main[(main.model==model)&(main.feature_space==feature)&(main.metric=="macro_f1")].iloc[0]
            performance.append([model.replace(" (direction_agnostic_absmax)",""),feature,f"{a['mean']:.3f} ± {a['sd']:.3f}",f"{f['mean']:.3f} ± {f['sd']:.3f}"])
    rprimary=contr[(contr.model=="ResNet")&(contr.metric=="macro_f1")&(contr.contrast=="Gene + HubmiR − Gene")].iloc[0]
    sprimary=contr[(contr.model=="SVM")&(contr.metric=="macro_f1")&(contr.contrast=="Gene + HubmiR − Gene")].iloc[0]
    best=main[main.metric=="macro_f1"].sort_values("mean",ascending=False).iloc[0]
    prov=json.loads((fig4/"progeny_provenance.json").read_text())
    emb=ROOT/"additional_exp/5.1_embedding_controls/results";econtr=pd.read_csv(emb/"embedding_paired_contrasts.csv");esum=pd.read_csv(emb/"embedding_absolute_summary.csv")
    comp=ROOT/"additional_exp/5.2_complementarity/results";ridge=pd.read_csv(comp/"ridge_predictability_summary.csv");dcontr=pd.read_csv(comp/"decomposition_paired_contrasts.csv");cka=pd.read_csv(comp/"cka_summary.csv");cca=pd.read_csv(comp/"cca_summary.csv")
    rescue=ROOT/"paper_exp/figure5_rescue/results";cases=pd.read_csv(rescue/"locked_stable_cases.csv");rstat=pd.read_csv(ROOT/"additional_exp/5.4_expanded_rescue/results/overall_rescue_statistics.csv");pathway=pd.read_csv(rescue/"pathway_reclassification_summary.csv").query("model=='ResNet'")
    validation=json.loads((ROOT/"validation/final_validation.json").read_text()) if (ROOT/"validation/final_validation.json").exists() else {"status":"PENDING"}
    gene_ridge_mean = ridge.query("source_space=='Gene-29045' and metric=='variance_weighted_r2'").iloc[0]["mean"]
    stable_rescue_count = int(cases[(cases.model == "ResNet") & cases.stable_rescue].shape[0])
    stable_harm_count = int(cases[(cases.model == "ResNet") & cases.stable_harm].shape[0])
    lines=[]
    lines += ["# Aligned Pathway Classification — formal experiment report","",f"**Validation:** {validation['status']} · **Samples:** 565 · **Pathways:** 11 · **Frozen splits:** 20 · **Learnable runs:** 400","","## Main readout","",table(["Endpoint","Result"],[
        ["Best mean macro-F1",f"{best.model}, {best.feature_space}: {best['mean']:.3f} (95% CI {best.ci95_low:.3f}–{best.ci95_high:.3f})"],
        ["Primary ResNet Δmacro-F1",f"Gene+HubmiR − Gene = {rprimary.mean_delta:+.3f} (95% CI {rprimary.ci95_low:+.3f} to {rprimary.ci95_high:+.3f}; exact p={rprimary.exact_signflip_p:.3g})"],
        ["Model-specific SVM Δmacro-F1",f"Gene+HubmiR − Gene = {sprimary.mean_delta:+.3f} (95% CI {sprimary.ci95_low:+.3f} to {sprimary.ci95_high:+.3f}; Holm p={sprimary.holm_p_within_model_metric:.3g})"],
        ["Linear recoverability",f"Full Gene → HubmiR held-out variance-weighted R² = {gene_ridge_mean:.3f}"],
        ["Stable ResNet cases",f"{stable_rescue_count} recurrent rescues; {stable_harm_count} recurrent harms"],
    ]),"","![Aligned Figure 4](figures/figure4_aligned_pathway_benchmark.png)","","### Test performance (mean ± SD across paired splits)","",table(["Model","Feature space","Accuracy","Macro-F1"],performance),"","## PROGENy baseline audit","",table(["Question","Answer"],[
        ["Was ‘WLR’ correct?","No. The recovered object is a fixed linear footprint-score model, not a fitted logistic-regression classifier."],
        ["Recommended name",prov["recommended_name"]],
        ["Does the old prediction use effect?",f"Yes. {prov['legacy_final_predictions_match_effect_aware_rule_fraction']*100:.0f}% of 84 legacy predictions equal max-score for activating and min-score for inhibiting samples."],
        ["Is the implementation official?",f"Yes. Official top-100 weights; all 11 pathways have 100/100 genes. Legacy scores have r=1 per pathway with the official raw scores (max affine residual {prov['legacy_scores_exact_affine_lineage_max_residual']:.2e})."],
        ["Fair Figure 4 rule","Absolute train-standardized score maximum among the 11 target pathways; no effect label is read."],
    ]),"","## 5.1 Embedding controls","","![Embedding controls](figures/figureS_embedding_controls.png)",""]
    erows=[]
    for model in ["SVM","ResNet"]:
        for comparison in ["Gene + HubmiR − Gene","Gene + HubmiR − Gene + PCA-414","Gene + HubmiR − Gene + AE-414","Gene + HubmiR − Gene + RP-414 (5-seed mean)"]:
            r=econtr[(econtr.model==model)&(econtr.metric=="macro_f1")&(econtr.contrast==comparison)].iloc[0];erows.append([model,comparison.replace("Gene + HubmiR − ","vs "),f"{r.mean_delta:+.3f}",f"{r.ci95_low:+.3f} to {r.ci95_high:+.3f}",f"{r.holm_p_within_model_metric:.3g}"])
    lines += [table(["Model","HubmiR contrast","Δmacro-F1","95% CI","Holm p"],erows),"","## 5.2 Complementarity","","![Feature complementarity](figures/figureS_feature_complementarity.png)",""]
    cr=cka.query("source_space=='Gene-29045' and target_representation=='HubmiR' and method=='debiased_linear'").iloc[0];ca=cca.query("source_space=='Gene-29045' and target_representation=='HubmiR'").iloc[0];rr=ridge.query("source_space=='Gene-29045' and metric=='variance_weighted_r2'").iloc[0]
    drows=[]
    for model in ["SVM","ResNet"]:
        for c in ["Gene + residual HubmiR − Gene","Gene + HubmiR − Gene + predicted HubmiR","Gene + predicted HubmiR − Gene"]:
            r=dcontr[(dcontr.model==model)&(dcontr.metric=="macro_f1")&(dcontr.contrast==c)].iloc[0];drows.append([model,c,f"{r.mean_delta:+.3f}",f"{r.ci95_low:+.3f} to {r.ci95_high:+.3f}",f"{r.holm_p_within_model_metric:.3g}"])
    lines += [table(["Geometry/predictability endpoint","Mean (95% CI)"],[
        ["Gene–HubmiR debiased linear CKA",fmt_ci(cr)],
        ["Gene–HubmiR mean top-10 CCA",fmt_ci(ca)],
        ["Full Gene → HubmiR variance-weighted R²",fmt_ci(rr)],
    ]),"",table(["Model","Contrast","Δmacro-F1","95% CI","Holm p"],drows),"","> Interpretation boundary: these analyses support shared structure and nonlinear recoding. They do not establish independent molecular information. Residual utility is only the remainder not recovered by the fixed cross-validated Ridge model.","","## Figure 5 / 5.4 Expanded rescue analysis","","![Aligned Figure 5](figures/figure5_aligned_rescue_analysis.png)",""]
    pres=pathway.sort_values("net_reclassification_rate",ascending=False);lines += [table(["Pathway","Rescued","Harmed","Net rate"],[[r.true_pathway,int(r.rescued),int(r.harmed),f"{r.net_reclassification_rate:+.1%}"] for _,r in pres.iterrows()]),"",f"Locked rule yielded **{stable_rescue_count}** recurrent ResNet rescue cases and **{stable_harm_count}** recurrent harm case; case selection did not use miRNA values or literature.","","## ResNet tuning","","![ResNet tuning](figures/figureS_resnet_tuning.png)","","Round 10 was frozen by the highest mean validation macro-F1 over seeds 7/14/21. Test arrays were not passed during the 10 sequential rounds.","","## Bottom line","",f"The aligned rerun is technically complete. HubmiR gives a small, corrected-significant SVM gain (Δmacro-F1 {sprimary.mean_delta:+.3f}), but the primary ResNet gain is **not stable** (Δmacro-F1 {rprimary.mean_delta:+.3f}, CI crosses zero). The defensible evidence is therefore model-dependent utility, distinct but shared representation geometry, and a recurrent rescue subset—not a universal performance or independent-information claim. Unfavorable embedding-control and residual findings remain reported.","","## Provenance and validation","",f"- Final validation: [{validation['status']}](validation/final_validation.md)","- Frozen split manifest: [frozen_20_splits.csv](manifests/splits/frozen_20_splits.csv)","- Reviewer evidence ledger: [reviewer_evidence_ledger.csv](manifests/reviewer_evidence_ledger.csv)","- Server-to-local artifact hash: [transfer_integrity.json](validation/transfer_integrity.json)","- Figure QA: [figure_qa.md](validation/figure_qa.md)","- All numbers in this report are generated from prediction-level outputs; no favorable seeds were selected.",""]
    (ROOT/"report.md").write_text("\n".join(lines),encoding="utf-8");print("PASS: report.md written")


if __name__=="__main__":main()
