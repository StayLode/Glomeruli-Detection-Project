"""Per-split clustering summaries; no invented clinical accuracy or grades."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.clustering.metrics import _metric_result, _cramers_v, _eta_squared, _profile_table
from src.clustering.labels import LABEL_LIMITATION
from src.utils.evaluation import write_json


def criteria(metrics):
    stability=metrics["stability"]; slide=metrics["slide_confounding"]["train"]
    z=metrics["random_baseline"]["silhouette"]["z_score"]
    flags={"ari_subsampling":stability["subsampling_ari_mean"]>=.75,
           "jaccard_min":stability["cluster_jaccard_min"]>=.60,
           "silhouette_z":z is not None and z>=2,
           "cramers_v":slide["cramers_v_cluster_slide"]<=.30,
           "dominant_slide":slide["max_dominant_slide_fraction"]<=.80}
    return flags


def evaluate_partition(frame, common, feature_names, train_frame, k, mapping):
    labels=frame.cluster.to_numpy(int)
    distances=frame.distance_to_center.to_numpy(float)
    result={"n_samples":len(frame), "n_slides":int(frame.slide_id.nunique()) if len(frame) else 0,
            "counts":{mapping[i]:int(np.count_nonzero(labels==i)) for i in range(k)},
            "occupied_clusters":int(len(np.unique(labels))),
            "internal_common":_metric_result(common,labels),
            "cramers_v":_cramers_v(frame) if len(frame) else {"status":"not_evaluable","value":None,"reason":"no samples"},
            "distance_median":float(np.median(distances)) if len(frame) else None,
            "distance_q95":float(np.quantile(distances,.95)) if len(frame) else None,
            "native_profiles":_profile_table(frame,feature_names,train_frame) if len(frame) else {},
            "technical_eta2":{name:_eta_squared(frame[name],labels) for name in ("completeness","border_distance_px","tissue_ratio","background_mean_intensity") if name in frame},
            "label_limitation":LABEL_LIMITATION}
    result["by_slide"]={str(slide):{"counts":{mapping[i]:int(np.count_nonzero(rows.cluster==i)) for i in range(k)},
        "n":len(rows),"internal_common":_metric_result(common[frame.slide_id.to_numpy()==slide],rows.cluster.to_numpy())}
        for slide,rows in frame.groupby("slide_id",sort=True)}
    return result


def save_gallery(path, crops, frame, k, mapping):
    selected=[]
    ordered_clusters=sorted(range(k),key=lambda cluster:mapping[cluster])
    for cluster in ordered_clusters:
        idx=np.flatnonzero(frame.cluster.to_numpy()==cluster)
        if "split" in frame.columns:
            for preferred_split in ("test","val","train"):
                preferred=idx[frame.iloc[idx]["split"].to_numpy()==preferred_split]
                if len(preferred):
                    idx=preferred
                    break
        if len(idx):
            representative=int(idx[np.argmin(frame.iloc[idx].distance_to_center.to_numpy())])
            selected.append((cluster,representative))
        else:
            selected.append((cluster,None))
    rows,columns={2:(1,2),4:(2,2),6:(2,3)}.get(k,(int(np.ceil(k/3)),min(3,k)))
    fig=plt.figure(figsize=(5.4*columns,5.8*rows),facecolor="white")
    grid=fig.add_gridspec(rows*2,columns,height_ratios=[1,.22]*rows,
                          left=.045,right=.955,bottom=.045,top=.955,
                          wspace=.13,hspace=.14)
    for position,(cluster,i) in enumerate(selected):
        grid_row=2*(position//columns)
        grid_column=position%columns
        image_ax=fig.add_subplot(grid[grid_row,grid_column])
        caption_ax=fig.add_subplot(grid[grid_row+1,grid_column])
        caption_ax.axis("off")
        if i is None:
            image_ax.text(.5,.5,"nessun campione",ha="center",va="center",fontsize=12)
            image_ax.axis("off")
            continue
        row=frame.iloc[i]
        score=row.get("peripheral_clear_fraction")
        score_text=f"clear proxy {score:.2f}" if score is not None and np.isfinite(score) else "clear proxy n.d."
        source=row.get("split",row.get("slide_id","source n.d."))
        if source=="val":source="validation"
        image_ax.imshow(crops[i])
        image_ax.axis("off")
        caption_ax.text(.5,.5,
            f"{mapping[cluster]} · {source}  |  {row.annotation_id}\n{score_text}  |  d={row.distance_to_center:.3f}",
            ha="center",va="center",fontsize=11,linespacing=1.35,wrap=True)
    fig.savefig(path,dpi=140);plt.close(fig)


def save_projection(path, projected, frame, mapping):
    from sklearn.metrics import silhouette_score

    selected=frame["split"].isin(["train","val"]).to_numpy()
    points=np.asarray(projected)[selected]
    labels=frame.loc[selected,"cluster"].to_numpy()
    silhouette=(float(silhouette_score(points,labels))
                if 2<=len(np.unique(labels))<len(labels) else None)
    silhouette_text=f"{silhouette:.3f}" if silhouette is not None else "non valutabile"
    fig,ax=plt.subplots(figsize=(8,6))
    palette=plt.get_cmap("tab10")
    for index,(cluster,label) in enumerate(sorted(mapping.items(),key=lambda item:item[1])):
        mask=labels==cluster
        if mask.any():
            ax.scatter(points[mask,0],points[mask,1],s=20,alpha=.65,
                       color=palette(index%10),label=f"{label} (n={int(mask.sum())})")
    ax.set(xlabel="PC1",ylabel="PC2",
           title=f"PCA20 · K={len(mapping)} · train + validation\nSilhouette PCA20: {silhouette_text}")
    if len(labels):ax.legend(title="Classe",fontsize=9)
    fig.tight_layout();fig.savefig(path,dpi=140);plt.close(fig)


def write_report(root, records, title, note=""):
    root=Path(root);pd.DataFrame(records).to_csv(root/"summary.csv",index=False)
    write_json(root/"summary.json",records)
    lines=[f"**{title}**","",LABEL_LIMITATION,"",note,"",
        "| Modello | Split/slide | N | Conteggi lettere | Silhouette comune | Mediana distanza | Criteri train |",
        "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in records:
        silhouette=f"{r['silhouette']:.4f}" if r.get("silhouette") is not None else "non valutabile"
        distance=f"{r['distance_median']:.4f}" if r.get("distance_median") is not None else "non valutabile"
        lines.append(f"| {r['model']} | {r['split']} | {r['n_samples']} | {r['counts']} | {silhouette} | {distance} | {r.get('criteria_passed','—')} |")
    lines += ["","Le lettere vengono ordinate sul train e mantenute fisse negli altri split. Non sono una classificazione clinica validata.",
              "I modelli ai diversi K sono indipendenti: lettere uguali non garantiscono identici sottogruppi fra K.",
              "Un solo cluster occupato rende silhouette/DB/CH non valutabili; una sola slide rende Cramér's V non stimabile.",
              "Le distanze appartengono allo spazio del modello e non sono probabilità. I cinque criteri sono criteri sul train.",
              "Gli XML descrivono posizioni/contorni, non classi di salute o necrosi. Le WSI test vengono elaborate da run_wsi_inference e predict_clustering."]
    (root/"report.md").write_text("\n".join(lines)+"\n")
