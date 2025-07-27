import os
import argparse
import scanpy as sc
import scvi
import torch
import time
import scib.metrics as sm
import pandas as pd
import matplotlib.pyplot as plt

torch.set_float32_matmul_precision('medium')

scvi.settings.seed = 0
scvi.settings.dl_num_workers = os.cpu_count() - 1  
scvi.settings.dl_pin_memory  = True 

print("Last run with scvi-tools version:", scvi.__version__)

if torch.cuda.is_available():
    print("Current device index:   ", torch.cuda.current_device())
    print("Device name:            ", torch.cuda.get_device_name(0))

class BatchCorrector:
    """
    Class to handle batch correction using scVI.
    """

    def __init__(
        self,
        train_path="fly_train.h5ad",
        eval_path="fly_eval.h5ad",          # raw hold‑out
    ):
        """
        Initialize with the path to the AnnData object.
        """   
        self.adata_train = sc.read_h5ad(train_path)
        # self.adata_train = self.adata_train[:100].copy()
        
        self.adata_eval  = sc.read_h5ad(eval_path)
        # self.adata_eval = self.adata_eval[:100].copy()
        
        self.batch_column = "dataset"
        self.SCVI_LATENT_KEY = "X_scVI"
        self.SCVI_NORMALIZED_KEY = "scvi_normalized"
        self.label_col = "afca_annotation_broad"
    
    def _prep_counts(self, ad):
        """Copy raw counts into .layers['counts']."""
        ad.layers["counts"] = ad.X.copy()

    def preprocess_data(self):
        self._prep_counts(self.adata_train)
        self._prep_counts(self.adata_eval)
    
    def setup_scvi(self):
        """
       ` Setup the AnnData object for scVI.
        """
        scvi.model.SCVI.setup_anndata(
            self.adata_train,
            layer="counts",
            batch_key=self.batch_column,
        )

    def train_model(self):
        """
        Train the scVI model.
        """
        model = scvi.model.SCVI(
            self.adata_train,
            n_latent=25,
            n_hidden=256,
            n_layers=2,
            dropout_rate=0.15
        )
        
        model.train(max_epochs=None,
                    plan_kwargs={"weight_decay": 1e-4}      # 0.0001 L2 penalty
                    )   # early‑stops automatically
        print("scVI model trained.")
        return model
    
    def add_scvi_outputs(self, model):
        """Add scVI latent & normalised output to train and eval AnnData."""
        # ----- TRAIN (already registered) -----
        self.adata_train.obsm[self.SCVI_LATENT_KEY] = model.get_latent_representation(self.adata_train)
        self.adata_train.layers[self.SCVI_NORMALIZED_KEY] = model.get_normalized_expression(
            self.adata_train, library_size=1e4
        )

        # ----- EVAL (register as query, *no* parameter updates) -----
        qadata = scvi.model.SCVI.prepare_query_anndata(
        self.adata_eval,         # ← your unseen cells
        reference_model=model    # ← the model you just trained
       )
    
        self.adata_eval.obsm[self.SCVI_LATENT_KEY] = model.get_latent_representation(qadata)
        self.adata_eval.layers[self.SCVI_NORMALIZED_KEY] = model.get_normalized_expression(
            qadata, library_size=1e4
        )

    def save_results(self, out_dir="."):
        """
        Save the corrected AnnData object.
        """
        self.adata_train.write_h5ad(os.path.join(out_dir, "fly_train_batch.h5ad"))
        self.adata_eval .write_h5ad(os.path.join(out_dir, "fly_eval_batch.h5ad"))
        print(f"Saved batch corrected data to {out_dir}")
    
    def run(self, out_dir="."):
        """
        Run the full batch correction process.
        """
        start_time = time.time()
        self.preprocess_data()
        self.setup_scvi()
        model = self.train_model()
        self.add_scvi_outputs(model)
        self.save_results(out_dir)

        print(f"batch_correct_scvi completed in {time.time() - start_time:.2f} s.")



    def umap_visualization(
            self,
            raw_path="fly_eval.h5ad",
            batch_path="fly_eval_batch.h5ad",
            out_prefix="fly_eval"
        ):
        """
        Visualise batch removal and biology preservation on UMAPs.

        Generates:
        • One PNG per obs column (`dataset`, `sex`, `age`) for *raw* counts
        • One PNG per obs column for *scVI latent*
        • One extra PNG with all three columns side‑by‑side on the scVI latent

        Files created (example):
            fly_eval_uncorrected_dataset.png
            fly_eval_uncorrected_sex.png
            fly_eval_uncorrected_age.png
            fly_eval_scvi_dataset.png
            fly_eval_scvi_sex.png
            fly_eval_scvi_age.png
            fly_eval_scvi_all.png          ← NEW
        """
        t0 = time.time()

        # 1 ─ Load data
        adata_raw   = sc.read_h5ad(raw_path)
        adata_batch = sc.read_h5ad(batch_path)

        # 2 ─ Build UMAP on raw log‑CP10K
        adata_raw.layers["log1p_cp10k"] = adata_raw.X.copy()
        sc.pp.normalize_total(adata_raw, target_sum=1e4, layer="log1p_cp10k")
        sc.pp.log1p(adata_raw, layer="log1p_cp10k")
        adata_raw.X = adata_raw.layers["log1p_cp10k"].copy()
        sc.pp.pca(adata_raw)
        sc.pp.neighbors(adata_raw, n_pcs=30)
        sc.tl.umap(adata_raw, min_dist=0.3)

        # 3 ─ Build UMAP on scVI latent
        adata_batch.obsm["X_int"] = adata_batch.obsm[self.SCVI_LATENT_KEY]
        sc.pp.neighbors(adata_batch, use_rep="X_int", n_neighbors=20)
        sc.tl.umap(adata_batch, min_dist=0.3)

        # 4 ─ Columns to plot
        batch_col = self.batch_column          # "dataset"
        bio_cols  = ["sex", "age"]
        all_cols  = [batch_col] + bio_cols

        # 5 ─ Helper for per‑column PNGs
        def plot_umap(adata, cols, label, prefix):
            for c in cols:
                if adata.obs[c].dtype.kind in "ifc":
                    if f"{c}_str" not in adata.obs.columns:
                        adata.obs[f"{c}_str"] = adata.obs[c].astype(str)
                    c = f"{c}_str"
                sc.pl.umap(
                    adata,
                    color=[c],
                    title=f"{label}: {c}",
                    save=f"_{prefix}_{c}.png",
                    show=False
                )
                plt.close()

        # 6 ─ Raw per‑column plots
        plot_umap(
            adata_raw,
            all_cols,
            label="Uncorrected (log‑CP10K)",
            prefix=f"{out_prefix}_uncorrected"
        )

        # 7 ─ scVI per‑column plots
        plot_umap(
            adata_batch,
            all_cols,
            label="scVI latent (batch‑corrected)",
            prefix=f"{out_prefix}_scvi"
        )

        # 8 ─ NEW: one figure with dataset / sex / age side‑by‑side (scVI)
        sc.pl.umap(
            adata_batch,
            color=all_cols,
            frameon=False,
            ncols=3,
            title=[f"scVI: {c}" for c in all_cols],
            save=f"_{out_prefix}_scvi_all.png",
            show=False
        )
        plt.close()

        print(f"UMAPs saved (elapsed {time.time() - t0:.1f} s).")


    def evaluate_metrics(
        self,
        raw_path="fly_eval.h5ad",          # raw hold‑out
        batch_path="fly_eval_batch.h5ad",  # same cells + scVI outputs
        out_csv="fly_eval_scvi_scores.csv",
    ):
        """
        Evaluate batch-integration quality on the hold-out set and
        store a CSV with two columns:  metric , score
        """
        t0 = time.time()
        adata_int = sc.read_h5ad(batch_path)   # has .obsm["X_scVI"]
        adata_raw = sc.read_h5ad(raw_path)     # counts in .X

        # scIB expects raw counts in X for the "batch" view
        adata_raw.layers["counts"] = adata_raw.X.copy()  # just in case
        adata_int.obsm["X_int"] = adata_int.obsm["X_scVI"]

        # neighbours on the integrated embedding
        sc.pp.neighbors(adata_int, use_rep="X_int", n_neighbors=15)

        # --- compute scores -------------------------------------------------
        scores_wide = sm.metrics_fast(
            adata_raw,
            adata_int,
            batch_key=self.batch_column,      # e.g. "dataset"
            label_key=self.label_col,    # e.g. "afca_annotation_broad"
            embed="X_int",
            type_="embed",
            n_cores=os.cpu_count() - 1
        )
        
        # ---- biology‑conservation metrics you care about ----
        adata_int.obs["age_str"] = adata_int.obs["age"].astype(str)   # silhouette wants categorical
        sil_sex = sm.silhouette(adata_int, label_key="sex",     embed="X_int")
        sil_age = sm.silhouette(adata_int, label_key="age_str", embed="X_int")

        # ---- tidy and save ---------------------------------
        scores_long = (
            scores_wide.melt(var_name="metric", value_name="score")
            .sort_values("metric")
            .reset_index(drop=True)
        )
        extra = pd.DataFrame({
            "metric": ["silhouette_sex", "silhouette_age"],
            "score" : [sil_sex, sil_age]
        })
        all_scores = pd.concat([scores_long, extra], ignore_index=True)
        all_scores.to_csv(out_csv, index=False)

        print(all_scores.head())           # sanity‑check
        print(f"Scores saved to {out_csv} (elapsed {time.time()-t0:.1f}s)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run scVI batch correction on fly dataset.")
    parser.add_argument("--train", action="store_true", help="Train the scVI model (otherwise just evaluate).")
    parser.add_argument("--umap", action="store_true", help="Run UMAP after training.")
    args = parser.parse_args()
    batch_corrector = BatchCorrector(
    train_path="fly_train.h5ad",
    eval_path="fly_eval.h5ad",
    )
    if args.train:
        batch_corrector.run(out_dir=".")
        print("Batch correction completed.")
    elif args.umap:
        batch_corrector.umap_visualization()
        print("Visualisation completed.")
    else:
        batch_corrector.evaluate_metrics()
        print("Metrics evaluation completed.")
