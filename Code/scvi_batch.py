import os
import argparse
import scanpy as sc
import scvi # type: ignore
import torch # type: ignore
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
        data_type="uncorrected",  # "uncorrected" or "batch_corrected"
        tissue="head",           # "head", "body", or "all"
        base_dir="../Data/h5ad"  # base directory for h5ad files
    ):
        """
        Initialize with the path to the AnnData object.
        """   
        self.base_dir = base_dir
        self.tissue = tissue
        
        if data_type == "uncorrected":
            data_dir = os.path.join(base_dir, tissue, "uncorrected")
            train_path = os.path.join(data_dir, "fly_train.h5ad")
            eval_path = os.path.join(data_dir, "fly_eval.h5ad")
        elif data_type == "batch_corrected":
            data_dir = os.path.join(base_dir, tissue, "batch_corrected")
            train_path = os.path.join(data_dir, "fly_train_batch.h5ad")
            eval_path = os.path.join(data_dir, "fly_eval_batch.h5ad")
        else:
            raise ValueError("data_type must be 'uncorrected' or 'batch_corrected'")
        
        self.adata_train = sc.read_h5ad(train_path)
        self.adata_eval  = sc.read_h5ad(eval_path)
        
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
        # Train data
        self.adata_train.obsm[self.SCVI_LATENT_KEY] = (
            model.get_latent_representation(self.adata_train)
        )
        self.adata_train.layers[self.SCVI_NORMALIZED_KEY] = (
            model.get_normalized_expression(self.adata_train, library_size=1e4)
        )

        # Evaluation data
        scvi.model.SCVI.prepare_query_anndata(
            self.adata_eval,        
            reference_model=model
        )

        # Latent representation for evaluation data
        latent = model.get_latent_representation(self.adata_eval)

        # Directly use the modified self.adata_eval
        self.adata_eval.obsm[self.SCVI_LATENT_KEY] = latent

        self.adata_eval.layers[self.SCVI_NORMALIZED_KEY] = (
            model.get_normalized_expression(self.adata_eval, library_size=1e4)
        )

    def save_results(self, out_dir=None):
        """
        Save the corrected AnnData object.
        """
        if out_dir is None:
            # Default to the batch_corrected directory structure
            out_dir = os.path.join(self.base_dir, self.tissue, "batch_corrected")
            os.makedirs(out_dir, exist_ok=True)
        
        self.adata_train.write_h5ad(os.path.join(out_dir, "fly_train_batch.h5ad"))
        self.adata_eval .write_h5ad(os.path.join(out_dir, "fly_eval_batch.h5ad"))
        print(f"Saved batch corrected data to {out_dir}")
    
    def run(self, out_dir=None):
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
            raw_path=None,
            batch_path=None,
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
            fly_eval_scvi_all.png         
        """
        t0 = time.time()

        # Set default paths if not provided
        if raw_path is None:
            raw_path = os.path.join(self.base_dir, self.tissue, "uncorrected", "fly_eval.h5ad")
        if batch_path is None:
            batch_path = os.path.join(self.base_dir, self.tissue, "batch_corrected", "fly_eval_batch.h5ad")

        # Load data
        adata_raw   = sc.read_h5ad(raw_path)
        adata_batch = sc.read_h5ad(batch_path)

        # Build UMAP on raw log‑CP10K
        adata_raw.layers["counts"] = adata_raw.X.copy()
        sc.pp.normalize_total(adata_raw, target_sum=1e4)   
        sc.pp.log1p(adata_raw)                            
        adata_raw.layers["log1p_cp10k"] = adata_raw.X.copy()  
        sc.pp.pca(adata_raw, n_comps=30)
        sc.pp.neighbors(adata_raw, n_pcs=30)
        sc.tl.umap(adata_raw, min_dist=0.3)

        # Build UMAP on scVI latent
        sc.pp.neighbors(adata_batch, use_rep=self.SCVI_LATENT_KEY, n_neighbors=20)  
        sc.tl.umap(adata_batch, min_dist=0.3)

        # Columns to plot
        batch_col = self.batch_column          
        bio_cols  = ["sex", "age"]
        all_cols  = [batch_col] + bio_cols

        # Set Scanpy's figure output directory
        sc.settings.figdir = os.path.join("..", "Analysis", "UMAP")
        os.makedirs(sc.settings.figdir, exist_ok=True)

        # Helper for per‑column PNGs
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

        # Raw per‑column plots
        plot_umap(
            adata_raw,
            all_cols,
            label="Uncorrected (log‑CP10K)",
            prefix=f"{out_prefix}_uncorrected"
        )

        # scVI per‑column plots
        plot_umap(
            adata_batch,
            all_cols,
            label="scVI latent (batch‑corrected)",
            prefix=f"{out_prefix}_scvi"
        )

        # figure with dataset / sex / age side‑by‑side (scVI)
        sc.pl.umap(
            adata_batch,
            color=all_cols,
            frameon=False,
            ncols=3,
            title=[f"scVI: {c}" for c in all_cols],
            save=f"_{out_prefix}_scvi_all.png",
            show=False
        )

        # figure with dataset / sex / age side‑by‑side (uncorrected)
        sc.pl.umap(
            adata_raw,
            color=all_cols,
            frameon=False,
            ncols=3,
            title=[f"scVI: {c}" for c in all_cols],
            save=f"_{out_prefix}_uncorrected_all.png",
            show=False
        )
        plt.close()

        print(f"UMAPs saved (elapsed {time.time() - t0:.1f} s).")


    def evaluate_metrics(
        self,
        raw_path=None,         
        batch_path=None,  
        out_csv=None,
    ):
        """
        Evaluate batch-integration quality on the hold-out set and
        store a CSV with two columns:  metric , score
        """
        t0 = time.time()
        
        # Set default paths if not provided
        if raw_path is None:
            raw_path = os.path.join(self.base_dir, self.tissue, "uncorrected", "fly_eval.h5ad")
        if batch_path is None:
            batch_path = os.path.join(self.base_dir, self.tissue, "batch_corrected", "fly_eval_batch.h5ad")
        if out_csv is None:
            # Save to Analysis/batch_corrected directory
            out_dir = os.path.join("..", "Analysis", "batch_corrected")
            os.makedirs(out_dir, exist_ok=True)
            out_csv = os.path.join(out_dir, "fly_eval_scvi_scores.csv")
            
        adata_int = sc.read_h5ad(batch_path)   # has .obsm["X_scVI"]
        adata_raw = sc.read_h5ad(raw_path)     # counts in .X

        # scIB expects raw counts in X for the "batch" view
        adata_raw.layers["counts"] = adata_raw.X.copy() 
        adata_int.obsm["X_int"] = adata_int.obsm["X_scVI"]

        # neighbours on the integrated embedding
        sc.pp.neighbors(adata_int, use_rep="X_int", n_neighbors=15)

        # compute scores 
        scores_wide = sm.metrics_fast(
            adata_raw,
            adata_int,
            batch_key=self.batch_column,
            label_key=self.label_col,
            embed="X_int",
            type_="embed",
            n_cores=os.cpu_count() - 1
        )

        # silhouette scores
        adata_int.obs["age_str"] = adata_int.obs["age"].astype(str)
        sil_sex = sm.silhouette(adata_int, label_key="sex", embed="X_int")
        sil_age = sm.silhouette(adata_int, label_key="age_str", embed="X_int")

        # Create a clean DataFrame with metric names and scores
        metrics_list = []
        scores_list = []
        
        # Add scIB metrics - extract from index and first column
        for metric_name in scores_wide.index:
            score_value = scores_wide.iloc[scores_wide.index.get_loc(metric_name), 0]
            # Only add metrics with non-NaN values
            if pd.notna(score_value):
                metrics_list.append(str(metric_name))
                scores_list.append(float(score_value))
        
        # Add silhouette scores
        metrics_list.extend(["silhouette_sex", "silhouette_age"])
        scores_list.extend([sil_sex, sil_age])
        
        # Create final DataFrame
        results_df = pd.DataFrame({
            "metric": metrics_list,
            "score": scores_list
        })
        
        # Save to CSV
        results_df.to_csv(out_csv, index=False)

        print(f"Evaluation metrics:")
        print(results_df.head(10))
        print(f"Scores saved to {out_csv} (elapsed {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run scVI batch correction on fly dataset.")
    parser.add_argument("--train", action="store_true", help="Train the scVI model (otherwise just evaluate).")
    parser.add_argument("--umap", action="store_true", help="Run UMAP after training.")
    parser.add_argument("--data_type", default="uncorrected", choices=["uncorrected", "batch_corrected"], 
                        help="Type of data to load: 'uncorrected' or 'batch_corrected'")
    parser.add_argument("--tissue", default="head", choices=["head", "body", "all"], 
                        help="Tissue type: 'head', 'body', or 'all'")
    args = parser.parse_args()
    
    batch_corrector = BatchCorrector(
        data_type=args.data_type,
        tissue=args.tissue
    )
    
    if args.train:
        batch_corrector.run()
        print("Batch correction completed.")
    elif args.umap:
        batch_corrector.umap_visualization()
        print("Visualisation completed.")
    else:
        batch_corrector.evaluate_metrics()
        print("Metrics evaluation completed.")
