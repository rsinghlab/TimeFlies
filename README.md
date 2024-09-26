# Project Folder Structure

This directory structure is organized to facilitate analysis and testing on different tissues, model types, and configuration settings for gene selection and balancing. Each folder corresponds to a specific combination of configurations, enabling easy access to results and reproducing experiments.

## Folder Breakdown:

- **uncorrected_or_batch_corrected/**: Top-level folder indicating whether batch correction was applied. The options are:
  - `uncorrected`
  - `batch_corrected`
  
- **tissue_type/**: Refers to the tissue used in training. Possible values:
  - `head`
  - `body`
  - `head&body`
    - `trainHead_testBody`
    - `trainBody_testHead`
    - `all`



- **model_type/**: Specifies the model architecture used for analysis. Examples:
  - `CNN`
  - `MLP`

- **target_variable/**: Represents the variable used for encoding:
  - `age`
  - `sex`
  - `sex_age`
  - Other variables can be added here based on the configuration.

- **gene_data/**: Specifies the type of gene data used in the experiment. Examples:
  - `full_data`: Full gene set used.
  - `no_sex`: Filtered or reduced gene sets.
  - `no_sex`: Filtered or reduced gene sets.
  - `no_sex`: Filtered or reduced gene sets.
  - `no_sex`: Filtered or reduced gene sets.
  - `select_batch_genes`: True or False, depending on whether batch gene selection was applied.
  - `highly_variable_genes`: True or False, indicating whether highly variable genes were used.
  - `balance_genes`: True or False, indicating whether genes were balanced during training.
  - `balance_lnc_genes`: Similar to `balance_genes` but for lncRNAs.

- **celltype/**: The cell types used in training, such as:
  - `all`
  - Specific cell types like `neuron`, `epithelial`, etc.

- **sextype/**: Indicates the type of sex data used in training:
  - `male`, `female`, etc.

- **Results_or_EDA/**: Stores either final results or exploratory data analysis outputs.
  - `Results`: Contains the final model outputs.
  - `EDA`: Holds exploratory analysis results.

- **Latest_or_Best/**: Subfolders to track either the latest experiment or the best-performing model.
  - `Latest`: Stores the most recent run.
  - `Best`: Contains results from the best-performing model based on specific criteria.

## Configuration Options and Corresponding Folders

This structure is directly linked to the configuration file used in the code. For instance:
- `select_batch_genes=True` creates a corresponding folder under `filters/`.
- If `balance_genes` is toggled, a folder is created to reflect that configuration.

By navigating the folder structure, one can easily trace the configuration used for any experiment, making replication and analysis more straightforward.

## Notes:
- Every experiment should have a corresponding README or metadata file within the `Results/` folder that details the configuration used for that run.
- Follow the naming conventions strictly to maintain consistency.