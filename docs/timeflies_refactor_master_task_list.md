# 📋 Timeflies Repo Refactor – Master Task List

## ✅ Functionality Goals
- Preserve all original functionality from the old repo/README (check old readme and old folder for previous code.)
- Add granular batch correction control.
- Ensure tests cover all critical workflows so future refactors don’t silently break things.
- Provide clear, production-grade documentation (README, comments, environment instructions).
- Make repo polished enough that an employer would see it as top-tier engineering.

---

## 🛠️ To-Do Items

### 1. CLI / Core Scripts
- [ ] Restore `run_timeflies.py`
  - Add back the main CLI entrypoint.
  - Support the following commands:
    ```bash
    python run_timeflies.py setup --tissue head
    python run_timeflies.py batch --train --tissue head
    python run_timeflies.py batch --evaluate --tissue head
    python run_timeflies.py batch --visualize --tissue head
    ```
  - Document these commands in the README.

- [ ] Granular Batch Correction  
  - Reintroduce fine-grained CLI options instead of just one integrated flag.  
  - Ensure parity with old `scvi_batch.py` functionality.  

---

### 2. Path & Config Management
- [ ] Path Manager Update  
  - Update `src/timeflies/utils/path_manager.py` to match new naming conventions (`data/raw/h5ad/...`).  
  - Remove any old hard-coded references.  

- [ ] Config Cleanup  
  - Currently both `core/config.py` and `configs/*.yaml` exist.  
  - Decide single source of truth (likely YAML configs for flexibility).  
  - Refactor `config.py` to just handle loading/validation if needed.  

---

### 3. Environment & Dependencies
- [ ] Primary Environments  
  - Use the two requirements files as the official environment setup:  
    - `requirements/requirements_batch.txt` → batch correction env  
    - `requirements/requirements_linux.txt` → Linux-specific env  
  - Document how/when to use each in the README.  

- [ ] Clarify Environment Workflow  
  - Make it explicit whether users need one or both environments.  
  - If batch correction needs isolation, explain why.  

---

### 4. Documentation
- [ ] README Overhaul  
  - Add installation instructions (using requirements files).  
  - Add CLI usage examples (setup, batch correction, training).  
  - Document path changes (new vs old).  
  - Clarify environments and config usage.  
  - Showcase improved analysis notebook.  

- [ ] Code Comments & Docstrings  
  - Add thorough comments + docstrings in all modules.  
  - Ensure functions are clear enough that any reviewer can follow logic instantly.  

- [ ] Employer-Polish Pass  
  - Write docs and comments to the standard of an “interview repo” (clear, clean, professional).  

---

### 5. Testing
- [ ] Add Comprehensive Tests  
  - Unit tests for preprocessing, batch correction, path manager, model factory.  
  - Integration tests for CLI commands (`setup`, `batch train/evaluate/visualize`).  
  - Smoke tests for training on small toy data.  

- [ ] Automated Test Workflow  
  - Add GitHub Actions/CI (if repo is public) to run tests automatically.  

---

### 6. Analysis & Results
- [ ] Analysis Notebook Update  
  - Rename `analysis.ipynb` → `model_analysis.ipynb`.  
  - Expand with:  
    - Clearer EDA & visualizations.  
    - UMAP/t-SNE plots.  
    - Batch correction before/after views.  
    - Performance metrics (loss curves, accuracy).  
  - Add markdown commentary so it reads like a mini-report.  

---

### 7. Multi-Project Structure
- [ ] Support Alzheimer’s Project  
  - Decide structure:  
    - Option A: Fork repo → keep Alzheimer’s separate.  
    - Option B: Single repo with submodules:  
      ```
      src/common/        # shared utils
      src/timeflies/     # main project
      src/alzheimers/    # Alzheimer’s-specific code
      ```  
  - Keep preprocessing pipelines flexible so datasets can be swapped easily.  

---

## 🧾 Final Deliverables
1. Restored + improved `run_timeflies.py`.  
2. Updated `path_manager.py` and unified config system.  
3. Clear environment setup (batch + Linux req files as source of truth).  
4. Polished README with usage examples + environment instructions.  
5. Fully commented/documented codebase.  
6. Comprehensive test suite with CI.  
7. Improved analysis notebook (`model_analysis.ipynb`).  
8. Clear repo structure to support both Timeflies + Alzheimer’s projects.  
