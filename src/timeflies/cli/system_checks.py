"""
System verification for TimeFlies.

Quick check: can we import things? Do data files exist?
"""

import importlib
import sys
from pathlib import Path


def verify_system(dev_mode: bool = None) -> bool:
    """Verify that TimeFlies can run: Python version, packages, directories."""
    if dev_mode is None:
        dev_mode = Path("tests").exists() and Path("src").exists()

    print("TimeFlies System Verification")
    print("=" * 40)

    passed = True

    # Python version
    v = sys.version_info
    required = (3, 12)
    ok = v >= required
    status = "OK" if ok else "FAIL"
    print(f"  [{status}] Python {v.major}.{v.minor}.{v.micro} (need >={required[0]}.{required[1]})")
    passed &= ok

    # Core packages
    packages = [
        ("tensorflow", "tensorflow"),
        ("scikit-learn", "sklearn"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("scanpy", "scanpy"),
        ("anndata", "anndata"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("shap", "shap"),
        ("pyyaml", "yaml"),
        ("dill", "dill"),
        ("xgboost", "xgboost"),
    ]
    print("\n  Packages:")
    for name, import_name in packages:
        try:
            mod = importlib.import_module(import_name)
            version = getattr(mod, "__version__", "?")
            print(f"    [OK]   {name} {version}")
        except ImportError:
            print(f"    [FAIL] {name} not installed")
            passed = False

    # Directories
    print("\n  Directories:")
    for d in ["configs", "data"]:
        exists = Path(d).exists()
        status = "OK" if exists else "MISSING"
        print(f"    [{status}] {d}/")
        if not exists and d == "data":
            print(f"           Create {d}/ and add your H5AD files")

    # Summary
    print("\n" + "=" * 40)
    if passed:
        print("All checks passed. TimeFlies is ready.")
    else:
        print("Some checks failed. Fix the issues above.")
    return passed


if __name__ == "__main__":
    verify_system()
