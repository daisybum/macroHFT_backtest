"""
Verification Script for MacroHFT Bitcoin Backtesting System
Tests all imports, dependencies, and model availability
"""

import sys
import os
from typing import List, Tuple

print("="*70)
print(" "*15 + "MACROHFT INSTALLATION VERIFICATION")
print("="*70)
print()

# Track results
passed_tests = []
failed_tests = []


def test_import(module_name: str, package_name: str = None) -> bool:
    """Test if a module can be imported"""
    try:
        __import__(module_name)
        return True
    except ImportError as e:
        return False


def check_file_exists(filepath: str) -> bool:
    """Check if a file exists"""
    return os.path.exists(filepath)


print("Testing Core Dependencies...")
print("-" * 70)

# Core dependencies
dependencies = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("torch", "pytorch"),
    ("yaml", "pyyaml"),
    ("binance", "python-binance"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("sklearn", "scikit-learn"),
    ("scipy", "scipy"),
    ("tqdm", "tqdm"),
]

for module, package in dependencies:
    if test_import(module):
        print(f"  [OK] {package}")
        passed_tests.append(f"{package} import")
    else:
        print(f"  [FAIL] {package} (install with: pip install {package})")
        failed_tests.append(f"{package} import")

print()
print("Testing MacroHFT Components...")
print("-" * 70)

# MacroHFT modules
macrohft_modules = [
    "MacroHFT.model.net",
    "MacroHFT.RL.util.memory",
    "MacroHFT.tools.demonstration",
]

for module in macrohft_modules:
    if test_import(module):
        print(f"  [OK] {module}")
        passed_tests.append(f"{module} import")
    else:
        print(f"  [FAIL] {module}")
        failed_tests.append(f"{module} import")

print()
print("Testing Backtest Components...")
print("-" * 70)

# Backtest modules
sys.path.insert(0, './backtest/src')
backtest_modules = [
    "data_fetcher",
    "feature_engineering",
    "strategy",
    "backtester",
    "analysis",
    "run_backtest",
]

for module in backtest_modules:
    if test_import(module):
        print(f"  [OK] {module}")
        passed_tests.append(f"{module} import")
    else:
        print(f"  [FAIL] {module}")
        failed_tests.append(f"{module} import")

print()
print("Checking Pre-trained Models...")
print("-" * 70)

# Model files
model_files = [
    "MacroHFT/result/low_level/ETHUSDT/best_model/slope/1/best_model.pkl",
    "MacroHFT/result/low_level/ETHUSDT/best_model/slope/2/best_model.pkl",
    "MacroHFT/result/low_level/ETHUSDT/best_model/slope/3/best_model.pkl",
    "MacroHFT/result/low_level/ETHUSDT/best_model/vol/1/best_model.pkl",
    "MacroHFT/result/low_level/ETHUSDT/best_model/vol/2/best_model.pkl",
    "MacroHFT/result/low_level/ETHUSDT/best_model/vol/3/best_model.pkl",
]

for model_file in model_files:
    if check_file_exists(model_file):
        print(f"  [OK] {model_file}")
        passed_tests.append(f"{model_file} exists")
    else:
        print(f"  [FAIL] {model_file}")
        failed_tests.append(f"{model_file} exists")

print()
print("Checking Configuration Files...")
print("-" * 70)

config_files = [
    "backtest/config/backtest_config.yaml",
    "MacroHFT/data/feature_list/single_features.npy",
    "MacroHFT/data/feature_list/trend_features.npy",
]

for config_file in config_files:
    if check_file_exists(config_file):
        print(f"  [OK] {config_file}")
        passed_tests.append(f"{config_file} exists")
    else:
        print(f"  [FAIL] {config_file}")
        failed_tests.append(f"{config_file} exists")

print()
print("Checking Directory Structure...")
print("-" * 70)

directories = [
    "backtest/src",
    "backtest/config",
    "MacroHFT/model",
    "MacroHFT/RL",
    "MacroHFT/env",
]

for directory in directories:
    if os.path.isdir(directory):
        print(f"  [OK] {directory}/")
        passed_tests.append(f"{directory}/ exists")
    else:
        print(f"  [FAIL] {directory}/")
        failed_tests.append(f"{directory}/ exists")

print()
print("Testing PyTorch CUDA Availability...")
print("-" * 70)

try:
    import torch
    if torch.cuda.is_available():
        print(f"  [OK] CUDA available: {torch.cuda.get_device_name(0)}")
        passed_tests.append("CUDA available")
    else:
        print("  [WARN] CUDA not available (will use CPU - slower but functional)")
        passed_tests.append("CPU mode")
except:
    print("  [FAIL] PyTorch not properly installed")
    failed_tests.append("PyTorch CUDA check")

print()
print("="*70)
print("SUMMARY")
print("="*70)
print(f"Passed: {len(passed_tests)}")
print(f"Failed: {len(failed_tests)}")
print()

if failed_tests:
    print("Failed Tests:")
    for test in failed_tests:
        print(f"  - {test}")
    print()
    print("[WARN] Some components are missing. Please:")
    print("  1. Install missing dependencies: pip install -r requirements_backtest.txt")
    print("  2. Ensure MacroHFT models are available")
    print("  3. Check directory structure")
    print()
    sys.exit(1)
else:
    print("[SUCCESS] All checks passed!")
    print()
    print("Your system is ready to run MacroHFT Bitcoin backtesting.")
    print()
    print("To get started:")
    print("  Windows: run_backtest.bat")
    print("  Linux/Mac: bash run_backtest.sh")
    print("  Python: python backtest/src/run_backtest.py")
    print()
    sys.exit(0)

