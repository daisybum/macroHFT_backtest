"""
MacroHFT Bitcoin Fine-tuning Pipeline
Manages the execution of low-level and high-level fine-tuning.
Supports resuming from where it left off.
"""

import subprocess
import os
import sys
import time
import torch

PYTHON_EXEC = sys.executable
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"[Pipeline] Using device: {DEVICE}", flush=True)

def run_command(cmd, description):
    print("\n" + "="*80, flush=True)
    print(f"RUNNING: {description}", flush=True)
    print(f"COMMAND: {cmd}", flush=True)
    print("="*80, flush=True)
    
    try:
        # Use subprocess.Popen to stream output to console
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip(), flush=True)
        
        ret_code = process.poll()
        
        if ret_code != 0:
            print(f"\n[ERROR] Task failed with exit code {ret_code}", flush=True)
            return False
    except KeyboardInterrupt:
        print("\n[INFO] Pipeline interrupted by user.", flush=True)
        return False
        
    return True

def main():
    print("Starting MacroHFT Bitcoin Fine-tuning Pipeline...", flush=True)
    print("This pipeline can be interrupted (Ctrl+C) and resumed later.", flush=True)
    
    # 1. Low-level Sub-agents
    subagents = [
        # (clf, alpha, label, pretrained_path)
        ('slope', 1, 'label_1', "MacroHFT/result/low_level/ETHUSDT/best_model/slope/1/best_model.pkl"),
        ('slope', 4, 'label_2', "MacroHFT/result/low_level/ETHUSDT/best_model/slope/2/best_model.pkl"),
        ('slope', 0, 'label_3', "MacroHFT/result/low_level/ETHUSDT/best_model/slope/3/best_model.pkl"),
        ('vol', 4, 'label_1', "MacroHFT/result/low_level/ETHUSDT/best_model/vol/1/best_model.pkl"),
        ('vol', 1, 'label_2', "MacroHFT/result/low_level/ETHUSDT/best_model/vol/2/best_model.pkl"),
        ('vol', 1, 'label_3', "MacroHFT/result/low_level/ETHUSDT/best_model/vol/3/best_model.pkl"),
    ]
    
    for clf, alpha, label, pretrained in subagents:
        print(f"\n[Pipeline] Checking Sub-agent: {clf}, alpha={alpha}, {label}", flush=True)
        
        # Check if already completed (optional optimization)
        # But we rely on script's internal resume check (or it will just run fast if epochs done?)
        # Actually the script will resume or restart. If we want to skip completed agents entirely we need a check here.
        # For now let's rely on script resume. But if epoch number is reached, script should exit?
        # Our script loops `range(start_epoch, epoch_number)`. If start_epoch >= epoch_number, it does nothing.
        
        cmd = (
            f'"{PYTHON_EXEC}" finetune/finetune_lowlevel.py '
            f"--alpha {alpha} "
            f"--clf {clf} "
            f"--label {label} "
            f"--device {DEVICE} "
            f"--epoch_number 3 "
            f"--pretrained_model \"{pretrained}\""
        )
        
        success = run_command(cmd, f"Fine-tuning Sub-agent ({clf}, {label})")
        if not success:
            print("[Pipeline] Stopping pipeline due to error or interruption.", flush=True)
            sys.exit(1)
            
    # 2. High-level Agent
    print("\n[Pipeline] Starting High-level Agent Fine-tuning", flush=True)
    cmd = (
        f'"{PYTHON_EXEC}" finetune/finetune_highlevel.py '
        f"--device {DEVICE} "
        f"--epoch_number 3 "
        f"--buffer_size 100000"
    )
    success = run_command(cmd, "Fine-tuning Hyperagent")
    if not success:
        print("[Pipeline] Stopping pipeline.", flush=True)
        sys.exit(1)
        
    print("\n" + "="*80, flush=True)
    print("PIPELINE COMPLETED SUCCESSFULLY!", flush=True)
    print("="*80, flush=True)

if __name__ == "__main__":
    main()
