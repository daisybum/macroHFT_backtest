# Google Colab Execution Guide

This guide explains how to run the MacroHFT project on Google Colab with GPU support and Google Drive integration.

## 1. Setup Google Drive

1.  **Upload Project**: Upload the entire `macroHFT_backtest` folder to your Google Drive.
    *   Recommended Path: `My Drive/MacroHFT_Project`
    *   Ensure the folder structure is preserved.

## 2. Data Preparation (New!)

Before fine-tuning, you need to download and prepare the Bitcoin data (including Order Book LOB).

1.  Open **`setup_colab.ipynb`** or create a new notebook.
2.  Run the data pipeline scripts in order:
    ```python
    # 1. Download Raw Data (Warning: Large download)
    !python finetune/download_binance_raw.py
    
    # 2. Process LOB Snapshots
    !python finetune/process_lob.py
    
    # 3. Merge and Create Final Dataset
    !python finetune/merge_data.py
    ```
3.  This will create the training data in `MacroHFT/data/BTCUSDT/`.

## 3. Fine-tuning on Colab

1.  Open **`MacroHFT_Colab_Finetune.ipynb`** in Google Colab.
2.  **Check Path**: Verify the `PROJECT_PATH` variable in the first code cell matches where you uploaded the folder.
    ```python
    PROJECT_PATH = '/content/drive/MyDrive/MacroHFT_Project'
    ```
3.  **Run All Cells**:
    *   It will mount your Google Drive.
    *   Install necessary dependencies (`requirements_colab.txt`).
    *   Start the fine-tuning pipeline (`finetune/run_pipeline.py`).
    *   The pipeline uses GPU (`cuda:0`) automatically if available.
    *   Checkpoints are saved directly to your Drive folder (`MacroHFT/result/...`), so progress is saved even if Colab disconnects.

## 3. Backtesting on Colab

1.  Once fine-tuning is complete, open **`MacroHFT_Colab_Backtest.ipynb`**.
2.  **Check Path**: Verify `PROJECT_PATH` again.
3.  **Run All Cells**:
    *   It will run the backtest using the fine-tuned models.
    *   Results (Equity Curve, Metrics) will be displayed directly in the notebook.
    *   Result files are also saved to `backtest/results/` in your Drive.

## Troubleshooting

*   **Path Errors**: If you see `FileNotFoundError`, double-check the `PROJECT_PATH` in the notebook and ensure your Drive is mounted correctly.
*   **GPU Not Found**: Go to **Runtime -> Change runtime type** in Colab menu and ensure "T4 GPU" (or better) is selected.
*   **Resume Training**: If training stops (e.g., Colab timeout), simply restart the `MacroHFT_Colab_Finetune.ipynb`. The script is designed to verify existing checkpoints and resume from the last saved state.

