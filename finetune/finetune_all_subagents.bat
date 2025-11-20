@echo off
REM Fine-tune all 6 sub-agents for Bitcoin
REM Windows batch script

echo ======================================================================
echo Fine-tuning All Sub-agents for Bitcoin
echo ======================================================================

cd /d "%~dp0.."
call .venv\Scripts\activate.bat

REM Create result directories
mkdir "MacroHFT\result\low_level\BTCUSDT\slope\1\label_1\seed_12345" 2>nul
mkdir "MacroHFT\result\low_level\BTCUSDT\slope\4\label_2\seed_12345" 2>nul
mkdir "MacroHFT\result\low_level\BTCUSDT\slope\0\label_3\seed_12345" 2>nul
mkdir "MacroHFT\result\low_level\BTCUSDT\vol\4\label_1\seed_12345" 2>nul
mkdir "MacroHFT\result\low_level\BTCUSDT\vol\1\label_2\seed_12345" 2>nul
mkdir "MacroHFT\result\low_level\BTCUSDT\vol\1\label_3\seed_12345" 2>nul

echo.
echo [1/6] Fine-tuning Slope Agent 1 (alpha=1, label_1)
echo ----------------------------------------------------------------------
python finetune\finetune_lowlevel.py ^
    --alpha 1 ^
    --clf slope ^
    --label label_1 ^
    --device cpu ^
    --epoch_number 3 ^
    --pretrained_model "MacroHFT/result/low_level/ETHUSDT/best_model/slope/1/best_model.pkl"

echo.
echo [2/6] Fine-tuning Slope Agent 2 (alpha=4, label_2)
echo ----------------------------------------------------------------------
python finetune\finetune_lowlevel.py ^
    --alpha 4 ^
    --clf slope ^
    --label label_2 ^
    --device cpu ^
    --epoch_number 3 ^
    --pretrained_model "MacroHFT/result/low_level/ETHUSDT/best_model/slope/2/best_model.pkl"

echo.
echo [3/6] Fine-tuning Slope Agent 3 (alpha=0, label_3)
echo ----------------------------------------------------------------------
python finetune\finetune_lowlevel.py ^
    --alpha 0 ^
    --clf slope ^
    --label label_3 ^
    --device cpu ^
    --epoch_number 3 ^
    --pretrained_model "MacroHFT/result/low_level/ETHUSDT/best_model/slope/3/best_model.pkl"

echo.
echo [4/6] Fine-tuning Vol Agent 1 (alpha=4, label_1)
echo ----------------------------------------------------------------------
python finetune\finetune_lowlevel.py ^
    --alpha 4 ^
    --clf vol ^
    --label label_1 ^
    --device cpu ^
    --epoch_number 3 ^
    --pretrained_model "MacroHFT/result/low_level/ETHUSDT/best_model/vol/1/best_model.pkl"

echo.
echo [5/6] Fine-tuning Vol Agent 2 (alpha=1, label_2)
echo ----------------------------------------------------------------------
python finetune\finetune_lowlevel.py ^
    --alpha 1 ^
    --clf vol ^
    --label label_2 ^
    --device cpu ^
    --epoch_number 3 ^
    --pretrained_model "MacroHFT/result/low_level/ETHUSDT/best_model/vol/2/best_model.pkl"

echo.
echo [6/6] Fine-tuning Vol Agent 3 (alpha=1, label_3)
echo ----------------------------------------------------------------------
python finetune\finetune_lowlevel.py ^
    --alpha 1 ^
    --clf vol ^
    --label label_3 ^
    --device cpu ^
    --epoch_number 3 ^
    --pretrained_model "MacroHFT/result/low_level/ETHUSDT/best_model/vol/3/best_model.pkl"

echo.
echo ======================================================================
echo All Sub-agents Fine-tuned!
echo ======================================================================
echo.
echo Fine-tuned models saved to:
echo   MacroHFT/result/low_level/BTCUSDT/slope/1/label_1/seed_12345/best_model.pkl
echo   MacroHFT/result/low_level/BTCUSDT/slope/4/label_2/seed_12345/best_model.pkl
echo   MacroHFT/result/low_level/BTCUSDT/slope/0/label_3/seed_12345/best_model.pkl
echo   MacroHFT/result/low_level/BTCUSDT/vol/4/label_1/seed_12345/best_model.pkl
echo   MacroHFT/result/low_level/BTCUSDT/vol/1/label_2/seed_12345/best_model.pkl
echo   MacroHFT/result/low_level/BTCUSDT/vol/1/label_3/seed_12345/best_model.pkl
echo.
pause

