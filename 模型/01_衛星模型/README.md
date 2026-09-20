# 衛星與波段差 GRU 實驗

這是三個目標模型中的第一組 `SAT`。新的 `SAT` 定義與舊實驗不同，不可將舊 `SAT` 權重當成此模型使用。

目前訓練架構已與混合模型對齊：`GRU hidden state (64) → Dense (32) → ReLU → Dropout (0.3) → Dense (1 logit)`，並使用 binary Focal Loss：`alpha=0.91`、`gamma=1.0`，正負類成本比約為 10.11:1，且不另外使用 `pos_weight`。若要比較其他設定，可直接修改 `train_gru.py` 上方的相關常數。

## 輸入

- 資料：`visibility_satellite_2023_btd.parquet`，完整保留 clean 檔的原有欄位及逐筆內容。
- 原始衛星波段：`B01`–`B16`，共 16 欄。
- 波段差：`B07_minus_B14`、`B07_minus_B13`、`B11_minus_B15`、`B13_minus_B15`、`B14_minus_B15`、`B15_minus_B16`，共 6 欄，直接讀取預先計算的資料。
- 時間：臺灣時間的 `hour_sin`、`hour_cos`、`month_sin`、`month_cos`。
- 地理：`Latitude`、`Longitude`、`Elevation_m`。

總計 **29 個輸入特徵**。能見度不作為模型輸入，僅用於未來霧標籤及 persistence 參考基準。

## 訓練與切分

沿用原流程：ACOS 測站、18 筆間隔 10 分鐘的輸入、最後輸入後 60 分鐘的霧標籤（能見度 < 1 km）。
輸入首末時刻相距 170 分鐘。缺時與 UTC 月界切開，視窗不跨測站或切分。
驗證月份為 UTC 3、9 月，時間測試為 6、12 月；C48 留作空間測試且只使用訓練月份。
標準化參數只取訓練視窗涵蓋的列，門檻只在驗證集選擇。
單層 GRU、64 hidden units、固定種子 42、最多 30 epochs，驗證 loss 連續 5 輪未改善即停止。

```powershell
.\.venv\Scripts\python.exe 模型\01_衛星模型\create_sequences.py
.\.venv\Scripts\python.exe 模型\01_衛星模型\train_gru.py --mode SAT --epochs 30
```

預設輸出至 `模型/01_衛星模型/results/`，可使用 `--output-dir` 指定其他目錄。同一目錄重跑會覆寫同名檔案。

- `best_gru_fog_sat_model.pt`：最佳模型 state_dict。
- `gru_fog_sat_metadata.json`：特徵順序、標準化參數、門檻、資料與切分設定、模型尺寸及每輪 loss。
- `fog_feature_results_sat.csv`：時間與空間測試指標。
- `training_validation_loss_fog_sat.png`：訓練及驗證 loss 曲線。
- `sat_spatial_threshold_metrics.csv`：C48 在 0.00–1.00 共 101 個門檻下的完整指標。
- `sat_spatial_threshold_precision.png`、`sat_spatial_threshold_recall.png`、`sat_spatial_threshold_csi.png`：C48 門檻診斷曲線。

`ALL` 在本程式中執行已定義的衛星特徵組合。地面及混合模型使用各自資料夾內的獨立訓練程式。

## 本次訓練結果

> 下列數值是修改 Focal Loss 前留下的歷史結果；重新訓練後應以 `results/` 最新輸出更新本節。

本次使用單層 GRU 與 64 hidden units。最多 30 輪的訓練於第 11 輪提前停止，最佳權重為第 6 輪；驗證集以 F1 選出的門檻為 0.50。

| 測試集 | TP | FP | FN | Precision | Recall | CSI |
|---|---:|---:|---:|---:|---:|---:|
| 時間測試 | 199 | 418 | 238 | 32.25% | 45.54% | 23.27% |
| 空間測試 C48 | 4 | 42 | 90 | 8.70% | 4.26% | 2.94% |

時間測試仍漏掉 238/437 筆霧，C48 仍漏掉 90/94 筆霧；模型已可運作，但跨站泛化仍弱。
以上指標使用驗證集選出的門檻，未使用測試集調整門檻。
