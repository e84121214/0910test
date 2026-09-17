# 衛星加地面氣象 GRU 實驗

這是第三組 `SAT_GROUND` 模型。訓練入口為 `train_combined_gru.py`，序列組合由 `create_combined_sequences.py` 負責；前兩種模型的訓練程式與結果不會被本模型覆寫。

## 輸入特徵

總計 71 個特徵：

- 22 個衛星特徵：`B01`–`B16` 及 6 個預先計算的波段差。
- 49 個地面模型特徵：核心氣象聚合、有效鄰站數、缺值遮罩、資料年齡、鄰站空間摘要、時間與能見度儀地理特徵。

時間、經緯度與海拔只使用地面特徵組中的一份，沒有重複輸入。模型不使用氣象站能見度 `VS01`，也不使用能見度儀當下的 `Visibility_km`。

地面氣象空間聚合沿用第二種模型：排除 9 個暫不採用站，對每個要素及時刻依有效鄰站重新正規化 `exp(-distance_km / 5)`。海拔作為模型特徵，不參與權重計算。

## 序列一致性

混合資料層組合前兩個已驗證的 Dataset，並逐個切分檢查：

- 序列索引完全一致。
- 標籤與測站順序完全一致。
- persistence 基準完全一致。
- 衛星與地面資料設定的 lookback、horizon、月份切分及留出站一致。

| 切分 | 序列數 | 霧標籤 | 測站數 |
|---|---:|---:|---:|
| Train | 140,290 | 1,992 | 25 |
| Validation | 37,515 | 489 | 25 |
| Temporal test | 38,744 | 437 | 25 |
| Spatial test C48 | 5,782 | 94 | 1 |

## 執行與輸出

從專案根目錄執行：

```powershell
.\.venv\Scripts\python.exe 模型\03_衛星加地面模型\create_combined_sequences.py
.\.venv\Scripts\python.exe 模型\03_衛星加地面模型\train_combined_gru.py --epochs 30
```

輸出位於本資料夾的 `results/`：

- `best_combined_satellite_ground_gru_model.pt`
- `combined_gru_metadata.json`
- `combined_results.csv`
- `training_validation_loss_combined.png`
- `training.log`
- `combined_spatial_threshold_metrics.csv`：C48 在 0.00–1.00 共 101 個門檻下的完整指標。
- `combined_spatial_threshold_precision.png`、`combined_spatial_threshold_recall.png`、`combined_spatial_threshold_csi.png`：C48 門檻診斷曲線。

## 本次結果

模型使用單層 GRU、64 hidden units、種子 42。訓練最多 30 輪，於第 8 輪提前停止，最佳權重為第 3 輪；validation 以 F1 選出的門檻為 0.61。

| 測試集 | TP | FP | FN | Precision | Recall | CSI |
|---|---:|---:|---:|---:|---:|---:|
| Temporal test | 283 | 572 | 154 | 33.10% | 64.76% | 28.05% |
| Spatial test C48 | 11 | 92 | 83 | 10.68% | 11.70% | 5.91% |

使用 F1 門檻後，混合模型的 temporal Recall 64.76% 高於純地面模型的 58.58%，但 temporal CSI 28.05% 低於純地面的 32.20%。C48 只偵測到 11／94 筆霧，跨站泛化仍然不足。兩個 test 均使用 validation 選出的門檻，未參與門檻調整。
