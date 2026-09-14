# 多測站 GRU 霧偵測實驗說明

這份文件說明本專案目前的 **多測站、60 分鐘後霧偵測**流程與已得到的結果。原本的 [README.md](README.md) 是 Parquet 資料集的來源、欄位及背景說明，兩份文件用途不同。

## 目標與演進

每筆預測使用同一測站過去 18 筆、間隔 10 分鐘的資料（共 180 分鐘），判斷**最後一筆輸入之後 60 分鐘**，該站能見度是否 **< 1 km**。標籤為 1（霧）或 0（非霧）；模型輸出先經 sigmoid 轉成 0–1 分數，再依驗證集選出的門檻判定。這個分數尚未經過機率校準。

研究流程從單站能見度回歸開始。單站版本無法回答「其他測站學到的規律能否共用」，且整體回歸誤差容易掩蓋霧事件的漏報，因此先將目標改為 `<1 km` 分類。分類初期曾用類別權重降低漏報，但誤報偏多，於是加入 validation 門檻選擇並限制正類權重。接著改為**每站獨立建立序列，合併後訓練同一個普通 GRU**；沒有把不同測站在同一時間攤成一個輸入，也沒有使用 GConvGRU 或測站鄰接矩陣。最後固定測站與切分，依序比較衛星、衛星加時間、衛星加時間與地理資訊。

目前只使用 `ACOS` 的 26 站。資料集說明指出 ACOS 能見度儀的量測範圍通常到 75 km；先排除量測上限各異的 Road 站，讓第一版比較較單純。`C48` 完全不進訓練，作為留出站；其餘 25 站提供訓練與時間測試。這是**測站間泛化**實驗，尚不是「任意無測站位置」的預測系統。

## 輸入、標籤與資料切分

| 特徵組合 | 維度 | 每個時間步的輸入 |
|---|---:|---|
| `SAT` | 16 | `B01`–`B16` |
| `SAT_TIME` | 20 | 波段，加 `hour_sin/cos`、`month_sin/cos` |
| `SAT_TIME_GEO` | 23 | 上述特徵，加 `Latitude`、`Longitude`、`Elevation_m` |

時間特徵從 `DateTime_UTC0` 加 8 小時取得臺灣當地時刻，再以 sin/cos 編碼日週期與年週期。經緯度與高程對同一站是常數，但在多站共同訓練時會隨測站而變。`Station_ID` 只負責分組、切分及追蹤樣本，**不是數值輸入**。`Visibility_km` 只用於建立未來標籤和 persistence 參考方法；模型輸入不含未來能見度。資料集內已有的 `is_fog` 欄位目前未直接使用，而是依目標時刻的 `Visibility_km < 1.0` 重建標籤。

先依測站排序，再於時間差不是 10 分鐘或跨 **UTC 月份**時切成新區段；輸入與目標不會跨站、跨缺時或跨 UTC 月份。含非有限特徵值、非有限目標值或非有限當下能見度的序列會排除。輸入形狀是 `[batch, 18, 特徵數]`，標籤形狀是 `[batch, 1]`。時間特徵用臺灣時間，但月份切分仍依資料的 UTC 月份，兩者在月界附近可能不同。

| 集合 | 測站 | UTC 月份 | 用途 | 有效序列／霧標籤 |
|---|---|---|---|---:|
| `train` | 除 `C48` 外 25 站 | 1、2、4、5、7、8、10、11 月 | 更新模型參數、計算標準化統計量 | 140,290／1,992 |
| `val` | 同 25 站 | 3、9 月 | 選最佳 epoch 與判定門檻 | 37,515／489 |
| `temporal_test` | 同 25 站 | 6、12 月 | 評估看過的測站在其他月份的表現 | 38,744／437 |
| `spatial_test` | 僅 `C48` | 與 `train` 相同的月份 | 評估未見測站，避免同時改變測站與月份 | 5,782／94 |

標準化均值和標準差只用**訓練序列涵蓋的原始時間列**計算；重複出現在多個重疊視窗的列只計一次。同一組統計量套用到該特徵組合的所有集合。不同特徵組合各自重新計算其訓練統計量。序列會重疊，因此上表的「霧標籤數」是**序列數**，不是互相獨立的霧事件數。

## 主要程式逐段說明

### `create_sequences.py`：建立四個資料集合

1. **設定區**指定 Parquet、`ACOS`、留出站 `C48`、`LOOKBACK=18`、`HORIZON=6`、霧門檻、三組特徵、切分月份及 batch size。更換站或月份應在此處調整。
2. **`SequenceDataset`**保存各測站連續區段與序列索引，記錄二元標籤、當下能見度（供 persistence 使用）和站碼。`__getitem__` 才切出視窗並標準化，避免先把所有重疊視窗複製成龐大陣列。
3. **`build_datasets()` 前半**只讀所需欄位、選 ACOS、衍生臺灣時間的週期特徵，並逐站排序、切開缺時及月界。
4. **候選序列與切分**以同一站同一連續區段產生視窗；`current` 是輸入最後時刻，`target` 是 60 分鐘後。排除無效值後，按站碼和 UTC 月份分到 train、val、temporal test 或 spatial test。程式逐站印出有效序列及霧標籤數，方便發現某站幾乎沒有霧。
5. **標準化與檢查**只從訓練視窗使用到的列計算統計量，建立四個 `SequenceDataset`，並斷言 `C48` 不在訓練站內。直接執行此檔時，會檢查 `SAT_TIME_GEO` 的第一個 batch；**不會寫出資料檔**。

### `train_gru.py`：訓練、選門檻與比較

1. **模型與輔助函式**：`VisibilityGRU` 為單層、64 hidden units 的普通 GRU，取最後 hidden state 經線性層輸出 logit。`evaluate_loss` 算 weighted BCE；`predict_probabilities` 對 logit 套 sigmoid；指標函式計算 TP、FP、FN、TN、precision、POD/recall 與 CSI。
2. **`choose_threshold`**在 validation 上搜尋 0.05–0.95 的門檻（步長 0.01），選 F0.5 最大值。F0.5 比 F1 更重視 precision；**時間與空間測試資料都不參與門檻選擇**。
3. **`run_experiment` 資料與訓練區**為指定特徵組合重建相同切分，固定種子 42，用 Adam（學習率 0.001）及 `BCEWithLogitsLoss` 訓練。`pos_weight` 取 `0.5 × 非霧/霧`，但上限設為 5，避免多站霧比例低時過度鼓勵報霧。預設訓練 30 epoch，以 validation weighted BCE 最低的 epoch 保存模型狀態。
4. **評估區**先印 validation 在固定 0.50 與選出門檻下的指標，再用同一選出門檻評估 temporal test 和 `C48` spatial test。兩組測試各自列出 persistence：若輸入最後時刻的**實測**能見度 < 1 km，就猜未來仍有霧。Persistence 使用模型沒有的地面能見度，屬參考基準，不是相同輸入條件的公平模型對照。
5. **輸出區**為每組產生最佳權重、loss 曲線，`main()` 再把三組的兩種測試結果合併為一份 CSV。CSV 使用各組在 validation 選出的門檻；各組門檻可以不同。

`TP` 是抓對霧，`FP` 是誤報霧，`FN` 是漏報霧，`TN` 是正確判無霧。`precision = TP/(TP+FP)`，`POD/recall = TP/(TP+FN)`，`CSI = TP/(TP+FP+FN)`。霧很少，不能只看 TN 或整體正確率。Weighted BCE 是模型選擇用的損失，不等於正確率或霧偵測率。

## 目前完整比較結果

以下為三組各跑 30 epoch、各自以 validation 選門檻後的測試結果；完整 TP、FP、FN、TN 等數值見 `fog_feature_comparison.csv`。三組使用相同資料切分與隨機種子，只有輸入特徵不同。

| 特徵組合 | 選出門檻 | Temporal TP / FP / FN | Temporal precision / recall / CSI | `C48` TP / FP / FN |
|---|---:|---:|---:|---:|
| `SAT` | 0.32 | 119 / 1,139 / 318 | 9.5% / 27.2% / 7.6% | 5 / 152 / 89 |
| `SAT_TIME` | 0.51 | 41 / 239 / 396 | 14.6% / 9.4% / 6.1% | 0 / 1 / 94 |
| `SAT_TIME_GEO` | 0.78 | 91 / 114 / 346 | 44.4% / 20.8% / 16.5% | 0 / 0 / 94 |

單加時間特徵提高了時間測試的 precision，卻降低 recall 與 CSI；**時間加地理**的完整版本相對衛星基線提高了時間測試 precision 與 CSI，且誤報下降，代價是仍漏掉 346／437 筆霧。對未見的 `C48`，完整版本在選出門檻下沒有報出任何霧，因此**尚不能宣稱地理資訊改善跨站泛化**。結果只涵蓋一個留出站，且各站霧樣本分布差異很大（例如訓練集中 `C06`、`C13` 的霧標籤遠多於多數站），後續應增加留站驗證，避免只憑 `C48` 下結論。

## 執行方式與輸出檔案

在專案根目錄、已具備相依套件及 Parquet 資料後執行：

```powershell
uv run python create_sequences.py  # 僅檢查資料及第一個 batch
uv run python train_gru.py         # 三組特徵各訓練 30 epoch
```

短程流程測試可於 PowerShell 先設定 `$env:FOG_EPOCHS='5'`，再執行訓練；此設定只改訓練輪數，**短跑結果不應與上述 30-epoch 結果混用**。模型可用 CUDA 時會使用 GPU，否則使用 CPU。`pyproject.toml` 定義相依套件，`uv.lock` 鎖定安裝版本。

| 輸出 | 內容與注意事項 |
|---|---|
| `fog_feature_comparison.csv` | 三組特徵 × temporal/spatial 兩個測試，共六列；含特徵數、最佳 epoch、validation 選出的門檻及分類指標。這是目前主要結果表。 |
| `best_gru_fog_sat_model.pt`、`best_gru_fog_sat_time_model.pt`、`best_gru_fog_sat_time_geo_model.pt` | 各組 validation loss 最佳的 PyTorch `state_dict`。**目前只存模型權重，沒有一起存標準化參數、特徵順序與門檻，不能單靠 `.pt` 檔獨立部署。** |
| `training_validation_loss_fog_sat.png`、`training_validation_loss_fog_sat_time.png`、`training_validation_loss_fog_sat_time_geo.png` | 三組各自的 train／validation weighted BCE 曲線，虛線標出最佳 epoch；用來觀察訓練與驗證損失是否分離，不能單靠曲線判定霧偵測品質。 |

每次執行 `train_gru.py` 會**覆寫**上述同名輸出。`.gitignore` 已忽略 `.pt` 與 `training_validation_loss*.png`，因此這些檔案可能存在本機卻不會出現在 Git 變更中；比較 CSV 目前未被忽略。專案目錄裡其他 `best_gru_*.pt` 和 `training_validation_loss*.png` 是先前單站或舊版實驗產物，請依檔名區分。

## 專案其他檔案

| 檔案 | 用途及與本流程的關係 |
|---|---|
| `visibility_satellite_2023_clean.parquet` | 2023 年測站、衛星波段、能見度、地理欄位的原始配對資料；本流程唯一資料輸入，Git 已忽略。 |
| `README.md` | 原有資料集文件：來源、欄位、儀器上限、缺值及過往特徵研究；不是目前模型執行說明。 |
| `create_sequences.py` | 目前多站資料準備與 DataLoader 所需的 `Dataset` 定義。 |
| `train_gru.py` | 目前三組特徵的 GRU 訓練、門檻選擇與兩種測試。 |
| `fog_feature_comparison.csv` | 目前三組 30-epoch 比較結果。 |
| `compare_experiments.py` | 早期**單站、衛星輸入、能見度回歸**實驗：比較 10／30 分鐘序列、驗證月份及 persistence，以 RMSE／MAE 輸出 CSV；與目前二元多站流程分開。 |
| `comparison_results.csv`、`comparison_results_C06.csv` | 上述單站比較腳本留下的結果，不是目前霧分類比較表。 |
| `0910test.py` | 早期探索腳本：檢查各站筆數、`C19` 時間間隔與連續區段，估算可形成的序列數。 |
| `example_usage.py` | 資料集的 Polars/Pandas 讀取、特徵工程及月份驗證範例；不是目前 GRU 的執行入口。 |
| `main.py` | 專案建立時的簡單 Hello World 入口，與實驗無關。 |
| `pyproject.toml`、`uv.lock` | Python 專案套件需求與鎖定版本。 |
| `.gitignore` | 忽略虛擬環境、Python 快取、Parquet、模型權重及 loss 圖。 |
