# ⚾ 棒球 AI 統一分析平台

整合 **比賽分析** 與 **季度診斷** 功能的 MLB 棒球 AI 分析系統。

---

## 🌟 功能特色

### 📅 比賽分析
- 選擇日期和球隊，抓取當日比賽資料
- 自動提取關鍵時刻，生成中英雙語比賽戰報
- 分析特定球員當日對戰策略

### 📊 季度診斷
- 輸入球員姓名，分析整季表現變化
- 將賽季切割為 Early(前10場) / Mid(季中10場) / Late(最後10場)
- 計算 10 項關鍵指標 (初速、Hard Hit Rate、wOBA 等)
- 互動式 Plotly 圖表視覺化
- AI 生成專業球探風格診斷報告

---

## 🚀 快速開始

### 1. 安裝套件

```bash
cd baseball_ai_unified
pip install -r requirements.txt
```

### 2. 設定 API Key

複製 `.env.example` 為 `.env`，填入您的 Gemini API Key：

```bash
cp .env.example .env
```

編輯 `.env`：
```
OPENAI_API_KEY=your_actual_gemini_api_key_here
```

### 3. 執行應用程式

```bash
streamlit run app.py
```

---

## 📁 專案結構

```
baseball_ai_unified/
├── app.py                    # 主程式 (Streamlit)
├── requirements.txt          # 套件依賴
├── .env.example             # API Key 範例
├── README.md                # 本文件
└── src/
    ├── __init__.py
    ├── data_fetcher.py      # 比賽資料抓取
    ├── data_engine.py       # 季度分析引擎
    ├── narrative_engine.py  # 比賽敘事生成
    └── narrative_logic.py   # 季度報告生成
```

---

## 📋 使用說明

### 比賽分析

1. 切換到「📅 比賽分析」標籤頁
2. 選擇比賽日期和球隊
3. 點擊「生成比賽戰報」
4. 查看中英雙語戰報
5. 選擇打者進行策略分析

### 季度診斷

1. 切換到「📊 季度診斷」標籤頁
2. 輸入球員英文姓名 (例如: Ohtani, Shohei)
3. 選擇賽季
4. 點擊「開始分析」
5. 查看圖表和快速摘要
6. 點擊「生成 AI 報告」取得完整診斷

---

## ⚠️ 注意事項

1. **網路需求**: 需連線抓取 MLB Statcast 資料
2. **樣本限制**: 季度診斷需球員至少出賽 30 場
3. **API Key 安全**: `.env` 檔案請勿上傳至 Git

---

## 📚 技術堆疊

- **資料來源**: pybaseball (MLB Statcast API)
- **AI 引擎**: Google Gemini API
- **介面框架**: Streamlit
- **圖表視覺化**: Plotly
- **資料處理**: Pandas, NumPy

---

## 授權

本專案僅供學術研究使用。MLB 資料版權歸 Major League Baseball 所有。
