"""
narrative_logic.py - AI 季度敘事引擎
負責將診斷 JSON 傳給 Gemini API，生成專業球探風格的分析報告

來源: baseball_ai_report專案
"""

import os
import json
from typing import Dict
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

# 載入環境變數
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)


# 專業球探 Prompt 範本
SCOUT_SYSTEM_PROMPT = """你是棒球解說員，正在向「完全不懂棒球的朋友」解釋一位球員的表現。

【MLB 聯盟平均值參考（用來對比）】
- 擊球速度 (Launch Speed): 平均約 88-89 mph，90+ 算不錯，95+ 算頂尖
- 扎實擊球率 (Hard Hit Rate): 平均約 35%，40%+ 算優秀，50%+ 算頂尖
- 被三振率 (K Rate): 平均約 22%，低於 18% 算很有選球眼，高於 28% 代表容易被三振
- 保送率 (BB Rate): 平均約 8%，12%+ 代表很會選球
- wOBA: 平均約 0.320，0.350+ 算優秀，0.400+ 算明星等級
- BABIP: 平均約 0.300，高於這個可能有運氣成分

【術語解釋方式】
當你提到數據時，要這樣解釋：
- 「他的擊球速度平均 95 mph，比聯盟平均的 88 mph 快了不少，這代表他打出去的球又快又扎實」
- 「他的保送率只有 5%，低於聯盟平均的 8%，表示他比較急著出棒，選球可以再耐心一點」
- 「wOBA 0.420，聯盟平均大概 0.320，等於他每次上場打擊能創造的價值比一般球員高出 30% 左右」

【寫作規則】
1. 數據要跟聯盟平均比較，讓讀者知道「好在哪、差在哪」
2. 不要用籠統的說法（如「打擊很好」），要具體說「因為 XX 數據比平均高多少」
3. 不要誇張形容詞
4. 語氣輕鬆，像跟朋友聊天
5. 400-600 字"""


ANALYSIS_PROMPT_TEMPLATE = """幫我用「跟朋友聊天」的方式，介紹 **{player_name}** 在 **{season_year} 賽季** 的表現：

他的數據：
- 球季剛開始時（前10場）: {early_data}
- 球季中段（中間10場）: {mid_data}  
- 球季尾聲（最後10場）: {late_data}

整體趨勢：擊球速度{launch_speed_trend}、扎實擊球{hard_hit_trend}、被三振{k_rate_trend}

---

請用 400-600 字介紹這位球員：

1. **一句話總結**：直接說結論，這季他表現如何？（不要寒暄）

2. **有沒有進步**：從開季到結束是越來越好還是退步了？用數據說。

3. **打擊風格**：大砲型還是技巧型？優缺點是什麼？

4. **最重要的一點**：只記住一件事，那是什麼？

重要：
- 不要開場白（不要「嘿」「你知道嗎」這種）
- 直接進入主題
- 提到數據時，要跟聯盟平均比較
- 解釋術語要具體
- 語氣自然但不廢話"""


def configure_gemini(api_key: str = None):
    """
    設定 Gemini API
    優先使用傳入的 api_key，若無則使用環境變數
    """
    if not api_key:
        api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key or api_key == 'your_gemini_api_key_here':
        raise ValueError(
            "請提供 Gemini API Key！\n"
            "可以在 Sidebar 輸入，或在 .env 檔案中設定 OPENAI_API_KEY"
        )
    
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.5-flash')


def _translate_trend(trend: str) -> str:
    """將趨勢轉換為中文"""
    translations = {
        'increasing': '上升 📈',
        'decreasing': '下降 📉',
        'stable': '穩定 ➡️',
        'insufficient_data': '數據不足'
    }
    return translations.get(trend, trend)


def generate_season_narrative(diagnosis_json: Dict, season_year: str = "2024", api_key: str = None) -> str:
    """
    使用 Gemini API 生成季度診斷報告
    
    Args:
        diagnosis_json: 由 data_engine 產生的診斷 JSON
        season_year: 賽季年份 (預設 2024)
        api_key: Gemini API Key (可選，若無則使用環境變數)
    
    Returns:
        str: AI 生成的診斷報告
    """
    model = configure_gemini(api_key)
    
    analysis_segments = diagnosis_json['analysis_segments']
    summary = diagnosis_json['summary']
    
    prompt = ANALYSIS_PROMPT_TEMPLATE.format(
        season_year=season_year,
        player_name=diagnosis_json['player_name'],
        early_data=json.dumps(analysis_segments['early'], indent=2, ensure_ascii=False),
        mid_data=json.dumps(analysis_segments['mid'], indent=2, ensure_ascii=False),
        late_data=json.dumps(analysis_segments['late'], indent=2, ensure_ascii=False),
        launch_speed_trend=_translate_trend(summary['launch_speed_trend']),
        hard_hit_trend=_translate_trend(summary['hard_hit_trend']),
        k_rate_trend=_translate_trend(summary['k_rate_trend'])
    )
    
    try:
        response = model.generate_content(
            [SCOUT_SYSTEM_PROMPT, prompt],
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=6000,
            )
        )
        return response.text
    
    except Exception as e:
        return f"AI 生成報告時發生錯誤: {str(e)}"


def generate_quick_summary(diagnosis_json: Dict) -> str:
    """
    生成快速摘要 (不使用 AI，用於快速預覽)
    """
    player_name = diagnosis_json['player_name']
    segments = diagnosis_json['analysis_segments']
    summary = diagnosis_json['summary']
    
    early = segments['early']
    late = segments['late']
    
    # 計算變化
    speed_change = ""
    if early.get('avg_launch_speed') and late.get('avg_launch_speed'):
        diff = late['avg_launch_speed'] - early['avg_launch_speed']
        if diff > 1:
            speed_change = f"初速提升 {diff:.1f} mph ⬆️"
        elif diff < -1:
            speed_change = f"初速下降 {abs(diff):.1f} mph ⬇️"
        else:
            speed_change = "初速維持穩定 ➡️"
    
    fatigue_indicator = ""
    if early.get('hard_hit_rate') and late.get('hard_hit_rate'):
        hh_diff = late['hard_hit_rate'] - early['hard_hit_rate']
        if hh_diff < -5:
            fatigue_indicator = "⚠️ 可能有季末疲勞跡象"
        elif hh_diff > 5:
            fatigue_indicator = "💪 季末表現提升"
    
    return f"""
## 📊 {player_name} 快速摘要

**分析場次**: {summary['total_games_analyzed']} 場

### 趨勢觀察
- 🔥 初速趨勢: {_translate_trend(summary['launch_speed_trend'])}
- 💥 Hard Hit 趨勢: {_translate_trend(summary['hard_hit_trend'])}
- ❌ 三振率趨勢: {_translate_trend(summary['k_rate_trend'])}

### 重點發現
- {speed_change}
- {fatigue_indicator if fatigue_indicator else "表現穩定"}

### 關鍵數據
| 指標 | 季初 | 季末 |
|------|------|------|
| 平均初速 | {early.get('avg_launch_speed', 'N/A')} mph | {late.get('avg_launch_speed', 'N/A')} mph |
| Hard Hit% | {early.get('hard_hit_rate', 'N/A')}% | {late.get('hard_hit_rate', 'N/A')}% |
| wOBA | {early.get('woba', 'N/A')} | {late.get('woba', 'N/A')} |
| 全壘打 | {early.get('home_runs', 'N/A')} | {late.get('home_runs', 'N/A')} |
"""
