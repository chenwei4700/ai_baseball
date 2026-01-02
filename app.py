"""
app.py - 棒球 AI 統一分析平台
整合比賽分析和季度診斷兩大功能

合併自: ai棒球 + baseball_ai_report
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date
import os
from pathlib import Path
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

# 導入自定義模組
from src.data_fetcher import get_game_data, get_mlb_teams, get_batters_from_game, get_player_history
from src.data_engine import get_full_analysis
from src.narrative_engine import extract_key_moments, generate_game_narrative, generate_player_analysis
from src.narrative_logic import generate_season_narrative, generate_quick_summary

# 頁面設定
st.set_page_config(
    page_title="⚾ 棒球 AI 統一分析平台",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定義 CSS - 現代化美化設計
st.markdown("""
<style>
    /* 全局樣式 */
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;500;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    
    /* 主標題 */
    .main-header {
        font-family: 'Noto Sans TC', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00d2ff, #3a7bd5, #00d2ff);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        animation: gradient 3s ease infinite;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* 副標題 */
    .sub-header {
        font-family: 'Noto Sans TC', sans-serif;
        font-size: 1.1rem;
        color: #a0aec0;
        text-align: center;
        margin-bottom: 2rem;
        letter-spacing: 2px;
    }
    
    /* 側邊欄美化 */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #e2e8f0;
    }
    
    /* 玻璃態容器 */
    .glass-container {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(255,255,255,0.1);
        margin: 1rem 0;
    }
    
    /* 按鈕美化 */
    .stButton > button {
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,210,255,0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,210,255,0.5);
    }
    
    /* 輸入框美化 */
    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.95) !important;
        border: 1px solid rgba(255,255,255,0.3);
        border-radius: 10px;
        color: #1a1a2e !important;
    }
    
    .stSelectbox > div > div {
        background: white !important;
        border-radius: 10px;
    }
    
    .stSelectbox div[data-baseweb="select"] > div {
        color: #000000 !important;
        background: white !important;
    }
    
    .stSelectbox div[data-baseweb="select"] span {
        color: #000000 !important;
    }
    
    /* 下拉選單選項 */
    [data-baseweb="popover"] {
        background: white !important;
    }
    
    [data-baseweb="menu"] li {
        color: #000000 !important;
        background: white !important;
    }
    
    /* 標籤頁美化 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #a0aec0;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        color: white;
    }
    
    /* 標題樣式 */
    h1, h2, h3 {
        color: #ffffff !important;
        font-family: 'Noto Sans TC', sans-serif;
    }
    
    /* Markdown 文字 */
    .stMarkdown {
        color: #f0f4f8 !important;
    }
    
    .stMarkdown p, .stMarkdown li, .stMarkdown span {
        color: #f0f4f8 !important;
    }
    
    /* 所有文字提高亮度 */
    p, span, div, li {
        color: #e8edf3;
    }
    
    label {
        color: #ffffff !important;
    }
    
    /* 隱藏 Streamlit 品牌 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


def get_api_key():
    """取得 API Key，優先使用環境變數，否則從 Sidebar 取得"""
    env_key = os.getenv('OPENAI_API_KEY')
    if env_key and env_key != 'your_gemini_api_key_here':
        return env_key
    return st.session_state.get('api_key', '')


def main():
    # 標題
    st.markdown('<p class="main-header">⚾ 棒球 AI 統一分析平台</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">整合比賽分析與季度診斷 | MLB Statcast 數據 + AI 智能分析</p>', unsafe_allow_html=True)
    
    # 側邊欄
    with st.sidebar:
        st.header("⚙️ 設定")
        
        with st.expander("📖 使用說明"):
            st.markdown("""
            **比賽分析** 📅
            - 選擇日期和球隊
            - 生成比賽戰報 (中英雙語)
            - 分析球員當日策略
            
            **季度診斷** 📊
            - 輸入球員姓名
            - 分析季初/季中/季末表現
            - 互動式圖表視覺化
            - AI 生成專業診斷報告
            
            **注意**: 球員需至少出賽 30 場才能進行季度分析
            """)
    
    # 主要內容區 - 兩個功能標籤頁
    tab1, tab2 = st.tabs(["📅 比賽分析", "📊 季度診斷"])
    
    # ==================== 比賽分析 ====================
    with tab1:
        game_analysis_ui(get_api_key())
    
    # ==================== 季度診斷 ====================
    with tab2:
        season_diagnosis_ui(get_api_key())


def game_analysis_ui(api_key):
    """比賽分析 UI"""
    st.header("📅 比賽分析")
    st.markdown("選擇日期和球隊，生成比賽戰報並分析球員表現")
    
    # Session State 初始化
    if 'game_df' not in st.session_state:
        st.session_state.game_df = None
    if 'batters' not in st.session_state:
        st.session_state.batters = {}
    if 'narrative_result' not in st.session_state:
        st.session_state.narrative_result = None
    if 'moments_data' not in st.session_state:
        st.session_state.moments_data = None
    if 'player_analysis_result' not in st.session_state:
        st.session_state.player_analysis_result = None
    
    # 輸入區
    col1, col2 = st.columns(2)
    
    with col1:
        game_date = st.date_input("比賽日期", date.today())
        
    with col2:
        teams = get_mlb_teams()
        team_name = st.selectbox("選擇球隊", list(teams.values()))
        team_code = [k for k, v in teams.items() if v == team_name][0]
    
    if st.button("🎯 生成比賽戰報", key="generate_game_report"):
        if not api_key:
            st.error("請在 .env 檔案中設定 OPENAI_API_KEY")
        else:
            with st.spinner("抓取資料並分析比賽..."):
                df = get_game_data(str(game_date), team_code)
                st.session_state.game_df = df
                
                if df is None:
                    st.error(f"找不到 {team_name} 在 {game_date} 的比賽資料")
                    st.session_state.narrative_result = None
                    st.session_state.moments_data = None
                else:
                    st.session_state.batters = get_batters_from_game(df)
                    moments, metadata = extract_key_moments(df)
                    
                    if not moments:
                        st.warning("未找到關鍵時刻 (可能資料不完整)")
                        st.session_state.narrative_result = None
                        st.session_state.moments_data = None
                    else:
                        narrative_data = generate_game_narrative(moments, metadata, api_key)
                        st.session_state.narrative_result = narrative_data
                        st.session_state.moments_data = moments
    
    # 顯示比賽戰報
    if st.session_state.narrative_result:
        st.success("戰報生成成功！")
        st.markdown("### 📝 比賽戰報")
        
        narrative_data = st.session_state.narrative_result
        if isinstance(narrative_data, dict):
            lang_tab1, lang_tab2 = st.tabs(["English", "中文"])
            with lang_tab1:
                st.write(narrative_data.get("english", ""))
            with lang_tab2:
                st.write(narrative_data.get("chinese", ""))
        else:
            st.write(narrative_data)
        
        if st.session_state.moments_data:
            with st.expander("查看關鍵時刻數據"):
                st.json(st.session_state.moments_data)
    
    # 球員策略分析區
    st.markdown("---")
    st.markdown("### 🎯 球員策略分析")
    st.markdown("選擇比賽中的打者，分析投手對他的投球策略")
    
    if st.session_state.batters:
        batter_options = {v: k for k, v in st.session_state.batters.items()}
        selected_batter_name = st.selectbox("選擇打者", list(batter_options.keys()))
        selected_batter_id = batter_options[selected_batter_name]
        
        if st.button("🔍 分析策略", key="analyze_player_strategy"):
            if not api_key:
                st.error("請在 .env 檔案中設定 OPENAI_API_KEY")
            elif st.session_state.game_df is None:
                st.error("請先生成比賽戰報")
            else:
                with st.spinner(f"分析 {selected_batter_name} 的對戰策略..."):
                    game_df = st.session_state.game_df
                    batter_game_df = game_df[game_df['batter'] == selected_batter_id]
                    history_df = get_player_history(selected_batter_id, str(game_date))
                    analysis = generate_player_analysis(batter_game_df, history_df, selected_batter_name, api_key)
                    st.session_state.player_analysis_result = analysis
                    st.session_state.analyzed_player_name = selected_batter_name
        
        if st.session_state.player_analysis_result:
            st.success("分析完成！")
            st.markdown(f"### 📊 策略分析: {st.session_state.get('analyzed_player_name', '')}")
            
            analysis = st.session_state.player_analysis_result
            if isinstance(analysis, dict):
                lang_tab_en, lang_tab_zh = st.tabs(["English", "中文"])
                with lang_tab_en:
                    st.write(analysis.get("english", ""))
                with lang_tab_zh:
                    st.write(analysis.get("chinese", ""))
            else:
                st.write(analysis)
    else:
        st.info("請先生成比賽戰報以查看可分析的打者")


def season_diagnosis_ui(api_key):
    """季度診斷 UI"""
    st.header("📊 季度診斷報告")
    st.markdown("分析球員整個賽季的表現變化 (前10場 / 季中10場 / 最後10場)")
    
    # 賽季對應的日期範圍
    SEASON_DATES = {
        "2024 賽季": ("2024-03-20", "2024-10-31"),
        "2023 賽季": ("2023-03-30", "2023-10-01"),
        "2022 賽季": ("2022-04-07", "2022-10-05"),
        "2021 賽季": ("2021-04-01", "2021-10-03"),
        "2020 賽季": ("2020-07-23", "2020-09-27"),
    }
    
    # 輸入區
    col1, col2, col3 = st.columns(3)
    
    with col1:
        last_name = st.text_input("姓氏 (Last Name)", value="Ohtani", help="例如: Judge, Ohtani")
    with col2:
        first_name = st.text_input("名字 (First Name)", value="Shohei", help="例如: Aaron, Shohei")
    with col3:
        selected_season = st.selectbox("選擇賽季", options=list(SEASON_DATES.keys()), index=0)
    
    start_date, end_date = SEASON_DATES[selected_season]
    st.caption(f"📆 資料範圍: {start_date} ~ {end_date}")
    
    if st.button("🚀 開始分析", key="start_season_analysis"):
        with st.spinner(f"正在分析 {first_name} {last_name} 的數據..."):
            try:
                season_year = selected_season.split()[0]
                diagnosis = get_full_analysis(last_name, first_name, start_date, end_date)
                
                st.session_state['diagnosis'] = diagnosis
                st.session_state['player_name'] = f"{first_name} {last_name}"
                st.session_state['season'] = season_year
                
                st.success(f"✅ 成功分析 {first_name} {last_name}！")
                
            except ValueError as e:
                st.error(f"❌ 分析失敗: {str(e)}")
                return
            except Exception as e:
                st.error(f"❌ 發生錯誤: {str(e)}")
                return
    
    # 顯示分析結果
    if 'diagnosis' in st.session_state:
        diagnosis = st.session_state['diagnosis']
        
        # 快速摘要
        st.markdown("---")
        quick_summary = generate_quick_summary(diagnosis)
        st.markdown(quick_summary)
        
        # 圖表區
        st.markdown("---")
        st.header("📈 數據視覺化")
        
        segments = diagnosis['analysis_segments']
        
        chart_tab1, chart_tab2, chart_tab3 = st.tabs(["🔥 物理指標", "📊 表現指標", "💯 進階指標"])
        
        with chart_tab1:
            col1, col2 = st.columns(2)
            with col1:
                fig_speed = create_bar_chart(segments, 'avg_launch_speed', '平均初速 (mph)', '三段時期初速對比')
                st.plotly_chart(fig_speed, use_container_width=True)
            with col2:
                fig_hh = create_bar_chart(segments, 'hard_hit_rate', 'Hard Hit Rate (%)', '三段時期 Hard Hit 對比')
                st.plotly_chart(fig_hh, use_container_width=True)
            
            fig_trend = create_trend_chart(segments)
            st.plotly_chart(fig_trend, use_container_width=True)
        
        with chart_tab2:
            col1, col2, col3 = st.columns(3)
            with col1:
                fig_hr = create_bar_chart(segments, 'home_runs', '全壘打數', '全壘打對比')
                st.plotly_chart(fig_hr, use_container_width=True)
            with col2:
                fig_k = create_bar_chart(segments, 'k_rate', '三振率 (%)', '三振率對比')
                st.plotly_chart(fig_k, use_container_width=True)
            with col3:
                fig_bb = create_bar_chart(segments, 'bb_rate', '保送率 (%)', '保送率對比')
                st.plotly_chart(fig_bb, use_container_width=True)
        
        with chart_tab3:
            col1, col2 = st.columns(2)
            with col1:
                fig_woba = create_bar_chart(segments, 'woba', 'wOBA', 'wOBA 對比')
                st.plotly_chart(fig_woba, use_container_width=True)
            with col2:
                fig_babip = create_bar_chart(segments, 'babip', 'BABIP', 'BABIP 對比')
                st.plotly_chart(fig_babip, use_container_width=True)
            
            fig_radar = create_radar_chart(segments)
            st.plotly_chart(fig_radar, use_container_width=True)
        
        # AI 診斷報告
        st.markdown("---")
        st.header("🤖 AI 專業診斷報告")
        
        if st.button("📝 生成 AI 報告", key="generate_ai_report"):
            if not api_key:
                st.error("請在 .env 檔案中設定 OPENAI_API_KEY")
            else:
                with st.spinner("AI 正在分析數據並撰寫報告..."):
                    try:
                        season = st.session_state.get('season', '2024')
                        ai_report = generate_season_narrative(diagnosis, season, api_key)
                        st.session_state['ai_report'] = ai_report
                    except Exception as e:
                        st.error(f"AI 報告生成失敗: {str(e)}")
        
        if 'ai_report' in st.session_state:
            st.markdown(st.session_state['ai_report'])
        
        # 原始數據
        st.markdown("---")
        with st.expander("📋 查看完整診斷 JSON"):
            st.json(diagnosis)


def create_bar_chart(segments: dict, metric: str, y_label: str, title: str):
    """建立長條圖"""
    data = {
        '時期': ['Early\n(前10場)', 'Mid\n(季中10場)', 'Late\n(最後10場)'],
        '數值': [
            segments['early'].get(metric, 0) or 0,
            segments['mid'].get(metric, 0) or 0,
            segments['late'].get(metric, 0) or 0
        ]
    }
    
    fig = px.bar(
        data,
        x='時期',
        y='數值',
        title=title,
        color='時期',
        color_discrete_sequence=['#00d2ff', '#a855f7', '#f43f5e']
    )
    
    fig.update_layout(
        yaxis_title=y_label,
        showlegend=False,
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0'),
        title_font=dict(color='#e2e8f0', size=16),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='#a0aec0')),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='#a0aec0'))
    )
    
    return fig


def create_trend_chart(segments: dict):
    """建立趨勢線圖"""
    metrics = ['avg_launch_speed', 'hard_hit_rate', 'whiff_rate']
    labels = ['平均初速 (mph)', 'Hard Hit Rate (%)', 'Whiff Rate (%)']
    
    fig = make_subplots(rows=1, cols=1)
    
    periods = ['Early', 'Mid', 'Late']
    colors = ['#3498db', '#e74c3c', '#f39c12']
    
    for metric, label, color in zip(metrics, labels, colors):
        values = [
            segments['early'].get(metric, 0) or 0,
            segments['mid'].get(metric, 0) or 0,
            segments['late'].get(metric, 0) or 0
        ]
        
        max_val = max(values) if max(values) > 0 else 1
        normalized = [v / max_val * 100 for v in values]
        
        fig.add_trace(go.Scatter(
            x=periods,
            y=normalized,
            mode='lines+markers',
            name=label,
            line=dict(color=color, width=3),
            marker=dict(size=10)
        ))
    
    fig.update_layout(
        title='關鍵指標趨勢變化 (標準化)',
        xaxis_title='賽季階段',
        yaxis_title='相對表現 (%)',
        height=400,
        hovermode='x unified',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0'),
        title_font=dict(color='#e2e8f0', size=16),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='#a0aec0')),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='#a0aec0')),
        legend=dict(font=dict(color='#e2e8f0'))
    )
    
    return fig


def create_radar_chart(segments: dict):
    """建立雷達圖"""
    categories = ['初速', 'Hard Hit', '選球', '抗三振', '長打力']
    
    def normalize(val, min_val, max_val):
        if val is None:
            return 50
        return min(100, max(0, (val - min_val) / (max_val - min_val) * 100))
    
    def get_scores(segment):
        return [
            normalize(segment.get('avg_launch_speed'), 80, 100),
            normalize(segment.get('hard_hit_rate'), 20, 60),
            normalize(segment.get('bb_rate'), 0, 15),
            100 - normalize(segment.get('k_rate'), 10, 35),
            normalize(segment.get('max_hit_distance'), 350, 450)
        ]
    
    fig = go.Figure()
    
    colors = [
        ('rgba(0,210,255,0.3)', 'rgb(0,210,255)'),
        ('rgba(168,85,247,0.3)', 'rgb(168,85,247)'),
        ('rgba(244,63,94,0.3)', 'rgb(244,63,94)')
    ]
    
    for (segment_key, segment_data), (fill_color, line_color), name in zip(
        segments.items(),
        colors,
        ['Early (前10場)', 'Mid (季中10場)', 'Late (最後10場)']
    ):
        scores = get_scores(segment_data)
        scores.append(scores[0])
        
        fig.add_trace(go.Scatterpolar(
            r=scores,
            theta=categories + [categories[0]],
            fill='toself',
            fillcolor=fill_color,
            line=dict(color=line_color, width=2),
            name=name
        ))
    
    fig.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor='rgba(255,255,255,0.1)',
                tickfont=dict(color='#a0aec0')
            ),
            angularaxis=dict(
                gridcolor='rgba(255,255,255,0.1)',
                tickfont=dict(color='#e2e8f0')
            )
        ),
        title='綜合能力雷達圖',
        height=500,
        showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0'),
        title_font=dict(color='#e2e8f0', size=16),
        legend=dict(font=dict(color='#e2e8f0'))
    )
    
    return fig


if __name__ == "__main__":
    main()
