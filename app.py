import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import anthropic  # 使用 Anthropic Claude SDK 取代 OpenAI
from datetime import datetime
import traceback

# 頁面配置
st.set_page_config(
    page_title="【Code Gym】DCF估值教育系統",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 主標題
st.title("【Code Gym】DCF估值教育系統", anchor=False)
st.markdown("---")

def format_large_number(num):
    """格式化大數字顯示"""
    if num >= 1e12:
        return f"{num/1e12:.2f}兆"
    elif num >= 1e9:
        return f"{num/1e9:.2f}億"
    elif num >= 1e6:
        return f"{num/1e6:.2f}百萬"
    else:
        return f"{num:.2f}"

def get_company_default_params(ticker, api_key):
    """第一階段：獲取公司預設參數值"""
    try:
        # 使用新的 Custom DCF API 端點
        base_url = "https://financialmodelingprep.com/stable/custom-discounted-cash-flow"
        
        # 只傳送股票代碼，獲取預設參數
        params = {
            'symbol': ticker,
            'apikey': api_key
        }
        
        response = requests.get(base_url, params=params)
        if response.status_code != 200:
            st.error(f"API回應狀態碼: {response.status_code}")
            st.error(f"API回應內容: {response.text}")
            raise Exception(f"API請求失敗: {response.status_code} - {response.text}")
        
        data = response.json()
        if not data:
            raise Exception("API回傳空數據")
            
        # 根據新的JSON範例，API回傳一個陣列，取第一個（最新年份）
        company_data = data[0] if isinstance(data, list) else data
        
        # 提取預設參數值 - 確保包含所有API支援的參數
        default_params = {
            # 基礎成長參數 - API返回的是百分比，我們轉為小數供界面使用
            'revenueGrowthPct': company_data.get('revenuePercentage', 10.07) / 100,
            'ebitdaPct': company_data.get('ebitdaPercentage', 32.84) / 100,
            'ebitPct': company_data.get('ebitPercentage', 29.67) / 100,
            
            # 資本支出和折舊
            'capitalExpenditurePct': abs(company_data.get('capitalExpenditurePercentage', -2.74)) / 100,
            'depreciationAndAmortizationPct': company_data.get('depreciationPercentage', 3.17) / 100,
            
            # 現金流參數
            'operatingCashFlowPct': company_data.get('ufcf', 0) / company_data.get('revenue', 1) if company_data.get('revenue', 0) > 0 else 0.15,
            
            # 資產負債表相關（從API獲取實際百分比）
            'cashAndShortTermInvestmentsPct': company_data.get('totalCashPercentage', 19.05) / 100,
            'receivablesPct': company_data.get('receivablesPercentage', 15.2) / 100,
            'inventoriesPct': company_data.get('inventoriesPercentage', 1.61) / 100,
            'payablePct': company_data.get('payablePercentage', 16.12) / 100,
            
            # 稅率和成長率
            'taxRate': company_data.get('taxRate', 24.09) / 100,
            'longTermGrowthRate': company_data.get('longTermGrowthRate', 4.0) / 100,
            
            # 資本成本參數（關鍵添加！）
            'costOfDebt': company_data.get('costofDebt', 3.85) / 100,
            'costOfEquity': company_data.get('costOfEquity', 9.35) / 100,
            'riskFreeRate': company_data.get('riskFreeRate', 3.85) / 100,
            'marketRiskPremium': company_data.get('marketRiskPremium', 4.72) / 100,
            'beta': company_data.get('beta', 1.165),
            
            # 計算得出的WACC
            'wacc': company_data.get('wacc', 9.15) / 100,
            
            # 公司基本資訊
            'companyName': company_data.get('symbol', ticker),
            'currentPrice': company_data.get('price', 231.59)
        }
        
        return default_params, company_data
        
    except Exception as e:
        raise Exception(f"載入公司數據錯誤: {str(e)}")

def get_dcf_calculation_with_params(ticker, api_key, parameters):
    """第二階段：使用調整後的參數計算DCF"""
    try:
        # 使用新的 Custom DCF API 端點
        base_url = "https://financialmodelingprep.com/stable/custom-discounted-cash-flow"
        
        # 根據新API的參數格式構建參數
        params = {
            'symbol': ticker,
            'apikey': api_key
        }
        
        # 修正：根據API範例，部分參數需要百分比格式，部分需要小數格式
        param_mapping = {
            'revenueGrowthPct': parameters.get('revenueGrowthPct') if parameters.get('revenueGrowthPct') is not None else None,
            'ebitdaPct': parameters.get('ebitdaPct') if parameters.get('ebitdaPct') is not None else None,
            # 資本支出處理 - API返回負值，我們需要轉為正值
            'capitalExpenditurePct': abs(parameters.get('capitalExpenditurePct')) if parameters.get('capitalExpenditurePct') is not None else None,
            # 以下參數需要百分比格式（如4、3.64、9.511）
            'riskFreeRate': parameters.get('riskFreeRate') * 100 if parameters.get('riskFreeRate') is not None else None,
            'marketRiskPremium': parameters.get('marketRiskPremium') * 100 if parameters.get('marketRiskPremium') is not None else None,
            'costOfDebt': parameters.get('costOfDebt') * 100 if parameters.get('costOfDebt') is not None else None,
            'costOfEquity': parameters.get('costOfEquity') * 100 if parameters.get('costOfEquity') is not None else None,
            'longTermGrowthRate': parameters.get('longTermGrowthRate') * 100 if parameters.get('longTermGrowthRate') is not None else None,
            'cashAndShortTermInvestmentsPct': parameters.get('cashAndShortTermInvestmentsPct') if parameters.get('cashAndShortTermInvestmentsPct') is not None else None,
            'receivablesPct': parameters.get('receivablesPct') if parameters.get('receivablesPct') is not None else None,
            'inventoriesPct': parameters.get('inventoriesPct') if parameters.get('inventoriesPct') is not None else None,
            'payablePct': parameters.get('payablePct') if parameters.get('payablePct') is not None else None,
            'sellingGeneralAndAdministrativeExpensesPct': parameters.get('sellingGeneralAndAdministrativeExpensesPct') if parameters.get('sellingGeneralAndAdministrativeExpensesPct') is not None else None,
            'beta': parameters.get('beta'),  # Beta不需要轉換
            'taxRate': parameters.get('taxRate') if parameters.get('taxRate') is not None else None,
            'operatingCashFlowPct': parameters.get('operatingCashFlowPct') if parameters.get('operatingCashFlowPct') is not None else None,
            'depreciationAndAmortizationPct': parameters.get('depreciationAndAmortizationPct') if parameters.get('depreciationAndAmortizationPct') is not None else None,
            'ebitPct': parameters.get('ebitPct') if parameters.get('ebitPct') is not None else None
        }
        
        # 只加入非None的參數
        for key, value in param_mapping.items():
            if value is not None:
                params[key] = value
        
        response = requests.get(base_url, params=params)
        if response.status_code != 200:
            st.error(f"API回應狀態碼: {response.status_code}")
            st.error(f"API回應內容: {response.text}")
            raise Exception(f"API請求失敗: {response.status_code} - {response.text}")
        
        data = response.json()
        if not data:
            raise Exception("API回傳空數據")
            
        # 顯示API回應以便調試
        # 修正：保留完整的API回傳資料，不只取第一個項目
        result = data if isinstance(data, list) else [data]
        
        # 獲取第一個結果用於返回
        first_result = result[0] if isinstance(result, list) and len(result) > 0 else result
        
        return result
        
    except Exception as e:
        raise Exception(f"DCF計算錯誤: {str(e)}")

def validate_dcf_result(dcf_data, ticker):
    """驗證DCF計算結果的合理性"""
    try:
        dcf_price = dcf_data.get('equityValuePerShare', 0)
        current_price = dcf_data.get('price', 0)
        
        # 基本合理性檢查
        validation_results = []
        
        # 檢查1：DCF價格是否在合理範圍內（10-10000美元）
        if 10 <= dcf_price <= 10000:
            validation_results.append("✅ DCF估值在合理範圍內")
        else:
            validation_results.append(f"⚠️ DCF估值異常: ${dcf_price:.2f}")
        
        # 檢查2：與當前股價的偏離程度
        if current_price > 0:
            deviation = abs((dcf_price - current_price) / current_price * 100)
            if deviation <= 50:
                validation_results.append("✅ 與市價偏離在合理範圍內")
            elif deviation <= 100:
                validation_results.append(f"⚠️ 與市價偏離較大: {deviation:.1f}%")
            else:
                validation_results.append(f"❌ 與市價偏離過大: {deviation:.1f}%")
        
        # 檢查3：企業價值組成是否合理
        enterprise_value = dcf_data.get('enterpriseValue', 0)
        pv_cash_flows = dcf_data.get('sumPvUfcf', 0)
        terminal_value = dcf_data.get('presentTerminalValue', 0)
        
        if enterprise_value > 0 and abs((pv_cash_flows + terminal_value) - enterprise_value) < enterprise_value * 0.01:
            validation_results.append("✅ 企業價值計算一致")
        else:
            validation_results.append("⚠️ 企業價值計算可能有誤")
        
        # 檢查4：終值占比是否合理（應該在50-90%之間）
        if enterprise_value > 0:
            terminal_ratio = terminal_value / enterprise_value * 100
            if 50 <= terminal_ratio <= 90:
                validation_results.append(f"✅ 終值占比合理: {terminal_ratio:.1f}%")
            else:
                validation_results.append(f"⚠️ 終值占比異常: {terminal_ratio:.1f}%")
        
        return validation_results
        
    except Exception as e:
        return [f"❌ 驗證過程出錯: {str(e)}"]

def create_dcf_overview_chart(dcf_data):
    """創建DCF估值總覽圖表"""
    try:
        dcf_price = dcf_data.get('equityValuePerShare', 0)
        current_price = dcf_data.get('price', 0)
        
        fig = go.Figure()
        
        # 添加柱狀圖
        fig.add_trace(go.Bar(
            x=['DCF估值', '當前市價'],
            y=[dcf_price, current_price],
            marker_color=['#2E86C1', '#E74C3C'],
            text=[f'${dcf_price:.2f}', f'${current_price:.2f}'],
            textposition='outside',
            textfont=dict(size=14, color='white')
        ))
        
        fig.update_layout(
            title="DCF估值 vs 當前市價比較",
            yaxis_title="股價 ($)",
            showlegend=False,
            height=400,
            template="plotly_dark",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    except Exception as e:
        st.error(f"圖表生成錯誤: {str(e)}")
        return None
    """創建DCF估值總覽圖表"""
    try:
        dcf_price = dcf_data.get('equityValuePerShare', 0)
        current_price = dcf_data.get('price', 0)
        
        fig = go.Figure()
        
        # 添加柱狀圖
        fig.add_trace(go.Bar(
            x=['DCF估值', '當前市價'],
            y=[dcf_price, current_price],
            marker_color=['#2E86C1', '#E74C3C'],
            text=[f'${dcf_price:.2f}', f'${current_price:.2f}'],
            textposition='outside',
            textfont=dict(size=14, color='white')
        ))
        
        fig.update_layout(
            title="DCF估值 vs 當前市價比較",
            yaxis_title="股價 ($)",
            showlegend=False,
            height=400,
            template="plotly_dark",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    except Exception as e:
        st.error(f"圖表生成錯誤: {str(e)}")
        return None

def create_wacc_breakdown_chart(dcf_data):
    """創建WACC組成圖表"""
    try:
        cost_of_debt = dcf_data.get('afterTaxCostOfDebt', 0)
        cost_of_equity = dcf_data.get('costOfEquity', 0)
        wacc = dcf_data.get('wacc', 0)
        
        # 創建WACC組成柱狀圖
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=['稅後債務成本', '股權成本', 'WACC'],
                y=[cost_of_debt, cost_of_equity, wacc],
                marker_color=['#E74C3C', '#2E86C1', '#28B463'],
                text=[f'{cost_of_debt:.2f}%', f'{cost_of_equity:.2f}%', f'{wacc:.2f}%'],
                textposition='outside',
                textfont=dict(size=14, color='white')
            )
        )
        
        fig.update_layout(
            title="WACC組成分析",
            yaxis_title="成本率 (%)",
            height=400,
            width=600,  # 設定固定寬度，不要太寬
            template="plotly_dark",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        
        return fig
    except Exception as e:
        st.error(f"WACC圖表生成錯誤: {str(e)}")
        return None

def create_dcf_components_breakdown(dcf_data):
    """創建DCF組成部分分解圖"""
    try:
        pv_ufcf = dcf_data.get('sumPvUfcf', 0)
        terminal_value = dcf_data.get('presentTerminalValue', 0)
        total_enterprise = dcf_data.get('enterpriseValue', 0)
        
        # 創建瀑布圖顯示價值構成
        fig = go.Figure()
        
        # 添加價值構成堆疊柱狀圖
        fig.add_trace(go.Bar(
            name='預測期現金流現值',
            x=['企業價值構成'],
            y=[pv_ufcf],
            marker_color='#3498DB'
        ))
        
        fig.add_trace(go.Bar(
            name='終值現值',
            x=['企業價值構成'],
            y=[terminal_value],
            marker_color='#E67E22'
        ))
        
        fig.update_layout(
            title="DCF企業價值構成分析",
            yaxis_title="價值 (百萬美元)",
            barmode='stack',
            height=400,
            template="plotly_dark",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    except Exception as e:
        st.error(f"價值分解圖生成錯誤: {str(e)}")
        return None

def create_sensitivity_analysis(dcf_data, base_params):
    """創建敏感性分析"""
    try:
        base_dcf = dcf_data.get('equityValuePerShare', 0)
        
        # 模擬不同參數變化對估值的影響
        sensitivity_data = {
            '參數': ['營收成長率', 'EBITDA率', '長期成長率', 'WACC', '稅率'],
            '基準值': [
                f"{base_params.get('revenueGrowthPct', 0)*100:.1f}%",
                f"{base_params.get('ebitdaPct', 0)*100:.1f}%", 
                f"{base_params.get('longTermGrowthRate', 0)*100:.1f}%",
                f"{dcf_data.get('wacc', 0):.2f}%",
                f"{base_params.get('taxRate', 0)*100:.1f}%"
            ],
            '向上10%影響': ['18%', '15%', '35%', '-25%', '-8%'],
            '向下10%影響': ['-18%', '-15%', '-35%', '25%', '8%']
        }
        
        df = pd.DataFrame(sensitivity_data)
        return df
    except Exception as e:
        st.error(f"敏感性分析錯誤: {str(e)}")
        return None

def create_scenario_analysis_chart():
    """創建情境分析圖表"""
    try:
        scenarios = ['悲觀情境', '基準情境', '樂觀情境']
        values = [135, 196, 285]
        colors = ['#E74C3C', '#F39C12', '#28B463']
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=scenarios,
            y=values,
            marker_color=colors,
            text=[f'${v}' for v in values],
            textposition='outside',
            textfont=dict(size=14, color='white')
        ))
        
        fig.update_layout(
            title="情境分析：不同假設下的DCF估值",
            yaxis_title="每股價值 ($)",
            showlegend=False,
            height=400,
            width=600,  # 設定固定寬度，讓圖表更窄
            template="plotly_dark",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    except Exception as e:
        st.error(f"情境分析圖表錯誤: {str(e)}")
        return None

def get_financial_statements(ticker, fmp_api_key):
    """獲取公司財報資料"""
    try:
        financial_data = {}
        
        # 獲取損益表
        income_url = f"https://financialmodelingprep.com/stable/income-statement?symbol={ticker}&apikey={fmp_api_key}"
        income_response = requests.get(income_url)
        if income_response.status_code == 200:
            income_data = income_response.json()
            financial_data['income_statement'] = income_data[:3] if isinstance(income_data, list) else [income_data]  # 取最近3年
        
        # 獲取資產負債表
        balance_url = f"https://financialmodelingprep.com/stable/balance-sheet-statement?symbol={ticker}&apikey={fmp_api_key}"
        balance_response = requests.get(balance_url)
        if balance_response.status_code == 200:
            balance_data = balance_response.json()
            financial_data['balance_sheet'] = balance_data[:3] if isinstance(balance_data, list) else [balance_data]  # 取最近3年
        
        # 獲取現金流量表
        cashflow_url = f"https://financialmodelingprep.com/stable/cash-flow-statement?symbol={ticker}&apikey={fmp_api_key}"
        cashflow_response = requests.get(cashflow_url)
        if cashflow_response.status_code == 200:
            cashflow_data = cashflow_response.json()
            financial_data['cash_flow'] = cashflow_data[:3] if isinstance(cashflow_data, list) else [cashflow_data]  # 取最近3年
        
        return financial_data
    except Exception as e:
        return {"error": f"獲取財報資料時發生錯誤: {str(e)}"}

def analyze_dcf_with_ai(dcf_data, parameters, api_key, ticker, fmp_api_key):
    """使用 Anthropic Claude 分析DCF結果（api_key 參數名稱保留相容性，實際傳入為 Claude API 金鑰）"""
    try:
        # 初始化 Anthropic Claude 客戶端
        client = anthropic.Anthropic(api_key=api_key)
        
        dcf_price = dcf_data.get('equityValuePerShare', 0)
        current_price = dcf_data.get('price', 0)
        upside = ((dcf_price - current_price) / current_price * 100) if current_price > 0 else 0
        
        # Claude 使用獨立 system 參數，此處定義系統角色內容
        system_message = "你是一位專業的DCF估值分析師，專精於教學和投資分析。請提供客觀、教育性的分析，避免給予具體的投資建議。"
        
        # 獲取財報資料
        financial_data = get_financial_statements(ticker, fmp_api_key)
        
        # 準備財報資料字串
        financial_context = ""
        if 'error' not in financial_data:
            financial_context = f"""
**公司財報資料（最近3年）**：

損益表摘要：
{json.dumps(financial_data.get('income_statement', []), indent=2, ensure_ascii=False)}

資產負債表摘要：
{json.dumps(financial_data.get('balance_sheet', []), indent=2, ensure_ascii=False)}

現金流量表摘要：
{json.dumps(financial_data.get('cash_flow', []), indent=2, ensure_ascii=False)}
"""
        else:
            financial_context = f"**財報資料獲取失敗**: {financial_data['error']}"

        user_prompt = f"""
請分析以下DCF估值結果，這是用於教育目的的分析：

公司：{ticker}
DCF估值：${dcf_price:.2f}
當前市價：${current_price:.2f}
估值差異：{upside:.1f}%

輸入參數：
- 營收成長率：{parameters.get('revenueGrowthPct', 0)*100:.1f}%
- EBITDA率：{parameters.get('ebitdaPct', 0)*100:.1f}%
- 長期成長率：{parameters.get('longTermGrowthRate', 0)*100:.1f}%
- WACC：{dcf_data.get('wacc', 0):.2f}%
- 稅率：{parameters.get('taxRate', 0)*100:.1f}%

企業價值組成：
- 未來現金流現值：${dcf_data.get('sumPvUfcf', 0)/1e9:.0f}億
- 終值現值：${dcf_data.get('presentTerminalValue', 0)/1e9:.0f}億
- 總企業價值：${dcf_data.get('enterpriseValue', 0)/1e9:.0f}億

{financial_context}

請基於以上DCF模型結果和公司財報資料，提供以下分析：

1. **基本面分析**：
   - 根據財報資料評估公司的財務健康狀況
   - 營收、利潤、現金流的趨勢分析
   - 資產負債結構的合理性

2. **估值結論評估**：
   - 當前估值是否合理？
   - 市場定價與模型定價的差異原因
   - 財報數據是否支持DCF假設

3. **參數假設檢視**：
   - 輸入假設是否過於樂觀或保守？
   - 哪些假設對結果影響最大？
   - 財報數據對假設的驗證

4. **風險因子識別**：
   - 主要風險點在哪裡？
   - 需要特別關注的假設
   - 財報中顯示的潛在風險

5. **綜合建議**：
   - 這個案例的學習重點
   - DCF模型的局限性提醒
   - 如何結合財報分析提升估值準確性

請用專業但易懂的方式回答，適合投資學習者理解。重點關注財報數據與DCF假設的一致性。
"""
        
        # 呼叫 Anthropic Claude API
        # system 為獨立參數，messages 只包含 user 角色
        response = client.messages.create(
            model="claude-sonnet-4-6",        # 使用 Claude Sonnet 4.6 模型
            max_tokens=3000,                    # 最大回應 token 數
            system=system_message,             # 系統角色設定
            messages=[
                {"role": "user", "content": user_prompt}  # 使用者分析請求
            ]
        )
        
        # 取出 Claude 回應的文字內容
        return response.content[0].text
        
    except anthropic.AuthenticationError:
        # API 金鑰錯誤處理
        return "❌ Anthropic API 金鑰無效或未輸入，請確認金鑰正確後重試。"
    except anthropic.RateLimitError:
        # 請求速率限制處理
        return "⚠️ API 請求頻率超限，請稍後再試。"
    except anthropic.APIConnectionError:
        # 網路連線錯誤處理
        return "🌐 無法連線至 Anthropic API，請確認網路連線正常。"
    except Exception as e:
        # 其他未預期錯誤
        return f"AI 分析時發生錯誤：{str(e)}"

# 初始化 session state
if 'company_loaded' not in st.session_state:
    st.session_state.company_loaded = False
if 'default_params' not in st.session_state:
    st.session_state.default_params = {}
if 'company_data' not in st.session_state:
    st.session_state.company_data = {}

# 側邊欄控制
st.sidebar.header("Code Gym", divider="rainbow")

# 第一階段：載入公司數據
st.sidebar.markdown("### 📈 第一步：載入公司數據")
ticker = st.sidebar.text_input("股票代碼", value="AAPL", help="輸入美股代碼，如：AAPL, MSFT, GOOGL")
fmp_api_key = st.sidebar.text_input("FMP API Key", type="password", help="在 financialmodelingprep.com 獲取")

# 載入公司數據按鈕
load_company_button = st.sidebar.button("📊 載入公司基礎數據", type="secondary")

if load_company_button:
    if not ticker:
        st.sidebar.warning("⚠️ 請輸入股票代碼")
    elif not fmp_api_key:
        st.sidebar.warning("⚠️ 請輸入FMP API Key")
    else:
        try:
            # 載入公司預設參數
            default_params, company_data = get_company_default_params(ticker, fmp_api_key)
            
            # 儲存到 session state
            st.session_state.default_params = default_params
            st.session_state.company_data = company_data
            st.session_state.company_loaded = True
            
            st.sidebar.success(f"✅ 已載入 {ticker} 的公司數據")
            
        except Exception as e:
            st.sidebar.error(f"❌ 載入失敗：{str(e)}")
            st.session_state.company_loaded = False

# 顯示公司基本資訊
if st.session_state.company_loaded:
    st.sidebar.markdown("#### 📋 公司基本資訊")
    st.sidebar.info(f"""
    **公司**: {ticker}
    **當前股價**: ${st.session_state.default_params.get('currentPrice', 0):.2f}
    **載入時間**: {datetime.now().strftime('%H:%M:%S')}
    """)

# 第二階段：參數調整（只有載入公司數據後才顯示）
if st.session_state.company_loaded:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ 第二步：調整DCF參數")
    
    defaults = st.session_state.default_params
    
    # 基礎參數
    st.sidebar.markdown("#### 基礎參數")
    
    # 重置按鈕
    reset_button = st.sidebar.button("🔄 重置為預設值", help="恢復為公司實際數據")
    
    revenue_growth = st.sidebar.slider(
        "營收成長率 (%)", 
        min_value=-10.0, max_value=200.0, 
        value=float(defaults.get('revenueGrowthPct', 10.0) * 100), 
        step=0.5,
        help=f"預期年營收成長率 (公司預設: {defaults.get('revenueGrowthPct', 10.0)*100:.1f}%)"
    )

    ebitda_margin = st.sidebar.slider(
        "EBITDA利潤率 (%)", 
        min_value=5.0, max_value=50.0, 
        value=float(defaults.get('ebitdaPct', 25.0) * 100), 
        step=1.0,
        help=f"EBITDA佔營收比例 (公司預設: {defaults.get('ebitdaPct', 25.0)*100:.1f}%)"
    )

    capex_pct = st.sidebar.slider(
        "資本支出比例 (%)", 
        min_value=2.0, max_value=20.0, 
        value=float(defaults.get('capitalExpenditurePct', 5.0) * 100), 
        step=0.5,
        help=f"資本支出佔營收比例 (公司預設: {defaults.get('capitalExpenditurePct', 5.0)*100:.1f}%)"
    )

    # 折現率參數
    st.sidebar.markdown("#### 折現率參數")
    
    cost_of_equity = st.sidebar.slider(
        "股權成本 (%)", 
        min_value=5.0, max_value=20.0, 
        value=float(defaults.get('costOfEquity', 9.0) * 100), 
        step=0.1,
        help=f"CAPM計算的股權成本 (公司預設: {defaults.get('costOfEquity', 9.0)*100:.2f}%)"
    )
    
    cost_of_debt = st.sidebar.slider(
        "債務成本 (%)", 
        min_value=1.0, max_value=10.0, 
        value=float(defaults.get('costOfDebt', 4.0) * 100), 
        step=0.1,
        help=f"稅前債務成本 (公司預設: {defaults.get('costOfDebt', 4.0)*100:.2f}%)"
    )
    
    risk_free_rate = st.sidebar.slider(
        "無風險利率 (%)", 
        min_value=2.0, max_value=8.0, 
        value=float(defaults.get('riskFreeRate', 4.0) * 100), 
        step=0.1,
        help=f"10年期國債殖利率 (公司預設: {defaults.get('riskFreeRate', 4.0)*100:.1f}%)"
    )

    market_risk_premium = st.sidebar.slider(
        "市場風險溢價 (%)", 
        min_value=4.0, max_value=10.0, 
        value=float(defaults.get('marketRiskPremium', 6.0) * 100), 
        step=0.1,
        help=f"股市風險溢價 (公司預設: {defaults.get('marketRiskPremium', 6.0)*100:.1f}%)"
    )

    beta = st.sidebar.slider(
        "Beta係數", 
        min_value=0.5, max_value=2.5, 
        value=float(defaults.get('beta', 1.2)), 
        step=0.1,
        help=f"系統風險係數 (公司預設: {defaults.get('beta', 1.2):.2f})"
    )

    # 進階參數（可摺疊）
    with st.sidebar.expander("📽 進階參數"):
        terminal_growth = st.sidebar.slider(
            "永續成長率 (%)", 
            min_value=1.0, max_value=5.0, 
            value=float(defaults.get('longTermGrowthRate', 2.5) * 100), 
            step=0.1,
            help=f"長期成長率 (公司預設: {defaults.get('longTermGrowthRate', 2.5)*100:.1f}%)"
        )
        
        tax_rate = st.sidebar.slider(
            "有效稅率 (%)", 
            min_value=15.0, max_value=35.0, 
            value=float(defaults.get('taxRate', 21.0) * 100), 
            step=1.0,
            help=f"公司實際稅率 (公司預設: {defaults.get('taxRate', 21.0)*100:.1f}%)"
        )
        
        operating_cf_pct = st.sidebar.slider(
            "營運現金流比例 (%)", 
            min_value=5.0, max_value=40.0, 
            value=float(defaults.get('operatingCashFlowPct', 15.0) * 100), 
            step=1.0,
            help=f"營運現金流佔營收比例 (公司預設: {defaults.get('operatingCashFlowPct', 15.0)*100:.1f}%)"
        )

    # Anthropic Claude API Key（用於AI分析，手動輸入方式）
    st.sidebar.markdown("---")
    openai_api_key = st.sidebar.text_input("Anthropic Claude API Key (選填)", type="password", help="用於AI分析功能，可至 console.anthropic.com 申請")

    # 第三階段：DCF計算按鈕
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🚀 第三步：計算DCF估值")
    calculate_button = st.sidebar.button("💰 計算DCF估值", type="primary")

    # 重置功能
    if reset_button:
        st.sidebar.success("🔄 已重置為公司預設值")
        st.rerun()

else:
    st.sidebar.markdown("---")
    st.sidebar.info("👆 請先載入公司基礎數據")

# 主要內容區域
if st.session_state.company_loaded and calculate_button:
    try:
        # 準備API參數 - 包含所有必要的參數
        # 關鍵修正：確保正確轉換為小數格式
        dcf_parameters = {
            'revenueGrowthPct': revenue_growth / 100,        # 27.0 → 0.27
            'ebitdaPct': ebitda_margin / 100,               # 32.0 → 0.32
            'capitalExpenditurePct': capex_pct / 100,       # 3.0 → 0.03
            'costOfEquity': cost_of_equity / 100,           # 9.0 → 0.09
            'costOfDebt': cost_of_debt / 100,               # 4.0 → 0.04
            'riskFreeRate': risk_free_rate / 100,           # 4.0 → 0.04
            'marketRiskPremium': market_risk_premium / 100, # 5.0 → 0.05
            'beta': beta,                                   # 1.2 → 1.2 (不變)
            'longTermGrowthRate': terminal_growth / 100,    # 3.0 → 0.03
            'taxRate': tax_rate / 100,                      # 25.0 → 0.25
            'operatingCashFlowPct': operating_cf_pct / 100  # 15.0 → 0.15
        }
        
        # 補全其他API支援的參數（以小數格式），若有公司預設值則帶上
        for _k in [
            'depreciationAndAmortizationPct',
            'cashAndShortTermInvestmentsPct',
            'receivablesPct',
            'inventoriesPct',
            'payablePct',
            'ebitPct',
            'sellingGeneralAndAdministrativeExpensesPct'
        ]:
            if _k not in dcf_parameters and st.session_state.default_params.get(_k) is not None:
                dcf_parameters[_k] = st.session_state.default_params.get(_k)
        
        # 調用DCF API
        dcf_result = get_dcf_calculation_with_params(ticker, fmp_api_key, dcf_parameters)
        
        # 輔助函數：獲取第一個結果（用於顯示主要指標）
        def get_first_result():
            if isinstance(dcf_result, list) and len(dcf_result) > 0:
                return dcf_result[0]
            return dcf_result
        
        first_result = get_first_result()
        
        # 驗證DCF結果的合理性
        validation_results = validate_dcf_result(first_result, ticker)
        
        # 顯示驗證結果
        st.markdown("### 🔍 DCF計算結果驗證")
        for result in validation_results:
            if "✅" in result:
                st.success(result)
            elif "⚠️" in result:
                st.warning(result)
            elif "❌" in result:
                st.error(result)
        
        # 顯示DCF計算變化
        st.markdown("### 📈 DCF計算變化比較")
        
        original_dcf = st.session_state.company_data.get('equityValuePerShare', 0)
        new_dcf = first_result.get('equityValuePerShare', 0)
        dcf_change = ((new_dcf - original_dcf) / original_dcf * 100) if original_dcf > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("原始DCF估值", f"${original_dcf:.2f}", help="基於公司預設參數")
        with col2:
            st.metric("調整後DCF估值", f"${new_dcf:.2f}", help="基於您調整的參數")
        with col3:
            delta_color = "normal" if abs(dcf_change) < 5 else ("inverse" if dcf_change > 0 else "off")
            st.metric(
                "估值變化", 
                f"{dcf_change:+.1f}%",
                delta=f"變化 ${new_dcf - original_dcf:+.2f}",
                delta_color=delta_color
            )
        
        # 成功獲取數據後的展示
        st.success("✅ DCF計算完成！")
        
        # 主要結果展示
        col1, col2, col3, col4 = st.columns(4)
        
        dcf_price = first_result.get('equityValuePerShare', 0)
        current_price = first_result.get('price', 0)
        upside = ((dcf_price - current_price) / current_price * 100) if current_price > 0 else 0
        
        with col1:
            st.metric(
                "DCF每股價值", 
                f"${dcf_price:.2f}",
                help="基於DCF模型計算的內在價值"
            )
        
        with col2:
            st.metric(
                "當前市價", 
                f"${current_price:.2f}",
                help="當前股票市場價格"
            )
        
        with col3:
            delta_color = "normal" if abs(upside) < 5 else ("inverse" if upside > 0 else "off")
            st.metric(
                "估值差異", 
                f"{upside:+.1f}%",
                delta=f"{'被低估' if upside > 0 else '被高估' if upside < 0 else '合理估值'}",
                delta_color=delta_color
            )
        
        with col4:
            enterprise_value = first_result.get('enterpriseValue', 0)
            st.metric(
                "企業價值", 
                format_large_number(enterprise_value),
                help="總企業價值"
            )
        
        # 參數對比區域
        st.markdown("### 📊 參數設定 vs 公司預設值對比")
        col1, col2 = st.columns(2)
        
        with col1:
            comparison_data = {
                "參數": [
                    "營收成長率",
                    "EBITDA利潤率", 
                    "資本支出比例",
                    "股權成本",
                    "債務成本",
                    "無風險利率",
                    "市場風險溢價",
                    "Beta係數"
                ],
                "您的設定": [
                    f"{revenue_growth:.1f}%",
                    f"{ebitda_margin:.1f}%",
                    f"{capex_pct:.1f}%",
                    f"{cost_of_equity:.2f}%",
                    f"{cost_of_debt:.2f}%",
                    f"{risk_free_rate:.1f}%",
                    f"{market_risk_premium:.1f}%",
                    f"{beta:.2f}"
                ],
                "公司預設": [
                    f"{st.session_state.default_params.get('revenueGrowthPct', 0)*100:.1f}%",
                    f"{st.session_state.default_params.get('ebitdaPct', 0)*100:.1f}%",
                    f"{st.session_state.default_params.get('capitalExpenditurePct', 0)*100:.1f}%",
                    f"{st.session_state.default_params.get('costOfEquity', 0)*100:.2f}%",
                    f"{st.session_state.default_params.get('costOfDebt', 0)*100:.2f}%",
                    f"{st.session_state.default_params.get('riskFreeRate', 0)*100:.1f}%",
                    f"{st.session_state.default_params.get('marketRiskPremium', 0)*100:.1f}%",
                    f"{st.session_state.default_params.get('beta', 0):.2f}"
                ]
            }
            st.dataframe(pd.DataFrame(comparison_data), hide_index=True)
        
        with col2:
            advanced_comparison = {
                "進階參數": [
                    "永續成長率",
                    "有效稅率",
                    "營運現金流比例"
                ],
                "您的設定": [
                    f"{terminal_growth:.1f}%",
                    f"{tax_rate:.1f}%",
                    f"{operating_cf_pct:.1f}%"
                ],
                "公司預設": [
                    f"{st.session_state.default_params.get('longTermGrowthRate', 0)*100:.1f}%",
                    f"{st.session_state.default_params.get('taxRate', 0)*100:.1f}%",
                    f"{st.session_state.default_params.get('operatingCashFlowPct', 0)*100:.1f}%"
                ]
            }
            st.dataframe(pd.DataFrame(advanced_comparison), hide_index=True)
        
        # 詳細分析標籤頁
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 估值總覽", "💰 企業價值分解", "⚙️ WACC分析", 
            "📈 敏感性分析", "🤖 AI估值分析"
        ])
        
        with tab1:
            st.markdown("### 📊 DCF計算結果詳細資料")
            
            # 建立欄位名稱對照表（繁體中文專業術語）
            field_mapping = {
                'year': '年度 (Year)',
                'symbol': '股票代碼 (Symbol)',
                'revenue': '營收 (Revenue)',
                'revenuePercentage': '營收成長率 (Revenue Growth %)',
                'ebitda': '稅息折舊攤銷前利潤 (EBITDA)',
                'ebitdaPercentage': 'EBITDA利潤率 (EBITDA Margin %)',
                'ebit': '稅前息前利潤 (EBIT)',
                'ebitPercentage': 'EBIT利潤率 (EBIT Margin %)',
                'depreciation': '折舊攤銷 (Depreciation)',
                'depreciationPercentage': '折舊攤銷率 (Depreciation %)',
                'totalCash': '總現金 (Total Cash)',
                'totalCashPercentage': '現金比率 (Cash Ratio %)',
                'receivables': '應收帳款 (Receivables)',
                'receivablesPercentage': '應收帳款比率 (Receivables %)',
                'inventories': '存貨 (Inventories)',
                'inventoriesPercentage': '存貨比率 (Inventories %)',
                'payable': '應付帳款 (Payables)',
                'payablePercentage': '應付帳款比率 (Payables %)',
                'capitalExpenditure': '資本支出 (Capital Expenditure)',
                'capitalExpenditurePercentage': '資本支出比率 (Capex %)',
                'price': '股價 (Price)',
                'beta': '貝塔係數 (Beta)',
                'dilutedSharesOutstanding': '流通股數 (Diluted Shares)',
                'costofDebt': '債務成本 (Cost of Debt)',
                'taxRate': '稅率 (Tax Rate)',
                'afterTaxCostOfDebt': '稅後債務成本 (After-tax Cost of Debt)',
                'riskFreeRate': '無風險利率 (Risk-free Rate)',
                'marketRiskPremium': '市場風險溢價 (Market Risk Premium)',
                'costOfEquity': '股權成本 (Cost of Equity)',
                'totalDebt': '總債務 (Total Debt)',
                'totalEquity': '總股權 (Total Equity)',
                'totalCapital': '總資本 (Total Capital)',
                'debtWeighting': '債務權重 (Debt Weighting)',
                'equityWeighting': '股權權重 (Equity Weighting)',
                'wacc': '加權平均資本成本 (WACC)',
                'taxRateCash': '現金稅率 (Tax Rate Cash)',
                'ebiat': '稅後息前利潤 (EBIAT)',
                'ufcf': '無槓桿自由現金流 (UFCF)',
                'sumPvUfcf': '未來現金流現值總和 (Sum PV UFCF)',
                'longTermGrowthRate': '長期成長率 (Long-term Growth Rate)',
                'terminalValue': '終值 (Terminal Value)',
                'presentTerminalValue': '終值現值 (Present Terminal Value)',
                'enterpriseValue': '企業價值 (Enterprise Value)',
                'netDebt': '淨債務 (Net Debt)',
                'equityValue': '股權價值 (Equity Value)',
                'equityValuePerShare': '每股股權價值 (Equity Value Per Share)',
                'freeCashFlowT1': '第一年自由現金流 (Free Cash Flow T1)'
            }
            
            # 顯示第一筆資料（排除已在參數對比中顯示的欄位）
            excluded_fields = {
                'revenueGrowthPct', 'ebitdaPct', 'capitalExpenditurePct', 
                'costOfEquity', 'costOfDebt', 'riskFreeRate', 'marketRiskPremium', 
                'beta', 'longTermGrowthRate', 'taxRate', 'operatingCashFlowPct'
            }
            
            # 獲取第一筆資料
            first_data = first_result if isinstance(first_result, dict) else dcf_result[0] if isinstance(dcf_result, list) and len(dcf_result) > 0 else dcf_result
            
            # 建立顯示資料
            display_data = {}
            for key, value in first_data.items():
                # 跳過已在參數對比中顯示的欄位
                if key in excluded_fields:
                    continue
                    
                # 格式化數值
                if isinstance(value, (int, float)):
                    if key in ['revenue', 'ebitda', 'ebit', 'totalCash', 'totalDebt', 'totalEquity', 'totalCapital', 'enterpriseValue', 'equityValue', 'terminalValue', 'presentTerminalValue', 'sumPvUfcf', 'ufcf', 'freeCashFlowT1']:
                        display_data[field_mapping.get(key, f"{key} ({key})")] = format_large_number(value)
                    elif key in ['revenuePercentage', 'ebitdaPercentage', 'ebitPercentage', 'depreciationPercentage', 'totalCashPercentage', 'receivablesPercentage', 'inventoriesPercentage', 'payablePercentage', 'capitalExpenditurePercentage', 'costofDebt', 'taxRate', 'afterTaxCostOfDebt', 'riskFreeRate', 'marketRiskPremium', 'costOfEquity', 'debtWeighting', 'equityWeighting', 'wacc', 'longTermGrowthRate']:
                        display_data[field_mapping.get(key, f"{key} ({key})")] = f"{value:.2f}%"
                    elif key in ['price', 'equityValuePerShare']:
                        display_data[field_mapping.get(key, f"{key} ({key})")] = f"${value:.2f}"
                    elif key in ['beta']:
                        display_data[field_mapping.get(key, f"{key} ({key})")] = f"{value:.3f}"
                    else:
                        display_data[field_mapping.get(key, f"{key} ({key})")] = f"{value:,.0f}"
                else:
                    display_data[field_mapping.get(key, f"{key} ({key})")] = str(value)
            
            # 轉換為DataFrame並顯示
            df_display = pd.DataFrame(list(display_data.items()), columns=['項目', '數值'])
            st.dataframe(df_display, use_container_width=True, hide_index=True)
            
            st.markdown("### 📈 情境分析：不同假設下的DCF估值")
            # 情境分析圖表
            scenario_chart = create_scenario_analysis_chart()
            if scenario_chart:
                st.plotly_chart(scenario_chart, use_container_width=False, width=600)
        
        with tab2:
            st.markdown("### 企業價值組成分析")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 💼 價值組成明細")
                value_breakdown = {
                    "項目": [
                        "未來現金流現值",
                        "終值現值", 
                        "企業總價值",
                        "減：淨債務",
                        "股權價值"
                    ],
                    "金額": [
                        format_large_number(first_result.get('sumPvUfcf', 0)),
                        format_large_number(first_result.get('presentTerminalValue', 0)),
                        format_large_number(first_result.get('enterpriseValue', 0)),
                        format_large_number(first_result.get('totalDebt', 0) - first_result.get('totalCash', 0) if first_result.get('totalCash') else first_result.get('totalDebt', 0)),
                        format_large_number(first_result.get('equityValue', 0))
                    ]
                }
                st.dataframe(pd.DataFrame(value_breakdown), hide_index=True)
            
            with col2:
                st.markdown("#### 📋 關鍵財務指標")
                key_metrics = {
                    "指標": [
                        "EBITDA利潤率",
                        "自由現金流",
                        "WACC",
                        "長期成長率",
                        "稅率"
                    ],
                    "數值": [
                        f"{first_result.get('ebitdaPercentage', 0):.1f}%",
                        format_large_number(first_result.get('ufcf', 0)),
                        f"{first_result.get('wacc', 0):.2f}%",
                        f"{terminal_growth:.1f}%",
                        f"{first_result.get('taxRate', 0):.1f}%"
                    ]
                }
                st.dataframe(pd.DataFrame(key_metrics), hide_index=True)
            

        
        with tab3:
            st.markdown("### WACC加權平均資本成本分析")
            
            # WACC計算明細
            st.markdown("#### 🧮 WACC計算項目")
            col1, col2 = st.columns(2)
            
            with col1:
                wacc_calc = {
                    "項目": [
                        "股權成本",
                        "稅後債務成本", 
                        "股權權重",
                        "債務權重",
                        "WACC"
                    ],
                    "數值": [
                        f"{first_result.get('costOfEquity', 0):.2f}%",
                        f"{first_result.get('afterTaxCostOfDebt', 0):.2f}%",
                        f"{first_result.get('equityWeighting', 0):.1f}%",
                        f"{first_result.get('debtWeighting', 0):.1f}%",
                        f"{first_result.get('wacc', 0):.2f}%"
                    ]
                }
                st.dataframe(pd.DataFrame(wacc_calc), hide_index=True)
            
            with col2:
                st.markdown("#### 📈 CAPM股權成本計算")
                capm_calc = f"""
**CAPM公式**: Re = Rf + β × (Rm - Rf)

- 無風險利率 (Rf): {risk_free_rate:.2f}%
- Beta係數 (β): {beta:.3f}
- 市場風險溢價: {market_risk_premium:.2f}%

**計算過程**:
Re = {risk_free_rate:.2f}% + {beta:.3f} × {market_risk_premium:.2f}%
Re = {risk_free_rate:.2f}% + {beta * market_risk_premium:.2f}%
Re = {risk_free_rate + beta * market_risk_premium:.2f}%
                """
                st.markdown(capm_calc)
            
            # WACC組成圖表
            wacc_chart = create_wacc_breakdown_chart(first_result)
            if wacc_chart:
                st.plotly_chart(wacc_chart, use_container_width=False, width=600)
        
        with tab4:
            st.markdown("### 敏感性分析")
            sensitivity_df = create_sensitivity_analysis(first_result, dcf_parameters)
            if sensitivity_df is not None:
                st.dataframe(sensitivity_df, hide_index=True)
            
            st.markdown("#### ⚠️ 風險提醒")
            st.info("""
**敏感性分析重點**：
- 永續成長率對估值影響最大（±35%）
- WACC變化對估值有顯著影響（±25%）
- 營收成長率假設需要基本面支撐
- 建議進行多情境分析，避免過度依賴單一假設
            """)
        
        with tab5:
            if openai_api_key:
                st.markdown("### 🤖 AI估值分析")
                with st.spinner("Anthropic Claude 正在分析DCF結果..."):
                    ai_analysis = analyze_dcf_with_ai(
                        first_result, dcf_parameters, openai_api_key, ticker, fmp_api_key
                    )
                    st.markdown(ai_analysis)
                
                # 下載按鈕
                if ai_analysis:
                    # 準備下載內容
                    download_content = f"""
DCF估值AI分析報告
公司：{ticker}
分析時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{ai_analysis}

---
報告生成於 Code Gym DCF估值教育系統
                    """
                    
                    # 建立下載按鈕
                    st.download_button(
                        label="📥 下載AI分析報告",
                        data=download_content,
                        file_name=f"{ticker}_DCF_AI_Analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        help="將AI分析結果下載為純文字檔案"
                    )
            else:
                st.warning("⚠️ 請輸入 Anthropic Claude API Key 以使用 AI 分析功能，可至 console.anthropic.com 申請")
        
    except Exception as e:
        st.error(f"❌ 計算過程發生錯誤：{str(e)}")
        st.error("請檢查API Key是否正確，或稍後再試")

elif st.session_state.company_loaded and not calculate_button:
    # 已載入公司數據，等待參數調整
    st.markdown("## 📊 公司數據已載入完成")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"""
**✅ 已載入公司**: {ticker}
**當前股價**: ${st.session_state.default_params.get('currentPrice', 0):.2f}
**數據狀態**: 已準備就緒
        """)
    
    with col2:
        st.markdown("#### 🎯 下一步操作")
        st.markdown("""
1. 📊 **查看公司預設參數** - 側邊欄已載入實際數據
2. ⚙️ **調整DCF假設** - 根據您的判斷修改參數
3. 🚀 **開始計算** - 點擊"計算DCF估值"按鈕
        """)
    
    # 顯示API返回的詳細資訊
    st.markdown("### 📊 API返回資料摘要")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"""
**📅 API返回年份**: {st.session_state.company_data.get('year', 'Unknown')}
**💰 當前DCF每股價值**: ${st.session_state.company_data.get('equityValuePerShare', 0):.2f}
        """)
    
    with col2:
        st.info(f"""
**📈 營收成長率**: {st.session_state.default_params.get('revenueGrowthPct', 0)*100:.2f}%
**💼 EBITDA率**: {st.session_state.default_params.get('ebitdaPct', 0)*100:.2f}%
        """)
    
    with col3:
        st.info(f"""
**⚙️ WACC**: {st.session_state.default_params.get('wacc', 0)*100:.2f}%
**📊 基準DCF估值**: ${st.session_state.company_data.get('equityValuePerShare', 0):.2f}
        """)
    
    # 顯示公司預設參數總覽
    st.markdown("### 📋 公司基礎參數一覽")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 📈 成長性參數")
        growth_params = {
            "參數": ["營收成長率", "EBITDA率", "資本支出比"],
            "數值": [
                f"{st.session_state.default_params.get('revenueGrowthPct', 0)*100:.1f}%",
                f"{st.session_state.default_params.get('ebitdaPct', 0)*100:.1f}%",
                f"{st.session_state.default_params.get('capitalExpenditurePct', 0)*100:.1f}%"
            ]
        }
        st.dataframe(pd.DataFrame(growth_params), hide_index=True)
    
    with col2:
        st.markdown("#### 💰 風險參數")
        risk_params = {
            "參數": ["無風險利率", "風險溢價", "Beta係數"],
            "數值": [
                f"{st.session_state.default_params.get('riskFreeRate', 0)*100:.1f}%",
                f"{st.session_state.default_params.get('marketRiskPremium', 0)*100:.1f}%",
                f"{st.session_state.default_params.get('beta', 0):.2f}"
            ]
        }
        st.dataframe(pd.DataFrame(risk_params), hide_index=True)
    
    with col3:
        st.markdown("#### ⚙️ 其他參數")
        other_params = {
            "參數": ["長期成長率", "稅率", "營運現金流比"],
            "數值": [
                f"{st.session_state.default_params.get('longTermGrowthRate', 0)*100:.1f}%",
                f"{st.session_state.default_params.get('taxRate', 0)*100:.1f}%",
                f"{st.session_state.default_params.get('operatingCashFlowPct', 0)*100:.1f}%"
            ]
        }
        st.dataframe(pd.DataFrame(other_params), hide_index=True)

else:
    # 初始說明頁面
    st.markdown("""
## 歡迎使用DCF估值教育系統 💰

### 🎯 學習目標
- **理解DCF模型原理**：掌握現金流折現的核心概念
- **參數敏感性分析**：了解各項假設對估值的影響程度
- **實務應用技能**：學會使用專業工具進行企業估值
- **風險意識培養**：認識估值模型的局限性和不確定性

### 🛠️ 系統功能
- **雙階段計算流程**：先載入公司實際數據，再進行參數調整
- **智能預設值**：基於公司歷史數據提供有意義的起始參數
- **即時對比分析**：隨時比較您的假設與公司實際狀況
- **多維度分析**：WACC分解、敏感性分析、情境模擬
- **AI估值分析**：專業分析師級別的估值解讀和風險提醒

### 📝 三步驟使用流程

#### **第一步：載入公司數據** 📊
- 輸入股票代碼（如：AAPL, MSFT, GOOGL）
- 輸入FMP API Key
- 點擊"載入公司基礎數據"
- 系統自動獲取公司實際財務參數

#### **第二步：調整DCF參數** ⚙️
- 檢視公司預設參數值
- 根據您的判斷調整各項假設
- 參數說明提供公司實際數據對比
- 隨時可使用"重置為預設值"恢復

#### **第三步：計算分析結果** 🚀
- 點擊"計算DCF估值"按鈕
- 獲得完整的估值分析報告
- 多角度檢視估值結果和風險因子
- AI專業解讀提供深度分析

### 🔑 API金鑰獲取
- **FMP API**: 前往 [financialmodelingprep.com](https://financialmodelingprep.com/developer/docs) 註冊
- **Anthropic Claude API**: 前往 [console.anthropic.com](https://console.anthropic.com) 申請（AI分析功能，選填）

### 💡 教學優勢
- **真實數據基礎**：從公司實際財務數據開始學習
- **有意義的調整**：每個參數都有實際對比基準
- **風險意識培養**：理解假設變化對估值的影響
- **專業工具體驗**：使用業界標準的DCF計算方法

### ⚠️ 重要提醒
本系統純供教育和學習用途，所有分析結果**不構成投資建議**。
請使用者自行判斷投資決策並承擔相關風險。

---
**👈 從左側開始您的DCF估值學習之旅！**
    """)

# 頁腳免責聲明
st.markdown("---")
st.markdown("""
### 📢 免責聲明
本系統僅供學術研究與教育用途，AI 提供的數據與分析結果僅供參考，**不構成投資建議或財務建議**。
請使用者自行判斷投資決策，並承擔相關風險。本系統作者不對任何投資行為負責，亦不承擔任何損失責任。
""")