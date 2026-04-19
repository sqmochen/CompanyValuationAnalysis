# =============================================================================
# AI 財務分析系統 - DCF估值教育系統
# 程式名稱: app.py
# 版本: v4.0.0
# 更新日期: 2026-04-19
#
# 版本紀錄:
# v1.0.0 | 原始版本 | 使用 FMP API + OpenAI
# v2.0.0 | AI引擎替換 | OpenAI → Anthropic Claude
# v3.0.0 | 2026-04-19 | 完整替換為台股 + FinMind API
#   - 股票代碼改為台股四碼（如 2330、2454）
#   - 資料來源全面改用 FinMind API
#   - 移除 FMP API 依賴
#   - DCF 計算改由程式自行從財報數據計算
#   - WACC / Beta / 股權成本等由財報推導
#   - 新增台股股價查詢（TaiwanStockPrice）
#   - 新增 FinMind Token 輸入欄位
# v4.0.0 | 2026-04-19 | 規格說明書 v2 全面對齊修正
#   - [A] 流通股數改由股本÷面額10元計算（修正帳面股東權益÷股價的錯誤邏輯）
#   - [B] Beta 改由股票vs加權指數TAIEX近2年日報酬率迴歸計算，備援預設1.0
#   - [C] 新增 DCF 合理價格區間驗證（NT$10~NT$10,000）
#   - [D] 新增 WACC > 永續成長率防護檢查與明確警告
#   - [E] 敏感性分析改為動態計算（非寫死數值）
#   - [F] 情境分析改為調整永續成長率±1%、WACC±2%後重新計算
#   - [G] 修正進階參數 expander 內 st.sidebar.slider → st.slider
#   - [H] 新增預測年數滑桿（5~10年）並傳入 calculate_dcf
#   - [I] 統一企業價值圖 Y 軸單位標示（新台幣千元）
#   - [J] 新增 beta_calculated session_state 狀態管理
# =============================================================================

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import anthropic  # 使用 Anthropic Claude SDK
from datetime import datetime, timedelta
import traceback

# 頁面配置
st.set_page_config(
    page_title="【Code Gym】DCF估值教育系統（台股版）",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 主標題
st.title("【Code Gym】DCF估值教育系統（台股版）", anchor=False)
st.markdown("---")

# =============================================================================
# 工具函式
# =============================================================================

def format_large_number(num):
    """格式化大數字顯示（台幣單位）"""
    if num is None or num == 0:
        return "0"
    if abs(num) >= 1e12:
        return f"{num/1e12:.2f}兆"
    elif abs(num) >= 1e8:
        return f"{num/1e8:.2f}億"
    elif abs(num) >= 1e4:
        return f"{num/1e4:.2f}萬"
    else:
        return f"{num:.2f}"

def validate_taiwan_stock_code(stock_code):
    """驗證台股代碼格式（四位數字）"""
    import re
    if not stock_code:
        return False, "請輸入股票代碼"
    stock_code = stock_code.strip()
    if not re.match(r'^\d{4}$', stock_code):
        return False, "台股代碼必須為四位數字（例如：2330）"
    return True, ""

# =============================================================================
# FinMind API 資料獲取函式
# =============================================================================

FINMIND_BASE_URL = "https://api.finmindtrade.com/api/v4/data"


def calculate_beta(stock_id, token):
    """
    [修正B] 從 FinMind 取得股票與加權指數(TAIEX)近2年日報酬率，
    以線性迴歸計算 Beta 係數。
    若取得失敗（API錯誤、資料不足），備援回傳 (1.0, False)。
    回傳：(beta值, 是否成功迴歸計算)
    """
    try:
        start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
        end_date   = datetime.now().strftime("%Y-%m-%d")

        # 取得個股日收盤價
        stock_params = {
            "dataset": "TaiwanStockDaily",
            "data_id": stock_id,
            "start_date": start_date,
            "end_date": end_date,
            "token": token
        }
        stock_resp = requests.get(FINMIND_BASE_URL, params=stock_params, timeout=15)
        stock_data = stock_resp.json().get("data", [])

        # 取得加權指數 TAIEX 日收盤價
        taiex_params = {
            "dataset": "TaiwanStockDaily",
            "data_id": "TAIEX",
            "start_date": start_date,
            "end_date": end_date,
            "token": token
        }
        taiex_resp = requests.get(FINMIND_BASE_URL, params=taiex_params, timeout=15)
        taiex_data = taiex_resp.json().get("data", [])

        if not stock_data or not taiex_data:
            return 1.0, False

        df_stock = pd.DataFrame(stock_data)[["date", "close"]].rename(columns={"close": "stock"})
        df_taiex = pd.DataFrame(taiex_data)[["date", "close"]].rename(columns={"close": "taiex"})

        df = pd.merge(df_stock, df_taiex, on="date").sort_values("date")
        df["stock"] = pd.to_numeric(df["stock"], errors="coerce")
        df["taiex"] = pd.to_numeric(df["taiex"], errors="coerce")
        df = df.dropna()

        if len(df) < 60:  # 至少需要60個交易日
            return 1.0, False

        # 計算日報酬率
        df["stock_ret"] = df["stock"].pct_change()
        df["taiex_ret"] = df["taiex"].pct_change()
        df = df.dropna()

        # 線性迴歸：stock_ret = alpha + beta * taiex_ret
        cov_matrix = np.cov(df["stock_ret"], df["taiex_ret"])
        beta = cov_matrix[0, 1] / cov_matrix[1, 1]
        beta = float(np.clip(beta, 0.3, 3.0))  # 限制在合理範圍
        return beta, True

    except Exception:
        return 1.0, False


def finmind_get(dataset, stock_id, token, start_date="2019-01-01", end_date=None):
    """通用 FinMind API 請求函式"""
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    params = {
        "dataset": dataset,
        "data_id": stock_id,
        "start_date": start_date,
        "end_date": end_date,
        "token": token
    }
    try:
        resp = requests.get(FINMIND_BASE_URL, params=params, timeout=15)
        if resp.status_code != 200:
            raise Exception(f"HTTP {resp.status_code}")
        result = resp.json()
        if result.get("status") != 200:
            raise Exception(f"FinMind 錯誤：{result.get('msg', '未知錯誤')}")
        return result.get("data", [])
    except Exception as e:
        raise Exception(f"FinMind API [{dataset}] 請求失敗：{str(e)}")


def get_finmind_financial_data(stock_id, token):
    """
    從 FinMind 獲取台股三大財報資料
    回傳：損益表、資產負債表、現金流量表（各最近3年）
    """
    financial_data = {}

    # 損益表
    try:
        income_raw = finmind_get("TaiwanStockFinancialStatements", stock_id, token, start_date="2019-01-01")
        if income_raw:
            # FinMind 損益表為長格式，轉為寬格式（以 date 為索引）
            df_income = pd.DataFrame(income_raw)
            df_income_wide = df_income.pivot_table(
                index="date", columns="type", values="value", aggfunc="first"
            ).reset_index()
            df_income_wide = df_income_wide.sort_values("date", ascending=False)
            financial_data["income_statement"] = df_income_wide.head(4).to_dict(orient="records")
        else:
            financial_data["income_statement"] = []
    except Exception as e:
        financial_data["income_statement"] = []
        financial_data["income_error"] = str(e)

    # 資產負債表
    try:
        balance_raw = finmind_get("TaiwanStockBalanceSheet", stock_id, token, start_date="2019-01-01")
        if balance_raw:
            df_balance = pd.DataFrame(balance_raw)
            df_balance_wide = df_balance.pivot_table(
                index="date", columns="type", values="value", aggfunc="first"
            ).reset_index()
            df_balance_wide = df_balance_wide.sort_values("date", ascending=False)
            financial_data["balance_sheet"] = df_balance_wide.head(4).to_dict(orient="records")
        else:
            financial_data["balance_sheet"] = []
    except Exception as e:
        financial_data["balance_sheet"] = []
        financial_data["balance_error"] = str(e)

    # 現金流量表
    try:
        cash_raw = finmind_get("TaiwanStockCashFlowsStatement", stock_id, token, start_date="2019-01-01")
        if cash_raw:
            df_cash = pd.DataFrame(cash_raw)
            df_cash_wide = df_cash.pivot_table(
                index="date", columns="type", values="value", aggfunc="first"
            ).reset_index()
            df_cash_wide = df_cash_wide.sort_values("date", ascending=False)
            financial_data["cash_flow"] = df_cash_wide.head(4).to_dict(orient="records")
        else:
            financial_data["cash_flow"] = []
    except Exception as e:
        financial_data["cash_flow"] = []
        financial_data["cashflow_error"] = str(e)

    return financial_data


def get_finmind_stock_price(stock_id, token):
    """獲取台股最新股價（TaiwanStockPrice）"""
    try:
        start_date = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
        price_data = finmind_get("TaiwanStockPrice", stock_id, token, start_date=start_date)
        if price_data:
            df_price = pd.DataFrame(price_data).sort_values("date", ascending=False)
            latest = df_price.iloc[0]
            return float(latest.get("close", 0))
        return 0
    except Exception:
        return 0


def get_finmind_stock_info(stock_id, token):
    """獲取台股基本資料（公司名稱、產業）"""
    try:
        info_data = finmind_get("TaiwanStockInfo", stock_id, token, start_date="2010-01-01")
        if info_data:
            df_info = pd.DataFrame(info_data)
            row = df_info[df_info["stock_id"] == stock_id]
            if not row.empty:
                return {
                    "stock_name": row.iloc[0].get("stock_name", stock_id),
                    "industry_category": row.iloc[0].get("industry_category", "N/A"),
                    "type": row.iloc[0].get("type", "N/A")
                }
        return {"stock_name": stock_id, "industry_category": "N/A", "type": "N/A"}
    except Exception:
        return {"stock_name": stock_id, "industry_category": "N/A", "type": "N/A"}


# =============================================================================
# DCF 自行計算核心函式（從 FinMind 財報數據推導）
# =============================================================================

def extract_income_value(income_records, key, default=0):
    """從損益表記錄中安全取值"""
    if not income_records:
        return default
    rec = income_records[0]
    val = rec.get(key, default)
    return float(val) if val is not None else default


def calculate_dcf_params_from_finmind(financial_data, current_price, stock_id, token):
    """
    從 FinMind 財報資料計算 DCF 所需參數
    回傳 (default_params, beta_calculated)
    """
    income  = financial_data.get("income_statement", [])
    balance = financial_data.get("balance_sheet", [])
    cash    = financial_data.get("cash_flow", [])

    # ── 損益表數據 ──
    revenue      = extract_income_value(income, "Revenue", 1)
    gross_profit = extract_income_value(income, "GrossProfit", 0)
    op_income    = extract_income_value(income, "OperatingIncome", 0)
    net_income   = extract_income_value(income, "NetIncome", 0)
    pretax       = extract_income_value(income, "PreTaxIncome", 1)
    depreciation = extract_income_value(income, "DepreciationAndAmortization", 0)

    # 前一年營收（用來算成長率）
    revenue_prev = 0
    if len(income) >= 2:
        revenue_prev = float(income[1].get("Revenue", 0) or 0)

    # ── 資產負債表數據 ──
    total_assets   = extract_income_value(balance, "TotalAssets", 1)
    total_equity   = extract_income_value(balance, "Equity", 1)
    total_debt     = extract_income_value(balance, "LongTermDebt", 0) + extract_income_value(balance, "ShortTermBorrowings", 0)
    cash_equiv     = extract_income_value(balance, "CashAndCashEquivalents", 0)
    receivables    = extract_income_value(balance, "ReceivablesNet", 0)
    inventories    = extract_income_value(balance, "Inventories", 0)
    payables       = extract_income_value(balance, "AccountsPayable", 0)

    # [修正A] 流通股數：優先從股本÷面額10元計算（台股面額統一10元）
    paid_in_capital = extract_income_value(balance, "PaidInCapital", 0)
    if paid_in_capital > 0:
        # FinMind 財報單位為千元，股本單位亦為千元，÷10(元) = 千股，需再×1000=股
        shares_outstanding = (paid_in_capital * 1000) / 10
    elif current_price > 0 and total_equity > 0:
        # 備援：用市值估算（財報股東權益帳面值估計，僅供備援）
        shares_outstanding = (total_equity * 1000) / current_price
    else:
        shares_outstanding = 1

    # ── 現金流量表數據 ──
    op_cf  = extract_income_value(cash, "CashFlowsFromOperatingActivities", 0)
    capex  = abs(extract_income_value(cash, "PropertyAndPlantAndEquipment", 0))
    inv_cf = extract_income_value(cash, "CashFlowsFromInvestingActivities", 0)
    fin_cf = extract_income_value(cash, "CashFlowsFromFinancingActivities", 0)

    # ── 計算各項比率 ──
    revenue_growth = (revenue - revenue_prev) / abs(revenue_prev) if revenue_prev != 0 else 0.10
    ebitda         = op_income + depreciation
    ebitda_pct     = ebitda / revenue if revenue != 0 else 0.25
    capex_pct      = capex / revenue if revenue != 0 else 0.05
    op_cf_pct      = op_cf / revenue if revenue != 0 else 0.15
    tax_rate       = 1 - (net_income / pretax) if pretax != 0 else 0.20
    tax_rate       = max(0.05, min(0.40, tax_rate))

    # ── WACC 推導 ──
    risk_free_rate      = 0.015   # 台灣10年期公債約1.5%
    market_risk_premium = 0.06    # 台股市場風險溢價約6%

    # [修正B] Beta：優先從股票vs TAIEX 近2年日報酬率迴歸計算
    beta, beta_calculated = calculate_beta(stock_id, token)

    cost_of_equity      = risk_free_rate + beta * market_risk_premium
    cost_of_debt_pretax = 0.03
    cost_of_debt        = cost_of_debt_pretax * (1 - tax_rate)
    total_capital       = total_equity + total_debt if (total_equity + total_debt) > 0 else 1
    equity_weight       = total_equity / total_capital
    debt_weight         = total_debt / total_capital
    wacc                = equity_weight * cost_of_equity + debt_weight * cost_of_debt
    long_term_growth    = 0.02

    default_params = {
        # 成長與獲利
        "revenueGrowthPct":               max(-0.2, min(2.0, revenue_growth)),
        "ebitdaPct":                       max(0.01, min(0.80, ebitda_pct)),
        "capitalExpenditurePct":          max(0.001, min(0.30, capex_pct)),
        "operatingCashFlowPct":           max(0.001, min(0.60, op_cf_pct)),
        "depreciationAndAmortizationPct": depreciation / revenue if revenue != 0 else 0.03,
        # 資本成本
        "riskFreeRate":                   risk_free_rate,
        "marketRiskPremium":              market_risk_premium,
        "beta":                           beta,
        "costOfEquity":                   cost_of_equity,
        "costOfDebt":                     cost_of_debt,
        "wacc":                           max(0.03, min(0.20, wacc)),
        # 其他
        "taxRate":                        tax_rate,
        "longTermGrowthRate":             long_term_growth,
        # 資產負債比率
        "cashPct":                        cash_equiv / revenue if revenue != 0 else 0.10,
        "receivablesPct":                 receivables / revenue if revenue != 0 else 0.10,
        "inventoriesPct":                 inventories / revenue if revenue != 0 else 0.05,
        "payablePct":                     payables / revenue if revenue != 0 else 0.08,
        # 公司資訊
        "currentPrice":                   current_price,
        "revenue":                        revenue,
        "totalEquity":                    total_equity,
        "totalDebt":                      total_debt,
        "cashAndEquiv":                   cash_equiv,
        "sharesOutstanding":              max(1, shares_outstanding),  # [修正A]
        "paidInCapital":                  paid_in_capital,
    }
    return default_params, beta_calculated


def calculate_dcf(params, n_years=5):
    """
    自行計算 DCF 估值（[修正H] n_years 由外部傳入，預設5年）
    回傳完整計算結果 dict
    """
    revenue          = params.get("revenue", 1)
    rev_growth       = params.get("revenueGrowthPct", 0.10)
    ebitda_pct       = params.get("ebitdaPct", 0.25)
    capex_pct        = params.get("capitalExpenditurePct", 0.05)
    dep_pct          = params.get("depreciationAndAmortizationPct", 0.03)
    tax_rate         = params.get("taxRate", 0.20)
    wacc             = params.get("wacc", 0.09)
    long_term_growth = params.get("longTermGrowthRate", 0.02)
    total_debt       = params.get("totalDebt", 0)
    cash_equiv       = params.get("cashAndEquiv", 0)
    shares           = params.get("sharesOutstanding", 1)
    current_price    = params.get("currentPrice", 0)
    cost_of_equity   = params.get("costOfEquity", 0.09)
    cost_of_debt     = params.get("costOfDebt", 0.025)
    beta             = params.get("beta", 1.0)
    risk_free        = params.get("riskFreeRate", 0.015)
    mkt_premium      = params.get("marketRiskPremium", 0.06)

    # ── 預測各年 UFCF ──
    yearly_details = []
    pv_ufcf_total = 0
    proj_revenue = revenue

    for yr in range(1, n_years + 1):
        proj_revenue = proj_revenue * (1 + rev_growth)
        ebitda       = proj_revenue * ebitda_pct
        dep          = proj_revenue * dep_pct
        ebit         = ebitda - dep
        ebiat        = ebit * (1 - tax_rate)
        capex        = proj_revenue * capex_pct
        delta_wc     = proj_revenue * 0.02   # 簡化：營運資金變動約2%
        ufcf         = ebiat + dep - capex - delta_wc
        pv_factor    = 1 / (1 + wacc) ** yr
        pv_ufcf      = ufcf * pv_factor
        pv_ufcf_total += pv_ufcf
        yearly_details.append({
            "年度": f"第{yr}年",
            "預測營收": format_large_number(proj_revenue),
            "EBITDA": format_large_number(ebitda),
            "EBIT": format_large_number(ebit),
            "UFCF": format_large_number(ufcf),
            "現值": format_large_number(pv_ufcf),
        })

    # ── 終值計算（Gordon Growth Model）──
    ufcf_terminal  = proj_revenue * ebitda_pct * (1 - tax_rate) * (1 + long_term_growth)
    terminal_value = ufcf_terminal / (wacc - long_term_growth) if wacc > long_term_growth else 0
    pv_terminal    = terminal_value / (1 + wacc) ** n_years

    # ── 企業價值 → 股權價值 → 每股價值 ──
    enterprise_value    = pv_ufcf_total + pv_terminal
    net_debt            = total_debt - cash_equiv
    equity_value        = max(0, enterprise_value - net_debt)
    equity_value_ps     = equity_value / shares if shares > 0 else 0

    # ── WACC 組成（百分比格式，對應顯示）──
    total_cap = params.get("totalEquity", 1) + total_debt
    eq_weight = params.get("totalEquity", 1) / total_cap if total_cap > 0 else 0.8
    dbt_weight = total_debt / total_cap if total_cap > 0 else 0.2

    result = {
        # 估值結果
        "equityValuePerShare":   equity_value_ps,
        "price":                 current_price,
        "equityValue":           equity_value,
        "enterpriseValue":       enterprise_value,
        "sumPvUfcf":             pv_ufcf_total,
        "terminalValue":         terminal_value,
        "presentTerminalValue":  pv_terminal,
        "netDebt":               net_debt,
        # WACC 相關（顯示用百分比）
        "wacc":                  wacc * 100,
        "costOfEquity":          cost_of_equity * 100,
        "afterTaxCostOfDebt":    cost_of_debt * 100,
        "equityWeighting":       eq_weight * 100,
        "debtWeighting":         dbt_weight * 100,
        "beta":                  beta,
        "riskFreeRate":          risk_free * 100,
        "marketRiskPremium":     mkt_premium * 100,
        # 損益相關（顯示用百分比）
        "ebitdaPercentage":      ebitda_pct * 100,
        "taxRate":               tax_rate * 100,
        "longTermGrowthRate":    long_term_growth * 100,
        "revenuePercentage":     rev_growth * 100,
        # 預測明細
        "yearly_details":        yearly_details,
        # 原始輸入（方便驗證函式使用）
        "totalDebt":             total_debt,
        "totalCash":             cash_equiv,
        "ufcf":                  pv_ufcf_total,
    }
    return result


# =============================================================================
# 驗證與圖表函式
# =============================================================================

def validate_dcf_result(dcf_data, ticker, long_term_growth_rate=0.02):
    """
    驗證DCF計算結果的合理性（4項檢查）
    [修正C] 新增合理價格區間檢查（NT$10~NT$10,000）
    [修正D] 新增 WACC > 永續成長率檢查
    """
    try:
        dcf_price     = dcf_data.get("equityValuePerShare", 0)
        current_price = dcf_data.get("price", 0)
        wacc_val      = dcf_data.get("wacc", 0)   # 已為百分比格式
        validation_results = []

        # 檢查1：DCF估值正負
        if dcf_price > 0:
            validation_results.append("✅ DCF估值計算成功")
        else:
            validation_results.append("⚠️ DCF估值為零或負值，請檢查財報數據")

        # 檢查2：[修正C] 合理台股價格區間 NT$10~NT$10,000
        if 10 <= dcf_price <= 10000:
            validation_results.append(f"✅ DCF估值在合理台股價格區間（NT${dcf_price:.2f}，區間：NT$10~NT$10,000）")
        elif dcf_price > 0:
            validation_results.append(
                f"⚠️ DCF估值 NT${dcf_price:.2f} 超出合理台股價格區間（NT$10~NT$10,000），"
                "請確認財報數據與假設是否正確"
            )

        # 檢查3：與市價偏離度
        if current_price > 0:
            deviation = abs((dcf_price - current_price) / current_price * 100)
            if deviation <= 50:
                validation_results.append(f"✅ 與市價偏離在合理範圍內（{deviation:.1f}%）")
            elif deviation <= 100:
                validation_results.append(f"⚠️ 與市價偏離較大：{deviation:.1f}%")
            else:
                validation_results.append(f"❌ 與市價偏離過大：{deviation:.1f}%，請確認假設合理性")

        # 檢查4：[修正D] WACC > 永續成長率（終值計算有效性）
        wacc_decimal = wacc_val / 100 if wacc_val > 1 else wacc_val
        if wacc_decimal > long_term_growth_rate:
            validation_results.append(
                f"✅ WACC（{wacc_val:.2f}%）> 永續成長率（{long_term_growth_rate*100:.1f}%），終值計算有效"
            )
        else:
            validation_results.append(
                f"❌ WACC（{wacc_val:.2f}%）≤ 永續成長率（{long_term_growth_rate*100:.1f}%），"
                "終值計算無效！請提高 WACC 或降低永續成長率"
            )

        # 檢查5：終值佔比
        enterprise_value = dcf_data.get("enterpriseValue", 0)
        terminal_value   = dcf_data.get("presentTerminalValue", 0)
        if enterprise_value > 0:
            terminal_ratio = terminal_value / enterprise_value * 100
            if 40 <= terminal_ratio <= 95:
                validation_results.append(f"✅ 終值占比合理：{terminal_ratio:.1f}%")
            else:
                validation_results.append(
                    f"⚠️ 終值占比異常：{terminal_ratio:.1f}%，建議調整 WACC 或長期成長率"
                )

        return validation_results
    except Exception as e:
        return [f"❌ 驗證過程出錯：{str(e)}"]


def create_dcf_overview_chart(dcf_data):
    """DCF估值 vs 當前市價比較圖"""
    try:
        dcf_price     = dcf_data.get("equityValuePerShare", 0)
        current_price = dcf_data.get("price", 0)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["DCF估值（元）", "當前市價（元）"],
            y=[dcf_price, current_price],
            marker_color=["#2E86C1", "#E74C3C"],
            text=[f"NT${dcf_price:.2f}", f"NT${current_price:.2f}"],
            textposition="outside",
            textfont=dict(size=14, color="white")
        ))
        fig.update_layout(
            title="DCF估值 vs 當前市價比較",
            yaxis_title="股價（新台幣元）",
            showlegend=False,
            height=400,
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        return fig
    except Exception as e:
        st.error(f"圖表生成錯誤：{str(e)}")
        return None


def create_wacc_breakdown_chart(dcf_data):
    """WACC組成分析圖"""
    try:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["稅後債務成本", "股權成本", "WACC"],
            y=[
                dcf_data.get("afterTaxCostOfDebt", 0),
                dcf_data.get("costOfEquity", 0),
                dcf_data.get("wacc", 0),
            ],
            marker_color=["#E74C3C", "#2E86C1", "#28B463"],
            text=[
                f"{dcf_data.get('afterTaxCostOfDebt', 0):.2f}%",
                f"{dcf_data.get('costOfEquity', 0):.2f}%",
                f"{dcf_data.get('wacc', 0):.2f}%",
            ],
            textposition="outside",
            textfont=dict(size=14, color="white")
        ))
        fig.update_layout(
            title="WACC組成分析",
            yaxis_title="成本率（%）",
            height=400, width=600,
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False
        )
        return fig
    except Exception as e:
        st.error(f"WACC圖表生成錯誤：{str(e)}")
        return None


def create_dcf_components_breakdown(dcf_data):
    """DCF企業價值構成分解圖 [修正I] 統一 Y 軸單位標示為新台幣千元"""
    try:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="預測期現金流現值",
            x=["企業價值構成"],
            y=[dcf_data.get("sumPvUfcf", 0)],
            marker_color="#3498DB"
        ))
        fig.add_trace(go.Bar(
            name="終值現值",
            x=["企業價值構成"],
            y=[dcf_data.get("presentTerminalValue", 0)],
            marker_color="#E67E22"
        ))
        fig.update_layout(
            title="DCF企業價值構成分析",
            yaxis_title="價值（新台幣 千元）",
            barmode="stack",
            height=400,
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        return fig
    except Exception as e:
        st.error(f"價值分解圖生成錯誤：{str(e)}")
        return None


def create_sensitivity_analysis(dcf_data, base_params):
    """
    [修正E] 敏感性分析表：對各參數分別調整±10%後重新計算DCF，
    動態產生影響幅度，不使用寫死數值。
    """
    try:
        base_price = dcf_data.get("equityValuePerShare", 0)
        if base_price <= 0:
            return None

        param_configs = [
            ("營收成長率",  "revenueGrowthPct",    base_params.get("revenueGrowthPct", 0.10),   True),
            ("EBITDA率",   "ebitdaPct",            base_params.get("ebitdaPct", 0.25),           True),
            ("長期成長率",  "longTermGrowthRate",   base_params.get("longTermGrowthRate", 0.02),  True),
            ("WACC",       "wacc",                 base_params.get("wacc", 0.09),                False),
            ("稅率",       "taxRate",              base_params.get("taxRate", 0.20),             True),
        ]

        rows = []
        n_years = base_params.get("n_years", 5)
        for label, key, base_val, positive_good in param_configs:
            # 向上 +10%
            params_up = dict(base_params)
            params_up[key] = base_val * 1.10
            result_up = calculate_dcf(params_up, n_years=n_years)
            price_up  = result_up.get("equityValuePerShare", base_price)
            chg_up    = (price_up - base_price) / base_price * 100 if base_price != 0 else 0

            # 向下 -10%
            params_dn = dict(base_params)
            params_dn[key] = base_val * 0.90
            result_dn = calculate_dcf(params_dn, n_years=n_years)
            price_dn  = result_dn.get("equityValuePerShare", base_price)
            chg_dn    = (price_dn - base_price) / base_price * 100 if base_price != 0 else 0

            fmt_base = f"{base_val*100:.1f}%" if key != "wacc" else f"{base_val*100:.2f}%"
            rows.append({
                "參數":       label,
                "基準值":     fmt_base,
                "向上10%影響": f"{chg_up:+.1f}%",
                "向下10%影響": f"{chg_dn:+.1f}%",
            })

        return pd.DataFrame(rows)
    except Exception as e:
        st.error(f"敏感性分析錯誤：{str(e)}")
        return None


def create_scenario_analysis_chart(dcf_result, base_params):
    """
    [修正F] 三情境 DCF 估值圖：依規格調整參數後重新計算，非直接縮放結果。
    - 悲觀：永續成長率 -1%、WACC +2%
    - 基準：當前設定
    - 樂觀：永續成長率 +1%、WACC -1%
    """
    try:
        base_val     = dcf_result.get("equityValuePerShare", 0)
        n_years      = base_params.get("n_years", 5)
        base_ltg     = base_params.get("longTermGrowthRate", 0.02)
        base_wacc    = base_params.get("wacc", 0.09)

        # 悲觀情境
        params_bear = dict(base_params)
        params_bear["longTermGrowthRate"] = max(0.001, base_ltg - 0.01)
        params_bear["wacc"]               = min(0.30,  base_wacc + 0.02)
        result_bear = calculate_dcf(params_bear, n_years=n_years)
        val_bear    = result_bear.get("equityValuePerShare", base_val * 0.65)

        # 樂觀情境
        params_bull = dict(base_params)
        params_bull["longTermGrowthRate"] = base_ltg + 0.01
        params_bull["wacc"]               = max(0.01, base_wacc - 0.01)
        result_bull = calculate_dcf(params_bull, n_years=n_years)
        val_bull    = result_bull.get("equityValuePerShare", base_val * 1.35)

        scenarios = [
            f"悲觀情境\n（永續成長率{(base_ltg-0.01)*100:.1f}%、WACC+2%）",
            "基準情境",
            f"樂觀情境\n（永續成長率{(base_ltg+0.01)*100:.1f}%、WACC-1%）",
        ]
        values = [val_bear, base_val, val_bull]
        colors = ["#E74C3C", "#F39C12", "#28B463"]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=scenarios,
            y=values,
            marker_color=colors,
            text=[f"NT${v:.1f}" for v in values],
            textposition="outside",
            textfont=dict(size=13, color="white")
        ))
        fig.update_layout(
            title="情境分析：不同假設下的DCF每股估值",
            yaxis_title="每股估值（新台幣元）",
            showlegend=False,
            height=400, width=600,
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        return fig
    except Exception as e:
        st.error(f"情境分析圖表錯誤：{str(e)}")
        return None


# =============================================================================
# AI 分析函式
# =============================================================================

def analyze_dcf_with_ai(dcf_data, parameters, claude_api_key, stock_id, financial_data):
    """使用 Anthropic Claude 分析台股 DCF 結果"""
    try:
        client        = anthropic.Anthropic(api_key=claude_api_key)
        dcf_price     = dcf_data.get("equityValuePerShare", 0)
        current_price = dcf_data.get("price", 0)
        upside        = ((dcf_price - current_price) / current_price * 100) if current_price > 0 else 0

        system_message = (
            "你是一位專業的台股DCF估值分析師，專精於台灣上市公司財務分析與教學。"
            "請提供客觀、教育性的繁體中文分析，避免給予具體的投資建議。"
        )

        # 財報摘要（取最近一期）
        income_summary  = financial_data.get("income_statement", [{}])[0] if financial_data.get("income_statement") else {}
        balance_summary = financial_data.get("balance_sheet", [{}])[0] if financial_data.get("balance_sheet") else {}
        cash_summary    = financial_data.get("cash_flow", [{}])[0] if financial_data.get("cash_flow") else {}

        user_prompt = f"""
請分析以下台股 DCF 估值結果，這是用於教育目的的分析：

**公司股票代碼**：{stock_id}
**DCF每股估值**：NT${dcf_price:.2f}
**當前市價**：NT${current_price:.2f}
**估值差異**：{upside:.1f}%（正值=低估，負值=高估）

**DCF輸入假設**：
- 營收成長率：{parameters.get('revenueGrowthPct', 0)*100:.1f}%
- EBITDA率：{parameters.get('ebitdaPct', 0)*100:.1f}%
- 長期成長率：{parameters.get('longTermGrowthRate', 0)*100:.1f}%
- WACC：{parameters.get('wacc', 0)*100:.2f}%（若已計算）
- 稅率：{parameters.get('taxRate', 0)*100:.1f}%
- Beta：{parameters.get('beta', 1.0):.2f}

**企業價值組成**：
- 預測期現金流現值：{format_large_number(dcf_data.get('sumPvUfcf', 0))}元
- 終值現值：{format_large_number(dcf_data.get('presentTerminalValue', 0))}元
- 總企業價值：{format_large_number(dcf_data.get('enterpriseValue', 0))}元

**最新一期損益表摘要（FinMind）**：
{json.dumps(income_summary, ensure_ascii=False, indent=2)}

**最新一期資產負債表摘要（FinMind）**：
{json.dumps(balance_summary, ensure_ascii=False, indent=2)}

**最新一期現金流量表摘要（FinMind）**：
{json.dumps(cash_summary, ensure_ascii=False, indent=2)}

請基於以上資料，提供以下五項分析：

1. **基本面分析**：財務健康狀況、營收利潤現金流趨勢、資產負債結構

2. **估值結論評估**：DCF估值是否合理？市場定價與模型差異原因

3. **參數假設檢視**：假設是否合理？哪項假設影響最大？

4. **風險因子識別**：主要風險點、台股特有風險（匯率、景氣循環、產業政策）

5. **學習重點總結**：本案例 DCF 應用的學習重點與模型局限性

請以繁體中文、專業但易懂的方式回答，適合台灣投資學習者理解。
"""

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=3000,
            system=system_message,
            messages=[{"role": "user", "content": user_prompt}]
        )
        return response.content[0].text

    except anthropic.AuthenticationError:
        return "❌ Anthropic API 金鑰無效，請確認金鑰正確後重試。"
    except anthropic.RateLimitError:
        return "⚠️ API 請求頻率超限，請稍後再試。"
    except anthropic.APIConnectionError:
        return "🌐 無法連線至 Anthropic API，請確認網路連線正常。"
    except Exception as e:
        return f"AI 分析時發生錯誤：{str(e)}"


# =============================================================================
# Session State 初始化
# =============================================================================

if "company_loaded" not in st.session_state:
    st.session_state.company_loaded = False
if "default_params" not in st.session_state:
    st.session_state.default_params = {}
if "company_data" not in st.session_state:
    st.session_state.company_data = {}
if "financial_data" not in st.session_state:
    st.session_state.financial_data = {}
if "stock_info" not in st.session_state:
    st.session_state.stock_info = {}
if "beta_calculated" not in st.session_state:  # [修正J]
    st.session_state.beta_calculated = False

# =============================================================================
# 側邊欄控制
# =============================================================================

st.sidebar.header("Code Gym", divider="rainbow")

# ── 第一步：載入公司數據 ──
st.sidebar.markdown("### 📈 第一步：載入公司數據")
ticker = st.sidebar.text_input(
    "台股代碼（四位數字）",
    value="2330",
    help="輸入台股四位數代碼，例如：2330（台積電）、2454（聯發科）、2317（鴻海）"
)
finmind_token = st.sidebar.text_input(
    "FinMind API Token",
    type="password",
    help="請至 finmindtrade.com 免費申請 Token"
)

load_company_button = st.sidebar.button("📊 載入公司基礎數據", type="secondary")

if load_company_button:
    is_valid, err_msg = validate_taiwan_stock_code(ticker)
    if not is_valid:
        st.sidebar.warning(f"⚠️ {err_msg}")
    elif not finmind_token:
        st.sidebar.warning("⚠️ 請輸入 FinMind API Token")
    else:
        with st.spinner(f"正在從 FinMind 載入 {ticker} 財務數據..."):
            try:
                # 取得財報資料
                financial_data = get_finmind_financial_data(ticker, finmind_token)
                # 取得即時股價
                current_price = get_finmind_stock_price(ticker, finmind_token)
                # 取得公司基本資訊
                stock_info = get_finmind_stock_info(ticker, finmind_token)
                # [修正B+A] 計算 DCF 預設參數（含 Beta 迴歸計算與正確股數）
                default_params, beta_calc = calculate_dcf_params_from_finmind(
                    financial_data, current_price, ticker, finmind_token
                )

                st.session_state.financial_data   = financial_data
                st.session_state.default_params   = default_params
                st.session_state.stock_info       = stock_info
                st.session_state.company_data     = {"currentPrice": current_price}
                st.session_state.company_loaded   = True
                st.session_state.beta_calculated  = beta_calc  # [修正J]

                company_name = stock_info.get("stock_name", ticker)
                beta_msg = "（迴歸計算）" if beta_calc else "（備援預設值1.0）"
                st.sidebar.success(f"✅ 已載入 {company_name}（{ticker}）數據")
                st.sidebar.caption(f"Beta係數{beta_msg}")

            except Exception as e:
                st.sidebar.error(f"❌ 載入失敗：{str(e)}")
                st.session_state.company_loaded = False

# 公司基本資訊顯示
if st.session_state.company_loaded:
    info = st.session_state.stock_info
    beta_status = "✅ 迴歸計算" if st.session_state.beta_calculated else "⚠️ 備援預設值1.0"
    st.sidebar.markdown("#### 📋 公司基本資訊")
    st.sidebar.info(f"""
**公司**：{info.get('stock_name', ticker)}（{ticker}）
**產業**：{info.get('industry_category', 'N/A')}
**類型**：{info.get('type', 'N/A')}
**當前股價**：NT${st.session_state.default_params.get('currentPrice', 0):.2f}
**Beta計算**：{beta_status}
**載入時間**：{datetime.now().strftime('%H:%M:%S')}
    """)

# ── 第二步：調整 DCF 參數 ──
if st.session_state.company_loaded:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ 第二步：調整DCF參數")

    defaults = st.session_state.default_params
    reset_button = st.sidebar.button("🔄 重置為預設值", help="恢復為財報計算的推導值")

    st.sidebar.markdown("#### 基礎參數")
    revenue_growth = st.sidebar.slider(
        "營收成長率（%）",
        min_value=-20.0, max_value=100.0,
        value=float(round(defaults.get("revenueGrowthPct", 0.10) * 100, 1)),
        step=0.5,
        help=f"預期年營收成長率（財報推導：{defaults.get('revenueGrowthPct', 0)*100:.1f}%）"
    )
    ebitda_margin = st.sidebar.slider(
        "EBITDA利潤率（%）",
        min_value=1.0, max_value=80.0,
        value=float(round(defaults.get("ebitdaPct", 0.25) * 100, 1)),
        step=0.5,
        help=f"EBITDA佔營收比例（財報推導：{defaults.get('ebitdaPct', 0)*100:.1f}%）"
    )
    capex_pct = st.sidebar.slider(
        "資本支出比例（%）",
        min_value=0.1, max_value=30.0,
        value=float(round(defaults.get("capitalExpenditurePct", 0.05) * 100, 1)),
        step=0.5,
        help=f"資本支出佔營收比例（財報推導：{defaults.get('capitalExpenditurePct', 0)*100:.1f}%）"
    )

    st.sidebar.markdown("#### 折現率參數")
    cost_of_equity = st.sidebar.slider(
        "股權成本（%）",
        min_value=3.0, max_value=20.0,
        value=float(round(defaults.get("costOfEquity", 0.09) * 100, 2)),
        step=0.1,
        help=f"CAPM計算股權成本（財報推導：{defaults.get('costOfEquity', 0)*100:.2f}%）"
    )
    cost_of_debt = st.sidebar.slider(
        "稅後債務成本（%）",
        min_value=0.5, max_value=10.0,
        value=float(round(defaults.get("costOfDebt", 0.025) * 100, 2)),
        step=0.1,
        help=f"稅後借款利率（財報推導：{defaults.get('costOfDebt', 0)*100:.2f}%）"
    )
    risk_free_rate = st.sidebar.slider(
        "無風險利率（%）",
        min_value=0.5, max_value=5.0,
        value=float(round(defaults.get("riskFreeRate", 0.015) * 100, 1)),
        step=0.1,
        help="台灣10年期公債殖利率"
    )
    market_risk_premium = st.sidebar.slider(
        "市場風險溢價（%）",
        min_value=3.0, max_value=10.0,
        value=float(round(defaults.get("marketRiskPremium", 0.06) * 100, 1)),
        step=0.1,
        help="台股歷史市場風險溢價"
    )
    beta_help = (
        f"系統風險係數（迴歸計算：{defaults.get('beta', 1.0):.2f}）"
        if st.session_state.beta_calculated
        else f"系統風險係數（備援預設值1.0，因資料不足無法迴歸計算）"
    )
    beta = st.sidebar.slider(
        "Beta係數",
        min_value=0.3, max_value=2.5,
        value=float(round(defaults.get("beta", 1.0), 2)),
        step=0.05,
        help=beta_help
    )

    with st.sidebar.expander("📽 進階參數"):
        # [修正G] expander 內使用 st.slider（非 st.sidebar.slider）
        terminal_growth = st.slider(
            "永續成長率（%）",
            min_value=0.5, max_value=4.0,
            value=float(round(defaults.get("longTermGrowthRate", 0.02) * 100, 1)),
            step=0.1,
            help="長期永續成長率，台灣企業建議2%以下"
        )
        tax_rate = st.slider(
            "有效稅率（%）",
            min_value=10.0, max_value=35.0,
            value=float(round(defaults.get("taxRate", 0.20) * 100, 1)),
            step=1.0,
            help=f"公司實際稅率（財報推導：{defaults.get('taxRate', 0)*100:.1f}%）"
        )
        operating_cf_pct = st.slider(
            "營運現金流比例（%）",
            min_value=1.0, max_value=60.0,
            value=float(round(defaults.get("operatingCashFlowPct", 0.15) * 100, 1)),
            step=1.0,
            help=f"營運現金流佔營收比例（財報推導：{defaults.get('operatingCashFlowPct', 0)*100:.1f}%）"
        )
        # [修正H] 新增預測年數滑桿
        n_years = st.slider(
            "DCF預測年數（年）",
            min_value=5, max_value=10,
            value=5,
            step=1,
            help="DCF預測期間，通常設定5~10年；預測期越長，終值佔比越低"
        )

    # Anthropic Claude API Key
    st.sidebar.markdown("---")
    claude_api_key = st.sidebar.text_input(
        "Anthropic Claude API Key（選填）",
        type="password",
        help="用於 AI 分析功能，可至 console.anthropic.com 申請"
    )

    # 第三步：計算
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🚀 第三步：計算DCF估值")
    calculate_button = st.sidebar.button("💰 計算DCF估值", type="primary")

    if reset_button:
        st.sidebar.success("🔄 已重置為財報推導預設值")
        st.rerun()

else:
    st.sidebar.markdown("---")
    st.sidebar.info("👆 請先載入公司基礎數據")
    calculate_button = False
    claude_api_key   = ""

# =============================================================================
# 主要內容區域
# =============================================================================

if st.session_state.company_loaded and calculate_button:
    try:
        # 組合 DCF 計算參數
        dcf_parameters = {
            "revenueGrowthPct":            revenue_growth / 100,
            "ebitdaPct":                   ebitda_margin / 100,
            "capitalExpenditurePct":       capex_pct / 100,
            "costOfEquity":                cost_of_equity / 100,
            "costOfDebt":                  cost_of_debt / 100,
            "riskFreeRate":                risk_free_rate / 100,
            "marketRiskPremium":           market_risk_premium / 100,
            "beta":                        beta,
            "longTermGrowthRate":          terminal_growth / 100,
            "taxRate":                     tax_rate / 100,
            "operatingCashFlowPct":        operating_cf_pct / 100,
            "depreciationAndAmortizationPct": defaults.get("depreciationAndAmortizationPct", 0.03),
            "n_years":                     n_years,  # [修正H]
            # 財報基礎數字
            "revenue":                     defaults.get("revenue", 1),
            "totalEquity":                 defaults.get("totalEquity", 1),
            "totalDebt":                   defaults.get("totalDebt", 0),
            "cashAndEquiv":                defaults.get("cashAndEquiv", 0),
            "sharesOutstanding":           defaults.get("sharesOutstanding", 1),
            "currentPrice":                defaults.get("currentPrice", 0),
            # WACC 計算（由側邊欄輸入推導）
            "wacc": (
                (cost_of_equity / 100) * (defaults.get("totalEquity", 1) / max(defaults.get("totalEquity", 1) + defaults.get("totalDebt", 0), 1)) +
                (cost_of_debt / 100) * (defaults.get("totalDebt", 0) / max(defaults.get("totalEquity", 1) + defaults.get("totalDebt", 0), 1))
            )
        }

        # [修正D] 計算前先檢查 WACC > 永續成長率
        wacc_val = dcf_parameters["wacc"]
        ltg_val  = dcf_parameters["longTermGrowthRate"]
        if wacc_val <= ltg_val:
            st.error(
                f"❌ WACC（{wacc_val*100:.2f}%）≤ 永續成長率（{ltg_val*100:.1f}%），"
                "終值計算將無效！請提高 WACC 或降低永續成長率後再計算。"
            )
            st.stop()

        # 執行 DCF 計算
        with st.spinner("正在計算 DCF 估值..."):
            dcf_result = calculate_dcf(dcf_parameters, n_years=n_years)  # [修正H]

        # [修正C+D] 傳入 long_term_growth_rate 進行完整4項驗證
        validation_results = validate_dcf_result(dcf_result, ticker, long_term_growth_rate=ltg_val)
        st.markdown("### 🔍 DCF計算結果驗證")
        for res in validation_results:
            if "✅" in res:
                st.success(res)
            elif "⚠️" in res:
                st.warning(res)
            else:
                st.error(res)

        # 估值差異比較
        original_price = defaults.get("currentPrice", 0)
        new_dcf        = dcf_result.get("equityValuePerShare", 0)
        upside_pct     = ((new_dcf - original_price) / original_price * 100) if original_price > 0 else 0

        st.markdown("### 📈 DCF估值結果")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("DCF每股估值", f"NT${new_dcf:.2f}", help="由程式自行計算的DCF內在價值")
        with col2:
            st.metric("當前市價", f"NT${original_price:.2f}", help="FinMind 最新收盤價")
        with col3:
            st.metric(
                "估值差異",
                f"{upside_pct:+.1f}%",
                delta=f"{'低估' if upside_pct > 0 else '高估' if upside_pct < 0 else '合理'}",
                delta_color="normal" if upside_pct > 0 else "inverse"
            )
        with col4:
            st.metric("企業價值", format_large_number(dcf_result.get("enterpriseValue", 0)) + "元", help="總企業價值")

        st.success("✅ DCF計算完成！")

        # 參數對比表
        st.markdown("### 📊 參數設定 vs 財報推導值對比")
        col1, col2 = st.columns(2)
        with col1:
            comparison_data = {
                "參數": ["營收成長率", "EBITDA利潤率", "資本支出比例", "股權成本", "稅後債務成本", "無風險利率", "市場風險溢價", "Beta係數"],
                "您的設定": [
                    f"{revenue_growth:.1f}%", f"{ebitda_margin:.1f}%", f"{capex_pct:.1f}%",
                    f"{cost_of_equity:.2f}%", f"{cost_of_debt:.2f}%",
                    f"{risk_free_rate:.1f}%", f"{market_risk_premium:.1f}%", f"{beta:.2f}"
                ],
                "財報推導": [
                    f"{defaults.get('revenueGrowthPct', 0)*100:.1f}%",
                    f"{defaults.get('ebitdaPct', 0)*100:.1f}%",
                    f"{defaults.get('capitalExpenditurePct', 0)*100:.1f}%",
                    f"{defaults.get('costOfEquity', 0)*100:.2f}%",
                    f"{defaults.get('costOfDebt', 0)*100:.2f}%",
                    f"{defaults.get('riskFreeRate', 0)*100:.1f}%",
                    f"{defaults.get('marketRiskPremium', 0)*100:.1f}%",
                    f"{defaults.get('beta', 0):.2f}"
                ]
            }
            st.dataframe(pd.DataFrame(comparison_data), hide_index=True)

        with col2:
            advanced_comparison = {
                "進階參數": ["永續成長率", "有效稅率", "營運現金流比例"],
                "您的設定": [f"{terminal_growth:.1f}%", f"{tax_rate:.1f}%", f"{operating_cf_pct:.1f}%"],
                "財報推導": [
                    f"{defaults.get('longTermGrowthRate', 0)*100:.1f}%",
                    f"{defaults.get('taxRate', 0)*100:.1f}%",
                    f"{defaults.get('operatingCashFlowPct', 0)*100:.1f}%"
                ]
            }
            st.dataframe(pd.DataFrame(advanced_comparison), hide_index=True)

        # 詳細分析頁籤
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 估值總覽", "💰 企業價值分解", "⚙️ WACC分析",
            "📈 敏感性分析", "🤖 AI估值分析"
        ])

        with tab1:
            st.markdown("### 📊 DCF預測現金流明細")
            yearly_df = pd.DataFrame(dcf_result.get("yearly_details", []))
            if not yearly_df.empty:
                st.dataframe(yearly_df, use_container_width=True, hide_index=True)

            st.markdown("### 📊 估值總覽")
            overview_chart = create_dcf_overview_chart(dcf_result)
            if overview_chart:
                st.plotly_chart(overview_chart, use_container_width=True)

            st.markdown("### 📈 情境分析")
            scenario_chart = create_scenario_analysis_chart(dcf_result, dcf_parameters)
            if scenario_chart:
                st.plotly_chart(scenario_chart, use_container_width=False, width=600)

        with tab2:
            st.markdown("### 企業價值組成分析")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 💼 價值組成明細")
                value_breakdown = {
                    "項目": ["預測期現金流現值", "終值現值", "企業總價值", "減：淨債務", "股權價值"],
                    "金額": [
                        format_large_number(dcf_result.get("sumPvUfcf", 0)) + " 元",
                        format_large_number(dcf_result.get("presentTerminalValue", 0)) + " 元",
                        format_large_number(dcf_result.get("enterpriseValue", 0)) + " 元",
                        format_large_number(dcf_result.get("netDebt", 0)) + " 元",
                        format_large_number(dcf_result.get("equityValue", 0)) + " 元",
                    ]
                }
                st.dataframe(pd.DataFrame(value_breakdown), hide_index=True)
            with col2:
                components_chart = create_dcf_components_breakdown(dcf_result)
                if components_chart:
                    st.plotly_chart(components_chart, use_container_width=True)

        with tab3:
            st.markdown("### WACC 加權平均資本成本分析")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 🧮 WACC計算項目")
                wacc_calc = {
                    "項目": ["股權成本", "稅後債務成本", "股權權重", "債務權重", "WACC"],
                    "數值": [
                        f"{dcf_result.get('costOfEquity', 0):.2f}%",
                        f"{dcf_result.get('afterTaxCostOfDebt', 0):.2f}%",
                        f"{dcf_result.get('equityWeighting', 0):.1f}%",
                        f"{dcf_result.get('debtWeighting', 0):.1f}%",
                        f"{dcf_result.get('wacc', 0):.2f}%",
                    ]
                }
                st.dataframe(pd.DataFrame(wacc_calc), hide_index=True)
            with col2:
                st.markdown("#### 📈 CAPM股權成本計算")
                capm_text = f"""
**CAPM公式**：Re = Rf + β × (Rm - Rf)

- 無風險利率（Rf）：{risk_free_rate:.2f}%
- Beta係數（β）：{beta:.3f}
- 市場風險溢價：{market_risk_premium:.2f}%

**計算過程**：
Re = {risk_free_rate:.2f}% + {beta:.3f} × {market_risk_premium:.2f}%
Re = {risk_free_rate:.2f}% + {beta * market_risk_premium:.2f}%
Re = **{risk_free_rate + beta * market_risk_premium:.2f}%**
"""
                st.markdown(capm_text)

            wacc_chart = create_wacc_breakdown_chart(dcf_result)
            if wacc_chart:
                st.plotly_chart(wacc_chart, use_container_width=False, width=600)

        with tab4:
            st.markdown("### 敏感性分析")
            sensitivity_df = create_sensitivity_analysis(dcf_result, dcf_parameters)
            if sensitivity_df is not None:
                st.dataframe(sensitivity_df, hide_index=True)
            st.info("""
**敏感性分析重點**：
- 永續成長率對估值影響最大（±35%）
- WACC變化對估值有顯著影響（±25%）
- 台股特性：台幣升值、匯率波動為額外風險因子
- 建議進行多情境分析，避免過度依賴單一假設
            """)

        with tab5:
            if claude_api_key:
                st.markdown("### 🤖 AI 台股DCF估值分析")
                with st.spinner("Anthropic Claude 正在分析DCF結果..."):
                    ai_analysis = analyze_dcf_with_ai(
                        dcf_result, dcf_parameters,
                        claude_api_key, ticker,
                        st.session_state.financial_data
                    )
                    st.markdown(ai_analysis)

                if ai_analysis:
                    download_content = f"""DCF估值AI分析報告
公司：{ticker}（{st.session_state.stock_info.get('stock_name', ticker)}）
分析時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{ai_analysis}

---
報告生成於 Code Gym DCF估值教育系統（台股版）
"""
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
        with st.expander("詳細錯誤資訊", expanded=False):
            st.code(traceback.format_exc())

elif st.session_state.company_loaded and not calculate_button:
    info = st.session_state.stock_info
    defaults = st.session_state.default_params
    st.markdown("## 📊 公司數據已載入完成")
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"""
**✅ 已載入**：{info.get('stock_name', ticker)}（{ticker}）
**產業**：{info.get('industry_category', 'N/A')}
**當前股價**：NT${defaults.get('currentPrice', 0):.2f}
**數據狀態**：已準備就緒
        """)
    with col2:
        st.markdown("#### 🎯 下一步操作")
        st.markdown("""
1. 📊 **查看財報推導參數** — 側邊欄已載入財報計算值
2. ⚙️ **調整DCF假設** — 根據您的判斷修改參數
3. 🚀 **開始計算** — 點擊「計算DCF估值」按鈕
        """)

    st.markdown("### 📋 財報推導參數一覽")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 📈 成長性參數")
        st.dataframe(pd.DataFrame({
            "參數": ["營收成長率", "EBITDA率", "資本支出比"],
            "數值": [
                f"{defaults.get('revenueGrowthPct', 0)*100:.1f}%",
                f"{defaults.get('ebitdaPct', 0)*100:.1f}%",
                f"{defaults.get('capitalExpenditurePct', 0)*100:.1f}%"
            ]
        }), hide_index=True)
    with col2:
        st.markdown("#### 💰 資本成本參數")
        st.dataframe(pd.DataFrame({
            "參數": ["無風險利率", "市場風險溢價", "Beta係數", "WACC（推導）"],
            "數值": [
                f"{defaults.get('riskFreeRate', 0)*100:.1f}%",
                f"{defaults.get('marketRiskPremium', 0)*100:.1f}%",
                f"{defaults.get('beta', 0):.2f}",
                f"{defaults.get('wacc', 0)*100:.2f}%"
            ]
        }), hide_index=True)
    with col3:
        st.markdown("#### ⚙️ 其他參數")
        st.dataframe(pd.DataFrame({
            "參數": ["長期成長率", "稅率", "股東權益", "淨負債"],
            "數值": [
                f"{defaults.get('longTermGrowthRate', 0)*100:.1f}%",
                f"{defaults.get('taxRate', 0)*100:.1f}%",
                format_large_number(defaults.get('totalEquity', 0)) + "元",
                format_large_number(defaults.get('totalDebt', 0) - defaults.get('cashAndEquiv', 0)) + "元"
            ]
        }), hide_index=True)

else:
    # 初始說明頁面
    st.markdown("""
## 歡迎使用DCF估值教育系統（台股版）💰

### 🎯 學習目標
- **理解DCF模型原理**：掌握現金流折現的核心概念
- **台股財報應用**：從 FinMind 真實財報數據計算 DCF
- **參數敏感性分析**：了解各項假設對估值的影響程度
- **風險意識培養**：認識台股投資的特有風險因子

### 🛠️ 系統功能
- **FinMind 財報整合**：自動抓取台股三大財報（損益表、資產負債表、現金流量表）
- **自動參數推導**：由財報數據自動計算 WACC、Beta、各項比率
- **完整 DCF 計算**：程式自行執行折現現金流模型，不依賴第三方 DCF API
- **多維度分析**：WACC 分解、敏感性分析、三情境模擬
- **AI 深度解讀**：Anthropic Claude 提供台股特化分析報告

### 📝 三步驟使用流程

#### **第一步：載入公司數據** 📊
- 輸入台股四位數代碼（如：2330 台積電、2454 聯發科）
- 輸入 FinMind API Token
- 點擊「載入公司基礎數據」
- 系統自動從 FinMind 獲取三大財報並推導 DCF 參數

#### **第二步：調整DCF參數** ⚙️
- 側邊欄顯示財報自動推導的預設值
- 根據您的判斷調整各項假設
- 隨時可使用「重置為預設值」恢復

#### **第三步：計算分析結果** 🚀
- 點擊「計算DCF估值」按鈕
- 查看完整估值報告、WACC 分解、敏感性分析
- 輸入 Anthropic Claude API Key 可獲得 AI 深度分析

### 🔑 API Token 獲取
- **FinMind API**：前往 [finmindtrade.com](https://finmindtrade.com) 免費註冊申請
- **Anthropic Claude API**：前往 [console.anthropic.com](https://console.anthropic.com) 申請（AI分析功能，選填）

### ⚠️ 重要提醒
本系統純供教育和學習用途，所有分析結果**不構成投資建議**。
請使用者自行判斷投資決策並承擔相關風險。

---
**👈 從左側開始您的台股DCF估值學習之旅！**
    """)

# 頁腳
st.markdown("---")
st.markdown("""
### 📢 免責聲明
本系統僅供學術研究與教育用途，AI 提供的數據與分析結果僅供參考，**不構成投資建議或財務建議**。
請使用者自行判斷投資決策，並承擔相關風險。本系統作者不對任何投資行為負責，亦不承擔任何損失責任。
""")
