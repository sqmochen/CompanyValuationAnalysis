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
# v5.0.0 | 2026-04-19 | 新增資料核對與推導過程面板
#   - [K] 修正 CapitalStock 欄位：FinMind 單位為「元」，直接÷10，移除×1000錯誤
#   - [K] 修正 DepreciationAndAmortization：損益表無此欄，改從現金流量表
#         Depreciation + AmortizationExpense 合計取得
#   - [K] 修正負債欄位：LongtermBorrowings / ShorttermBorrowings（非LongTermDebt/ShortTermBorrowings）
#   - [K] 修正 net_income 來源：改用 EquityAttributableToOwnersOfParent（歸屬母公司）
#   - [K] 修正 taxRate 計算：改用 TAX / PreTaxIncome（非 1 - NetIncome/PreTaxIncome）
#   - [L] 新增「📋 原始數據核對面板」：載入後顯示六大資料來源所有取用欄位與實際數值
#   - [L] 新增「🔢 推導過程面板」：逐步展示 STEP 0~9 完整計算過程（含 EPS 交叉驗證）
#   - [L] 新增各季 EPS 拆解（累計→單季差分）與說明
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
    [修正K] 修正欄位名稱、股數計算、折舊來源、稅率計算
    回傳 (default_params, beta_calculated, raw_data)
    """
    income  = financial_data.get("income_statement", [])
    balance = financial_data.get("balance_sheet", [])
    cash    = financial_data.get("cash_flow", [])

    # ── 損益表數據 ──
    revenue           = extract_income_value(income, "Revenue", 1)
    op_income         = extract_income_value(income, "OperatingIncome", 0)
    gross_profit      = extract_income_value(income, "GrossProfit", 0)
    pretax            = extract_income_value(income, "PreTaxIncome", 1)
    # [修正K] 歸屬母公司淨利
    net_income_parent = extract_income_value(income, "EquityAttributableToOwnersOfParent", 0)
    # [修正K] 直接取 TAX 欄位
    tax_amount        = extract_income_value(income, "TAX", 0)
    eps_reported      = extract_income_value(income, "EPS", 0)

    # 前一年營收
    revenue_prev = 0
    if len(income) >= 2:
        revenue_prev = float(income[1].get("Revenue", 0) or 0)

    # 各期累計淨利（歸屬母公司），供EPS季度拆解
    income_by_date = {}
    for rec in income:
        d = rec.get("date", "")
        v = rec.get("EquityAttributableToOwnersOfParent", 0)
        if d and v is not None:
            income_by_date[d] = float(v) if v else 0

    # ── 資產負債表數據 ──
    total_assets  = extract_income_value(balance, "TotalAssets", 1)
    total_equity  = extract_income_value(balance, "Equity", 1)
    # [修正K] 正確欄位名稱
    total_debt_lt = extract_income_value(balance, "LongtermBorrowings", 0)
    total_debt_st = extract_income_value(balance, "ShorttermBorrowings", 0)
    total_debt    = total_debt_lt + total_debt_st
    cash_equiv    = extract_income_value(balance, "CashAndCashEquivalents", 0)
    receivables   = extract_income_value(balance, "AccountsReceivableNet", 0)
    inventories   = extract_income_value(balance, "Inventories", 0)
    payables      = extract_income_value(balance, "AccountsPayable", 0)
    # [修正K] CapitalStock 單位為「元」，直接÷10（不×1000）
    capital_stock = extract_income_value(balance, "CapitalStock", 0)
    if capital_stock > 0:
        shares_outstanding = capital_stock / 10
    elif eps_reported > 0 and net_income_parent > 0:
        shares_outstanding = net_income_parent / eps_reported
    else:
        shares_outstanding = 1

    # ── 現金流量表數據 ──
    op_cf  = extract_income_value(cash, "CashFlowsFromOperatingActivities", 0)
    capex  = abs(extract_income_value(cash, "PropertyAndPlantAndEquipment", 0))
    inv_cf = extract_income_value(cash, "CashProvidedByInvestingActivities", 0)
    fin_cf = extract_income_value(cash, "CashFlowsProvidedFromFinancingActivities", 0)
    # [修正K] 折舊攤銷從現金流量表取得
    dep    = extract_income_value(cash, "Depreciation", 0)
    amort  = extract_income_value(cash, "AmortizationExpense", 0)
    dep_total = dep + amort

    # ── 計算各項比率 ──
    revenue_growth = (revenue - revenue_prev) / abs(revenue_prev) if revenue_prev != 0 else 0.10
    ebitda         = op_income + dep_total
    ebitda_pct     = ebitda / revenue if revenue != 0 else 0.25
    capex_pct      = capex / revenue if revenue != 0 else 0.05
    op_cf_pct      = op_cf / revenue if revenue != 0 else 0.15
    dep_pct        = dep_total / revenue if revenue != 0 else 0.03
    # [修正K] 稅率：TAX / PreTaxIncome
    tax_rate_raw   = tax_amount / pretax if pretax != 0 else 0.20
    tax_rate       = max(0.05, min(0.40, tax_rate_raw))

    # EPS 交叉驗證
    eps_derived  = net_income_parent / shares_outstanding if shares_outstanding > 0 else 0
    eps_diff_pct = abs(eps_derived - eps_reported) / eps_reported * 100 if eps_reported > 0 else 0

    # ── WACC 推導 ──
    risk_free_rate       = 0.015
    market_risk_premium  = 0.06
    beta, beta_calculated = calculate_beta(stock_id, token)
    cost_of_equity       = risk_free_rate + beta * market_risk_premium
    cost_of_debt_pretax  = 0.03
    cost_of_debt         = cost_of_debt_pretax * (1 - tax_rate)
    total_capital        = total_equity + total_debt if (total_equity + total_debt) > 0 else 1
    equity_weight        = total_equity / total_capital
    debt_weight          = total_debt / total_capital
    wacc                 = equity_weight * cost_of_equity + debt_weight * cost_of_debt
    long_term_growth     = 0.02

    # ── 整理原始數據（供核對面板使用）[修正L] ──
    raw_data = {
        "revenue": revenue, "revenue_prev": revenue_prev,
        "op_income": op_income, "pretax": pretax,
        "net_income_parent": net_income_parent, "tax_amount": tax_amount,
        "eps_reported": eps_reported, "gross_profit": gross_profit,
        "total_assets": total_assets, "total_equity": total_equity,
        "total_debt_lt": total_debt_lt, "total_debt_st": total_debt_st,
        "total_debt": total_debt, "cash_equiv": cash_equiv,
        "receivables": receivables, "inventories": inventories,
        "payables": payables, "capital_stock": capital_stock,
        "shares_outstanding": shares_outstanding,
        "op_cf": op_cf, "capex": capex, "dep": dep, "amort": amort,
        "dep_total": dep_total, "inv_cf": inv_cf, "fin_cf": fin_cf,
        "current_price": current_price,
        "revenue_growth": revenue_growth, "ebitda": ebitda,
        "ebitda_pct": ebitda_pct, "capex_pct": capex_pct,
        "op_cf_pct": op_cf_pct, "dep_pct": dep_pct,
        "tax_rate_raw": tax_rate_raw, "tax_rate": tax_rate,
        "eps_derived": eps_derived, "eps_diff_pct": eps_diff_pct,
        "risk_free_rate": risk_free_rate,
        "market_risk_premium": market_risk_premium,
        "beta": beta, "beta_calculated": beta_calculated,
        "cost_of_equity": cost_of_equity,
        "cost_of_debt_pretax": cost_of_debt_pretax,
        "cost_of_debt": cost_of_debt, "total_capital": total_capital,
        "equity_weight": equity_weight, "debt_weight": debt_weight,
        "wacc": wacc, "long_term_growth": long_term_growth,
        "income_by_date": income_by_date,
    }

    default_params = {
        "revenueGrowthPct":               max(-0.2, min(2.0, revenue_growth)),
        "ebitdaPct":                       max(0.01, min(0.80, ebitda_pct)),
        "capitalExpenditurePct":          max(0.001, min(0.30, capex_pct)),
        "operatingCashFlowPct":           max(0.001, min(0.60, op_cf_pct)),
        "depreciationAndAmortizationPct": dep_pct,
        "riskFreeRate":                   risk_free_rate,
        "marketRiskPremium":              market_risk_premium,
        "beta":                           beta,
        "costOfEquity":                   cost_of_equity,
        "costOfDebt":                     cost_of_debt,
        "wacc":                           max(0.03, min(0.20, wacc)),
        "taxRate":                        tax_rate,
        "longTermGrowthRate":             long_term_growth,
        "cashPct":                        cash_equiv / revenue if revenue != 0 else 0.10,
        "receivablesPct":                 receivables / revenue if revenue != 0 else 0.10,
        "inventoriesPct":                 inventories / revenue if revenue != 0 else 0.05,
        "payablePct":                     payables / revenue if revenue != 0 else 0.08,
        "currentPrice":                   current_price,
        "revenue":                        revenue,
        "totalEquity":                    total_equity,
        "totalDebt":                      total_debt,
        "cashAndEquiv":                   cash_equiv,
        "sharesOutstanding":              max(1, shares_outstanding),
        "capitalStock":                   capital_stock,
    }
    return default_params, beta_calculated, raw_data


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
# [修正L] 原始數據核對面板 & 推導過程面板
# =============================================================================

def show_raw_data_panel(raw, stock_id, stock_name, beta_calculated):
    """
    載入後顯示六大資料來源所有取用欄位與實際數值，供人工核對。
    """
    st.markdown("## 📋 原始數據核對面板")
    st.info("以下為從 FinMind 六個資料來源取出的所有欄位與實際數值，請逐一確認後再進行計算。")

    # ① 公司基本資料
    with st.expander("① 公司基本資料（TaiwanStockInfo）", expanded=True):
        st.dataframe(pd.DataFrame([{
            "欄位": "stock_id",   "說明": "股票代號",   "數值": stock_id
        }, {
            "欄位": "stock_name", "說明": "公司名稱",   "數值": stock_name
        }]), hide_index=True, use_container_width=True)

    # ② 綜合損益表
    with st.expander("② 綜合損益表（TaiwanStockFinancialStatements）", expanded=True):
        rows = [
            ("Revenue",                              "本期營業收入",         raw["revenue"],           f"{raw['revenue']/1e6:.2f} 百萬元"),
            ("Revenue（前期）",                       "前期營業收入",         raw["revenue_prev"],       f"{raw['revenue_prev']/1e6:.2f} 百萬元"),
            ("GrossProfit",                           "營業毛利",             raw["gross_profit"],       f"{raw['gross_profit']/1e6:.2f} 百萬元"),
            ("OperatingIncome",                       "營業利益",             raw["op_income"],          f"{raw['op_income']/1e6:.2f} 百萬元"),
            ("PreTaxIncome",                          "稅前淨利",             raw["pretax"],             f"{raw['pretax']/1e6:.2f} 百萬元"),
            ("TAX",                                   "所得稅費用",           raw["tax_amount"],         f"{raw['tax_amount']/1e6:.2f} 百萬元"),
            ("EquityAttributableToOwnersOfParent",    "歸屬母公司稅後淨利",   raw["net_income_parent"],  f"{raw['net_income_parent']/1e6:.2f} 百萬元"),
            ("EPS",                                   "每股盈餘（財報）",     raw["eps_reported"],       f"NT${raw['eps_reported']:.2f}"),
        ]
        st.dataframe(pd.DataFrame(rows, columns=["FinMind欄位", "說明", "原始數值（元）", "格式化"]),
                     hide_index=True, use_container_width=True)

    # ③ 資產負債表
    with st.expander("③ 資產負債表（TaiwanStockBalanceSheet）", expanded=True):
        rows = [
            ("TotalAssets",          "總資產",         raw["total_assets"],   f"{raw['total_assets']/1e6:.2f} 百萬元"),
            ("Equity",               "股東權益",        raw["total_equity"],   f"{raw['total_equity']/1e6:.2f} 百萬元"),
            ("LongtermBorrowings",   "長期借款",        raw["total_debt_lt"],  f"{raw['total_debt_lt']/1e6:.2f} 百萬元"),
            ("ShorttermBorrowings",  "短期借款",        raw["total_debt_st"],  f"{raw['total_debt_st']/1e6:.2f} 百萬元"),
            ("（合計）總負債",        "長期+短期借款",   raw["total_debt"],     f"{raw['total_debt']/1e6:.2f} 百萬元"),
            ("CashAndCashEquivalents","現金及約當現金",  raw["cash_equiv"],     f"{raw['cash_equiv']/1e6:.2f} 百萬元"),
            ("AccountsReceivableNet","應收帳款（淨）",  raw["receivables"],    f"{raw['receivables']/1e6:.2f} 百萬元"),
            ("Inventories",          "存貨",            raw["inventories"],    f"{raw['inventories']/1e6:.2f} 百萬元"),
            ("AccountsPayable",      "應付帳款",        raw["payables"],       f"{raw['payables']/1e6:.2f} 百萬元"),
            ("CapitalStock",         "股本（元）",      raw["capital_stock"],  f"{raw['capital_stock']/1e6:.2f} 百萬元"),
            ("（推算）流通股數",      "股本÷面額10元",   raw["shares_outstanding"], f"{raw['shares_outstanding']/1e6:.4f} 百萬股"),
        ]
        st.dataframe(pd.DataFrame(rows, columns=["FinMind欄位", "說明", "原始數值（元）", "格式化"]),
                     hide_index=True, use_container_width=True)

    # ④ 現金流量表
    with st.expander("④ 現金流量表（TaiwanStockCashFlowsStatement）", expanded=True):
        rows = [
            ("CashFlowsFromOperatingActivities",     "營業活動現金流",   raw["op_cf"],      f"{raw['op_cf']/1e6:.2f} 百萬元"),
            ("PropertyAndPlantAndEquipment",          "資本支出（原始負值）", -raw["capex"], f"{-raw['capex']/1e6:.2f} 百萬元"),
            ("Depreciation",                          "折舊",             raw["dep"],        f"{raw['dep']/1e6:.2f} 百萬元"),
            ("AmortizationExpense",                   "攤銷",             raw["amort"],      f"{raw['amort']/1e6:.2f} 百萬元"),
            ("（合計）折舊攤銷",                       "Dep+Amort",        raw["dep_total"],  f"{raw['dep_total']/1e6:.2f} 百萬元"),
            ("CashProvidedByInvestingActivities",     "投資活動現金流",   raw["inv_cf"],     f"{raw['inv_cf']/1e6:.2f} 百萬元"),
            ("CashFlowsProvidedFromFinancingActivities","融資活動現金流", raw["fin_cf"],     f"{raw['fin_cf']/1e6:.2f} 百萬元"),
        ]
        st.dataframe(pd.DataFrame(rows, columns=["FinMind欄位", "說明", "原始數值（元）", "格式化"]),
                     hide_index=True, use_container_width=True)

    # ⑤⑥ 個股 & 大盤股價（Beta）
    with st.expander("⑤⑥ 個股股價 & 大盤指數（TaiwanStockPrice）", expanded=True):
        beta_src = "✅ 迴歸計算（個股 vs TAIEX 日報酬率）" if beta_calculated else "⚠️ 備援預設值 1.0（資料不足）"
        rows = [
            ("close（最新）",    "當前收盤價",      f"NT${raw['current_price']:.2f}", ""),
            ("Beta 計算方式",    "計算來源",        beta_src,                          ""),
            ("Beta 值",          "迴歸結果",        f"{raw['beta']:.4f}",              "Cov(股,大盤)/Var(大盤)"),
        ]
        st.dataframe(pd.DataFrame(rows, columns=["項目", "說明", "數值", "備註"]),
                     hide_index=True, use_container_width=True)

    st.success("✅ 以上為所有資料來源的原始數值，請確認無誤後即可進行計算。")


def show_derivation_panel(raw, stock_id, n_years=5):
    """
    [修正L] 展示完整 STEP 0~9 推導過程，含各季 EPS 拆解與 EPS 交叉驗證。
    """
    st.markdown("## 🔢 推導過程詳解")
    st.info("以下為根據原始數據逐步推導的完整計算過程，可供教學與人工驗證使用。")

    # ── STEP 1：財務比率 ──
    with st.expander("STEP 1｜財務比率計算", expanded=True):
        r = raw
        rows = [
            ("1-1", "營收成長率",
             f"({r['revenue']/1e6:.2f} - {r['revenue_prev']/1e6:.2f}) / {r['revenue_prev']/1e6:.2f}",
             f"{r['revenue_growth']*100:.2f}%"),
            ("1-2", "EBITDA = 營業利益 + 折舊攤銷",
             f"{r['op_income']/1e6:.2f} + {r['dep_total']/1e6:.2f}",
             f"{r['ebitda']/1e6:.2f} 百萬元"),
            ("1-3", "EBITDA利潤率 = EBITDA / 營收",
             f"{r['ebitda']/1e6:.2f} / {r['revenue']/1e6:.2f}",
             f"{r['ebitda_pct']*100:.2f}%"),
            ("1-4", "資本支出比例 = |資本支出| / 營收",
             f"{r['capex']/1e6:.2f} / {r['revenue']/1e6:.2f}",
             f"{r['capex_pct']*100:.2f}%"),
            ("1-5", "折舊攤銷比例 = 折舊攤銷 / 營收",
             f"{r['dep_total']/1e6:.2f} / {r['revenue']/1e6:.2f}",
             f"{r['dep_pct']*100:.2f}%"),
            ("1-6", "營業現金流比例 = 營業現金流 / 營收",
             f"{r['op_cf']/1e6:.2f} / {r['revenue']/1e6:.2f}",
             f"{r['op_cf_pct']*100:.2f}%"),
            ("1-7", "有效稅率 = TAX / 稅前淨利",
             f"{r['tax_amount']/1e6:.2f} / {r['pretax']/1e6:.2f}",
             f"{r['tax_rate_raw']*100:.2f}% → 限制後 {r['tax_rate']*100:.2f}%"),
        ]
        st.dataframe(pd.DataFrame(rows, columns=["步驟", "項目", "計算式（百萬元）", "結果"]),
                     hide_index=True, use_container_width=True)

    # ── STEP 2：Beta ──
    with st.expander("STEP 2｜Beta 係數計算", expanded=True):
        beta_src = "迴歸計算 ✅" if raw["beta_calculated"] else "備援預設值 1.0 ⚠️"
        st.markdown(f"""
**公式**：Beta = Cov(個股日報酬, 大盤日報酬) / Var(大盤日報酬)

| 項目 | 數值 |
|------|------|
| 計算來源 | 個股（{stock_id}）vs 加權指數（TAIEX） |
| 計算方式 | {beta_src} |
| Beta 值 | **{raw['beta']:.4f}** |
| 限制範圍 | 0.3 ~ 3.0 |
""")

    # ── STEP 3：WACC ──
    with st.expander("STEP 3｜WACC 計算", expanded=True):
        r = raw
        rows = [
            ("3-1", "股權成本 Re（CAPM）",
             f"Rf + β×(Rm-Rf) = {r['risk_free_rate']*100:.1f}% + {r['beta']:.4f}×{r['market_risk_premium']*100:.1f}%",
             f"{r['cost_of_equity']*100:.4f}%"),
            ("3-2", "稅後債務成本 Rd",
             f"{r['cost_of_debt_pretax']*100:.1f}% × (1-{r['tax_rate']*100:.2f}%)",
             f"{r['cost_of_debt']*100:.4f}%"),
            ("3-3", "總資本 = 股東權益 + 總負債",
             f"{r['total_equity']/1e6:.2f} + {r['total_debt']/1e6:.2f}",
             f"{r['total_capital']/1e6:.2f} 百萬元"),
            ("3-4", "股權權重 We",
             f"{r['total_equity']/1e6:.2f} / {r['total_capital']/1e6:.2f}",
             f"{r['equity_weight']*100:.2f}%"),
            ("3-5", "債務權重 Wd",
             f"{r['total_debt']/1e6:.2f} / {r['total_capital']/1e6:.2f}",
             f"{r['debt_weight']*100:.2f}%"),
            ("3-6", "WACC = We×Re + Wd×Rd",
             f"{r['equity_weight']:.4f}×{r['cost_of_equity']*100:.4f}% + {r['debt_weight']:.4f}×{r['cost_of_debt']*100:.4f}%",
             f"{r['wacc']*100:.4f}%"),
        ]
        st.dataframe(pd.DataFrame(rows, columns=["步驟", "項目", "計算式", "結果"]),
                     hide_index=True, use_container_width=True)

    # ── STEP 4：FCFF ──
    with st.expander("STEP 4｜自由現金流（FCFF）參考", expanded=False):
        r = raw
        fcff = r["op_cf"] - r["capex"]
        st.markdown(f"""
**FCFF = 營業現金流 - 資本支出**
= {r['op_cf']/1e6:.2f} - {r['capex']/1e6:.2f} = **{fcff/1e6:.2f} 百萬元**

FCFF利潤率 = {fcff/1e6:.2f} / {r['revenue']/1e6:.2f} = **{fcff/r['revenue']*100:.2f}%**

> 本模型以 EBITDA 法推算各年 FCF，FCFF 僅供參考驗證使用。
""")

    # ── STEP 5：預測期 DCF ──
    with st.expander("STEP 5｜預測期現金流折現（5年）", expanded=True):
        r = raw
        rev  = r["revenue"]
        rows = []
        pv_total = 0
        for yr in range(1, n_years + 1):
            rev     = rev * (1 + r["revenue_growth"])
            ebitda  = rev * r["ebitda_pct"]
            dep     = rev * r["dep_pct"]
            ebit    = ebitda - dep
            ebiat   = ebit * (1 - r["tax_rate"])
            capex   = rev * r["capex_pct"]
            delta_wc= rev * 0.02
            ufcf    = ebiat + dep - capex - delta_wc
            pv_f    = 1 / (1 + r["wacc"]) ** yr
            pv_ufcf = ufcf * pv_f
            pv_total += pv_ufcf
            rows.append({
                "年度": f"第{yr}年",
                "預測營收(M)": f"{rev/1e6:.1f}",
                "EBITDA(M)": f"{ebitda/1e6:.1f}",
                "折舊(M)": f"{dep/1e6:.1f}",
                "EBIT(M)": f"{ebit/1e6:.1f}",
                "EBIAT(M)": f"{ebiat/1e6:.1f}",
                "資本支出(M)": f"{capex/1e6:.1f}",
                "△WC(M)": f"{delta_wc/1e6:.1f}",
                "UFCF(M)": f"{ufcf/1e6:.1f}",
                "折現係數": f"{pv_f:.6f}",
                "現值(M)": f"{pv_ufcf/1e6:.1f}",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.metric("預測期現金流現值合計", f"{pv_total/1e6:,.2f} 百萬元")

    # ── STEP 6：終值 ──
    with st.expander("STEP 6｜終值計算（Gordon Growth Model）", expanded=True):
        r = raw
        last_rev   = raw["revenue"] * (1 + r["revenue_growth"]) ** n_years
        fcf_term   = last_rev * r["ebitda_pct"] * (1 - r["tax_rate"]) * (1 + r["long_term_growth"])
        tv         = fcf_term / (r["wacc"] - r["long_term_growth"]) if r["wacc"] > r["long_term_growth"] else 0
        pv_tv      = tv / (1 + r["wacc"]) ** n_years
        # 重算 pv_total
        rev2 = raw["revenue"]
        pv_total2 = 0
        for yr in range(1, n_years + 1):
            rev2   = rev2 * (1 + r["revenue_growth"])
            ebitda = rev2 * r["ebitda_pct"]
            dep    = rev2 * r["dep_pct"]
            ebit   = ebitda - dep
            ebiat  = ebit * (1 - r["tax_rate"])
            capex  = rev2 * r["capex_pct"]
            ufcf   = ebiat + dep - capex - rev2 * 0.02
            pv_total2 += ufcf / (1 + r["wacc"]) ** yr

        ev          = pv_total2 + pv_tv
        net_debt    = r["total_debt"] - r["cash_equiv"]
        eq_val      = max(0, ev - net_debt)
        eq_val_ps   = eq_val / r["shares_outstanding"] if r["shares_outstanding"] > 0 else 0
        tv_ratio    = pv_tv / ev * 100 if ev > 0 else 0

        st.markdown(f"""
| 項目 | 計算式 | 結果 |
|------|--------|------|
| 第{n_years}年預測營收 | 基準 × (1+{r['revenue_growth']*100:.2f}%)^{n_years} | {last_rev/1e6:.2f} 百萬元 |
| FCF_terminal | 第{n_years}年營收 × EBITDA率 × (1-稅率) × (1+{r['long_term_growth']*100:.1f}%) | {fcf_term/1e6:.2f} 百萬元 |
| 終值 TV | FCF_term / (WACC - 永續成長率) | {tv/1e6:.2f} 百萬元 |
| 終值現值 PV_TV | TV / (1+WACC)^{n_years} | {pv_tv/1e6:.2f} 百萬元 |
| 企業價值 EV | PV_FCF + PV_TV | {ev/1e6:.2f} 百萬元 |
| 淨債務 | 總負債 - 現金 | {net_debt/1e6:.2f} 百萬元 |
| 股權價值 | EV - 淨債務 | {eq_val/1e6:.2f} 百萬元 |
| **DCF每股價值** | 股權價值 / 流通股數 | **NT${eq_val_ps:.2f}** |
| 終值佔比 | PV_TV / EV | {tv_ratio:.1f}% |
""")

    # ── STEP 7：EPS 驗證與季度拆解 ──
    with st.expander("STEP 7｜EPS 交叉驗證 & 各季拆解", expanded=True):
        r = raw
        eps_ok = "✅" if r["eps_diff_pct"] < 2 else "⚠️"
        st.markdown(f"""
**EPS 交叉驗證**

| 項目 | 計算式 | 結果 |
|------|--------|------|
| 推導 EPS | 歸屬母公司淨利 / 流通股數 | NT${r['eps_derived']:.4f} |
| 財報揭露 EPS | FinMind EPS 欄位 | NT${r['eps_reported']:.2f} |
| 誤差 | |abs(推導-財報)/財報| | {r['eps_diff_pct']:.2f}% {eps_ok} |

> ⚠️ 財報 EPS 以**加權平均股數**計算，推導使用期末股數，誤差 <2% 屬正常。
""")

        # 各季 EPS 拆解
        income_dates = sorted(raw["income_by_date"].keys(), reverse=True)
        if len(income_dates) >= 4:
            latest_year = income_dates[0][:4]
            year_dates  = [d for d in income_dates if d.startswith(latest_year)]
            year_dates  = sorted(year_dates)

            season_rows = []
            prev_cum = 0
            labels   = {"-03-31": "Q1", "-06-30": "Q2", "-09-30": "Q3", "-12-31": "Q4"}
            for d in year_dates:
                suffix  = d[4:]
                label   = labels.get(suffix, suffix)
                cum_ni  = raw["income_by_date"].get(d, 0)
                qtr_ni  = cum_ni - prev_cum
                qtr_eps = qtr_ni / r["shares_outstanding"] if r["shares_outstanding"] > 0 else 0
                season_rows.append({
                    "期間": f"{latest_year} {label}（{d}）",
                    "累計淨利（M）": f"{cum_ni/1e6:.2f}",
                    "單季淨利（M）": f"{qtr_ni/1e6:.2f}",
                    "單季EPS（NT$）": f"{qtr_eps:.4f}",
                })
                prev_cum = cum_ni

            total_eps = sum(float(row["單季EPS（NT$）"]) for row in season_rows)
            st.markdown(f"**{latest_year} 各季 EPS 拆解**（FinMind累計差分法）")
            st.dataframe(pd.DataFrame(season_rows), hide_index=True, use_container_width=True)
            st.metric(f"{latest_year} 全年 EPS 加總", f"NT${total_eps:.4f}")
            st.caption("FinMind 損益表為累計格式：各季單季數字 = 本期累計 - 前期累計")


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
if "beta_calculated" not in st.session_state:
    st.session_state.beta_calculated = False
if "raw_data" not in st.session_state:          # [修正L]
    st.session_state.raw_data = {}

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
                financial_data = get_finmind_financial_data(ticker, finmind_token)
                current_price  = get_finmind_stock_price(ticker, finmind_token)
                stock_info     = get_finmind_stock_info(ticker, finmind_token)
                # [修正K+L] 新版函式回傳三個值
                default_params, beta_calc, raw_data = calculate_dcf_params_from_finmind(
                    financial_data, current_price, ticker, finmind_token
                )

                st.session_state.financial_data   = financial_data
                st.session_state.default_params   = default_params
                st.session_state.stock_info       = stock_info
                st.session_state.company_data     = {"currentPrice": current_price}
                st.session_state.company_loaded   = True
                st.session_state.beta_calculated  = beta_calc
                st.session_state.raw_data         = raw_data  # [修正L]

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
    info     = st.session_state.stock_info
    defaults = st.session_state.default_params
    raw      = st.session_state.raw_data

    # ── 頂部狀態列 ──
    st.markdown("## 📊 公司數據已載入完成")
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"""
**✅ 已載入**：{info.get('stock_name', ticker)}（{ticker}）
**產業**：{info.get('industry_category', 'N/A')}
**當前股價**：NT${defaults.get('currentPrice', 0):.2f}
**Beta計算**：{'✅ 迴歸計算' if st.session_state.beta_calculated else '⚠️ 備援預設值1.0'}
        """)
    with col2:
        st.markdown("#### 🎯 下一步操作")
        st.markdown("""
1. 📋 **核對原始數據** — 確認下方六大來源欄位與數值
2. 🔢 **查看推導過程** — 確認各步驟計算邏輯
3. ⚙️ **調整DCF假設** — 根據判斷修改側邊欄參數
4. 🚀 **開始計算** — 點擊「計算DCF估值」按鈕
        """)

    st.markdown("---")

    # ── [修正L] 原始數據核對面板 ──
    if raw:
        show_raw_data_panel(
            raw,
            ticker,
            info.get("stock_name", ticker),
            st.session_state.beta_calculated
        )
        st.markdown("---")
        # ── [修正L] 推導過程面板 ──
        show_derivation_panel(raw, ticker, n_years=5)
    else:
        st.warning("⚠️ 原始數據尚未載入，請重新點擊「載入公司基礎數據」")

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
