"""
QQQ 三层信号仪表盘 - Flask 后端
支持 Railway / Render 部署
"""
from flask import Flask, jsonify, send_from_directory, send_file
from flask_cors import CORS
import json, math, datetime, traceback, os
import yfinance as yf
import pandas as pd

app = Flask(__name__, static_folder="static")
CORS(app)

PORT = int(os.environ.get("PORT", 8765))

TICKERS_7 = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"]
TICKERS_ALL = ["QQQ"] + TICKERS_7

# ─── 工具函数 ────────────────────────────────────────────────────────────────
def safe(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return round(float(v), 4)

def rsi_series(s, period=14):
    delta = s.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss
    return 100 - 100 / (1 + rs)

def compute_signals(stock, qqq, tnx_r, irx_r, hyg_r, lqd_r, vix_r):
    """给一支股票计算全部因子，返回 records 列表"""
    stk_rsi = rsi_series(stock)
    records = []
    for i in range(252, len(stock)):
        d = stock.index[i]
        # MRS
        curve  = safe(tnx_r.iloc[i] - irx_r.iloc[i])
        h21    = safe(hyg_r.iloc[i]/hyg_r.iloc[i-21]-1) if i>=21 else 0
        l21    = safe(lqd_r.iloc[i]/lqd_r.iloc[i-21]-1) if i>=21 else 0
        credit = safe((h21 or 0)-(l21 or 0))
        f1_6m  = safe(stock.iloc[i]/stock.iloc[i-126]-1) if i>=126 else 0
        vix_v  = safe(vix_r.iloc[i])
        vix_m  = safe(vix_r.rolling(20).mean().iloc[i])
        mrs_c  = max(-1,min(1,(curve or 0)/2.0))
        mrs_cr = max(-1,min(1,(credit or 0)/0.02))
        mrs_m  = max(-1,min(1,(f1_6m or 0)/0.15))
        mrs_v  = 1.0 if (vix_v or 25)<(vix_m or 25) else -0.5
        MRS = safe(0.40*mrs_c+0.25*mrs_cr+0.20*mrs_m+0.15*mrs_v)
        # TFS
        f1_12  = safe(stock.iloc[i]/stock.iloc[i-231]-1) if i>=231 else 0
        m1     = safe(stock.iloc[i]/stock.iloc[i-22]-1) if i>=22 else 0
        ma200  = stock.rolling(200).mean().iloc[i]
        above  = 1.0 if stock.iloc[i]>ma200 else -1.0
        TFS = safe(0.50*max(-1,min(1,(f1_12 or 0)/0.30))
                  +0.25*max(-1,min(1,(m1 or 0)/0.08))
                  +0.25*above)
        # TSS
        rv     = float(stk_rsi.iloc[i]) if not math.isnan(float(stk_rsi.iloc[i])) else 50
        ma20   = stock.rolling(20).mean().iloc[i]
        dev    = (stock.iloc[i]-ma20)/ma20 if ma20 else 0
        tss_r  = (50-rv)/50.0
        tss_v  = max(-1,min(1,((vix_v or 20)-20)/15.0))
        tss_d  = max(-1,min(1,-dev/0.05))
        TSS = safe(0.40*tss_r+0.35*tss_v+0.25*tss_d)
        CAS = safe(0.50*(MRS or 0)+0.30*(TFS or 0)+0.20*(TSS or 0))
        # EPS代理
        f2 = safe((stock.iloc[i]/stock.iloc[i-63]-1)-(qqq.iloc[i]/qqq.iloc[i-63]-1)) if i>=63 else None
        fwd_1w = safe(stock.iloc[min(i+5,  len(stock)-1)]/stock.iloc[i]-1)
        fwd_1m = safe(stock.iloc[min(i+22, len(stock)-1)]/stock.iloc[i]-1)
        fwd_3m = safe(stock.iloc[min(i+63, len(stock)-1)]/stock.iloc[i]-1)
        records.append({
            "date":  d.strftime("%Y-%m-%d"),
            "price": safe(stock.iloc[i]),
            "MRS": MRS, "TFS": TFS, "TSS": TSS, "CAS": CAS,
            "f1": safe(f1_12), "f2": f2, "rsi": safe(rv),
            "fwd_1w": fwd_1w, "fwd_1m": fwd_1m, "fwd_3m": fwd_3m,
        })
    return records

def win_rate(recs, sig, th, direction, fwd, year=None):
    r2  = [r for r in recs if (year is None or r['date'].startswith(str(year)))]
    hit = [r for r in r2 if r[sig] is not None and r[fwd] is not None
           and (r[sig]>th if direction=='gt' else r[sig]<th)]
    if not hit: return {"n":0,"wr":0,"avg":0}
    wins = [r for r in hit if r[fwd]>0]
    return {"n":len(hit),
            "wr":round(len(wins)/len(hit)*100,1),
            "avg":round(sum(r[fwd] for r in hit)/len(hit)*100,2)}

# ─── 主数据接口 ──────────────────────────────────────────────────────────────
@app.route("/api/data")
def api_data():
    try:
        tickers_pull = TICKERS_ALL + ["^TNX","^IRX","HYG","LQD","^VIX"]
        raw  = yf.download(tickers_pull, period="3y", interval="1d",
                           auto_adjust=True, progress=False)
        cl   = raw["Close"]
        qqq  = cl["QQQ"].dropna()
        vix  = cl["^VIX"]

        # QQQ 当前信号
        q_last  = len(qqq)-1
        tnx_r   = cl["^TNX"].reindex(qqq.index, method='ffill')
        irx_r   = cl["^IRX"].reindex(qqq.index, method='ffill')
        hyg_r   = cl["HYG"].reindex(qqq.index, method='ffill')
        lqd_r   = cl["LQD"].reindex(qqq.index, method='ffill')
        vix_r   = vix.reindex(qqq.index, method='ffill')

        qqq_records = compute_signals(qqq, qqq, tnx_r, irx_r, hyg_r, lqd_r, vix_r)
        qqq_today   = qqq_records[-1] if qqq_records else {}

        # 七巨头今日 TSS
        stocks_today = []
        for tk in TICKERS_7:
            s   = cl[tk].dropna()
            recs = compute_signals(s, qqq.reindex(s.index, method='ffill'),
                                   cl["^TNX"].reindex(s.index,method='ffill'),
                                   cl["^IRX"].reindex(s.index,method='ffill'),
                                   cl["HYG"].reindex(s.index,method='ffill'),
                                   cl["LQD"].reindex(s.index,method='ffill'),
                                   vix.reindex(s.index,method='ffill'))
            if recs:
                t = recs[-1]
                stocks_today.append({
                    "ticker": tk,
                    "price": t["price"],
                    "MRS": t["MRS"], "TFS": t["TFS"], "TSS": t["TSS"],
                    "rsi": t["rsi"], "f2": t["f2"],
                })

        # QQQ 历史（近52周）
        history = []
        for r in qqq_records[-260::5]:
            history.append({"date":r["date"],"price":r["price"],"MRS":r["MRS"],"TFS":r["TFS"],"TSS":r["TSS"]})

        # 信号状态
        MRS = qqq_today.get("MRS",0) or 0
        TFS = qqq_today.get("TFS",0) or 0
        TSS = qqq_today.get("TSS",0) or 0
        CAS = qqq_today.get("CAS",0) or 0

        def state(v):
            if v>0.15: return "Bull"
            if v<-0.15: return "Bear"
            return "Neutral"
        triple = f"{state(MRS)}/{state(TFS)}/{state(TSS)}"

        if triple=="Bull/Bull/Bull":
            signal_label,signal_color = "Triple Bull 🟢","#27ae60"
        elif triple=="Bear/Bear/Bear":
            signal_label,signal_color = "Triple Bear 🔴","#c0392b"
        else:
            signal_label,signal_color = f"分歧 ({triple})","#e67e22"

        pos = 100 if CAS>0.3 else 70 if CAS>0 else 30 if CAS>-0.3 else 0

        vix_now = safe(vix.dropna().iloc[-1])
        curve   = safe(cl["^TNX"].dropna().iloc[-1]-cl["^IRX"].dropna().iloc[-1])

        return jsonify({
            "updated":      datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "qqq_price":    qqq_today.get("price"),
            "qqq_change_1d":safe(qqq.iloc[-1]/qqq.iloc[-2]-1) if len(qqq)>1 else None,
            "vix":          vix_now,
            "curve_slope":  curve,
            "MRS": safe(MRS), "TFS": safe(TFS), "TSS": safe(TSS), "CAS": safe(CAS),
            "position":     pos,
            "triple":       triple,
            "signal_label": signal_label,
            "signal_color": signal_color,
            "stocks":       stocks_today,
            "history":      history,
        })
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


# ─── 七巨头回测接口（带年份）────────────────────────────────────────────────
@app.route("/api/backtest/<ticker>")
def api_backtest(ticker):
    ticker = ticker.upper()
    valid  = TICKERS_ALL
    if ticker not in valid:
        return jsonify({"error": "Unknown ticker"}), 400
    try:
        tickers_pull = [ticker,"QQQ","^TNX","^IRX","HYG","LQD","^VIX"]
        raw  = yf.download(tickers_pull, period="5y", interval="1d",
                           auto_adjust=True, progress=False)
        cl   = raw["Close"]
        stock = cl[ticker].dropna()
        qqq   = cl["QQQ"].reindex(stock.index, method='ffill')
        recs  = compute_signals(
            stock, qqq,
            cl["^TNX"].reindex(stock.index, method='ffill'),
            cl["^IRX"].reindex(stock.index, method='ffill'),
            cl["HYG"].reindex(stock.index, method='ffill'),
            cl["LQD"].reindex(stock.index, method='ffill'),
            cl["^VIX"].reindex(stock.index, method='ffill'),
        )
        stats_by_year = {}
        for yr in [2022, 2023, 2024, 2025, None]:
            label = str(yr) if yr else "all"
            stats_by_year[label] = {
                "TSS_buy_1m": win_rate(recs,"TSS",0.3,"gt","fwd_1m",yr),
                "TSS_buy_3m": win_rate(recs,"TSS",0.3,"gt","fwd_3m",yr),
                "MRS_bull_1m": win_rate(recs,"MRS",0.2,"gt","fwd_1m",yr),
                "TFS_bull_1m": win_rate(recs,"TFS",0.2,"gt","fwd_1m",yr),
                "MRS_bull_3m": win_rate(recs,"MRS",0.2,"gt","fwd_3m",yr),
                "TFS_bull_3m": win_rate(recs,"TFS",0.2,"gt","fwd_3m",yr),
                "TSS_buy_1w":  win_rate(recs,"TSS",0.3,"gt","fwd_1w",yr),
                "MRS_bull_1w": win_rate(recs,"MRS",0.2,"gt","fwd_1w",yr),
                "TFS_bull_1w": win_rate(recs,"TFS",0.2,"gt","fwd_1w",yr),
            }
        # 精简 records（只返回近260条）
        slim = [{"date":r["date"],"price":r["price"],
                 "MRS":r["MRS"],"TFS":r["TFS"],"TSS":r["TSS"],
                 "rsi":r["rsi"],"f1":r["f1"],"f2":r["f2"],
                 "fwd_1m":r["fwd_1m"],"fwd_3m":r["fwd_3m"],"fwd_1w":r["fwd_1w"]}
                for r in recs[-260:]]
        return jsonify({"ticker":ticker,"records":slim,"stats_by_year":stats_by_year,
                        "today":recs[-1] if recs else {}})
    except Exception as e:
        return jsonify({"error":str(e),"trace":traceback.format_exc()}), 500


# ─── 静态页面服务 ────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory("static", path)


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
