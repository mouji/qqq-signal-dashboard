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

NAMES = {
    "QQQ":"纳斯达克ETF","AAPL":"苹果","MSFT":"微软","NVDA":"英伟达",
    "AMZN":"亚马逊","GOOGL":"谷歌","META":"Meta","TSLA":"特斯拉"
}

# ─── AI 分析师点评生成 ──────────────────────────────────────────────────────
def gen_commentary(ticker, MRS, TFS, TSS, CAS, rsi, price, tss_3m_wr, tss_3m_avg, tss_3m_n):
    """根据信号值生成拟人化分析师点评"""
    name = NAMES.get(ticker, ticker)

    # 宏观判断
    if MRS > 0.3:
        macro = "宏观环境强劲偏多"
    elif MRS > 0.1:
        macro = "宏观环境温和偏多"
    elif MRS > -0.1:
        macro = "宏观环境中性，方向待定"
    elif MRS > -0.3:
        macro = "宏观环境偏空，需谨慎"
    else:
        macro = "宏观环境明显偏空，建议控制仓位"

    # 趋势判断
    if TFS > 0.5:
        trend = "中期趋势强劲向上"
    elif TFS > 0.2:
        trend = "中期趋势偏多"
    elif TFS > -0.2:
        trend = "中期趋势中性"
    else:
        trend = "中期趋势向下"

    # 情绪/时机判断
    if TSS > 0.5:
        timing = f"技术情绪极度悲观（RSI {rsi:.0f}），历史上这类超卖信号触发后 3 个月上涨概率高达 {tss_3m_wr}%，平均涨幅 {tss_3m_avg:+.1f}%，是不可多得的战略买点"
        action = "【建议操作】当前为超卖买入窗口，可考虑分批建仓"
    elif TSS > 0.3:
        timing = f"技术情绪偏向悲观（RSI {rsi:.0f}），接近历史超卖区间，历史胜率 {tss_3m_wr}%"
        action = "【建议操作】可观察是否进一步走弱，等待更佳买点确认"
    elif TSS > -0.3:
        timing = f"技术情绪中性（RSI {rsi:.0f}），无明显超买或超卖信号"
        action = "【建议操作】持仓观望，等待信号明确"
    elif TSS > -0.5:
        timing = f"技术情绪偏向乐观（RSI {rsi:.0f}），市场情绪略有过热"
        action = "【建议操作】不宜追高，可适当止盈部分仓位"
    else:
        timing = f"技术情绪过热（RSI {rsi:.0f}），市场短期涨幅较大，情绪透支"
        action = "【建议操作】建议减仓或等待回调，避免高位追涨"

    # 综合结论
    pos_text = "满仓（100%）" if CAS > 0.3 else "七成仓（70%）" if CAS > 0 else "三成仓（30%）" if CAS > -0.3 else "空仓观望（0%）"

    # MRS 熊市警告
    bear_warning = ""
    if MRS < -0.2:
        bear_warning = " ⚠ 注意：当前宏观政权偏空（MRS 为负），TSS 超卖信号在此环境下历史失效率较高，建议降低仓位权重。"

    commentary = (
        f"{name}（{ticker}）· 今日信号解读｜${price:.2f}\n\n"
        f"📊 {macro}，{trend}。{timing}。\n\n"
        f"{action}，模型建议综合仓位：{pos_text}。{bear_warning}"
    )
    return commentary

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
        # 分批下载：股票 + 指数分开，避免 yfinance 多 ticker 列名问题
        tickers_stocks = TICKERS_ALL + ["HYG","LQD"]
        tickers_index  = ["^TNX","^IRX","^VIX"]

        raw_s = yf.download(tickers_stocks, period="3y", interval="1d",
                            auto_adjust=True, progress=False)
        raw_i = yf.download(tickers_index,  period="3y", interval="1d",
                            auto_adjust=True, progress=False)

        cl_s = raw_s["Close"]
        cl_i = raw_i["Close"]

        # 统一 columns（yfinance 新版可能返回 MultiIndex）
        def flatten_cols(df):
            if hasattr(df.columns, 'levels'):
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            return df
        cl_s = flatten_cols(cl_s)
        cl_i = flatten_cols(cl_i)

        qqq = cl_s["QQQ"].dropna()
        vix = cl_i.get("^VIX", cl_i.get("VIX", cl_i.iloc[:,2] if cl_i.shape[1]>2 else cl_i.iloc[:,0]))
        tnx = cl_i.get("^TNX", cl_i.get("TNX", cl_i.iloc[:,0]))
        irx = cl_i.get("^IRX", cl_i.get("IRX", cl_i.iloc[:,1] if cl_i.shape[1]>1 else cl_i.iloc[:,0]))
        hyg = cl_s.get("HYG", cl_s.iloc[:,0])
        lqd = cl_s.get("LQD", cl_s.iloc[:,0])

        tnx_r = tnx.reindex(qqq.index, method='ffill')
        irx_r = irx.reindex(qqq.index, method='ffill')
        hyg_r = hyg.reindex(qqq.index, method='ffill')
        lqd_r = lqd.reindex(qqq.index, method='ffill')
        vix_r = vix.reindex(qqq.index, method='ffill')

        qqq_records = compute_signals(qqq, qqq, tnx_r, irx_r, hyg_r, lqd_r, vix_r)
        qqq_today   = qqq_records[-1] if qqq_records else {}

        # 七巨头今日 TSS + 点评
        stocks_today = []
        for tk in TICKERS_7:
            s   = cl_s[tk].dropna()
            recs = compute_signals(s, qqq.reindex(s.index, method='ffill'),
                                   tnx.reindex(s.index, method='ffill'),
                                   irx.reindex(s.index, method='ffill'),
                                   hyg.reindex(s.index, method='ffill'),
                                   lqd.reindex(s.index, method='ffill'),
                                   vix.reindex(s.index, method='ffill'))
            if recs:
                t = recs[-1]
                # 历史胜率（快速计算近2年）
                tss_wr = win_rate(recs[-500:],"TSS",0.3,"gt","fwd_3m")
                comment = gen_commentary(
                    tk, t["MRS"] or 0, t["TFS"] or 0, t["TSS"] or 0, t["CAS"] or 0,
                    t["rsi"] or 50, t["price"] or 0,
                    tss_wr["wr"], tss_wr["avg"], tss_wr["n"]
                )
                stocks_today.append({
                    "ticker": tk,
                    "name": NAMES.get(tk, tk),
                    "price": t["price"],
                    "MRS": t["MRS"], "TFS": t["TFS"], "TSS": t["TSS"], "CAS": t["CAS"],
                    "rsi": t["rsi"], "f2": t["f2"],
                    "tss_3m_wr": tss_wr["wr"],
                    "commentary": comment,
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

        vix_now = safe(float(vix.dropna().iloc[-1])) if len(vix.dropna()) > 0 else None
        curve   = None
        try:
            curve = safe(float(tnx.dropna().iloc[-1]) - float(irx.dropna().iloc[-1]))
        except Exception:
            curve = None

        qqq_change = None
        try:
            if len(qqq) > 1:
                qqq_change = safe(float(qqq.iloc[-1])/float(qqq.iloc[-2])-1)
        except Exception:
            qqq_change = None
        qqq_wr = win_rate(qqq_records[-500:],"TSS",0.3,"gt","fwd_3m")
        qqq_commentary = gen_commentary(
            "QQQ", MRS, TFS, TSS, CAS,
            qqq_today.get("rsi") or 50,
            qqq_today.get("price") or 0,
            qqq_wr["wr"], qqq_wr["avg"], qqq_wr["n"]
        )

        return jsonify({
            "updated":        datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "qqq_price":      qqq_today.get("price"),
            "qqq_change_1d":  qqq_change,
            "vix":            vix_now,
            "curve_slope":    curve,
            "MRS": safe(MRS), "TFS": safe(TFS), "TSS": safe(TSS), "CAS": safe(CAS),
            "position":       pos,
            "triple":         triple,
            "signal_label":   signal_label,
            "signal_color":   signal_color,
            "qqq_commentary": qqq_commentary,
            "stocks":         stocks_today,
            "history":        history,
        })
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


# ─── 任意股票回测接口（支持搜索）────────────────────────────────────────────
@app.route("/api/backtest/<ticker>")
def api_backtest(ticker):
    ticker = ticker.upper().strip()
    try:
        raw_s = yf.download([ticker,"QQQ","HYG","LQD"], period="5y", interval="1d",
                            auto_adjust=True, progress=False)
        raw_i = yf.download(["^TNX","^IRX","^VIX"], period="5y", interval="1d",
                            auto_adjust=True, progress=False)

        def flatten_cols(df):
            if hasattr(df.columns, 'levels'):
                df.columns = [c[0] if isinstance(c,tuple) else c for c in df.columns]
            return df
        cl_s = flatten_cols(raw_s["Close"])
        cl_i = flatten_cols(raw_i["Close"])

        if ticker not in cl_s.columns:
            return jsonify({"error": f"找不到股票代码 {ticker}，请确认输入正确"}), 404

        stock = cl_s[ticker].dropna()
        if len(stock) < 300:
            return jsonify({"error": f"{ticker} 历史数据不足，无法回测"}), 400

        qqq = cl_s["QQQ"].reindex(stock.index, method='ffill')
        hyg = cl_s.get("HYG", cl_s.iloc[:,0]).reindex(stock.index, method='ffill')
        lqd = cl_s.get("LQD", cl_s.iloc[:,0]).reindex(stock.index, method='ffill')
        tnx = cl_i.get("^TNX", cl_i.get("TNX", cl_i.iloc[:,0])).reindex(stock.index, method='ffill')
        irx = cl_i.get("^IRX", cl_i.get("IRX", cl_i.iloc[:,0])).reindex(stock.index, method='ffill')
        vix = cl_i.get("^VIX", cl_i.get("VIX", cl_i.iloc[:,0])).reindex(stock.index, method='ffill')

        recs = compute_signals(stock, qqq, tnx, irx, hyg, lqd, vix)
        stats_by_year = {}
        for yr in [2022, 2023, 2024, 2025, None]:
            label = str(yr) if yr else "all"
            stats_by_year[label] = {
                "TSS_buy_1m":  win_rate(recs,"TSS",0.3,"gt","fwd_1m",yr),
                "TSS_buy_3m":  win_rate(recs,"TSS",0.3,"gt","fwd_3m",yr),
                "MRS_bull_1m": win_rate(recs,"MRS",0.2,"gt","fwd_1m",yr),
                "TFS_bull_1m": win_rate(recs,"TFS",0.2,"gt","fwd_1m",yr),
                "MRS_bull_3m": win_rate(recs,"MRS",0.2,"gt","fwd_3m",yr),
                "TFS_bull_3m": win_rate(recs,"TFS",0.2,"gt","fwd_3m",yr),
                "TSS_buy_1w":  win_rate(recs,"TSS",0.3,"gt","fwd_1w",yr),
                "MRS_bull_1w": win_rate(recs,"MRS",0.2,"gt","fwd_1w",yr),
                "TFS_bull_1w": win_rate(recs,"TFS",0.2,"gt","fwd_1w",yr),
            }
        slim = [{"date":r["date"],"price":r["price"],
                 "MRS":r["MRS"],"TFS":r["TFS"],"TSS":r["TSS"],
                 "rsi":r["rsi"],"f1":r["f1"],"f2":r["f2"],
                 "fwd_1m":r["fwd_1m"],"fwd_3m":r["fwd_3m"],"fwd_1w":r["fwd_1w"]}
                for r in recs[-260:]]
        today = recs[-1] if recs else {}
        tss_wr = stats_by_year["all"]["TSS_buy_3m"]
        commentary = gen_commentary(
            ticker, today.get("MRS") or 0, today.get("TFS") or 0,
            today.get("TSS") or 0, today.get("CAS") or 0,
            today.get("rsi") or 50, today.get("price") or 0,
            tss_wr["wr"], tss_wr["avg"], tss_wr["n"]
        )
        return jsonify({
            "ticker": ticker,
            "name": NAMES.get(ticker, ticker),
            "records": slim,
            "stats_by_year": stats_by_year,
            "today": today,
            "commentary": commentary,
        })
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
