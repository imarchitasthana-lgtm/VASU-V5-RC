# ==============================================================================
#  VASU AI v3.1 - Self-Learning Autonomous Trading Bot
#  RENDER / GITHUB EDITION — cloud-ready, cross-platform
#
#  NEW in v3.1:
#  + Stochastic RSI  + Williams %R  + CCI  (3 new indicators)
#  + RSI divergence detection (bullish/bearish)
#  + News sentiment scoring (keyword-based)
#  + Market hours awareness (no ghost trades after close)
#  + /health endpoint for Render health checks
#  + Non-blocking startup (server starts in <2s, training in background)
#  + Mobile-responsive dashboard
#  + Live auto-refresh without full page reload
#  + Improved composite scoring (+divergence bonus)
#  + Better Kelly Criterion (symmetric win/loss ratio)
#  + Smarter self-coding engine (5 analysis types → 7)
#  + Pattern recognition: hammer, engulfing, doji
#  + Volatility-adjusted position sizing
# ==============================================================================

from config import *
import yfinance as yf
import ta, time, threading, json, numpy as np, os, copy, re
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from urllib.parse import parse_qs, urlparse

# ==============================================================================
#  CLOUD-SAFE PATHS & SERVER CONFIG
# ==============================================================================
DATA_DIR      = os.environ.get("DATA_DIR", ".")
BOT_FILE      = os.path.join(DATA_DIR, "bot_data.json")
BRAIN_FILE    = os.path.join(DATA_DIR, "vasu_brain.json")
EQUITY_FILE   = os.path.join(DATA_DIR, "vasu_equity.json")
SELFCODE_FILE = os.path.join(DATA_DIR, "vasu_selfcode.json")

PORT = int(os.environ.get("PORT", 5050))
HOST = "0.0.0.0"

bot_lock      = threading.Lock()
training_done = threading.Event()   # signals when AI models are ready

# ==============================================================================
#  TRADING MODES
# ==============================================================================
MODE_CONFIGS = {
    "auto":        {"label":"AUTO",        "color":"#a78bfa","buy_confidence":65.0,"stop_loss_pct":3.0, "take_profit_pct":5.0, "max_positions":6,"position_size_pct":15.0,"require_surge":False,"require_oversold":False,"min_signal":"BUY",       "trailing_stop":True, "max_heat":0.80,"desc":"Reads market every scan and auto-switches mode."},
    "sniper":      {"label":"SNIPER",      "color":"#00ff88","buy_confidence":80.0,"stop_loss_pct":2.5, "take_profit_pct":8.0, "max_positions":3,"position_size_pct":22.0,"require_surge":False,"require_oversold":False,"min_signal":"STRONG BUY", "trailing_stop":True, "max_heat":0.66,"desc":"Only the absolute best setups. High confidence."},
    "aggressive":  {"label":"AGGRESSIVE",  "color":"#ff4757","buy_confidence":55.0,"stop_loss_pct":4.0, "take_profit_pct":7.0, "max_positions":8,"position_size_pct":18.0,"require_surge":False,"require_oversold":False,"min_signal":"BUY",       "trailing_stop":True, "max_heat":0.90,"desc":"Low threshold, more positions. High risk/reward."},
    "balanced":    {"label":"BALANCED",    "color":"#3b82f6","buy_confidence":65.0,"stop_loss_pct":3.0, "take_profit_pct":5.0, "max_positions":6,"position_size_pct":15.0,"require_surge":False,"require_oversold":False,"min_signal":"BUY",       "trailing_stop":True, "max_heat":0.80,"desc":"Default. Good for most market conditions."},
    "conservative":{"label":"CONSERVATIVE","color":"#f59e0b","buy_confidence":75.0,"stop_loss_pct":2.0, "take_profit_pct":4.0, "max_positions":4,"position_size_pct":10.0,"require_surge":False,"require_oversold":False,"min_signal":"BUY",       "trailing_stop":True, "max_heat":0.50,"desc":"Capital preservation first. High threshold."},
    "momentum":    {"label":"MOMENTUM",    "color":"#facc15","buy_confidence":60.0,"stop_loss_pct":3.5, "take_profit_pct":7.0, "max_positions":5,"position_size_pct":16.0,"require_surge":True, "require_oversold":False,"min_signal":"BUY",       "trailing_stop":True, "max_heat":0.80,"desc":"Hunts stocks with momentum AND volume surges."},
    "contrarian":  {"label":"CONTRARIAN",  "color":"#10b981","buy_confidence":58.0,"stop_loss_pct":4.5, "take_profit_pct":10.0,"max_positions":4,"position_size_pct":14.0,"require_surge":False,"require_oversold":True, "min_signal":"BUY",       "trailing_stop":False,"max_heat":0.70,"desc":"Buys when everyone sells. Targets RSI<35."},
    "bear":        {"label":"BEAR MODE",   "color":"#6b7280","buy_confidence":92.0,"stop_loss_pct":1.5, "take_profit_pct":3.0, "max_positions":2,"position_size_pct":8.0, "require_surge":False,"require_oversold":False,"min_signal":"STRONG BUY","trailing_stop":True, "max_heat":0.25,"desc":"Market crashing. No new buys. Protect cash."},
}

# ==============================================================================
#  GLOBAL STATE
# ==============================================================================
latest_data    = {k:[] for k in ["tech","finance","healthcare","consumer","energy","mypicks"]}
last_updated   = "Initializing..."
ai_models      = {}
win_tracker    = {}
analyst_advice = {k:[] for k in ["claude","arya","magnus","zeus","bear_analyst"]}
chat_history   = []
market_mood    = {"label":"Scanning...","color":"#6b7280","score":0}
vasu_daily     = ""
watchlist      = []
dream_trades_cache = []

SECTOR_MAP = {}
for _sec, _syms in [("tech",TECH),("finance",FINANCE),("healthcare",HEALTHCARE),
                    ("consumer",CONSUMER),("energy",ENERGY),("mypicks",MY_PICKS)]:
    for _s in _syms: SECTOR_MAP[_s] = _sec

# ==============================================================================
#  BRAIN DEFAULT
# ==============================================================================
DEFAULT_BRAIN = {
    "buy_confidence":65.0,"stop_loss_pct":3.0,"take_profit_pct":5.0,
    "max_positions":6,"position_size_pct":15.0,
    "active_mode":"auto","auto_target":"balanced",
    "mode_history":[],"mode_performance":{},
    "market_regime":"unknown","regime_history":[],"market_score":0,
    "sector_scores":{k:0 for k in ["tech","finance","healthcare","consumer","energy","mypicks"]},
    "sector_blacklist":[],"stock_scores":{},
    "time_scores":{},"rsi_scores":{},"confidence_decay":{},
    "consecutive_losses":0,"trading_paused":False,
    "peak_value":1000.0,"current_drawdown_pct":0.0,
    "position_size_by_conf":True,"min_volume_ratio":0.8,
    "trailing_highs":{},"total_adjustments":0,
    "personality":"balanced","mood":"neutral","lessons":[],
    "ai_blend":{"rf":0.30,"gb":0.35,"et":0.35},
    "weights":{"technical":25.0,"ai_model":30.0,"fundamental":25.0,"sector_macro":10.0,"self_learned":10.0},
    "weight_performance":{k:{"correct":0,"total":0} for k in ["technical","ai_model","fundamental","sector_macro","self_learned"]},
    "weight_history":[],"last_weight_update":"",
    "selfcode_version":1,"selfcode_improvements":0,"selfcode_last_run":"",
    "sector_macro_scores":{},
}

# ==============================================================================
#  PERSISTENCE
# ==============================================================================
def _load_json(fp, default):
    if os.path.exists(fp):
        try:
            with open(fp,"r") as f: return json.load(f)
        except: pass
    return default() if callable(default) else copy.deepcopy(default)

def _save_json(fp, data):
    try:
        tmp = fp + ".tmp"
        with open(tmp,"w") as f: json.dump(data, f, indent=2)
        os.replace(tmp, fp)
    except Exception as e: print("Save error:", e)

def load_brain():
    d = _load_json(BRAIN_FILE, None)
    if d:
        for k,v in DEFAULT_BRAIN.items():
            if k not in d: d[k] = copy.deepcopy(v)
        return d
    return copy.deepcopy(DEFAULT_BRAIN)

def save_brain(): _save_json(BRAIN_FILE, vasu_brain)

def load_bot():
    d = _load_json(BOT_FILE, None)
    if d: return d
    return {"enabled":BOT_ENABLED,"cash":1000.0,"holdings":{},"trades":[],"wins":0,"losses":0,"start_value":1000.0}

def save_bot(): _save_json(BOT_FILE, bot)
def load_equity(): return _load_json(EQUITY_FILE, list)

def save_equity(val):
    eq = load_equity()
    eq.append({"time":datetime.now().strftime("%m/%d %H:%M"),"value":round(val,2)})
    if len(eq) > 200: eq = eq[-200:]
    _save_json(EQUITY_FILE, eq)

vasu_brain = load_brain()
bot        = load_bot()

# ==============================================================================
#  MARKET HOURS (NEW v3.1)
#  Avoids phantom signals outside trading hours
# ==============================================================================
def market_is_open():
    """Returns True during NYSE market hours (Mon-Fri 9:30-16:00 ET)."""
    try:
        from datetime import timezone as tz
        import zoneinfo
        et = zoneinfo.ZoneInfo("America/New_York")
        now = datetime.now(et)
        if now.weekday() >= 5: return False          # weekend
        open_t  = now.replace(hour=9, minute=30, second=0, microsecond=0)
        close_t = now.replace(hour=16, minute=0,  second=0, microsecond=0)
        return open_t <= now <= close_t
    except:
        return True   # if timezone check fails, don't block trading

def market_hours_label():
    try:
        from zoneinfo import ZoneInfo
        et = ZoneInfo("America/New_York")
        now = datetime.now(et)
        if now.weekday() >= 5: return "Weekend - Market Closed"
        if now.hour < 9 or (now.hour == 9 and now.minute < 30): return "Pre-Market"
        if now.hour >= 16: return "After Hours"
        return "Market Open"
    except:
        return "Unknown"

# ==============================================================================
#  MODE SYSTEM
# ==============================================================================
def get_active_mode_config():
    mode = vasu_brain.get("active_mode","auto")
    if mode == "auto":
        return MODE_CONFIGS.get(vasu_brain.get("auto_target","balanced"), MODE_CONFIGS["balanced"])
    return MODE_CONFIGS.get(mode, MODE_CONFIGS["balanced"])

def get_mode_label():
    mode = vasu_brain.get("active_mode","auto")
    if mode == "auto":
        return "AUTO -> " + vasu_brain.get("auto_target","balanced").upper()
    return MODE_CONFIGS.get(mode, MODE_CONFIGS["balanced"])["label"]

def set_mode(name, reason="Manual"):
    name = name.lower().strip()
    if name not in MODE_CONFIGS: return False
    old = vasu_brain.get("active_mode","auto")
    vasu_brain["active_mode"] = name
    vasu_brain["mode_history"].append({"time":_ts(),"from":old,"to":name,"reason":reason})
    vasu_brain["mode_history"] = vasu_brain["mode_history"][-30:]
    save_brain(); return True

def update_auto_mode():
    if vasu_brain.get("active_mode") != "auto": return
    items = all_items_flat()
    if not items: return
    bulls       = [i for i in items if i["signal"] in ("BUY","STRONG BUY")]
    bears       = [i for i in items if i["signal"] in ("SELL","STRONG SELL")]
    strong_buys = [i for i in items if i["signal"] == "STRONG BUY"]
    oversold    = [i for i in items if i["rsi"] < 35]
    surges      = [i for i in items if i.get("volume_surge")]
    total       = len(items)
    bull_ratio  = len(bulls)/total if total else 0
    bear_ratio  = len(bears)/total if total else 0
    over_ratio  = len(oversold)/total if total else 0
    surge_ratio = len(surges)/total if total else 0
    score = round((bull_ratio - bear_ratio) * 10, 1)
    vasu_brain["market_score"] = score
    if   score >= 6:  regime = "bull"
    elif score >= 2:  regime = "mixed_bull"
    elif score > -2:  regime = "mixed"
    elif score > -6:  regime = "mixed_bear"
    else:             regime = "bear"
    vasu_brain["market_regime"] = regime
    rh = vasu_brain["regime_history"]; rh.append(regime)
    vasu_brain["regime_history"] = rh[-10:]
    if len(rh) >= 3 and all("bear" in r for r in rh[-3:]): regime = "crash"
    tv   = get_bot_value(); peak = vasu_brain.get("peak_value", tv)
    if tv > peak: vasu_brain["peak_value"] = tv; peak = tv
    dd   = (peak-tv)/peak*100 if peak > 0 else 0
    vasu_brain["current_drawdown_pct"] = round(dd,1)
    old  = vasu_brain.get("auto_target","balanced")
    if   dd >= 8:                           new,why = "conservative","Portfolio down "+str(round(dd,1))+"% from peak."
    elif regime == "crash":                 new,why = "bear","Crash detected."
    elif regime == "bear":                  new,why = "conservative","Bear regime."
    elif regime in ("bull","mixed_bull") and surge_ratio > 0.12: new,why = "momentum","Bull + surges."
    elif regime == "bull" and len(strong_buys) >= 6:             new,why = "aggressive","Strong bull + "+str(len(strong_buys))+" STRONG BUYs."
    elif over_ratio > 0.25 and bear_ratio < 0.3:                 new,why = "contrarian",str(len(oversold))+" oversold."
    elif len(strong_buys) <= 1 and bull_ratio < 0.3:             new,why = "sniper","Quiet market."
    elif regime in ("bull","mixed_bull"):   new,why = "balanced","Solid bull."
    elif regime == "mixed_bear":            new,why = "conservative","Mixed bearish."
    else:                                   new,why = "balanced","Mixed market."
    if new != old:
        vasu_brain["auto_target"] = new
        vasu_brain["mode_history"].append({"time":_ts(),"from":"auto->"+old,"to":"auto->"+new,"reason":why})
        vasu_brain["mode_history"] = vasu_brain["mode_history"][-30:]
        print(f"[AUTO] {old.upper()} -> {new.upper()} | {why}")

# ==============================================================================
#  HELPERS
# ==============================================================================
def _ts(): return datetime.now().strftime("%m/%d %H:%M")
def all_items_flat():
    items = []
    for v in latest_data.values(): items.extend(v)
    return items

def get_bot_value():
    pm    = {i["symbol"]:i["price"] for i in all_items_flat()}
    total = bot["cash"]
    for sym,h in bot["holdings"].items():
        total += h["shares"] * pm.get(sym, h["buy_price"])
    return round(total, 2)

def _portfolio_heat():
    tv = get_bot_value()
    return 1.0 - (bot["cash"] / tv) if tv > 0 else 0

def _is_blacklisted(sym):
    return vasu_brain.get("stock_scores",{}).get(sym, 0) <= -4

# ==============================================================================
#  NEW INDICATORS v3.1
# ==============================================================================
def get_stochastic(close, high, low, k_period=14, d_period=3):
    """Stochastic %K and %D — measures where price sits in recent high/low range."""
    try:
        lowest_low   = low.rolling(k_period).min()
        highest_high = high.rolling(k_period).max()
        denom = (highest_high - lowest_low).replace(0, float("nan"))
        k = ((close - lowest_low) / denom * 100).rolling(d_period).mean()
        d = k.rolling(d_period).mean()
        return float(k.iloc[-1]), float(d.iloc[-1])
    except: return 50.0, 50.0

def get_williams_r(close, high, low, period=14):
    """Williams %R — momentum oscillator (-100 to 0, oversold < -80)."""
    try:
        hh   = high.rolling(period).max()
        ll   = low.rolling(period).min()
        denom = (hh - ll).replace(0, float("nan"))
        wr = ((hh - close) / denom * -100)
        return float(wr.iloc[-1])
    except: return -50.0

def get_cci(close, high, low, period=20):
    """Commodity Channel Index — overbought > +100, oversold < -100."""
    try:
        tp  = (high + low + close) / 3
        sma = tp.rolling(period).mean()
        mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        cci = (tp - sma) / (0.015 * mad.replace(0, float("nan")))
        return float(cci.iloc[-1])
    except: return 0.0

def detect_rsi_divergence(close, rsi_series, window=10):
    """
    Detects bullish/bearish RSI divergence.
    Bullish: price makes lower low, RSI makes higher low → reversal signal.
    Bearish: price makes higher high, RSI makes lower high → reversal signal.
    Returns: 'bullish', 'bearish', or 'none'
    """
    try:
        prices = close.values[-window:]
        rsiv   = rsi_series.dropna().values[-window:]
        if len(prices) < window or len(rsiv) < window: return "none"
        # Bullish divergence
        if prices[-1] < prices[0] and rsiv[-1] > rsiv[0]: return "bullish"
        # Bearish divergence
        if prices[-1] > prices[0] and rsiv[-1] < rsiv[0]: return "bearish"
    except: pass
    return "none"

def detect_candlestick_pattern(open_s, close_s, high_s, low_s):
    """Basic candlestick pattern recognition."""
    try:
        o,c,h,l = float(open_s.iloc[-1]),float(close_s.iloc[-1]),float(high_s.iloc[-1]),float(low_s.iloc[-1])
        o2,c2   = float(open_s.iloc[-2]),float(close_s.iloc[-2])
        body    = abs(c-o); rng = h-l; bullish = c > o
        if rng == 0: return "none"
        # Hammer: small body, long lower wick, bullish after downtrend
        lower_wick = min(o,c) - l
        if body < rng*0.3 and lower_wick > rng*0.6 and bullish:
            return "hammer"
        # Doji: very small body
        if body < rng * 0.1:
            return "doji"
        # Bullish engulfing: current bullish candle engulfs previous bearish
        if bullish and c2 < o2 and c > o2 and o < c2:
            return "bullish_engulfing"
        # Bearish engulfing
        if not bullish and c2 > o2 and o > c2 and c < o2:
            return "bearish_engulfing"
        # Shooting star: small body top, long upper wick
        upper_wick = h - max(o,c)
        if upper_wick > rng*0.6 and body < rng*0.3:
            return "shooting_star"
    except: pass
    return "none"

def score_news_sentiment(news_items):
    """
    Keyword-based news sentiment scoring.
    Returns: score -3 to +3
    """
    if not news_items: return 0
    POSITIVE = ["beat","beats","record","upgrade","strong","growth","profit","raises","buyback",
                "dividend","revenue","exceeds","surges","rally","bullish","outperform","buy"]
    NEGATIVE = ["miss","misses","downgrade","loss","warn","layoff","lawsuit","probe","recall",
                "fraud","decline","cut","downside","sell","bearish","underperform","crash","debt"]
    score = 0
    for item in news_items[:5]:
        title = item.get("title","").lower()
        pos = sum(1 for w in POSITIVE if w in title)
        neg = sum(1 for w in NEGATIVE if w in title)
        score += pos - neg
    return max(-3, min(3, score))

# ==============================================================================
#  TRIPLE AI ENGINE
# ==============================================================================
FEATURE_COLS = ["rsi","macd","bb","sma_ratio","atr","mom5","mom20","vwap_ratio","vol_ratio","pos52",
                "stoch_k","williams_r","cci"]   # 3 new features in v3.1

def train_model(symbol):
    try:
        import pandas as pd
        df = yf.download(symbol, period="5y", interval="1d", progress=False, auto_adjust=True)
        if df.empty or len(df) < 150: return None
        close  = df["Close"].squeeze()
        high   = df["High"].squeeze()
        low    = df["Low"].squeeze()
        open_  = df["Open"].squeeze()
        volume = df["Volume"].squeeze()

        rsi    = ta.momentum.RSIIndicator(close).rsi()
        macd   = ta.trend.MACD(close).macd_diff()
        bb     = ta.volatility.BollingerBands(close).bollinger_pband()
        atr    = ta.volatility.AverageTrueRange(high, low, close).average_true_range()
        sma20  = close.rolling(20).mean()
        sma50  = close.rolling(50).mean()
        vol_ma = volume.rolling(20).mean().replace(0, float("nan"))
        vwap   = (close*volume).rolling(20).sum() / volume.rolling(20).sum().replace(0, float("nan"))
        h52    = close.rolling(252).max()
        l52    = close.rolling(252).min()
        # Stochastic
        ll14   = low.rolling(14).min()
        hh14   = high.rolling(14).max()
        stoch_k = ((close-ll14)/(hh14-ll14+1e-10)*100).rolling(3).mean()
        # Williams %R
        will_r  = ((hh14-close)/(hh14-ll14+1e-10)*-100)
        # CCI
        tp = (high+low+close)/3
        tp_sma = tp.rolling(20).mean()
        tp_mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x-x.mean())), raw=True)
        cci    = (tp - tp_sma) / (0.015 * (tp_mad + 1e-10))

        feat = pd.DataFrame({
            "rsi":rsi,"macd":macd,"bb":bb,
            "sma_ratio":sma20/sma50.replace(0,float("nan")),
            "atr":atr,"mom5":close.pct_change(5),"mom20":close.pct_change(20),
            "vwap_ratio":close/vwap.replace(0,float("nan")),
            "vol_ratio":volume/vol_ma,
            "pos52":(close-l52)/(h52-l52+1e-10),
            "stoch_k":stoch_k,"williams_r":will_r,"cci":cci,
        }).dropna()
        future = close.shift(-5)/close - 1
        feat   = feat.join(future.rename("future")).dropna()
        cols   = [c for c in FEATURE_COLS if c in feat.columns]
        X      = feat[cols].values
        y      = (feat["future"] > 0.01).astype(int).values
        if len(X) < 60: return None
        sc  = StandardScaler()
        Xs  = sc.fit_transform(X)
        rf  = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1, max_features="sqrt", min_samples_leaf=3)
        gb  = GradientBoostingClassifier(n_estimators=150, random_state=42, learning_rate=0.04, max_depth=4, min_samples_leaf=3)
        et  = ExtraTreesClassifier(n_estimators=150, random_state=42, n_jobs=-1, max_features="sqrt", min_samples_leaf=2, bootstrap=True)
        rf.fit(Xs, y); gb.fit(Xs, y); et.fit(Xs, y)
        return {"rf":rf,"gb":gb,"et":et,"scaler":sc,"cols":cols}
    except Exception as e:
        print(f"  Train error {symbol}: {e}"); return None

def get_ai_confidence(symbol, fdict):
    m = ai_models.get(symbol)
    if not m: return None
    try:
        X  = np.array([[fdict.get(c,0.0) for c in m["cols"]]])
        Xs = m["scaler"].transform(X)
        blend = vasu_brain.get("ai_blend",{"rf":0.30,"gb":0.35,"et":0.35})
        rf_p  = float(m["rf"].predict_proba(Xs)[0][1])
        gb_p  = float(m["gb"].predict_proba(Xs)[0][1])
        et_p  = float(m["et"].predict_proba(Xs)[0][1]) if "et" in m else gb_p
        tw = blend.get("rf",0.30) + blend.get("gb",0.35) + blend.get("et",0.35)
        blended = (rf_p*blend.get("rf",0.30) + gb_p*blend.get("gb",0.35) + et_p*blend.get("et",0.35)) / tw
        return round(blended * 100, 1)
    except: return None

def get_ai_agreement(symbol, fdict):
    m = ai_models.get(symbol)
    if not m: return None
    try:
        X  = np.array([[fdict.get(c,0.0) for c in m["cols"]]])
        Xs = m["scaler"].transform(X)
        rf_p = float(m["rf"].predict_proba(Xs)[0][1])
        gb_p = float(m["gb"].predict_proba(Xs)[0][1])
        et_p = float(m["et"].predict_proba(Xs)[0][1]) if "et" in m else gb_p
        probs = [rf_p, gb_p, et_p]
        std   = float(np.std(probs))
        agree = round(max(0, 1-std*4), 3)
        avg   = float(np.mean(probs))
        return {
            "rf":round(rf_p*100,1),"gb":round(gb_p*100,1),"et":round(et_p*100,1),
            "agreement":agree,
            "consensus":"bullish" if avg>0.6 else "bearish" if avg<0.4 else "split",
            "all_agree_bull": all(p>0.55 for p in probs),
            "all_agree_bear": all(p<0.45 for p in probs),
        }
    except: return None

def train_all():
    print(f"[VASU] Training Triple AI on {len(ALL_STOCKS)} symbols...")
    for s in ALL_STOCKS:
        ai_models[s] = train_model(s)
        print(f"  {'OK' if ai_models[s] else 'XX'} {s}")
    ready = sum(1 for v in ai_models.values() if v)
    print(f"[VASU] Training done! {ready}/{len(ALL_STOCKS)} symbols ready.")
    training_done.set()

# ==============================================================================
#  FUNDAMENTALS ENGINE (10 Wall Street Frameworks)
# ==============================================================================
_fund_cache      = {}
_fund_cache_time = {}

def get_fundamentals(symbol):
    now  = datetime.now()
    last = _fund_cache_time.get(symbol)
    if last and (now-last).seconds < 1800 and symbol in _fund_cache:
        return _fund_cache[symbol]
    result = {
        "pe_ratio":None,"forward_pe":None,"pb_ratio":None,"ps_ratio":None,
        "peg_ratio":None,"ev_ebitda":None,"dcf_signal":0,
        "profit_margin":None,"revenue_growth":None,"earnings_growth":None,
        "return_on_equity":None,"debt_to_equity":None,"current_ratio":None,
        "quality_score":0,"dividend_yield":0.0,"payout_ratio":None,"dividend_score":0,
        "earnings_date":"","earnings_surprise":None,"revenue_beat_streak":0,"earnings_score":0,
        "institutional_pct":None,"short_interest":None,"insider_score":0,
        "beta":None,"risk_score":0,"fundamental_score":0,"fundamental_grade":"C",
    }
    try:
        info = yf.Ticker(symbol).info or {}
        pe=info.get("trailingPE") or info.get("forwardPE"); fpe=info.get("forwardPE")
        pb=info.get("priceToBook"); ps=info.get("priceToSalesTrailing12Months")
        peg=info.get("pegRatio"); ev_eb=info.get("enterpriseToEbitda")
        result.update({
            "pe_ratio":round(pe,2) if pe else None,
            "forward_pe":round(fpe,2) if fpe else None,
            "pb_ratio":round(pb,2) if pb else None,
            "ps_ratio":round(ps,2) if ps else None,
            "peg_ratio":round(peg,2) if peg else None,
            "ev_ebitda":round(ev_eb,2) if ev_eb else None,
        })
        # DCF signal
        ds = 0
        if pe:
            if pe<15:ds+=2
            elif pe<25:ds+=1
            elif pe>50:ds-=2
            elif pe>35:ds-=1
        if peg: ds += (1 if peg<1 else -1 if peg>2.5 else 0)
        if ps:  ds += (1 if ps<2 else -1 if ps>10 else 0)
        result["dcf_signal"] = max(-2,min(2,ds))
        # Quality
        pm=info.get("profitMargins"); rg=info.get("revenueGrowth")
        eg=info.get("earningsGrowth"); roe=info.get("returnOnEquity")
        de=info.get("debtToEquity"); cr=info.get("currentRatio")
        result.update({
            "profit_margin":round(pm*100,1) if pm else None,
            "revenue_growth":round(rg*100,1) if rg else None,
            "earnings_growth":round(eg*100,1) if eg else None,
            "return_on_equity":round(roe*100,1) if roe else None,
            "debt_to_equity":round(de,2) if de else None,
            "current_ratio":round(cr,2) if cr else None,
        })
        qs = 0
        if pm: qs += (2 if pm>0.20 else 1 if pm>0.10 else -2 if pm<0 else 0)
        if rg: qs += (1 if rg>0.20 else -1 if rg<-0.10 else 0)
        if roe: qs += (1 if roe>0.20 else -1 if roe<0.05 else 0)
        if de:  qs += (1 if de<0.5 else -1 if de>2.0 else 0)
        if cr:  qs += (1 if cr>2.0 else -1 if cr<1.0 else 0)
        result["quality_score"] = max(-3,min(3,qs))
        # Dividend
        dy=info.get("dividendYield") or 0; pr=info.get("payoutRatio")
        result["dividend_yield"] = round(dy*100,2) if dy else 0.0
        result["payout_ratio"]   = round(pr*100,1) if pr else None
        result["dividend_score"] = (1 if dy and dy>0.02 else 0) + (1 if pr and 0<pr<0.6 else 0)
        # Institutional / short
        ip=info.get("heldPercentInstitutions"); sp=info.get("shortPercentOfFloat")
        result["institutional_pct"] = round(ip*100,1) if ip else None
        result["short_interest"]    = round(sp*100,1) if sp else None
        ins = (1 if ip and ip>0.6 else 0) + (-1 if sp and sp>0.15 else 0)
        result["insider_score"] = ins
        # Beta
        beta=info.get("beta")
        result["beta"] = round(beta,2) if beta else None
        if beta: result["risk_score"] = (2 if beta<0.8 else 1 if beta<1.2 else -2 if beta>1.8 else -1)
        # Earnings history
        try:
            hist = yf.Ticker(symbol).earnings_history
            if hist is not None and not hist.empty:
                surprises=[]
                for _,row in hist.tail(4).iterrows():
                    est=row.get("epsEstimate") or row.get("EPS Estimate")
                    act=row.get("epsActual") or row.get("Reported EPS")
                    if est and act and est!=0: surprises.append((act-est)/abs(est))
                if surprises:
                    avg=sum(surprises)/len(surprises)
                    result["earnings_surprise"]   = round(avg*100,1)
                    result["revenue_beat_streak"] = len([s for s in surprises if s>0])
                    result["earnings_score"] = (2 if avg>0.05 else 1 if avg>0 else -2 if avg<-0.05 else -1)
        except: pass
        # Earnings date
        try:
            cal = yf.Ticker(symbol).calendar
            if cal is not None and not cal.empty:
                ed = cal.get("Earnings Date")
                if ed is not None and len(ed)>0: result["earnings_date"] = str(ed[0])[:10]
        except: pass
        # Composite fundamental score
        fs = (result["dcf_signal"] + result["quality_score"] + result["dividend_score"]
              + result["earnings_score"] + result["insider_score"] + result["risk_score"])
        result["fundamental_score"] = fs
        result["fundamental_grade"] = "A" if fs>=7 else "B" if fs>=4 else "C" if fs>=1 else "D" if fs>=-1 else "F"
    except Exception as e:
        print(f"  Fundamentals error {symbol}: {e}")
    _fund_cache[symbol] = result; _fund_cache_time[symbol] = now
    return result

def get_sector_macro_score(sector):
    cache = vasu_brain.get("sector_macro_scores",{})
    items = latest_data.get(sector,[])
    if not items: return cache.get(sector, 0)
    changes = [i["change_pct"] for i in items if i.get("change_pct") is not None]
    avg_chg = sum(changes)/len(changes) if changes else 0
    bulls   = len([i for i in items if i["signal"] in ("BUY","STRONG BUY")])
    bull_r  = bulls/len(items) if items else 0
    rsis    = [i["rsi"] for i in items if i.get("rsi") is not None]
    avg_rsi = sum(rsis)/len(rsis) if rsis else 50
    s = 0
    s += (2 if avg_chg>1.5 else 1 if avg_chg>0.5 else -2 if avg_chg<-1.5 else -1 if avg_chg<-0.5 else 0)
    s += (1 if bull_r>0.6 else -1 if bull_r<0.3 else 0)
    s += (1 if avg_rsi<35 else -1 if avg_rsi>65 else 0)
    brain_s = vasu_brain.get("sector_scores",{}).get(sector,0)
    s += (1 if brain_s>=3 else -1 if brain_s<=-3 else 0)
    s = max(-2,min(2,s))
    cache[sector] = s; vasu_brain["sector_macro_scores"] = cache
    return s

def _get_weights():
    w = vasu_brain.get("weights",{})
    if not w: w = DEFAULT_BRAIN["weights"]
    total = sum(w.values())
    return {k: v/total*100 for k,v in w.items()} if total > 0 else w

def compute_composite(symbol, tech_score, ai_conf, fund, sector, divergence, pattern, sent_score):
    w = _get_weights(); score = 50.0
    tech_c  = tech_score * (w.get("technical",25)/6.0); score += tech_c
    ai_c    = 0.0
    if ai_conf is not None:
        ai_c = (ai_conf-50)*(w.get("ai_model",30)/100.0); score += ai_c
    fs      = fund.get("fundamental_score",0)
    fund_c  = fs * (w.get("fundamental",25)/10.0); score += fund_c
    macro_c = get_sector_macro_score(sector) * (w.get("sector_macro",10)/2.0); score += macro_c
    ss      = vasu_brain.get("stock_scores",{}).get(symbol,0)
    h       = datetime.now().hour
    ts_     = vasu_brain.get("time_scores",{}).get(str(h)+"h",0)
    self_c  = (min(ss*1.5,5)+ts_) * (w.get("self_learned",10)/10.0); score += self_c
    # NEW v3.1 bonuses
    if divergence == "bullish":  score += 5  # RSI divergence = early reversal
    if divergence == "bearish":  score -= 5
    if pattern in ("hammer","bullish_engulfing"):  score += 3
    if pattern in ("shooting_star","bearish_engulfing"): score -= 3
    if pattern == "doji":        score -= 1  # uncertainty
    score += min(3, max(-3, sent_score))  # news sentiment
    # Hard bonuses
    ed = fund.get("earnings_date","")
    if ed:
        try:
            days = (datetime.strptime(ed,"%Y-%m-%d")-datetime.now()).days
            if 0<=days<=3: score -= 15
            elif 0<=days<=7: score -= 5
        except: pass
    si = fund.get("short_interest")
    if si and si>20: score -= 8
    ip = fund.get("institutional_pct")
    if ip and ip>70: score += 3
    if fund.get("revenue_beat_streak",0)>=3: score += 5
    grade_bonus = {"A":8,"B":4,"C":0,"D":-4,"F":-10}
    score += grade_bonus.get(fund.get("fundamental_grade","C"),0)
    fund["_contrib"] = {"technical":round(tech_c,2),"ai_model":round(ai_c,2),
                        "fundamental":round(fund_c,2),"sector_macro":round(macro_c,2),"self_learned":round(self_c,2)}
    return max(0, min(100, round(score,1)))

# ==============================================================================
#  ANALYZE — Triple AI + 13 Indicators + Divergence + Patterns + Sentiment
# ==============================================================================
def analyze(symbol):
    try:
        import pandas as pd
        df = yf.download(symbol, period="5d", interval="15m", progress=False, auto_adjust=True)
        if df.empty or len(df)<20: return None
        close  = df["Close"].squeeze()
        high   = df["High"].squeeze()
        low    = df["Low"].squeeze()
        open_  = df["Open"].squeeze()
        volume = df["Volume"].squeeze() if "Volume" in df.columns else None
        rsi_ind  = ta.momentum.RSIIndicator(close)
        rsi_val  = float(rsi_ind.rsi().iloc[-1])
        macd_val = float(ta.trend.MACD(close).macd_diff().iloc[-1])
        bb_val   = float(ta.volatility.BollingerBands(close).bollinger_pband().iloc[-1])
        price    = float(close.iloc[-1]); prev = float(close.iloc[-2])
        change_pct = (price-prev)/prev*100 if prev else 0.0
        vol_ratio  = 1.0
        if volume is not None and len(volume)>20:
            avg_v = float(volume.iloc[-20:].mean()); cur_v = float(volume.iloc[-1])
            vol_ratio = cur_v/avg_v if avg_v>0 else 1.0
        daily = yf.download(symbol, period="1y", interval="1d", progress=False, auto_adjust=True)
        sma20=sma50=price; atr=mom5=mom20=pos52=0.0; history=[]
        stoch_k_v=50.0; williams_v=-50.0; cci_v=0.0; divergence="none"; pattern="none"
        if not daily.empty and len(daily)>=50:
            dc=daily["Close"].squeeze(); dh=daily["High"].squeeze()
            dl=daily["Low"].squeeze(); do_=daily["Open"].squeeze()
            sma20 = float(dc.rolling(20).mean().iloc[-1])
            sma50 = float(dc.rolling(50).mean().iloc[-1])
            atr   = float(ta.volatility.AverageTrueRange(dh,dl,dc).average_true_range().iloc[-1])
            mom5  = float(dc.pct_change(5).iloc[-1])
            mom20 = float(dc.pct_change(20).iloc[-1])
            h52   = float(dc.rolling(min(252,len(dc))).max().iloc[-1])
            l52   = float(dc.rolling(min(252,len(dc))).min().iloc[-1])
            pos52 = (price-l52)/(h52-l52) if (h52-l52)>0 else 0.5
            history = [round(float(x),4) for x in dc.tolist()[-30:]]
            stoch_k_v, _ = get_stochastic(dc, dh, dl)
            williams_v   = get_williams_r(dc, dh, dl)
            cci_v        = get_cci(dc, dh, dl)
            rsi_full     = ta.momentum.RSIIndicator(dc).rsi()
            divergence   = detect_rsi_divergence(dc, rsi_full)
            if len(dc) >= 2: pattern = detect_candlestick_pattern(do_, dc, dh, dl)
        sma_ratio  = sma20/sma50 if sma50!=0 else 1.0
        vwap_ratio = 1.0
        if volume is not None:
            tv2 = float(volume.sum())
            if tv2>0: vwap_ratio = price / (float((close*volume).sum())/tv2)
        fdict = {
            "rsi":rsi_val,"macd":macd_val,"bb":bb_val,"sma_ratio":sma_ratio,
            "atr":atr,"mom5":mom5,"mom20":mom20,"vwap_ratio":vwap_ratio,
            "vol_ratio":vol_ratio,"pos52":pos52,
            "stoch_k":stoch_k_v,"williams_r":williams_v,"cci":cci_v,
        }
        # Technical score (5 indicators → 8 indicators now)
        ts = 0
        if rsi_val<30:  ts+=2
        elif rsi_val<45: ts+=1
        elif rsi_val>70: ts-=2
        elif rsi_val>55: ts-=1
        ts += (1 if macd_val>0 else -1)
        ts += (1 if bb_val<0.2 else -1 if bb_val>0.8 else 0)
        ts += (1 if mom5>0.03 else -1 if mom5<-0.03 else 0)
        ts += (1 if sma_ratio>1.02 else -1 if sma_ratio<0.98 else 0)
        # NEW indicators
        ts += (1 if stoch_k_v<20 else -1 if stoch_k_v>80 else 0)   # Stochastic
        ts += (1 if williams_v<-80 else -1 if williams_v>-20 else 0) # Williams %R
        ts += (1 if cci_v<-100 else -1 if cci_v>100 else 0)          # CCI
        volume_surge = vol_ratio >= 1.5
        # AI confidence with decay
        ai_conf     = get_ai_confidence(symbol, fdict)
        ai_agree    = get_ai_agreement(symbol, fdict)
        decay = vasu_brain.get("confidence_decay",{})
        if ai_conf and symbol not in bot["holdings"]:
            decay[symbol] = decay.get(symbol,0)+1
            if decay[symbol]>3: ai_conf = max(0.0, round(ai_conf - decay[symbol]*1.5, 1))
        else: decay[symbol] = 0
        vasu_brain["confidence_decay"] = decay
        sector = SECTOR_MAP.get(symbol,"tech")
        fund   = get_fundamentals(symbol)
        # News sentiment
        news_items = []
        try:
            raw = yf.Ticker(symbol).news or []
            news_items = [{"title":n.get("content",{}).get("title",""),
                           "link":n.get("content",{}).get("canonicalUrl",{}).get("url","#")}
                          for n in raw[:5] if n.get("content",{}).get("title")]
        except: pass
        sent_score = score_news_sentiment(news_items)
        composite  = compute_composite(symbol, ts, ai_conf, fund, sector, divergence, pattern, sent_score)
        # AI agreement adjustments
        if ai_agree:
            if ai_agree.get("all_agree_bull"): composite = min(100, composite+3)
            elif ai_agree.get("all_agree_bear"): composite = max(0, composite-3)
        grade = fund.get("fundamental_grade","C")
        if   composite>=72: signal,color = "STRONG BUY","#00ff88"
        elif composite>=58: signal,color = "BUY","#a8ff78"
        elif composite<=28: signal,color = "STRONG SELL","#ff4757"
        elif composite<=42: signal,color = "SELL","#ff9f43"
        else:               signal,color = "HOLD","#aaaaaa"
        if grade == "F" and signal in ("BUY","STRONG BUY"):
            signal,color = "HOLD","#aaaaaa"
        # Block buys right before earnings
        ed = fund.get("earnings_date","")
        if ed:
            try:
                days = (datetime.strptime(ed,"%Y-%m-%d")-datetime.now()).days
                if 0<=days<=2 and signal in ("BUY","STRONG BUY"):
                    signal,color = "HOLD","#aaaaaa"
            except: pass
        win_tracker[symbol] = {"signal":signal,"price":price,"composite":composite}
        return {
            "symbol":symbol,"price":round(price,4),"change_pct":round(change_pct,2),
            "rsi":round(rsi_val,1),"score":ts,"signal":signal,"color":color,
            "history":history,"ai_conf":ai_conf,"volume_surge":volume_surge,
            "vol_ratio":round(vol_ratio,2),"news":news_items[:3],
            "earnings_date":fund.get("earnings_date",""),
            "div_amount":fund.get("dividend_yield",0.0) or 0.0,
            "composite":composite,"fundamental_grade":grade,
            "fundamental_score":fund.get("fundamental_score",0),
            "pe_ratio":fund.get("pe_ratio"),"profit_margin":fund.get("profit_margin"),
            "revenue_growth":fund.get("revenue_growth"),"debt_to_equity":fund.get("debt_to_equity"),
            "earnings_surprise":fund.get("earnings_surprise"),
            "short_interest":fund.get("short_interest"),"institutional_pct":fund.get("institutional_pct"),
            "beta":fund.get("beta"),"sector_macro":get_sector_macro_score(sector),
            "dcf_signal":fund.get("dcf_signal",0),"quality_score":fund.get("quality_score",0),
            "ai_agreement":ai_agree,
            # NEW v3.1 fields
            "divergence":divergence,"pattern":pattern,"sentiment":sent_score,
            "stoch_k":round(stoch_k_v,1),"williams_r":round(williams_v,1),"cci":round(cci_v,1),
        }
    except Exception as e:
        print(f"  Analyze error {symbol}: {e}"); return None

# ==============================================================================
#  SELF-LEARNING ENGINE
# ==============================================================================
def vasu_learn():
    completed = [t for t in bot["trades"] if t.get("pnl") is not None]
    if len(completed) < 3: return
    recent = completed[-10:]
    wins   = [t for t in recent if t["pnl"]>0]
    losses = [t for t in recent if t["pnl"]<0]
    wr     = len(wins)/len(recent) if recent else 0
    new_lessons = []
    streak = 0
    for t in reversed(completed[-8:]):
        if t["pnl"]<0: streak+=1
        else: break
    vasu_brain["consecutive_losses"] = streak
    if streak>=4 and not vasu_brain.get("trading_paused"):
        vasu_brain["trading_paused"] = True
        new_lessons.append("4 losses in a row. Auto-trading paused. Switching to SNIPER.")
        if vasu_brain.get("active_mode")=="auto": vasu_brain["auto_target"]="sniper"
    elif streak==0 and vasu_brain.get("trading_paused"):
        vasu_brain["trading_paused"] = False
        new_lessons.append("Back to winning. Resuming trading.")
    mode_cfg  = get_active_mode_config()
    base_conf = mode_cfg["buy_confidence"]
    if wr<0.4 and vasu_brain["buy_confidence"]<base_conf+15:
        vasu_brain["buy_confidence"] = min(base_conf+15, vasu_brain["buy_confidence"]+2.5)
        vasu_brain["mood"] = "cautious"
        new_lessons.append(f"Win rate {round(wr*100)}%. Tightened threshold.")
    elif wr>0.7 and vasu_brain["buy_confidence"]>base_conf-5:
        vasu_brain["buy_confidence"] = max(base_conf-5, vasu_brain["buy_confidence"]-1.0)
        vasu_brain["mood"] = "aggressive"
        new_lessons.append(f"Win rate {round(wr*100)}%. Loosened threshold.")
    else: vasu_brain["mood"] = "neutral"
    avg_loss = abs(sum(t["pnl"] for t in losses)/len(losses)) if losses else 0
    if avg_loss>20 and vasu_brain["stop_loss_pct"]>1.5:
        vasu_brain["stop_loss_pct"] = max(1.5, round(vasu_brain["stop_loss_pct"]-0.5,1))
        new_lessons.append(f"Tightened stop to {vasu_brain['stop_loss_pct']}%")
    avg_win = sum(t["pnl"] for t in wins)/len(wins) if wins else 0
    if avg_win>30 and vasu_brain["take_profit_pct"]<15:
        vasu_brain["take_profit_pct"] = min(15, round(vasu_brain["take_profit_pct"]+0.5,1))
        new_lessons.append(f"Extended TP to {vasu_brain['take_profit_pct']}%")
    # Sector / stock / time scoring
    for t in completed[-20:]:
        sym=t["symbol"]; sec=SECTOR_MAP.get(sym)
        ss=vasu_brain["stock_scores"]; ss[sym]=ss.get(sym,0)+(2 if t["pnl"]>0 else -1)
        if sec: vasu_brain["sector_scores"][sec]=vasu_brain["sector_scores"].get(sec,0)+(1 if t["pnl"]>0 else -1)
        try:
            h=int(t["time"].split(" ")[1].split(":")[0])
            vasu_brain["time_scores"][str(h)+"h"]=vasu_brain["time_scores"].get(str(h)+"h",0)+(1 if t["pnl"]>0 else -1)
        except: pass
    # Sector blacklist
    scores = vasu_brain["sector_scores"]
    if scores:
        worst = min(scores, key=scores.get)
        if scores[worst]<=-3 and worst not in vasu_brain["sector_blacklist"]:
            vasu_brain["sector_blacklist"].append(worst); new_lessons.append(f"Blacklisted {worst} sector.")
        for s in list(vasu_brain["sector_blacklist"]):
            if scores.get(s,0)>=0: vasu_brain["sector_blacklist"].remove(s)
    vasu_brain["personality"] = "confident" if sum(t["pnl"] for t in completed)>100 else "cautious" if sum(t["pnl"] for t in completed)<-50 else "balanced"
    vasu_brain["total_adjustments"] += 1
    for lesson in new_lessons:
        vasu_brain["lessons"].append({"time":_ts(),"lesson":lesson})
        print(f"[VASU] Learned: {lesson}")
    vasu_brain["lessons"] = vasu_brain["lessons"][-25:]
    if len(completed)%10==0 and len(completed)>=10: evolve_weights()
    if len(completed)%20==0 and len(completed)>=20: run_self_coding_engine()
    save_brain()

def evolve_weights():
    completed = [t for t in bot["trades"] if t.get("pnl") is not None and t.get("layer_contribs")]
    if len(completed) < 10: return
    recent = completed[-30:]
    perf = {k:{"correct":0,"total":0} for k in ["technical","ai_model","fundamental","sector_macro","self_learned"]}
    for t in recent:
        contribs = t.get("layer_contribs",{}); won = t["pnl"]>0
        for layer,contrib in contribs.items():
            if layer not in perf: continue
            perf[layer]["total"] += 1
            if (won and contrib>0) or (not won and contrib<0): perf[layer]["correct"] += 1
    current = _get_weights(); new_weights = {}; changes = []
    for layer in current:
        d = perf.get(layer,{}); tot = d.get("total",0); acc = d["correct"]/tot if tot>=3 else 0.5
        old_w = current[layer]
        if acc>0.60:   new_w=min(55,old_w+min(2,(acc-0.5)*10)); changes.append(f"{layer}↑")
        elif acc<0.40: new_w=max(5,old_w-min(2,(0.5-acc)*10)); changes.append(f"{layer}↓")
        else: new_w=old_w
        new_weights[layer] = round(new_w,1)
    total = sum(new_weights.values())
    if total>0: new_weights = {k:round(v/total*100,1) for k,v in new_weights.items()}
    vasu_brain["weights"] = new_weights; vasu_brain["last_weight_update"] = _ts()
    if changes:
        vasu_brain["weight_history"].append({"time":_ts(),"changes":changes,"new":new_weights})
        vasu_brain["weight_history"] = vasu_brain["weight_history"][-20:]
        print(f"[WEIGHTS] Evolved: {' | '.join(changes)}"); save_brain()

def run_self_coding_engine():
    completed = [t for t in bot["trades"] if t.get("pnl") is not None]
    if len(completed) < 20: return
    log = _load_json(SELFCODE_FILE, lambda:{"version":1,"changes":[],"total_self_mods":0,"last_run":""})
    changes_made = []
    print(f"[SELFCODE] Running on {len(completed)} trades...")

    # Analysis 1: AI blend optimization
    model_acc = {m:{"c":0,"t":0} for m in ["rf","gb","et"]}
    for t in completed[-30:]:
        ai_d = t.get("ai_detail",{}); won = t["pnl"]>0
        for m in ["rf","gb","et"]:
            p = ai_d.get(m,0)/100
            model_acc[m]["t"] += 1
            if (won and p>0.55) or (not won and p<0.45): model_acc[m]["c"] += 1
    blend = dict(vasu_brain.get("ai_blend",{"rf":0.30,"gb":0.35,"et":0.35}))
    for m,stats in model_acc.items():
        if stats["t"]>=5:
            acc=stats["c"]/stats["t"]; old_w=blend.get(m,0.33)
            if acc>0.62:   new_w=min(0.55,old_w+0.03)
            elif acc<0.38: new_w=max(0.10,old_w-0.03)
            else: continue
            if abs(new_w-old_w)>0.01:
                blend[m]=new_w
                changes_made.append({"type":"ai_blend","model":m,"old":round(old_w,3),"new":round(new_w,3),
                                      "reason":f"{m.upper()} acc {round(acc*100)}%","timestamp":_ts()})
    total_b=sum(blend.values())
    if total_b>0: blend={k:round(v/total_b,3) for k,v in blend.items()}
    vasu_brain["ai_blend"] = blend

    # Analysis 2: RSI zone optimization
    rsi_b = {}
    for t in completed[-50:]:
        rv=t.get("rsi"); 
        if rv is None: continue
        bk=f"rsi_{int(rv//10)*10}"
        rsi_b.setdefault(bk,{"w":0,"t":0})
        rsi_b[bk]["t"]+=1
        if t["pnl"]>0: rsi_b[bk]["w"]+=1
    for bk,d in rsi_b.items():
        if d["t"]>=3:
            wr=d["w"]/d["t"]
            if wr>=0.65:
                changes_made.append({"type":"rsi_insight","bucket":bk,"win_rate":round(wr*100,1),
                                      "insight":f"{bk} zone has {round(wr*100)}% win rate","timestamp":_ts()})

    # Analysis 3: Divergence effectiveness (NEW v3.1)
    div_wins = {"bullish":{"w":0,"t":0},"none":{"w":0,"t":0}}
    for t in completed[-40:]:
        div=t.get("divergence","none")
        k = "bullish" if div=="bullish" else "none"
        div_wins[k]["t"]+=1
        if t["pnl"]>0: div_wins[k]["w"]+=1
    for div_type,d in div_wins.items():
        if d["t"]>=5:
            wr=d["w"]/d["t"]
            if div_type=="bullish" and wr>=0.60:
                changes_made.append({"type":"divergence_insight","divergence":div_type,"win_rate":round(wr*100,1),
                                      "insight":f"Bullish RSI divergence = {round(wr*100)}% win rate","timestamp":_ts()})

    # Analysis 4: Pattern effectiveness (NEW v3.1)
    pat_data = {}
    for t in completed[-40:]:
        pat=t.get("pattern","none")
        if pat=="none": continue
        pat_data.setdefault(pat,{"w":0,"t":0})
        pat_data[pat]["t"]+=1
        if t["pnl"]>0: pat_data[pat]["w"]+=1
    for pat,d in pat_data.items():
        if d["t"]>=3:
            wr=d["w"]/d["t"]
            if wr>=0.65:
                changes_made.append({"type":"pattern_insight","pattern":pat,"win_rate":round(wr*100,1),
                                      "insight":f"{pat} pattern = {round(wr*100)}% win rate","timestamp":_ts()})

    # Analysis 5: Hold duration optimization
    dur_data={}
    for t in completed[-30:]:
        if t.get("buy_time"):
            try:
                bt=datetime.strptime(t["buy_time"],"%m/%d %H:%M"); st=datetime.strptime(t["time"],"%m/%d %H:%M")
                hrs=max(0,int((st-bt).total_seconds()/3600))
                bk="fast" if hrs<2 else "medium" if hrs<24 else "long"
                dur_data.setdefault(bk,{"w":0,"t":0})
                dur_data[bk]["t"]+=1
                if t["pnl"]>0: dur_data[bk]["w"]+=1
            except: pass
    for dur,d in dur_data.items():
        if d["t"]>=3:
            wr=d["w"]/d["t"]
            if wr>=0.65:
                changes_made.append({"type":"duration_insight","duration":dur,"win_rate":round(wr*100,1),
                                      "insight":f"{dur} trades = {round(wr*100)}% win rate","timestamp":_ts()})

    if changes_made:
        log["total_self_mods"] += len(changes_made)
        log["changes"].extend(changes_made); log["changes"]=log["changes"][-100:]
        vasu_brain["selfcode_version"] = log["version"] = log.get("version",1)+1
        vasu_brain["selfcode_improvements"] = log["total_self_mods"]
        vasu_brain["selfcode_last_run"] = log["last_run"] = _ts()
        for ch in changes_made:
            vasu_brain["lessons"].append({"time":_ts(),"lesson":f"[SELFCODE] {ch['type']}: {ch.get('insight',ch.get('reason',''))}"})
        vasu_brain["lessons"] = vasu_brain["lessons"][-25:]
        _save_json(SELFCODE_FILE, log); save_brain()
        print(f"[SELFCODE] Applied {len(changes_made)} mods. Version: {log['version']}")
    return changes_made

# ==============================================================================
#  KELLY CRITERION (improved v3.1)
# ==============================================================================
def kelly_position_size(symbol):
    try:
        trades = [t for t in bot.get("trades",[]) if t.get("pnl") is not None]
        sym_t  = [t for t in trades if t.get("symbol")==symbol]
        use    = sym_t if len(sym_t)>=4 else trades
        if len(use) < 3: return 0.15
        wins   = [t["pnl"] for t in use if t["pnl"]>0]
        losses = [t["pnl"] for t in use if t["pnl"]<0]
        if not wins or not losses: return 0.15
        p=len(wins)/len(use); q=1-p
        b=abs(sum(wins)/len(wins)) / abs(sum(losses)/len(losses))
        f=(b*p-q)/b
        return round(max(0.05, min(0.25, f*0.25)), 3)
    except: return 0.15

def _position_size(ai_conf):
    mode_cfg = get_active_mode_config()
    base = mode_cfg["position_size_pct"]/100
    if not vasu_brain.get("position_size_by_conf",True) or ai_conf is None: return base
    if ai_conf>=85: return min(0.28, base*1.6)
    if ai_conf>=75: return min(0.22, base*1.3)
    if ai_conf<60:  return max(0.07, base*0.7)
    return base

# ==============================================================================
#  WATCHLIST / MARKET MOOD / DAILY BRIEF
# ==============================================================================
def update_market_mood():
    global market_mood
    items = all_items_flat()
    if not items: return
    bulls = [i for i in items if i["signal"] in ("BUY","STRONG BUY")]
    bears = [i for i in items if i["signal"] in ("SELL","STRONG SELL")]
    score = len(bulls)-len(bears); pct = round(len(bulls)/len(items)*100) if items else 50
    if   score>=8:  market_mood={"label":f"Very Bullish — {pct}% positive","color":"#00ff88","score":score}
    elif score>=4:  market_mood={"label":f"Bullish — {pct}% positive","color":"#a8ff78","score":score}
    elif score<=-8: market_mood={"label":f"Very Bearish — {pct}% negative","color":"#ff4757","score":score}
    elif score<=-4: market_mood={"label":f"Bearish — {pct}% negative","color":"#ff9f43","score":score}
    else:           market_mood={"label":"Neutral — Market mixed","color":"#aaaaaa","score":score}

def update_watchlist():
    global watchlist
    items = all_items_flat(); held = set(bot["holdings"].keys())
    thresh = vasu_brain["buy_confidence"]; bl = set(vasu_brain.get("sector_blacklist",[]))
    cands = []
    for item in items:
        sym=item["symbol"]; conf=item.get("ai_conf") or 0; gap=thresh-conf
        sec=SECTOR_MAP.get(sym,""); comp=item.get("composite",50); grade=item.get("fundamental_grade","C")
        if sym not in held and sec not in bl and 0<gap<30 and item["signal"] in ("BUY","STRONG BUY") and grade!="F":
            cands.append({"symbol":sym,"signal":item["signal"],"price":item["price"],
                          "ai_conf":conf,"gap":round(gap,1),"rsi":item["rsi"],"color":item["color"],
                          "surge":item.get("volume_surge",False),"composite":comp,"grade":grade,
                          "pe":item.get("pe_ratio"),"divergence":item.get("divergence","none"),
                          "pattern":item.get("pattern","none")})
    watchlist = sorted(cands, key=lambda x:x["composite"], reverse=True)[:8]

def generate_vasu_daily():
    global vasu_daily
    items=all_items_flat(); tv=get_bot_value()
    ts_=bot["wins"]+bot["losses"]; wr=round(bot["wins"]/ts_*100,1) if ts_>0 else 0
    bulls=[i for i in items if i["signal"] in ("BUY","STRONG BUY")]
    sec_scores=vasu_brain["sector_scores"]; best_s=max(sec_scores,key=sec_scores.get) if sec_scores else "tech"
    bl=vasu_brain.get("sector_blacklist",[]); bl_note=f" Avoiding {', '.join(bl)}." if bl else ""
    dd=vasu_brain.get("current_drawdown_pct",0); dd_note=f" Portfolio down {dd}% from peak." if dd>3 else ""
    pause=vasu_brain.get("trading_paused"); pause_note=" AUTO-TRADING PAUSED." if pause else ""
    blend=vasu_brain.get("ai_blend",{"rf":0.30,"gb":0.35,"et":0.35})
    sc_v=vasu_brain.get("selfcode_version",1); sc_m=vasu_brain.get("selfcode_improvements",0)
    hours_label = market_hours_label()
    divergence_stocks=[i["symbol"] for i in items if i.get("divergence")=="bullish"]
    div_note=f" Bullish RSI divergence: {', '.join(divergence_stocks[:3])}." if divergence_stocks else ""
    vasu_daily=(f"Mode: {get_mode_label()}. Regime: {vasu_brain.get('market_regime','unknown')} "
                f"(score {vasu_brain.get('market_score',0)}). "
                f"${round(tv,2)} | {len(bot['holdings'])} positions | ${round(bot['cash'],2)} cash. "
                f"{len(bulls)} buy signals. Threshold {round(vasu_brain['buy_confidence'],1)}%. "
                f"{best_s.upper()} my best sector.{bl_note}{dd_note}{pause_note}{div_note} "
                f"Market: {hours_label}. SelfCode v{sc_v} | {sc_m} mods. "
                f"AI: RF{round(blend.get('rf',0)*100)}% GB{round(blend.get('gb',0)*100)}% ET{round(blend.get('et',0)*100)}%")

# ==============================================================================
#  ANALYST COUNCIL
# ==============================================================================
def generate_analyst_advice():
    global analyst_advice
    items=all_items_flat()
    if not items: return
    pm={i["symbol"]:i["price"] for i in items}; sm={i["symbol"]:i for i in items}
    tv=get_bot_value(); pnl=tv-bot["start_value"]; pnl_pct=pnl/bot["start_value"]*100
    ts_=bot["wins"]+bot["losses"]; wr=round(bot["wins"]/ts_*100,1) if ts_>0 else 0
    cash=bot["cash"]; cash_pct=(cash/tv)*100 if tv>0 else 100
    bulls=[i for i in items if i["signal"] in ("BUY","STRONG BUY")]
    bears=[i for i in items if i["signal"] in ("SELL","STRONG SELL")]
    strong_buys=[i for i in items if i["signal"]=="STRONG BUY"]
    oversold=[i for i in items if i["rsi"]<30]; overbought=[i for i in items if i["rsi"]>70]
    high_conf=[i for i in items if (i.get("ai_conf") or 0)>=70 and i["signal"] in ("BUY","STRONG BUY")]
    movers=[i for i in sorted(items,key=lambda x:x["change_pct"],reverse=True) if i["change_pct"]>1.5][:3]
    hd=[]
    for sym,h in bot["holdings"].items():
        cur=pm.get(sym,h["buy_price"]); pp=((cur-h["buy_price"])/h["buy_price"])*100
        item=sm.get(sym)
        hd.append({"sym":sym,"pp":pp,"cur":cur,"buy":h["buy_price"],
                   "signal":item["signal"] if item else "N/A","rsi":item["rsi"] if item else 50})
    at_risk=[h for h in hd if h["pp"]<=-2]; winning=[h for h in hd if h["pp"]>=3]
    stop=vasu_brain["stop_loss_pct"]; tp=vasu_brain["take_profit_pct"]
    mode_label=get_mode_label(); regime=vasu_brain.get("market_regime","unknown")
    drawdown=vasu_brain.get("current_drawdown_pct",0); heat=round(_portfolio_heat()*100)
    blend=vasu_brain.get("ai_blend",{"rf":0.30,"gb":0.35,"et":0.35})
    sc_mods=vasu_brain.get("selfcode_improvements",0)
    divergence_stocks=[i["symbol"] for i in items if i.get("divergence")=="bullish"]
    pattern_stocks=[i for i in items if i.get("pattern") in ("hammer","bullish_engulfing")]

    claude=[]
    if vasu_brain.get("trading_paused"): claude.append(f"TRADING PAUSED: {vasu_brain.get('consecutive_losses',0)} consecutive losses.")
    if drawdown>=5: claude.append(f"Portfolio {drawdown}% below peak. Protective mode active.")
    if cash_pct<15: claude.append(f"Cash at {round(cash_pct)}% — fully deployed.")
    elif cash_pct>75: claude.append(f"Cash at {round(cash_pct)}%. {len(bulls)} signals but threshold not met.")
    else: claude.append(f"Cash at {round(cash_pct)}%. Heat: {heat}%. Room to deploy.")
    if ts_>5:
        if wr>=60: claude.append(f"Win rate {wr}% — strong. Staying the course.")
        elif wr>=50: claude.append(f"Win rate {wr}% — marginal. STRONG BUY + high confidence only.")
        else: claude.append(f"Win rate {wr}% — below 50%. Mode: {mode_label}.")
    if at_risk: claude.append(f"At risk: {', '.join([h['sym']+' ('+str(round(h['pp'],1))+'%)' for h in at_risk])}.")
    if winning: claude.append(f"{', '.join([h['sym'] for h in winning])} working. TP at +{tp}%.")
    if sc_mods>0: claude.append(f"Self-coding v{vasu_brain.get('selfcode_version',1)}: {sc_mods} modifications. Blend: RF{round(blend.get('rf',0)*100)}% GB{round(blend.get('gb',0)*100)}% ET{round(blend.get('et',0)*100)}%")
    analyst_advice["claude"]=claude

    arya=[]
    arya.append(f"{'Bullish' if len(bulls)>len(bears) else 'Bearish'}. {len(strong_buys)} STRONG BUY, {len(bulls)} buys vs {len(bears)} sells.")
    if oversold: arya.append(f"Oversold: {', '.join([i['symbol']+' RSI'+str(i['rsi']) for i in oversold[:4]])}.")
    if divergence_stocks: arya.append(f"Bullish RSI divergence (v3.1): {', '.join(divergence_stocks[:4])} — reversal signal.")
    if pattern_stocks: arya.append(f"Bullish candle patterns: {', '.join([i['symbol']+' ('+i['pattern']+')' for i in pattern_stocks[:3]])}.")
    agree_items=[i for i in items if i.get("ai_agreement",{}).get("all_agree_bull")]
    if agree_items: arya.append(f"Triple AI UNANIMOUS BUY: {', '.join([i['symbol'] for i in agree_items[:4]])} — all 3 models agree.")
    if strong_buys:
        best=sorted(strong_buys,key=lambda x:(x.get("ai_conf") or 0),reverse=True)[0]
        arya.append(f"Best setup: {best['symbol']} STRONG BUY, RSI {best['rsi']}, AI {best.get('ai_conf','N/A')}%.")
    analyst_advice["arya"]=arya

    magnus=[]
    magnus.append(f"{'Risk-on' if len(bulls)>len(bears) else 'Risk-off'}. {round(len(bulls)/len(items)*100) if items else 0}% positive. Regime: {regime}.")
    earn_soon=[i for i in items if i.get("earnings_date") and i["earnings_date"]>datetime.now().strftime("%Y-%m-%d")]
    if earn_soon: magnus.append(f"Earnings soon: {', '.join([i['symbol']+' ('+i['earnings_date']+')' for i in earn_soon[:4]])}. Extra risk.")
    mp=vasu_brain.get("mode_performance",{})
    if mp:
        bm=max(mp,key=lambda m:(mp[m]["wins"]/(mp[m]["wins"]+mp[m]["losses"])) if (mp[m]["wins"]+mp[m]["losses"])>0 else 0)
        bd=mp[bm]; bwr=round(bd["wins"]/(bd["wins"]+bd["losses"])*100) if (bd["wins"]+bd["losses"])>0 else 0
        magnus.append(f"Best mode historically: {bm} ({bwr}% win rate).")
    magnus.append(f"Hours status: {market_hours_label()}.")
    analyst_advice["magnus"]=magnus

    zeus=[]
    if len(strong_buys)>=5: zeus.append(f"Signal cluster — {len(strong_buys)} STRONG BUYs. VASU in {mode_label}.")
    elif len(strong_buys)>=2: zeus.append(f"{len(strong_buys)} STRONG BUYs. Cherry-pick highest confidence.")
    else: zeus.append(f"Only {len(strong_buys)} STRONG BUY. Momentum quiet.")
    if high_conf:
        bp=sorted(high_conf,key=lambda x:x.get("ai_conf",0),reverse=True)[:3]
        zeus.append(f"Highest conviction: {', '.join([i['symbol']+' ('+str(i.get('ai_conf','?'))+'%)' for i in bp])}.")
    if movers: zeus.append(f"Momentum: {', '.join([i['symbol']+' +'+str(round(i['change_pct'],1))+'%' for i in movers])}.")
    zeus.append(f"Threshold {round(vasu_brain['buy_confidence'],1)}% in {mode_label}.")
    analyst_advice["zeus"]=zeus

    bear_a=[]
    bear_a.append(f"{'Up' if pnl>=0 else 'Down'} {round(abs(pnl_pct),1)}% across {len(bot['trades'])} trades.")
    if hd:
        wh=min(hd,key=lambda x:x["pp"]); bh=max(hd,key=lambda x:x["pp"])
        if wh["pp"]<-1.5: bear_a.append(f"{wh['sym']} down {round(abs(wh['pp']),1)}% — patience or denial?")
        if bh["pp"]>3:    bear_a.append(f"{bh['sym']} up {round(bh['pp'],1)}% — paper profits disappear fast.")
    else: bear_a.append("No positions open.")
    bear_a.append(f"Win rate {'solid — stay humble.' if wr>=55 else 'below 55%. Raise the bar.'}")
    analyst_advice["bear_analyst"]=bear_a

# ==============================================================================
#  DREAM TRADE FINDER
# ==============================================================================
def find_dream_trades():
    items=all_items_flat(); dreams=[]
    for item in items:
        sym=item["symbol"]; sc=0; reasons=[]
        grade=item.get("fundamental_grade","C")
        if grade=="A": sc+=3; reasons.append("Grade A fundamentals")
        elif grade=="B": sc+=2; reasons.append("Grade B fundamentals")
        elif grade in ("D","F"): continue
        if item["signal"]=="STRONG BUY": sc+=3; reasons.append("STRONG BUY")
        elif item["signal"]=="BUY": sc+=1; reasons.append("BUY")
        else: continue
        comp=item.get("composite",0)
        if comp>=75: sc+=3; reasons.append(f"Composite {comp}/100")
        elif comp>=60: sc+=1
        if item["rsi"]<35: sc+=3; reasons.append(f"Oversold RSI {item['rsi']}")
        elif item["rsi"]<45: sc+=1
        if item.get("volume_surge"): sc+=2; reasons.append("Volume surge")
        if item.get("ai_agreement",{}).get("all_agree_bull"): sc+=3; reasons.append("TRIPLE AI UNANIMOUS")
        if item.get("divergence")=="bullish": sc+=2; reasons.append("RSI bullish divergence")
        if item.get("pattern") in ("hammer","bullish_engulfing"): sc+=2; reasons.append(f"Pattern: {item['pattern']}")
        if item.get("sentiment",0)>=2: sc+=1; reasons.append("Positive news")
        ed=item.get("earnings_date","")
        if ed:
            try:
                days=(datetime.strptime(ed,"%Y-%m-%d")-datetime.now()).days
                if 0<=days<=5: sc-=5
            except: pass
        if SECTOR_MAP.get(sym,"") in vasu_brain.get("sector_blacklist",[]): continue
        ss=vasu_brain.get("stock_scores",{}).get(sym,0)
        if ss>0: sc+=1; reasons.append("Positive track record")
        elif ss<-3: sc-=2
        if sc>=10: dreams.append({"symbol":sym,"dream_score":sc,"signal":item["signal"],
                                   "composite":comp,"grade":grade,"rsi":item["rsi"],
                                   "price":item["price"],"reasons":reasons,"sector":SECTOR_MAP.get(sym,"")})
    return sorted(dreams, key=lambda x:x["dream_score"], reverse=True)

def refresh_dream_trades():
    global dream_trades_cache
    dream_trades_cache = find_dream_trades()
    if dream_trades_cache:
        best=dream_trades_cache[0]
        if best["dream_score"]>=14: print(f"[DREAM] {best['symbol']} score:{best['dream_score']} | {' | '.join(best['reasons'][:3])}")

# ==============================================================================
#  BOT TRADING ENGINE
# ==============================================================================
def bot_trade():
    if not bot["enabled"] or vasu_brain.get("trading_paused"): return
    items=all_items_flat()
    if not items: return
    # Only trade during market hours (skip if closed)
    if not market_is_open():
        return
    mode_cfg=get_active_mode_config()
    now_h=datetime.now().hour
    worst_hours={int(k.replace("h","")) for k,v in vasu_brain.get("time_scores",{}).items() if v<=-3}
    max_heat=mode_cfg.get("max_heat",0.80); stop_pct=vasu_brain["stop_loss_pct"]
    tp_pct=vasu_brain["take_profit_pct"]; use_trail=mode_cfg.get("trailing_stop",True)
    min_sig=mode_cfg.get("min_signal","BUY"); req_surge=mode_cfg.get("require_surge",False)
    req_over=mode_cfg.get("require_oversold",False); min_vol=vasu_brain.get("min_volume_ratio",0.8)
    trail_h=vasu_brain.get("trailing_highs",{})

    for item in items:
        sym=item["symbol"]; price=item["price"]; signal=item["signal"]
        ai_conf=item.get("ai_conf") or 0; sector=SECTOR_MAP.get(sym,"")
        surge=item.get("volume_surge",False); comp=item.get("composite",50)
        grade=item.get("fundamental_grade","C"); ai_agree=item.get("ai_agreement",{})
        divergence=item.get("divergence","none"); pattern=item.get("pattern","none")

        # EXIT LOGIC
        if sym in bot["holdings"]:
            h=bot["holdings"][sym]; pp=((price-h["buy_price"])/h["buy_price"])*100
            if use_trail:
                if sym not in trail_h or price>trail_h[sym]: trail_h[sym]=price
                high=trail_h[sym]; drop=((price-high)/high)*100
                peak_pp=((high-h["buy_price"])/h["buy_price"])*100
                if peak_pp>=2 and drop<=-stop_pct:
                    _close(sym,price,h,"TRAILING STOP","Trail stop: "+str(round(abs(drop),1))+"% from peak"); continue
            if pp<=-stop_pct:
                _close(sym,price,h,"STOP LOSS","Stop loss "+str(round(pp,1))+"%"); continue
            eff_tp=tp_pct+(3 if grade=="A" else 0)
            if pp>=eff_tp:
                _close(sym,price,h,"TAKE PROFIT","TP +"+str(round(pp,1))+"%"); continue
            if signal in ("SELL","STRONG SELL"):
                _close(sym,price,h,"SELL",f"{signal} signal"); continue
            if comp<30 and pp>1:
                _close(sym,price,h,"SELL","Composite collapsed to "+str(comp)); continue

        # ENTRY LOGIC
        if _portfolio_heat()>=max_heat: continue
        is_dream=any(d["symbol"]==sym for d in dream_trades_cache)
        eff_thresh=vasu_brain["buy_confidence"]
        eff_thresh -= (3 if surge else 0)
        eff_thresh -= (5 if is_dream else 0)
        eff_thresh -= (5 if ai_agree.get("all_agree_bull") else 0)
        eff_thresh -= (3 if divergence=="bullish" else 0)  # NEW v3.1
        eff_thresh -= (2 if pattern in ("hammer","bullish_engulfing") else 0)  # NEW v3.1
        eff_thresh += {"A":-5,"B":-2.5,"C":0,"D":3,"F":999}.get(grade,0)
        eff_thresh -= get_sector_macro_score(sector)*1.5
        if ai_agree.get("consensus")=="split": eff_thresh += 3
        comp_min=60 if min_sig=="STRONG BUY" else 45
        sig_ok  = (signal=="STRONG BUY" if min_sig=="STRONG BUY" else signal in ("BUY","STRONG BUY"))
        if (sig_ok and ai_conf>=eff_thresh and sym not in bot["holdings"] and bot["cash"]>=50
                and len(bot["holdings"])<mode_cfg["max_positions"]
                and sector not in vasu_brain.get("sector_blacklist",[])
                and now_h not in worst_hours and item.get("vol_ratio",1)>=min_vol
                and not _is_blacklisted(sym)
                and (surge if req_surge else True)
                and (item["rsi"]<35 if req_over else True)
                and comp>=comp_min and grade!="F"):
            # Position sizing: Kelly + mode + volatility
            kelly=kelly_position_size(sym); mode_sz=_position_size(ai_conf)
            pos=round(mode_sz*0.70+kelly*0.30, 4)
            if comp>=80: pos=min(0.30,pos*1.4)
            elif comp>=70: pos=min(0.25,pos*1.2)
            if is_dream: pos=min(0.30,pos*1.15)
            if ai_agree.get("all_agree_bull"): pos=min(0.30,pos*1.10)
            if divergence=="bullish": pos=min(0.30,pos*1.05)   # NEW v3.1 boost
            pos=max(0.05,min(0.30,pos))
            invest=bot["cash"]*pos; shares=invest/price
            bot["holdings"][sym]={"shares":shares,"buy_price":price,"invested":invest,
                                   "buy_grade":grade,"buy_composite":comp}
            bot["cash"]-=invest; trail_h[sym]=price
            ai_det={}
            if item.get("ai_agreement"):
                ag=item["ai_agreement"]; ai_det={"rf":ag["rf"],"gb":ag["gb"],"et":ag["et"]}
            reason=(f"{signal} | Composite:{comp}/100 | Grade:{grade} | AI:{ai_conf}% | "
                    f"RSI:{item['rsi']} | Mode:{get_mode_label()}"
                    +(" [TRIPLE AI]" if ai_agree.get("all_agree_bull") else "")
                    +(" [DIVERGENCE]" if divergence=="bullish" else "")
                    +(" ["+pattern.upper()+"]" if pattern!="none" else "")
                    +(" [DREAM]" if is_dream else "")
                    +(" [SURGE]" if surge else ""))
            lc=get_fundamentals(sym).get("_contrib",{})
            bot["trades"].append({"type":"BUY","symbol":sym,"price":price,"shares":round(shares,4),
                                  "amount":round(invest,2),"time":_ts(),"pnl":None,"reason":reason,
                                  "rsi":item["rsi"],"ai_conf":ai_conf,"vol_ratio":round(item.get("vol_ratio",1),2),
                                  "mode":vasu_brain.get("active_mode","auto"),"composite":comp,
                                  "fundamental_grade":grade,"layer_contribs":lc,"ai_detail":ai_det,
                                  "buy_time":_ts(),"divergence":divergence,"pattern":pattern})
            save_bot()
            print(f"[BUY] {sym} @ ${price} [{grade}] score:{comp}{' [TRIPLE AI]' if ai_agree.get('all_agree_bull') else ''}{' [DIV]' if divergence=='bullish' else ''}")

    vasu_brain["trailing_highs"]=trail_h; save_brain()

def _close(sym, price, h, trade_type, reason):
    value=h["shares"]*price; pnl=value-h["invested"]
    bot["cash"]+=value; del bot["holdings"][sym]
    if pnl>0: bot["wins"]+=1
    else: bot["losses"]+=1
    bot["trades"].append({"type":trade_type,"symbol":sym,"price":price,"shares":round(h["shares"],4),
                          "amount":round(value,2),"time":_ts(),"pnl":round(pnl,2),"reason":reason})
    th=vasu_brain.get("trailing_highs",{})
    if sym in th: del th[sym]
    vasu_brain["trailing_highs"]=th
    save_bot(); vasu_learn()
    print(f"[{trade_type}] {sym} P&L:${round(pnl,2)}")

# ==============================================================================
#  CHAT COMMANDS
# ==============================================================================
def _extract_sym(q):
    AMBIG={"A","I","BE","GO","AT","IT","ON","OR","IN","IS","DO","SO","MA","ME","RE","V","C","M","T"}
    CTX={"buy","sell","stock","price","rsi","signal","shares","invest","trading","bullish","bearish","earnings","chart","hold","long","short","ticker","dividend","analysis"}
    q_up=q.upper(); qw=set(q_up.split()); has_ctx=bool(CTX.intersection(set(q.lower().split())))
    for sym in sorted(ALL_STOCKS, key=len, reverse=True):
        matched=(sym in qw) or ((" "+sym+" ") in (" "+q_up+" "))
        if matched:
            if sym in AMBIG and not has_ctx: continue
            return sym
    return None

def _extract_amt(q):
    q2=q.replace(",","").replace("$","")
    import re; m=re.search(r'\b(\d+(?:\.\d+)?)\b',q2)
    return float(m.group(1)) if m else None

def cmd_buy(sym, amt=None):
    items=all_items_flat(); pm={i["symbol"]:i["price"] for i in items}; sm={i["symbol"]:i for i in items}
    sym=sym.upper()
    if sym not in ALL_STOCKS: return f"{sym} not in watchlist. Add to config.py first."
    if sym in bot["holdings"]: return f"Already holding {sym}. One position per stock."
    item=sm.get(sym); price=item["price"] if item else None
    if not price: return f"No live price for {sym} yet."
    invest=min(amt if amt else bot["cash"]*_position_size(item.get("ai_conf") if item else None), bot["cash"])
    if invest<10: return f"Not enough cash. Only ${round(bot['cash'],2)} available."
    shares=invest/price
    bot["holdings"][sym]={"shares":shares,"buy_price":price,"invested":invest}
    bot["cash"]-=invest
    ai_agree=item.get("ai_agreement",{}) if item else {}
    bot["trades"].append({"type":"BUY","symbol":sym,"price":price,"shares":round(shares,4),
                          "amount":round(invest,2),"time":_ts(),"pnl":None,"reason":"Manual buy",
                          "rsi":item["rsi"] if item else 0,"ai_conf":item.get("ai_conf") if item else None,
                          "vol_ratio":item.get("vol_ratio",1) if item else 1,
                          "mode":vasu_brain.get("active_mode","auto"),
                          "buy_time":_ts(),"composite":item.get("composite",50) if item else 50,
                          "fundamental_grade":item.get("fundamental_grade","C") if item else "C"})
    save_bot()
    tag="[TRIPLE AI UNANIMOUS] " if ai_agree.get("all_agree_bull") else ""
    return f"{tag}Bought {round(shares,4)} shares of {sym} @ ${round(price,4)}. Invested ${round(invest,2)}."

def cmd_sell(sym):
    sym=sym.upper()
    if sym not in bot["holdings"]: return f"Not holding {sym}."
    items=all_items_flat(); pm={i["symbol"]:i["price"] for i in items}
    h=bot["holdings"][sym]; price=pm.get(sym,h["buy_price"])
    pnl_v=round((price-h["buy_price"])*h["shares"],2)
    _close(sym,price,h,"MANUAL SELL","Sold via chat")
    return f"Sold {sym} @ ${round(price,4)}. {'Made' if pnl_v>=0 else 'Lost'} ${abs(pnl_v)}. Cash: ${round(bot['cash'],2)}."

def cmd_sell_all():
    if not bot["holdings"]: return "No open positions."
    items=all_items_flat(); pm={i["symbol"]:i["price"] for i in items}; sold=[]
    for sym in list(bot["holdings"].keys()):
        h=bot["holdings"][sym]; price=pm.get(sym,h["buy_price"])
        pnl_v=round((price-h["buy_price"])*h["shares"],2)
        _close(sym,price,h,"MANUAL SELL","Sell all via chat")
        sold.append(sym+(" +$" if pnl_v>=0 else " -$")+str(abs(pnl_v)))
    return "Closed all: "+", ".join(sold)+f". Cash: ${round(bot['cash'],2)}."

def vasu_respond(question):
    q=question.lower().strip(); qw=set(q.split())
    items=all_items_flat(); pm={i["symbol"]:i["price"] for i in items}; sm={i["symbol"]:i for i in items}
    tv=get_bot_value(); pnl=tv-bot["start_value"]; pnl_pct=pnl/bot["start_value"]*100
    ts_=bot["wins"]+bot["losses"]; wr=round(bot["wins"]/ts_*100,1) if ts_>0 else 0
    cash=bot["cash"]; num_h=len(bot["holdings"])
    completed=[t for t in bot["trades"] if t.get("pnl") is not None]
    mode_label=get_mode_label(); mode_cfg=get_active_mode_config()
    conf=vasu_brain["buy_confidence"]; stop=vasu_brain["stop_loss_pct"]; tp=vasu_brain["take_profit_pct"]
    regime=vasu_brain.get("market_regime","unknown"); blend=vasu_brain.get("ai_blend",{"rf":0.30,"gb":0.35,"et":0.35})
    bulls=[i for i in items if i["signal"] in ("BUY","STRONG BUY")]
    sym_found=_extract_sym(q); amt_found=_extract_amt(q)
    def has(*phrases):
        for p in phrases:
            if " " in p: return p in q
            return p in qw
    # Mode switches
    for mode_name,triggers in {
        "auto":["auto mode","go auto","switch to auto","let vasu decide","automatic","back to auto"],
        "sniper":["sniper","go sniper","switch to sniper","only best","be picky","be selective"],
        "aggressive":["be aggressive","go aggressive","aggressive mode","yolo","high risk","go hard","full send"],
        "balanced":["balanced mode","go balanced","switch to balanced","normal mode","default mode","back to normal"],
        "conservative":["be conservative","go conservative","conservative mode","play it safe","low risk","protect"],
        "momentum":["momentum mode","go momentum","switch to momentum","chase momentum","ride the wave"],
        "contrarian":["contrarian mode","go contrarian","switch to contrarian","buy the dip","buy fear"],
        "bear":["bear mode","go bear","switch to bear","survival mode","protect cash","crash mode","defensive"],
    }.items():
        if any(t in q for t in triggers):
            old=vasu_brain.get("active_mode","auto"); set_mode(mode_name,"Manual chat switch")
            cfg=MODE_CONFIGS[mode_name]
            if mode_name!="auto":
                vasu_brain["buy_confidence"]=cfg["buy_confidence"]
                vasu_brain["stop_loss_pct"]=cfg["stop_loss_pct"]
                vasu_brain["take_profit_pct"]=cfg["take_profit_pct"]
                save_brain()
            extra="Now reading market automatically." if mode_name=="auto" else f"Threshold: {cfg['buy_confidence']}%, Stop: -{cfg['stop_loss_pct']}%, TP: +{cfg['take_profit_pct']}%. Max {cfg['max_positions']} positions."
            return f"Switched from {old.upper()} to {cfg['label']}.\n{cfg['desc']}\n{extra}"

    # Trade commands
    if any(w in qw for w in ["buy","purchase","grab"]) and sym_found and not any(w in qw for w in ["should","would","why","worth","think","opinion","what","how"]):
        return cmd_buy(sym_found, amt_found)
    if any(w in qw for w in ["sell","close","dump","exit"]) and not any(w in qw for w in ["should","would","why","think","opinion","what","how","don"]):
        if any(p in q for p in ["everything","sell all","close all","close everything"]): return cmd_sell_all()
        if sym_found and sym_found in bot["holdings"]: return cmd_sell(sym_found)
        if sym_found: return f"Not holding {sym_found}."
        if bot["holdings"]: return "Which one? Holding: "+", ".join(bot["holdings"].keys())

    # Self-coding
    if any(p in q for p in ["self code","selfcode","self coding","how are you evolving","ai blend","model weights","which model"]):
        log=_load_json(SELFCODE_FILE, lambda:{"version":1,"changes":[],"total_self_mods":0,"last_run":""})
        sc_v=log.get("version",1); sc_m=log.get("total_self_mods",0)
        parts=[f"=== VASU SELF-CODING ENGINE v{sc_v} ===",f"Total self-modifications: {sc_m} | Last run: {log.get('last_run','Never')}",
               f"Triggers every 20 closed trades. Trades until next: {20-(len(completed)%20) if completed else 20}",
               f"\nCurrent AI blend: RF={round(blend.get('rf',0)*100)}% | GB={round(blend.get('gb',0)*100)}% | ET={round(blend.get('et',0)*100)}%",
               "The engine adjusts blend weights based on which model predicts wins best.",
               "\nv3.1 Analysis types: AI blend, RSI zones, RSI divergence, candlestick patterns, hold duration"]
        recent=log.get("changes",[])[-5:]
        if recent:
            parts.append("\nLast 5 mods:")
            for c in recent: parts.append(f"  [{c.get('timestamp','?')}] {c.get('type','?')}: {c.get('insight',c.get('reason',''))[:80]}")
        return "\n".join(parts)

    # Stock lookup
    CTX={"buy","sell","hold","price","rsi","signal","confidence","stock","trade","invest","worth","good","should","think","about","thoughts","opinion","going","trend","news","earnings","dividend","chart","analysis","grade","composite","fundamental","pe","margin","growth","beta","short","triple","divergence","pattern"}
    do_lookup = sym_found and (q.upper().strip()==sym_found or bool(CTX.intersection(qw)) or sym_found in bot["holdings"])
    if do_lookup:
        item=sm.get(sym_found)
        if not item: return f"{sym_found}: no live data yet."
        chg_note=f"({'+'if item['change_pct']>=0 else ''}{item['change_pct']}% today)"
        comp=item.get("composite"); grade=item.get("fundamental_grade","")
        comp_note=""
        if comp is not None:
            cc=f"\nVASU composite: {comp}/100"
            cc+=(" — strong conviction." if comp>=75 else " — above average." if comp>=60 else " — neutral." if comp>=45 else " — weak. Wouldn't buy.")
            comp_note=cc
        # NEW v3.1 fields
        div_note=""; pat_note=""
        if item.get("divergence")=="bullish": div_note="\nBullish RSI divergence detected — potential reversal signal."
        elif item.get("divergence")=="bearish": div_note="\nBearish RSI divergence — caution."
        if item.get("pattern") not in (None,"none"):
            pat_map={"hammer":"🔨 Hammer — bullish reversal","bullish_engulfing":"Bullish engulfing — strong reversal",
                     "shooting_star":"Shooting star — bearish reversal","bearish_engulfing":"Bearish engulfing — reversal",
                     "doji":"Doji — indecision"}
            pat_note="\nPattern: "+pat_map.get(item["pattern"],item["pattern"])
        sent=item.get("sentiment",0)
        sent_note=("\nNews: "+("positive" if sent>0 else "negative" if sent<0 else "neutral")+f" (score {sent:+d})") if sent else ""
        # AI agreement
        agree_note=""
        if item.get("ai_agreement"):
            ag=item["ai_agreement"]
            agree_note=f"\nTriple AI: RF {ag['rf']}% | GB {ag['gb']}% | ET {ag['et']}%"
            agree_note+=(" — ALL THREE BULLISH" if ag.get("all_agree_bull") else " — ALL THREE BEARISH" if ag.get("all_agree_bear") else " — MODELS SPLIT")
        # Fundamentals
        fl=[]
        pe=item.get("pe_ratio"); pm_v=item.get("profit_margin"); rg=item.get("revenue_growth"); de=item.get("debt_to_equity")
        beta=item.get("beta"); si=item.get("short_interest"); ip=item.get("institutional_pct"); ed=item.get("earnings_date","")
        if pe: fl.append(f"P/E: {pe} ({'cheap' if pe<15 else 'fair' if pe<25 else 'stretched' if pe<40 else 'expensive'})")
        if pm_v is not None: fl.append(f"Profit margin: {pm_v}% ({'excellent' if pm_v>20 else 'good' if pm_v>10 else 'thin' if pm_v>0 else 'losing money'})")
        if rg is not None: fl.append(f"Revenue growth: {'+' if rg>=0 else ''}{rg}% ({'high growth' if rg>20 else 'growing' if rg>5 else 'flat' if rg>-5 else 'shrinking'})")
        if de is not None: fl.append(f"Debt/Equity: {de} ({'clean' if de<0.5 else 'manageable' if de<1.5 else 'leveraged'})")
        if beta: fl.append(f"Beta: {beta} ({'stable' if beta<0.8 else 'market-like' if beta<1.2 else 'volatile'})")
        if si and si>3: fl.append(f"Short interest: {si}% {'(danger zone)' if si>20 else ''}")
        if ip: fl.append(f"Institutions: {ip}% {'(smart money)' if ip>70 else ''}")
        if ed:
            try:
                days=(datetime.strptime(ed,"%Y-%m-%d")-datetime.now()).days
                if 0<=days<=3: fl.append(f"⚠️ EARNINGS IN {days} DAYS — high risk")
                elif 0<=days<=14: fl.append(f"Earnings in {days} days ({ed})")
            except: pass
        fund_note=("\n\nFundamentals:\n"+"\n".join([f"  {f}" for f in fl])) if fl else ""
        gap=conf-(item.get("ai_conf") or 0)
        thresh_note=f"\nPast {conf}% threshold." if gap<=0 else f"\nNeeds {round(gap,1)}% more to clear {conf}% threshold."
        hold_note=""
        if sym_found in bot["holdings"]:
            h=bot["holdings"][sym_found]; cur=pm.get(sym_found,h["buy_price"]); pp=(cur-h["buy_price"])/h["buy_price"]*100
            hold_note=f"\n\nHolding: bought ${round(h['buy_price'],4)}, now {'up' if pp>=0 else 'down'} {round(pp,1)}%."
            if item["signal"] in ("SELL","STRONG SELL"): hold_note+=" Signal turned negative — exit on radar."
        return (f"{sym_found}: ${round(item['price'],4)} {chg_note}\nSignal: {item['signal']} | RSI: {item['rsi']}"
                +comp_note+div_note+pat_note+sent_note+agree_note+thresh_note+fund_note+hold_note)

    # Greetings
    if qw.intersection({"hello","hi","hey","sup","yo","hiya"}):
        hours_=market_hours_label()
        return (f"Running clean. ${round(tv,2)} in {mode_label}. {num_h} positions, ${round(cash,2)} cash. "
                f"{len(bulls)} buy signals. Regime: {regime}. Market: {hours_}. "
                f"Self-coding v{vasu_brain.get('selfcode_version',1)}. "
                f"AI: RF{round(blend.get('rf',0)*100)}% GB{round(blend.get('gb',0)*100)}% ET{round(blend.get('et',0)*100)}%")

    # Performance
    if any(p in q for p in ["pnl","profit","performance","how am i","how are we","returns","portfolio value"]):
        track=""
        if completed:
            wins_l=[t for t in completed if t["pnl"]>0]; losses_l=[t for t in completed if t["pnl"]<0]
            best=max(completed,key=lambda t:t["pnl"]); worst=min(completed,key=lambda t:t["pnl"])
            aw=sum(t["pnl"] for t in wins_l)/len(wins_l) if wins_l else 0
            al=sum(t["pnl"] for t in losses_l)/len(losses_l) if losses_l else 0
            track=f"\n\nBest: {best['symbol']} +${round(best['pnl'],2)} | Worst: {worst['symbol']} -${abs(round(worst['pnl'],2))}\nAvg win: +${round(aw,2)} | Avg loss: -${abs(round(al,2))}"
        return (f"Started $1,000. Now ${round(tv,2)} — {'up' if pnl>=0 else 'down'} ${round(abs(pnl),2)} "
                f"({'+'if pnl>=0 else ''}{round(pnl_pct,1)}%).\n"
                f"Win rate: {wr}% ({bot['wins']}W/{bot['losses']}L). Mode: {mode_label}. Cash: ${round(cash,2)}."
                +track+f"\n\nSelf-coding engine v{vasu_brain.get('selfcode_version',1)}: {vasu_brain.get('selfcode_improvements',0)} modifications.")

    # Holdings
    if any(p in q for p in ["holding","holdings","positions","what do you own","open positions"]):
        if not bot["holdings"]: return f"Nothing open. ${round(cash,2)} cash. {len(bulls)} signals but none clearing {round(conf,1)}% threshold."
        lines=[]
        for sym,h in bot["holdings"].items():
            cur=pm.get(sym,h["buy_price"]); pp=(cur-h["buy_price"])/h["buy_price"]*100
            item=sm.get(sym); agree_tag=" [TRIPLE AI]" if item and item.get("ai_agreement",{}).get("all_agree_bull") else ""
            lines.append(f"{sym}: ${round(h['buy_price'],4)} -> ${round(cur,4)} ({'+'if pp>=0 else ''}{round(pp,1)}%) | {item['signal'] if item else 'N/A'}{agree_tag}")
        return f"{num_h} positions:\n"+"\n".join(lines)+f"\n\nStop: -{stop}% | TP: +{tp}% | Mode: {mode_label}"

    # Market
    if any(p in q for p in ["market","how are stocks","stocks today","market today"]):
        up=[i for i in items if i["change_pct"]>0]; dn=[i for i in items if i["change_pct"]<0]
        strong_s=[i for i in items if i["signal"]=="STRONG BUY"]
        triple_b=len([i for i in items if i.get("ai_agreement",{}).get("all_agree_bull")])
        div_s=len([i for i in items if i.get("divergence")=="bullish"])
        top=[i for i in sorted(items,key=lambda x:x.get("change_pct",0),reverse=True) if i["change_pct"]>0][:3]
        return (f"Regime: {regime} | Score: {vasu_brain.get('market_score',0)}/10\n"
                f"{len(up)} up, {len(dn)} down. {len(strong_s)} STRONG BUY vs {len([i for i in items if i['signal'] in ('SELL','STRONG SELL')])} sells.\n"
                +("Movers: "+", ".join([i["symbol"]+" +"+str(round(i["change_pct"],1))+"%" for i in top])+"\n" if top else "")
                +(f"Triple AI unanimous bulls: {triple_b}\n" if triple_b else "")
                +(f"Bullish RSI divergence signals: {div_s}\n" if div_s else "")
                +f"Market: {market_hours_label()}"
                +("\nAUTO chose: "+vasu_brain.get("auto_target","balanced").upper() if vasu_brain.get("active_mode")=="auto" else ""))

    # Watchlist
    if any(p in q for p in ["watchlist","watch list","next buy","about to buy"]):
        if not watchlist: return f"Nothing close to {round(conf,1)}% threshold yet."
        lines=[f"{w['symbol']}: {w['signal']} | Score:{w.get('composite','?')} | AI:{w['ai_conf']}% | Gap:{w['gap']}%"
               +(f" | SURGE" if w.get("surge") else "")
               +(f" | DIVERGENCE" if w.get("divergence")=="bullish" else "")
               +(f" | {w['pattern'].upper()}" if w.get("pattern") and w["pattern"]!="none" else "")
               for w in watchlist[:6]]
        return "Watching:\n"+"\n".join(lines)+f"\n\nFirst to clear {round(conf,1)}% I'm in."

    # Who is VASU
    if any(p in q for p in ["who are you","what are you","what is vasu","introduce yourself"]):
        return (f"I'm VASU AI v3.1 — self-learning, self-coding autonomous trading bot.\n\n"
                f"TRIPLE AI ENGINE:\n  Random Forest ({round(blend.get('rf',0)*100)}%) | Gradient Boost ({round(blend.get('gb',0)*100)}%) | Extra Trees ({round(blend.get('et',0)*100)}%)\n\n"
                f"v3.1 UPGRADES:\n  + Stochastic RSI | Williams %R | CCI (3 new indicators)\n"
                f"  + RSI divergence detection | Candlestick patterns\n"
                f"  + News sentiment scoring | Market hours awareness\n"
                f"  + Non-blocking startup | /health endpoint\n\n"
                f"10 WALL STREET FRAMEWORKS: Goldman, Morgan Stanley, Bridgewater, JPMorgan, BlackRock, Citadel, Harvard, Bain, Renaissance, McKinsey\n\n"
                f"SELF-CODING: Every 20 trades I analyze my own performance and modify my parameters.\n"
                f"  Version: {vasu_brain.get('selfcode_version',1)} | Modifications: {vasu_brain.get('selfcode_improvements',0)}\n\n"
                f"Currently: ${round(tv,2)} | {mode_label} | {wr}% win rate | {market_hours_label()}")

    # Fallback
    top=sorted([i for i in items if i["signal"] in ("BUY","STRONG BUY")],key=lambda x:x.get("composite",0),reverse=True)
    top_note=f" Best: {top[0]['symbol']} score:{top[0].get('composite','?')}." if top else " No standout setups."
    return (f"${round(tv,2)} in {mode_label} | {'+'if pnl>=0 else ''}{round(pnl_pct,1)}%. "
            f"{num_h} positions. Regime: {regime}.{top_note}\n\n"
            "Commands: buy NVDA | sell AAPL | sell everything | market | watchlist\n"
            "  holdings | performance | brain report | self code | who are you\n"
            "  switch to [sniper/aggressive/balanced/conservative/momentum/contrarian/bear/auto]")

# ==============================================================================
#  REFRESH LOOP
# ==============================================================================
def refresh_data():
    global latest_data, last_updated
    cats=[("tech",TECH),("finance",FINANCE),("healthcare",HEALTHCARE),("consumer",CONSUMER),("energy",ENERGY),("mypicks",MY_PICKS)]
    while True:
        print(f"[SCAN] Scanning {len(ALL_STOCKS)} symbols...")
        new_data={}
        for key,syms in cats:
            results=[]
            for s in syms:
                r=analyze(s)
                if r: results.append(r)
            new_data[key]=results
        with bot_lock:
            latest_data.update(new_data)
            update_auto_mode()
            bot_trade()
        generate_analyst_advice(); update_market_mood(); update_watchlist()
        generate_vasu_daily(); refresh_dream_trades()
        tv=get_bot_value()
        if tv>vasu_brain.get("peak_value",tv): vasu_brain["peak_value"]=tv
        save_equity(tv)
        last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[SCAN] Done: ${tv} | {get_mode_label()} | {market_hours_label()}")
        time.sleep(REFRESH_SECONDS)

# ==============================================================================
#  HTML DASHBOARD
# ==============================================================================
def ai_color(c): return "#00ff88" if c and c>=70 else "#a8ff78" if c and c>=50 else "#ff9f43" if c and c>=35 else "#ff4757" if c else "#374151"
def grade_color(g): return {"A":"#00ff88","B":"#a8ff78","C":"#f59e0b","D":"#ff9f43","F":"#ff4757"}.get(g,"#6b7280")
def comp_color(c): return "#00ff88" if c and c>=75 else "#a8ff78" if c and c>=60 else "#ff9f43" if c and c>=45 else "#ff4757" if c else "#374151"

def mk_card(item):
    chg=item["change_pct"]; ac="#00ff88" if chg>=0 else "#ff4757"; sym=item["symbol"]
    h=json.dumps(item["history"]); comp=item.get("composite"); grade=item.get("fundamental_grade","")
    conf=item.get("ai_conf"); surge=' <span style="color:#facc15;font-size:.6rem">▲VOL</span>' if item.get("volume_surge") else ""
    div_tag=' <span style="color:#818cf8;font-size:.6rem">DIV</span>' if item.get("divergence")=="bullish" else ""
    pat=item.get("pattern","none"); pat_tag=(' <span style="color:#34d399;font-size:.6rem">'+pat[:3].upper()+'</span>' if pat and pat!="none" else "")
    ag=item.get("ai_agreement",{})
    triple_tag=(' <span style="font-size:.6rem;background:#00ff8822;color:#00ff88;padding:1px 4px;border-radius:3px">3X✓</span>' if ag.get("all_agree_bull")
                else ' <span style="font-size:.6rem;background:#ff475722;color:#ff4757;padding:1px 4px;border-radius:3px">3X✗</span>' if ag.get("all_agree_bear") else "")
    if comp is not None:
        cc=comp_color(comp); cw=str(comp)
        grade_chip=f'<span style="font-size:.6rem;background:{grade_color(grade)}22;color:{grade_color(grade)};padding:1px 5px;border-radius:3px;font-weight:700;margin-left:3px">{grade}</span>' if grade else ""
        ci=(f'<div class="ai-bar-wrap"><div class="ai-bar" style="width:{cw}%;background:{cc}"></div></div>'
            f'<div class="ai-label" style="color:{cc}">Score:{cw}/100{surge}{grade_chip}</div>')
    elif conf is not None:
        ci=(f'<div class="ai-bar-wrap"><div class="ai-bar" style="width:{conf}%;background:{ai_color(conf)}"></div></div>'
            f'<div class="ai-label" style="color:{ai_color(conf)}">AI {conf}%{surge}</div>')
    else: ci='<div class="ai-label" style="color:#374151">Training...</div>'
    fund_p=[]
    pe=item.get("pe_ratio"); pm_v=item.get("profit_margin"); rg=item.get("revenue_growth")
    if pe: fund_p.append(f'<span style="color:#6b7280">PE:{pe}</span>')
    if pm_v is not None: fund_p.append(f'<span style="color:{"#00ff88" if pm_v>15 else "#ff9f43" if pm_v<0 else "#9ca3af"}">M:{pm_v}%</span>')
    if rg is not None: fund_p.append(f'<span style="color:{"#00ff88" if rg>15 else "#ff4757" if rg<-10 else "#9ca3af"}">G:{rg}%</span>')
    fund_h=f'<div style="font-size:.6rem;margin-top:3px;display:flex;gap:4px;flex-wrap:wrap">{"".join(fund_p)}</div>' if fund_p else ""
    news_h="".join([f'<div class="news-item"><a href="{n["link"]}" target="_blank">{n["title"][:50]}...</a></div>' for n in item.get("news",[])])
    earn_h=f'<div class="earn">Earnings: {item["earnings_date"]}</div>' if item.get("earnings_date") else ""
    ss=vasu_brain.get("stock_scores",{}); sc=ss.get(sym,0)
    badge=f' <span style="font-size:.6rem;color:{"#00ff88" if sc>0 else "#ff4757"}">{("+" if sc>0 else "")}{sc}</span>' if sc!=0 else ""
    # New indicators
    stoch=item.get("stoch_k"); will=item.get("williams_r")
    ind_h=(f'<div style="font-size:.58rem;color:#6b7280;margin-top:2px">'
           +(f'K:{round(stoch,0)}' if stoch else "")+(" " if stoch and will else "")+(f'W%:{round(will,0)}' if will else "")+"</div>") if (stoch or will) else ""
    return (f'<div class="card"><div onclick="showChart(\'{sym}\',{h},\'{item["color"]}\')" style="cursor:pointer">'
            f'<div class="sym">{sym}{badge}{triple_tag}{div_tag}{pat_tag}</div>'
            f'<div class="price">${item["price"]}</div>'
            f'<div style="color:{ac}">{"+" if chg>=0 else ""}{round(abs(chg),2)}%</div>'
            f'<div class="rsi">RSI:{item["rsi"]}</div>'
            f'<div class="sig" style="background:{item["color"]}">{item["signal"]}</div>'
            +ci+ind_h+fund_h+earn_h+
            f'</div><div class="news-wrap">{news_h}</div></div>')

def mk_section(title, items, tid):
    cards="".join([mk_card(i) for i in items])
    return (f'<div class="sec"><div class="sh"><h2>{title}</h2>'
            f'<label class="tog"><input type="checkbox" checked onchange="tog(\'{tid}\')"><span class="sl"></span></label></div>'
            f'<div class="cards" id="{tid}">{cards}</div></div>')

def cs(label, sid, content):
    return (f'<div class="cs"><div class="ch" onclick="toggleSection(\'{sid}\')">'
            f'<span>{label}</span><span id="arr_{sid}">▼</span></div>'
            f'<div id="sec_{sid}" style="display:none;padding-top:12px">{content}</div></div>')

def alloc_json():
    buys=[i for i in all_items_flat() if i["signal"] in ("BUY","STRONG BUY")]
    if not buys: return "[]"
    total=sum(max(i.get("composite",i["score"]),1) for i in buys)
    return json.dumps([{"symbol":i["symbol"],"signal":i["signal"],"price":i["price"],
                        "pct":round(max(i.get("composite",1),1)/total*100,1),"color":i["color"],
                        "history":i["history"],"ai_conf":i.get("ai_conf")} for i in buys])

def build_html():
    items=all_items_flat(); sm_={i["symbol"]:i for i in items}
    tv=get_bot_value(); pnl=tv-bot["start_value"]; pnl_pct=pnl/bot["start_value"]*100
    pc="#00ff88" if pnl>=0 else "#ff4757"
    ts_=bot["wins"]+bot["losses"]; wr=round(bot["wins"]/ts_*100,1) if ts_>0 else 0
    mode_cfg=get_active_mode_config(); mode_label=get_mode_label(); mc=mode_cfg["color"]
    regime=vasu_brain.get("market_regime","unknown")
    blend=vasu_brain.get("ai_blend",{"rf":0.30,"gb":0.35,"et":0.35})
    sc_v=vasu_brain.get("selfcode_version",1); sc_m=vasu_brain.get("selfcode_improvements",0)
    pause_badge=('<span style="background:#ff4757;color:#fff;padding:2px 8px;border-radius:8px;font-size:.7rem;margin-left:8px">PAUSED</span>' if vasu_brain.get("trading_paused") else "")
    hours_label=market_hours_label()
    hours_color={"Market Open":"#00ff88","Pre-Market":"#f59e0b","After Hours":"#6b7280","Weekend - Market Closed":"#374151"}.get(hours_label,"#6b7280")
    mood_c=market_mood["color"]
    # Mode panel
    mode_ab=(f'<div class="mode-card" data-mode="auto" style="background:#0a0e1a;border:1px solid {"#8b5cf6" if vasu_brain.get("active_mode")=="auto" else "#374151"};border-radius:12px;padding:14px 18px;margin-bottom:16px;cursor:pointer">'
             f'<div style="display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:12px">'
             f'<div style="display:flex;align-items:center;gap:14px">'
             f'<div style="background:#8b5cf6;color:#fff;padding:7px 18px;border-radius:8px;font-weight:700;font-size:.9rem">AUTO</div>'
             f'<div><div style="font-size:.85rem;font-weight:700;color:#e5e7eb">{"AUTO ACTIVE — reads market every scan" if vasu_brain.get("active_mode")=="auto" else "AUTO OFF — manual mode"}</div>'
             f'<div style="font-size:.72rem;color:#9ca3af;margin-top:3px">{"Chose "+vasu_brain.get("auto_target","balanced").upper() if vasu_brain.get("active_mode")=="auto" else "Click AUTO to re-enable"}</div>'
             f'<div style="font-size:.68rem;color:#a78bfa;margin-top:4px">SelfCode v{sc_v} | {sc_m} mods | AI: RF{round(blend.get("rf",0)*100)}% GB{round(blend.get("gb",0)*100)}% ET{round(blend.get("et",0)*100)}%</div>'
             f'</div></div>'
             f'<div style="display:flex;gap:10px;flex-wrap:wrap;font-size:.75rem;align-items:center">'
             f'<span>Regime: <b style="color:{"#00ff88" if "bull" in regime else "#ff4757" if "bear" in regime else "#aaaaaa"}">{regime.upper()}</b></span>'
             f'<span>Score: <b style="color:{"#00ff88" if vasu_brain.get("market_score",0)>0 else "#ff4757"}">{vasu_brain.get("market_score",0)}/10</b></span>'
             f'<span>Heat: <b style="color:{"#ff4757" if _portfolio_heat()>=0.8 else "#ff9f43" if _portfolio_heat()>=0.6 else "#00ff88"}">{round(_portfolio_heat()*100)}%</b></span>'
             f'<span style="color:{hours_color}">⬤ {hours_label}</span>'
             f'</div></div></div>')
    mode_cards=""
    for name,[m] in [(n,[v]) for n,v in MODE_CONFIGS.items() if n!="auto"]:
        is_a=vasu_brain.get("active_mode")==name
        mode_cards+=(f'<div class="mode-card" data-mode="{name}" style="flex:1;min-width:130px;max-width:200px;background:{""+m["color"]+"18" if is_a else "#0a0e1a"};border:2px solid {""+m["color"] if is_a else "#1f2937"};border-radius:12px;padding:12px;cursor:pointer">'
                     +(f'<div style="font-size:.6rem;background:{m["color"]};color:#000;padding:1px 7px;border-radius:3px;display:inline-block;margin-bottom:5px;font-weight:700">ACTIVE</div><br>' if is_a else "")
                     +f'<div style="color:{m["color"]};font-weight:700;font-size:.88rem;margin-bottom:4px">{m["label"]}</div>'
                     +f'<div style="color:#9ca3af;font-size:.7rem;line-height:1.4;margin-bottom:6px">{m["desc"]}</div>'
                     +f'<div style="color:#6b7280;font-size:.64rem">Buy:{m["buy_confidence"]}% | Stop:-{m["stop_loss_pct"]}% | TP:+{m["take_profit_pct"]}%</div></div>')
    # Holdings table
    hrows=""
    for sym,h in bot["holdings"].items():
        it=sm_.get(sym); cur=it["price"] if it else h["buy_price"]
        pp=((cur-h["buy_price"])/h["buy_price"])*100; hc="#00ff88" if pp>=0 else "#ff4757"
        ag_tag=' <span style="font-size:.65rem;color:#00ff88">[3X✓]</span>' if (it and it.get("ai_agreement",{}).get("all_agree_bull")) else ""
        hrows+=(f'<tr><td><b>{sym}</b>{ag_tag}</td><td>${round(h["buy_price"],4)}</td><td>${round(cur,4)}</td>'
                f'<td style="color:{hc}">{"+"if pp>=0 else ""}{round(pp,1)}%</td></tr>')
    # Watchlist table
    wrows=""
    for w in watchlist:
        div_t=' <span style="color:#818cf8;font-size:.65rem">DIV</span>' if w.get("divergence")=="bullish" else ""
        pat_t=(' <span style="color:#34d399;font-size:.65rem">'+w["pattern"][:3].upper()+'</span>' if w.get("pattern") and w["pattern"]!="none" else "")
        wrows+=(f'<tr><td><b>{w["symbol"]}</b>{div_t}{pat_t}</td>'
                f'<td><span style="background:{w["color"]};color:#000;padding:2px 8px;border-radius:10px;font-size:.7rem;font-weight:700">{w["signal"]}</span></td>'
                f'<td>${round(w["price"],4)}</td><td style="color:#a78bfa">{w["ai_conf"]}%</td>'
                f'<td style="color:#ff9f43">Need {round(vasu_brain["buy_confidence"])}%</td>'
                f'<td>Score:{w.get("composite","?")}</td></tr>')
    # Trade log
    tc_map={"BUY":"#00ff88","SELL":"#ff9f43","STOP LOSS":"#ff4757","TAKE PROFIT":"#a78bfa","TRAILING STOP":"#f59e0b","MANUAL SELL":"#6b7280"}
    trows=""
    for t in reversed(bot["trades"][-25:]):
        tc3=tc_map.get(t["type"],"#fff"); pc3="#00ff88" if (t.get("pnl") or 0)>0 else "#ff4757" if (t.get("pnl") or 0)<0 else "#6b7280"
        ps="$"+str(t["pnl"]) if t.get("pnl") is not None else "Open"
        mtag=f' <span style="font-size:.62rem;color:#8b5cf6">[{t.get("mode","?").upper()}]</span>' if t.get("mode") else ""
        trows+=(f'<tr><td style="color:{tc3};font-weight:700">{t["type"]}</td>'
                f'<td><b>{t["symbol"]}</b>{mtag}</td><td>${round(t["price"],4)}</td>'
                f'<td>${round(t["amount"],2)}</td><td style="color:{pc3}">{ps}</td>'
                f'<td style="color:#6b7280;font-size:.7rem">{t.get("reason","")[:40]}</td>'
                f'<td style="color:#6b7280;font-size:.7rem">{t["time"]}</td></tr>')
    # Self-coding panel
    sc_log=_load_json(SELFCODE_FILE, lambda:{"version":1,"changes":[],"total_self_mods":0,"last_run":""})
    sc_rows="".join([f'<div style="padding:6px 0;border-bottom:1px solid #1f2937;font-size:.78rem;color:#9ca3af">'
                     f'<span style="color:#a78bfa">[{c.get("timestamp","?")}]</span> '
                     f'<span style="color:#00ff88">{c.get("type","?")}</span>: '
                     f'{c.get("insight",c.get("reason",""))[:80]}</div>' for c in reversed(sc_log.get("changes",[])[-8:])]) or \
            '<div style="color:#6b7280;font-size:.85rem">No self-modifications yet. Need 20 closed trades.</div>'
    blend_h=(f'<div style="display:flex;gap:16px;margin-bottom:16px;flex-wrap:wrap">'
             +f'<div style="flex:1;min-width:100px"><div style="color:#6b7280;font-size:.7rem;margin-bottom:4px">RF ({round(blend.get("rf",0)*100)}%)</div><div style="background:#1f2937;border-radius:4px;height:8px"><div style="width:{round(blend.get("rf",0)*100)}%;background:#60a5fa;height:8px;border-radius:4px"></div></div></div>'
             +f'<div style="flex:1;min-width:100px"><div style="color:#6b7280;font-size:.7rem;margin-bottom:4px">GB ({round(blend.get("gb",0)*100)}%)</div><div style="background:#1f2937;border-radius:4px;height:8px"><div style="width:{round(blend.get("gb",0)*100)}%;background:#34d399;height:8px;border-radius:4px"></div></div></div>'
             +f'<div style="flex:1;min-width:100px"><div style="color:#6b7280;font-size:.7rem;margin-bottom:4px">ET ({round(blend.get("et",0)*100)}%)</div><div style="background:#1f2937;border-radius:4px;height:8px"><div style="width:{round(blend.get("et",0)*100)}%;background:#f97316;height:8px;border-radius:4px"></div></div></div>'
             +f'</div>')
    sc_html=(f'<div style="font-size:.82rem;color:#9ca3af;margin-bottom:10px">Every 20 trades VASU analyzes his own performance and modifies his parameters. v{sc_v} | {sc_m} total mods | v3.1: +divergence, +patterns</div>'
             +blend_h+sc_rows)
    lessons_h=("".join([f'<div style="font-size:.82rem;color:#9ca3af;padding:6px 0;border-bottom:1px solid #1f2937">[{l["time"]}] {l["lesson"]}</div>' for l in vasu_brain["lessons"][-10:]])
               or "<p style='color:#6b7280;font-size:.85rem'>No lessons yet.</p>")
    dream_h="".join([f'<div style="padding:8px 0;border-bottom:1px solid #1f2937"><b style="color:#facc15">{d["symbol"]}</b> <span style="color:#6b7280">Score:</span> <b style="color:#00ff88">{d["dream_score"]}</b> | Grade:{d["grade"]} | Composite:{d["composite"]} | ${d["price"]}<br><span style="font-size:.73rem;color:#9ca3af">{" + ".join(d["reasons"][:3])}</span></div>' for d in dream_trades_cache[:5]]) or \
              '<div style="color:#6b7280;font-size:.85rem">No dream trades found. Need Grade A/B + STRONG BUY + high composite.</div>'
    # Analyst council
    analysts=[("claude","1. Claude","Risk + Self-Code","#f59e0b"),("arya","2. Arya","Technical + v3.1","#3b82f6"),
              ("magnus","3. Magnus","Fundamentals","#10b981"),("zeus","4. Zeus","Momentum","#8b5cf6"),
              ("bear_analyst","5. Bear","Devil's Advocate","#ef4444")]
    arows=""
    for aid,name,role,col in analysts:
        alist=analyst_advice.get(aid,[])
        ih="".join([f'<div style="padding:10px 0;border-bottom:1px solid #1f2937;font-size:.85rem;color:#d1d5db;line-height:1.6">{a}</div>' for a in alist]) or \
           '<div style="color:#6b7280;font-size:.85rem;padding:8px 0">Analyzing...</div>'
        arows+=(f'<div style="border:1px solid #1f2937;border-radius:10px;margin-bottom:8px;overflow:hidden">'
                f'<div onclick="toggleAnalyst(\'{aid}\')" style="display:flex;justify-content:space-between;align-items:center;padding:14px 18px;cursor:pointer;background:#0a0e1a">'
                f'<div style="display:flex;align-items:center;gap:10px"><span style="font-weight:700;color:{col}">{name}</span>'
                f'<span style="font-size:.7rem;color:#6b7280;background:#111827;padding:2px 8px;border-radius:10px">{role}</span></div>'
                f'<span id="arr_{aid}" style="color:#6b7280">▼</span></div>'
                f'<div id="body_{aid}" style="display:none;padding:14px 18px;background:#111827;border-top:1px solid #1f2937">{ih}</div></div>')
    eq=load_equity(); eq_labels=json.dumps([e["time"] for e in eq[-40:]]); eq_values=json.dumps([e["value"] for e in eq[-40:]])
    daily_html=(f'<div style="background:#0a0e1a;border-left:3px solid {mc};padding:12px 16px;border-radius:8px;margin-bottom:16px;font-size:.83rem;color:#9ca3af;line-height:1.6">'
                f'<b style="color:{mc}">VASU v3.1 [{mode_label}]:</b> {vasu_daily}</div>') if vasu_daily else ""
    chat_msgs="".join([('<div class="msg ')+("user-msg" if m["role"]=="user" else "vasu-msg")+('">')+("<b>You:</b> " if m["role"]=="user" else "<b>VASU:</b> ")+m["text"]+"</div>" for m in chat_history[-20:]])
    css=("*{margin:0;padding:0;box-sizing:border-box}body{background:#0a0e1a;color:#fff;font-family:'Segoe UI',sans-serif;padding:16px;max-width:1600px;margin:0 auto}"
         "header{display:flex;justify-content:space-between;align-items:center;padding:16px 24px;background:#111827;border-radius:16px;margin-bottom:18px;border:1px solid #1f2937;flex-wrap:wrap;gap:10px}"
         "h1{font-size:1.4rem;color:#00ff88}.upd{color:#6b7280;font-size:.78rem;text-align:right}"
         ".sec{margin-bottom:28px}.sh{display:flex;align-items:center;gap:14px;margin-bottom:12px}.sh h2{font-size:1.1rem;color:#e5e7eb}"
         ".cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(175px,1fr));gap:12px}"
         ".card{background:#111827;border:1px solid #1f2937;border-radius:12px;padding:14px;transition:transform .18s,box-shadow .18s}.card:hover{transform:translateY(-3px);box-shadow:0 6px 24px rgba(0,0,0,.5)}"
         ".sym{font-size:.95rem;font-weight:700;margin-bottom:3px}.price{color:#9ca3af;margin-bottom:3px;font-size:.88rem}.rsi{font-size:.7rem;color:#6b7280;margin:3px 0 6px}"
         ".sig{display:inline-block;padding:3px 10px;border-radius:20px;font-size:.68rem;font-weight:700;color:#000;margin-bottom:6px}"
         ".ai-bar-wrap{background:#1f2937;border-radius:4px;height:5px;width:100%;margin:4px 0 2px}.ai-bar{height:5px;border-radius:4px}.ai-label{font-size:.66rem;margin-bottom:4px}"
         ".earn{font-size:.66rem;color:#9ca3af;margin-top:2px}.news-wrap{margin-top:5px;border-top:1px solid #1f2937;padding-top:5px}"
         ".news-item{font-size:.66rem;color:#6b7280;margin-bottom:2px;line-height:1.3}.news-item a{color:#6b7280;text-decoration:none}.news-item a:hover{color:#00ff88}"
         ".tog{position:relative;display:inline-block;width:44px;height:22px}.tog input{opacity:0;width:0;height:0}.sl{position:absolute;cursor:pointer;inset:0;background:#374151;border-radius:22px;transition:.3s}.sl:before{position:absolute;content:'';height:16px;width:16px;left:3px;bottom:3px;background:white;border-radius:50%;transition:.3s}input:checked+.sl{background:#00ff88}input:checked+.sl:before{transform:translateX(22px)}"
         ".alloc{background:#111827;border:1px solid #1f2937;border-radius:16px;padding:22px;margin-bottom:26px}.alloc h2{font-size:1.1rem;margin-bottom:12px;color:#e5e7eb}"
         "table{width:100%;border-collapse:collapse}th{text-align:left;color:#6b7280;font-size:.72rem;padding:7px 10px;border-bottom:1px solid #1f2937;text-transform:uppercase}td{padding:9px 10px;border-bottom:1px solid #1f2937;font-size:.83rem}"
         ".stat-box{background:#0a0e1a;padding:10px 14px;border-radius:10px;min-width:82px}.stat-label{color:#6b7280;font-size:.64rem;margin-bottom:2px;text-transform:uppercase}.stat-val{font-size:.92rem;font-weight:700}"
         ".cs{border:1px solid #1f2937;border-radius:10px;margin-bottom:7px;overflow:hidden}.ch{display:flex;justify-content:space-between;align-items:center;padding:11px 14px;cursor:pointer;background:#0a0e1a;font-size:.85rem;font-weight:600;color:#e5e7eb}.ch:hover{background:#1f2937}"
         ".mode-card{transition:transform .15s,box-shadow .15s}.mode-card:hover{transform:translateY(-2px);box-shadow:0 4px 16px rgba(0,0,0,.4)}"
         "#chat-box{position:fixed;bottom:20px;right:20px;width:340px;background:#111827;border:1px solid #3b82f6;border-radius:16px;z-index:999;box-shadow:0 8px 32px rgba(0,0,0,.6)}"
         "#chat-header{padding:11px 16px;border-bottom:1px solid #1f2937;cursor:pointer;display:flex;justify-content:space-between;align-items:center}#chat-header h3{font-size:.88rem;color:#00ff88;margin:0}"
         "#chat-body{height:280px;overflow-y:auto;padding:10px;display:none}#chat-input-row{padding:8px;border-top:1px solid #1f2937;display:none;flex-direction:column;gap:6px}"
         ".msg{padding:7px 9px;border-radius:8px;margin-bottom:5px;font-size:.8rem;line-height:1.6;white-space:pre-wrap}.user-msg{background:#1f2937;text-align:right;color:#9ca3af}.vasu-msg{background:rgba(0,255,136,0.06);color:#d1d5db;border-left:2px solid #00ff88}"
         "#chat-input{background:#0a0e1a;border:1px solid #374151;color:#fff;padding:9px 12px;border-radius:8px;font-size:.8rem;outline:none;resize:none;height:52px;width:100%;font-family:'Segoe UI',sans-serif}#chat-input:focus{border-color:#3b82f6}"
         "#chat-send{background:#3b82f6;color:#fff;border:none;padding:7px 14px;border-radius:8px;cursor:pointer;font-weight:700;font-size:.8rem;align-self:flex-end}#chat-send:hover{background:#2563eb}#chat-send:disabled{background:#374151}"
         ".irow{display:flex;gap:10px;align-items:center;margin-bottom:18px;flex-wrap:wrap}.irow input{background:#1f2937;border:1px solid #374151;color:#fff;padding:10px 14px;border-radius:10px;font-size:.95rem;width:180px;outline:none}.irow input:focus{border-color:#00ff88}.irow button{background:#00ff88;color:#000;border:none;padding:10px 20px;border-radius:10px;font-size:.9rem;font-weight:700;cursor:pointer}"
         "#cm{display:none;position:fixed;inset:0;background:rgba(0,0,0,.85);z-index:1000;align-items:center;justify-content:center}#cm.open{display:flex}.cb{background:#111827;border:1px solid #1f2937;border-radius:20px;padding:24px;width:660px;max-width:95vw}.cb h3{font-size:1.2rem;margin-bottom:16px;color:#f9fafb}.xbtn{float:right;background:#1f2937;border:none;color:#fff;padding:4px 10px;border-radius:8px;cursor:pointer}"
         "@media(max-width:640px){.cards{grid-template-columns:repeat(2,1fr)}#chat-box{width:calc(100vw - 24px);right:12px}}")

    return (f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>VASU AI v3.1</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>{css}</style></head><body>
<header>
  <div><h1>VASU AI v3.1</h1><div style="color:#6b7280;font-size:.72rem">Triple AI + Self-Coding + RSI Divergence + Candlestick Patterns</div></div>
  <div class="upd">Updated: {last_updated}<br><span style="color:{mc};font-size:.7rem">{mode_label}</span> <span style="color:{hours_color};font-size:.7rem">| {hours_label}</span></div>
</header>
<div style="background:{mood_c}22;border:1px solid {mood_c};border-radius:12px;padding:12px 20px;margin-bottom:20px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px">
  <div style="font-size:.95rem;font-weight:700;color:{mood_c}">{market_mood["label"]}</div>
  <div style="font-size:.75rem;color:#6b7280">Score: {vasu_brain.get("market_score",0)} | Regime: {regime} | Scan: {last_updated}</div>
</div>
<div class="alloc" style="border-color:#8b5cf6;margin-bottom:22px">
  <h2>VASU Mode Control</h2>
  {mode_ab}
  <div style="font-size:.78rem;color:#6b7280;font-weight:600;margin-bottom:10px">MANUAL OVERRIDE:</div>
  <div style="display:flex;gap:8px;flex-wrap:wrap">{mode_cards}</div>
</div>
<div class="alloc" style="border-color:{mc}">
  <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;margin-bottom:14px">
    <div><h2>VASU AI v3.1 <span style="font-size:.73rem;background:{mc}33;color:{mc};padding:3px 10px;border-radius:20px;margin-left:8px;border:1px solid {mc}44">{mode_label}</span>{pause_badge}</h2>
    <div style="color:#6b7280;font-size:.72rem;margin-top:3px">{vasu_brain["total_adjustments"]} self-adjustments | SelfCode v{sc_v} | {"ON" if bot["enabled"] else "OFF"}</div></div>
    <label class="tog"><input type="checkbox" {"checked" if bot["enabled"] else ""} onchange="toggleBot()"><span class="sl"></span></label>
  </div>
  <div style="display:flex;gap:10px;margin-bottom:14px;flex-wrap:wrap">
    <div class="stat-box"><div class="stat-label">VALUE</div><div class="stat-val">${round(tv,2)}</div></div>
    <div class="stat-box"><div class="stat-label">P&L</div><div class="stat-val" style="color:{pc}">{"+"if pnl>=0 else ""}${round(pnl,2)}</div></div>
    <div class="stat-box"><div class="stat-label">CASH</div><div class="stat-val">${round(bot["cash"],2)}</div></div>
    <div class="stat-box"><div class="stat-label">WIN RATE</div><div class="stat-val" style="color:#00ff88">{wr}%</div></div>
    <div class="stat-box"><div class="stat-label">THRESHOLD</div><div class="stat-val" style="color:#a78bfa">{round(vasu_brain["buy_confidence"],1)}%</div></div>
    <div class="stat-box"><div class="stat-label">SELF-CODE</div><div class="stat-val" style="color:#f97316">v{sc_v}</div></div>
  </div>
  {daily_html}
  {cs(f"Dream Trades ({len(dream_trades_cache)})", "dreams", dream_h)}
  {cs("Self-Coding Engine (v3.1)", "selfcode", sc_html)}
  {cs(f"Open Positions ({len(bot['holdings'])})", "holdings", '<table><thead><tr><th>Symbol</th><th>Buy</th><th>Now</th><th>P&L</th></tr></thead><tbody>'+hrows+'</tbody></table>' if hrows else "<p style='color:#6b7280'>No positions.</p>")}
  {cs(f"Watchlist ({len(watchlist)})", "watchlist", '<table><thead><tr><th>Symbol</th><th>Signal</th><th>Price</th><th>AI</th><th>Need</th><th>Score</th></tr></thead><tbody>'+wrows+'</tbody></table>' if wrows else "<p style='color:#6b7280'>Nothing close yet.</p>")}
  {cs("Equity Curve", "equity", '<canvas id="equityChart" height="100"></canvas>')}
  {cs(f"VASU Lessons ({len(vasu_brain['lessons'])})", "lessons", lessons_h)}
  {cs(f"Trade Log ({len(bot['trades'])})", "tradelog", '<div style="overflow-x:auto"><table><thead><tr><th>Type</th><th>Symbol</th><th>Price</th><th>Amount</th><th>P&L</th><th>Reason</th><th>Time</th></tr></thead><tbody>'+trows+'</tbody></table></div>' if trows else "<p style='color:#6b7280'>No trades yet.</p>")}
</div>
<div style="margin-bottom:32px">
  <button onclick="toggleCouncil()" style="width:100%;background:#111827;border:1px solid #374151;color:#e5e7eb;padding:14px 22px;border-radius:12px;font-size:.95rem;cursor:pointer;font-weight:600;display:flex;justify-content:space-between;align-items:center">
    <span>Analyst Council (5 analysts)</span><span id="council_arr" style="color:#6b7280">▼</span>
  </button>
  <div id="council_box" style="display:none;background:#111827;border:1px solid #374151;border-top:none;border-radius:0 0 12px 12px;padding:14px">
    {arows}
  </div>
</div>
<div class="alloc"><h2>Portfolio Allocator</h2>
  <div class="irow"><input type="text" id="amt" placeholder="Budget e.g. 5000" onkeydown="if(event.key==='Enter')calc()"/><button onclick="calc()">Calculate</button></div>
  <div id="ar"></div>
</div>
{mk_section("Tech", latest_data["tech"], "tech")}
{mk_section("Finance", latest_data["finance"], "finance")}
{mk_section("Healthcare", latest_data["healthcare"], "healthcare")}
{mk_section("Consumer", latest_data["consumer"], "consumer")}
{mk_section("Energy", latest_data["energy"], "energy")}
{mk_section("My Picks", latest_data["mypicks"], "mypicks")}
<div id="cm"><div class="cb"><button class="xbtn" onclick="closeChart()">✕ Close</button><h3 id="ct"></h3><canvas id="pc" height="260"></canvas></div></div>
<div id="chat-box">
  <div id="chat-header" onclick="toggleChat()"><h3>💬 Talk to VASU v3.1</h3><span id="chat-arr" style="color:#6b7280;font-size:.8rem">▲</span></div>
  <div id="chat-body">{chat_msgs}</div>
  <div id="chat-input-row"><textarea id="chat-input" placeholder="Try: NVDA / market / watchlist / switch to sniper / self code / divergence" onkeydown="if(event.key==='Enter'&amp;&amp;!event.shiftKey){{event.preventDefault();sendMsg();}}"></textarea><button id="chat-send" onclick="sendMsg()">Send</button></div>
</div>
""" + """<script>
window.__EQ_LABELS=""" + eq_labels + """;window.__EQ_VALUES=""" + eq_values + """;window.__AD_DATA=""" + alloc_json() + """;
var chart=null,chatOpen=false,eqChart=null;
function toggleBot(){fetch('/toggle_bot').then(r=>r.json()).then(()=>setTimeout(()=>location.reload(),500));}
function toggleCouncil(){var b=document.getElementById('council_box'),a=document.getElementById('council_arr'),o=b.style.display==='none';b.style.display=o?'block':'none';a.textContent=o?'▲':'▼';}
function toggleAnalyst(id){var b=document.getElementById('body_'+id),a=document.getElementById('arr_'+id),o=b.style.display==='none';b.style.display=o?'block':'none';a.textContent=o?'▲':'▼';}
function toggleSection(id){var s=document.getElementById('sec_'+id),a=document.getElementById('arr_'+id);if(!s||!a)return;var o=s.style.display==='none';s.style.display=o?'block':'none';a.textContent=o?'▲':'▼';if(o&&id==='equity')initEquity();}
function initEquity(){if(eqChart)return;var c=document.getElementById('equityChart'),ev=window.__EQ_VALUES||[],el=window.__EQ_LABELS||[];if(!c||ev.length<2)return;eqChart=new Chart(c.getContext('2d'),{type:'line',data:{labels:el,datasets:[{data:ev,borderColor:'#3b82f6',backgroundColor:'rgba(59,130,246,0.08)',borderWidth:2,pointRadius:1,fill:true,tension:0.3}]},options:{responsive:true,plugins:{legend:{display:false}},scales:{x:{ticks:{color:'#6b7280',maxTicksLimit:6},grid:{color:'#1f2937'}},y:{ticks:{color:'#6b7280'},grid:{color:'#1f2937'}}}}});}
function switchMode(mode){var t=document.createElement('div');t.textContent='Switching to '+mode.toUpperCase()+'...';t.style.cssText='position:fixed;top:20px;right:20px;background:#8b5cf6;color:#fff;padding:10px 20px;border-radius:10px;font-weight:700;z-index:9999;font-size:.9rem';document.body.appendChild(t);fetch('/chat?q='+encodeURIComponent('switch to '+mode)).then(r=>r.json()).then(()=>{t.textContent='Done \u2713';setTimeout(()=>{document.body.removeChild(t);location.reload();},1200);}).catch(()=>document.body.removeChild(t));}
document.addEventListener('DOMContentLoaded',()=>{document.addEventListener('click',e=>{var c=e.target.closest('.mode-card');if(c){var m=c.getAttribute('data-mode');if(m)switchMode(m);}});});
function toggleChat(){chatOpen=!chatOpen;document.getElementById('chat-body').style.display=chatOpen?'block':'none';document.getElementById('chat-input-row').style.display=chatOpen?'flex':'none';document.getElementById('chat-arr').textContent=chatOpen?'▼':'▲';if(chatOpen){var b=document.getElementById('chat-body');b.scrollTop=b.scrollHeight;}}
function sendMsg(){var inp=document.getElementById('chat-input'),btn=document.getElementById('chat-send'),msg=inp.value.trim();if(!msg)return;var body=document.getElementById('chat-body');body.innerHTML+='<div class="msg user-msg"><b>You:</b> '+msg+'</div>';var tid='t'+Date.now();body.innerHTML+='<div class="msg" id="'+tid+'" style="color:#6b7280;font-style:italic">VASU thinking...</div>';inp.value='';btn.disabled=true;body.scrollTop=body.scrollHeight;fetch('/chat?q='+encodeURIComponent(msg)).then(r=>r.json()).then(d=>{var el=document.getElementById(tid);if(el)el.remove();body.innerHTML+='<div class="msg vasu-msg"><b>VASU:</b> '+d.response+'</div>';body.scrollTop=body.scrollHeight;btn.disabled=false;}).catch(()=>{var el=document.getElementById(tid);if(el)el.remove();btn.disabled=false;});}
function calc(){var AD=window.__AD_DATA||[],amt=parseFloat(document.getElementById('amt').value.trim());if(!amt||amt<=0){document.getElementById('ar').innerHTML='<p style="color:#6b7280">Enter a valid amount.</p>';return;}if(!AD.length){document.getElementById('ar').innerHTML='<p style="color:#6b7280">No BUY signals yet.</p>';return;}var rows='';AD.forEach(i=>{var d=((i.pct/100)*amt).toFixed(2),sh=(d/i.price).toFixed(4),ai=i.ai_conf!=null?'<span style="color:#00ff88">'+i.ai_conf+'%</span>':'';rows+='<tr><td><b>'+i.symbol+'</b></td><td><span style="background:'+i.color+';color:#000;padding:3px 10px;border-radius:20px;font-size:.7rem;font-weight:700">'+i.signal+'</span></td><td style="color:#00ff88;font-weight:700">$'+d+'</td><td>'+i.pct+'%</td><td style="color:#9ca3af">'+sh+' shares</td><td>'+ai+'</td></tr>';});document.getElementById('ar').innerHTML='<p style="margin-bottom:10px;color:#9ca3af">Splitting $'+amt.toLocaleString()+' across '+AD.length+' signals:</p><table><thead><tr><th>Symbol</th><th>Signal</th><th>Amount</th><th>%</th><th>Shares</th><th>AI</th></tr></thead><tbody>'+rows+'</tbody></table>';}
function showChart(sym,hist,color){if(!hist||!hist.length){alert('No chart data yet');return;}document.getElementById('ct').textContent=sym+' \u2014 30 Day History';document.getElementById('cm').classList.add('open');if(chart)chart.destroy();chart=new Chart(document.getElementById('pc').getContext('2d'),{type:'line',data:{labels:hist.map((_,i)=>'Day '+(i+1)),datasets:[{label:sym,data:hist,borderColor:color,backgroundColor:color+'18',borderWidth:2,pointRadius:2,fill:true,tension:0.3}]},options:{responsive:true,plugins:{legend:{display:false}},scales:{x:{ticks:{color:'#6b7280'},grid:{color:'#1f2937'}},y:{ticks:{color:'#6b7280'},grid:{color:'#1f2937'}}}}});}
function closeChart(){document.getElementById('cm').classList.remove('open');if(chart){chart.destroy();chart=null;}}
function tog(id){document.getElementById(id).classList.toggle('hidden');}
function refreshStatus(){fetch('/data').then(r=>r.json()).then(d=>{var el=document.querySelector('.upd');if(el)el.innerHTML='Updated: '+d.updated+'<br><span style="color:#a78bfa;font-size:.7rem">'+d.mode+'</span> <span style="font-size:.7rem">| '+d.hours+'</span>';}).catch(()=>{});setTimeout(refreshStatus,70000);}
setTimeout(refreshStatus,70000);
</script></body></html>""")

# ==============================================================================
#  HTTP SERVER
# ==============================================================================
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path); path = parsed.path
        if path == "/health":
            # Render health check — always responds immediately
            self._j({"status":"ok","version":"3.1","trained":training_done.is_set()})
        elif path == "/toggle_bot":
            with bot_lock: bot["enabled"] = not bot["enabled"]; save_bot()
            self._j({"enabled":bot["enabled"]})
        elif path == "/chat":
            q = parse_qs(parsed.query).get("q",[""])[0].strip()
            if not q: resp = "Ask me anything. Try: market, watchlist, NVDA, self code, switch to sniper"
            elif not training_done.is_set(): resp = "Still training AI models... ask me again in a minute."
            else: resp = vasu_respond(q)
            chat_history.append({"role":"user","text":q})
            chat_history.append({"role":"vasu","text":resp})
            if len(chat_history)>100: chat_history[:]=chat_history[-80:]
            self._j({"response":resp})
        elif path == "/data":
            self._j({"updated":last_updated,"value":get_bot_value(),"mode":get_mode_label(),
                     "regime":vasu_brain.get("market_regime","unknown"),"hours":market_hours_label()})
        else:
            html = build_html().encode("utf-8")
            self.send_response(200); self.send_header("Content-Type","text/html; charset=utf-8")
            self.send_header("Content-Length",str(len(html))); self.end_headers(); self.wfile.write(html)
    def _j(self, data):
        b = json.dumps(data).encode()
        self.send_response(200); self.send_header("Content-Type","application/json")
        self.send_header("Content-Length",str(len(b))); self.end_headers(); self.wfile.write(b)
    def log_message(self, *a): pass  # silent logging

# ==============================================================================
#  MAIN
# ==============================================================================
if __name__ == "__main__":
    print("")
    print("  ╔═══════════════════════════════════════════════╗")
    print("  ║  VASU AI v3.1 — Render/GitHub Edition         ║")
    print("  ║  Triple AI + Self-Coding + RSI Divergence     ║")
    print("  ╚═══════════════════════════════════════════════╝")
    print(f"  Host: {HOST}:{PORT}")
    print("")

    os.makedirs(DATA_DIR, exist_ok=True)

    # Start server IMMEDIATELY so Render health check passes
    server = HTTPServer((HOST, PORT), Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    print(f"[BOOT] Server live at http://{HOST}:{PORT}")
    print("[BOOT] Training Triple AI in background (server already responding)...")

    # Train AI models in background — doesn't block server startup
    threading.Thread(target=train_all, daemon=True).start()

    # Wait for first training pass before starting market scanner
    threading.Thread(target=lambda: (training_done.wait(), refresh_data()), daemon=True).start()

    print("[LIVE] VASU AI v3.1 is running.")
    print("[LIVE] /health → Render health check")
    print("[LIVE] / → Dashboard")
    print("")

    try:
        heartbeat = 0
        while True:
            time.sleep(60); heartbeat += 1
            if heartbeat % 60 == 0:
                tv = get_bot_value(); pnl = tv-bot["start_value"]
                blend = vasu_brain.get("ai_blend",{"rf":0.30,"gb":0.35,"et":0.35})
                print(f"[HEARTBEAT] ${tv} | {'+'if pnl>=0 else ''}${round(pnl,2)} | {get_mode_label()} | "
                      f"SelfCode v{vasu_brain.get('selfcode_version',1)} | "
                      f"RF{round(blend.get('rf',0)*100)}% GB{round(blend.get('gb',0)*100)}% ET{round(blend.get('et',0)*100)}%")
    except KeyboardInterrupt:
        print("\n[STOP] Saving and shutting down...")
        save_brain(); save_bot()
        tv=get_bot_value(); pnl=tv-bot["start_value"]
        ts_=bot["wins"]+bot["losses"]; wr=round(bot["wins"]/ts_*100,1) if ts_>0 else 0
        print(f"  Final: ${tv} | {'+'if pnl>=0 else ''}${round(abs(pnl),2)} | Win rate: {wr}%")
        print("  VASU saved. See you next time. 🙏")
