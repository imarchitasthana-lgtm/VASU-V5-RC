# ==============================================================================
#  VASU AI v3.0 - Configuration
#  Edit this file to change your watchlist and settings
# ==============================================================================

BOT_ENABLED      = True   # Auto-trading on at startup
REFRESH_SECONDS  = 60     # Market scan interval (seconds)

# ── Watchlists ────────────────────────────────────────────────────────────────
TECH        = ["AAPL","MSFT","NVDA","GOOGL","META","AMD","NFLX","UBER","CRM","ORCL","ADBE"]
FINANCE     = ["JPM","BAC","V","MA","GS","WFC","C","AXP","MS"]
HEALTHCARE  = ["JNJ","PFE","UNH","ABBV","MRK","CVS","AMGN"]
CONSUMER    = ["AMZN","TSLA","WMT","TGT","NKE","MCD","SBUX","HD","COST"]
ENERGY      = ["XOM","CVX","OXY","SLB","COP"]
MY_PICKS    = ["SOFI","SCHD","CLOV"]

ALL_STOCKS  = list(dict.fromkeys(TECH + FINANCE + HEALTHCARE + CONSUMER + ENERGY + MY_PICKS))
