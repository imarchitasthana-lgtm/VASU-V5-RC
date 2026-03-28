# VASU AI v3.1 — Deployment Instructions
### GitHub + Render (Free Cloud Hosting)

---

## What You Need
- A free [GitHub](https://github.com) account
- A free [Render](https://render.com) account
- Your 4 files: `trading_tool.py`, `config.py`, `requirements.txt`, `render.yaml`

---

## Step 1 — Create a GitHub Repo

1. Go to [github.com](https://github.com) and sign in
2. Click the **+** button (top right) → **New repository**
3. Name it `VASU` (or anything you want)
4. Set it to **Private** (so your config stays yours)
5. Click **Create repository**

---

## Step 2 — Upload Your Files

On the new repo page:

1. Click **uploading an existing file**
2. Drag and drop all 4 files:
   - `trading_tool.py`
   - `config.py`
   - `requirements.txt`
   - `render.yaml`
3. Scroll down, click **Commit changes**

Your repo should now show all 4 files.

---

## Step 3 — Deploy on Render

1. Go to [render.com](https://render.com) and sign in (use your GitHub account to make it easier)
2. Click **New +** → **Web Service**
3. Click **Connect a repository** → select your `VASU` repo
4. Render will auto-detect `render.yaml` and fill in the settings. You should see:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python trading_tool.py`
5. Scroll down and click **Create Web Service**

That's it. Render starts building.

---

## Step 4 — Watch It Deploy

Render shows you a live build log. You'll see something like:

```
==> Installing dependencies...
==> pip install -r requirements.txt
==> Build successful
==> Starting service...
  ╔═══════════════════════════════════════════════╗
  ║  VASU AI v3.1 — Render/GitHub Edition         ║
  ╚═══════════════════════════════════════════════╝
[BOOT] Server live at http://0.0.0.0:10000
[BOOT] Training Triple AI in background...
[LIVE] VASU AI v3.1 is running.
```

**First deploy takes ~3-5 minutes** (installing packages + training AI on 47 stocks).

Once you see "Your service is live", click the URL at the top — it'll look like:
`https://vasu-ai.onrender.com`

---

## Step 5 — Using Your Live Dashboard

Open the URL in any browser. You'll see the full VASU dashboard with:

- Live stock signals updating every 60 seconds
- Mode control panel (click any mode card to switch)
- Chat box (bottom right) — talk to VASU live
- Portfolio tracker, trade log, analyst council
- Dream Trades, Watchlist, Self-Coding panel

**Chat commands to try first:**
```
market
watchlist
NVDA
holdings
switch to sniper
who are you
self code
performance
```

---

## Updating VASU in the Future

When you get a new `trading_tool.py` from me:

1. Go to your GitHub repo
2. Click on `trading_tool.py`
3. Click the **pencil icon** (Edit) or drag-and-drop a new file
4. Commit the change
5. Render **auto-detects the change and redeploys** within ~2 minutes

---

## Free Plan Notes

Render's free tier spins down after 15 minutes of inactivity. When you visit the URL after it's been idle, it takes ~30 seconds to wake up — that's normal. Once awake it runs full speed.

If you want it always-on, upgrade to Render's $7/month Starter plan.

---

## Troubleshooting

**"Exited with status 1" on deploy:**
Make sure all 4 files are in the repo root (not inside a folder).

**Dashboard shows "Initializing..." for a long time:**
Normal — AI is training. Wait 2-3 minutes and refresh.

**"Not enough cash" in chat:**
Starting balance is $1,000. Reset by deleting `bot_data.json` in Render's shell (Dashboard → Shell tab).

**Want to change your watchlist:**
Edit `config.py` in GitHub, commit → Render auto-redeploys.

---

## Your File Summary

| File | Purpose |
|------|---------|
| `trading_tool.py` | The entire bot — 1,977 lines |
| `config.py` | Your watchlist (edit this to add/remove stocks) |
| `requirements.txt` | Python packages Render installs |
| `render.yaml` | Tells Render how to run the app |

---

*VASU AI v3.1 — Triple AI + Self-Coding + RSI Divergence + Candlestick Patterns*
