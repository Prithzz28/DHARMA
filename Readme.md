# Web3 Transaction Firewall 🛡️

A wallet RPC firewall for Ethereum. Users add its **RPC URL to their wallet** (MetaMask or any EVM wallet). Every transaction they send reaches the firewall before the network and gets two independent checks, while all other wallet calls pass straight through:

1. **Counterparty checks** — *who does this transaction pay or give permissions to?* Payments and approvals to any of 1,763 wallets labelled fraudulent in BCCC-DeFiFraudTrans-2025, and unlimited / approve-all grants to plain wallets (a common drainer pattern), are **blocked**.
2. **DeFiGuard** — a model of *how fraud-labelled wallets transact*. Its training labels are per wallet (every transaction sent by a flagged wallet counts as fraud), so it scores how closely a transaction resembles ones **sent by** fraud-labelled wallets. That is evidence about the sender, not proof that the transaction harms the person sending it, so by default the model **warns** and never blocks on its own.

DeFiGuard uses transaction-local features only (no address history, block height or identity). With a receipt (mined or simulated) it also analyses the receipt's token-flow ego-graph with a GATv2 graph neural network.

---

## 🌟 Key Features

- **Wallet RPC Firewall:** `POST /rpc` is a standard Ethereum JSON-RPC endpoint. `eth_sendRawTransaction` / `eth_sendTransaction` are decoded (signed transactions of every type, sender recovered from the signature) and checked; blocked ones are refused with a JSON-RPC error the wallet shows, and nothing is broadcast. Fails closed. See [Wallet RPC firewall](#-wallet-rpc-firewall).
- **Counterparty Checks:** recipient, called contract, approved spender / operator and token recipient (decoded from `approve`, `increaseAllowance`, `setApprovalForAll`, `transfer`, `transferFrom`, `safeTransferFrom`) are checked against the fraud-labelled wallets; `eth_getCode` on the node tells a plain wallet from a contract.
- **DeFiGuard v2 Model (sender-pattern signal):** two modes — *pending* (pre-execution features → MLP) and *receipt* (GATv2 receipt ego-graph + tabular encoder with cross-attention fusion). Model, feature and graph code are byte-identical copies of the research repository's, so inference matches training exactly. Each mode's decision threshold was selected on address-disjoint validation data — see [Model evaluation](#-model-evaluation).
- **Rate Limited:** Scoring endpoints are limited per client IP (`RATE_LIMIT_PER_MINUTE`, default 60) and answer `429` with `Retry-After` when exceeded.
- **Fast Inference:** In-memory scoring API (`:8001`); well under 100 ms per scan on CPU.
- **Server-side Feature Extraction:** `POST /scan` turns user-level transaction parameters into a JSON-RPC pending transaction and extracts its features on the server with the training code (`backend/models/features.py`).
- **Validated Thresholds:** the model flags a transaction when its probability reaches the per-mode threshold selected on validation data; `FRAUD_THRESHOLD` can override it. In the wallet firewall a flag is a warning unless `RPC_BLOCK_THRESHOLD` is lowered below 1.
- **Explainable Verdicts:** every verdict names its cause — the counterparty rule that fired, or the transaction signals behind the model score (e.g. fresh sender, self-transfer, approval calls), optionally written up by an LLM.
- **Interactive UI Dashboard:** Modern React + Vite + Tailwind CSS frontend (`:5173`) featuring wallet setup (network details, *Add to MetaMask*, code snippets), a live feed of firewall verdicts, an interactive transaction scanner, attack simulation presets, risk analytics, and real-time engine health indicators.
- **1-Click Demo Launcher:** Automated shell script (`demo.sh`) to launch backend and frontend services with a single command.

---

## 🏗️ Architecture

```
 Wallet (MetaMask, ethers, web3.py)              Web dashboard (:5173)
        │ JSON-RPC                                 │ /scan (model only), /rpc/info, /rpc/recent
        ▼                                          ▼
 ┌───────────────────────────────────────────────────────────────────────┐
 │                FastAPI inference engine (:8001)  POST /rpc              │
 ├───────────────────────────────────────────────────────────────────────┤
 │  eth_sendRawTransaction / eth_sendTransaction       every other method │
 │  decode signed tx, recover sender                   ─────────────────┐ │
 │        │                                                             │ │
 │        ├─► Counterparty checks ── flagged wallets (1,763)            │ │
 │        │     recipient · spender · operator · token recipient         │ │
 │        │     unlimited / approve-all to a plain wallet (eth_getCode)  │ │
 │        │                                                             │ │
 │        └─► DeFiGuard (pending mode) ── sender-pattern score          │ │
 │                                                                      │ │
 │  verdict = stricter of the two                                       │ │
 │    blocked ─► JSON-RPC error -32003, not broadcast                    │ │
 │    warned / allowed ─► forward ─────────────────────────────────────┤ │
 └──────────────────────────────────────────────────────────────────────┼─┘
                                                                        ▼
                                                     Ethereum node (UPSTREAM_RPC_URL)
```

---

## 📁 Project Structure

```
.
├── backend/
│   ├── models/                        # DeFiGuard model (.joblib, versioned) + model code
│   │   ├── defi_guard_model.joblib
│   │   ├── defiguard.py               # DeFiGuard network (GATv2 + cross-attention fusion)
│   │   └── predictor.py               # DeFiGuardPredictor + ego-graph builder
│   ├── feature_extractor/             # Payload normalisation (tx / receipt) → models/features.py
│   │   ├── extractor.py               # Raw RPC & tx payload parser
│   ├── inference_engine/              # FastAPI inference & explainability service
│   │   ├── api/                       # API routes (main.py, rpc.py = wallet RPC, schemas.py, dependencies.py)
│   │   ├── core/                      # Settings (config.py) & rate limiting (rate_limit.py)
│   │   ├── services/                  # Prediction, explainer, rpc_firewall.py, counterparty.py, raw_transaction.py
│   │   └── resources/                 # flagged_addresses.json (counterparty denylist, generated)
│   ├── scripts/                       # evaluate_model.py, generate_defiguard_golden.py, build_scanner_presets.py,
│   │                                  # build_flagged_addresses.py, demo_rpc_firewall.py
│   ├── tests/                         # Backend test suite (+ fixtures/defiguard_golden.json)
│   ├── conftest.py                    # Puts backend/ on sys.path for pytest
│   ├── requirements.txt               # Runtime dependencies
│   └── requirements-dev.txt           # + pytest, pandas, gdown
├── frontend/                          # React + Vite + Tailwind CSS dashboard
│   ├── src/
│   │   ├── components/                # UI components (AIExplanation, Analytics, Hero, etc.)
│   │   ├── services/api.js            # API client (POST /scan, GET /health) + form validation
│   │   └── App.jsx                    # Root application layout
│   └── package.json                   # Frontend dependencies and scripts
├── demo.sh                            # 1-Click demo launcher for backend & frontend
├── .env.example                       # Template for the single root .env (backend + frontend)
└── Readme.md                          # Project documentation
```

---

## ⚡ Quick Start

### Prerequisites
- **Python** 3.10+
- **Node.js** 20+ & **npm** 11+ (npm 10 cannot install the test dependencies; use `npm i -g npm@11` or prefix commands with `npx npm@11`)
- **Git**

---

### Option 1: 1-Click Demo Launcher (Recommended)

Start all services (Inference Engine on `:8001`, Vite Frontend on `:5173`):

```bash
bash demo.sh
```
The DeFiGuard model (`backend/models/defi_guard_model.joblib`, 1.5 MB) is versioned with the code; `demo.sh` stops with instructions if it is missing.

**Helpful Demo Commands:**
```bash
bash demo.sh --curls    # Print ready-to-use attack simulation cURL commands
bash demo.sh --stop     # Gracefully stop all background services
```

Access the Web Dashboard at: **`http://localhost:5173`**

---

### Option 2: Manual Step-by-Step Setup

#### 1. Configure Environment Variables
```bash
cp .env.example .env   # one file in the repository root, read by both backend and frontend
```

#### 2. Start the Backend Inference Engine
```bash
python -m venv .venv && source .venv/bin/activate   # from the repository root
cd backend
pip install -r requirements-dev.txt                 # runtime deps + test/eval tools
uvicorn inference_engine.api.main:app --host 0.0.0.0 --port 8001 --reload
```
*Interactive Swagger documentation available at: `http://localhost:8001/docs`*

#### 3. Start the Frontend Dashboard (in a new terminal)
```bash
cd frontend
npm ci        # installs the exact versions in package-lock.json (npm ≥ 11)
npm run dev
```
*Open `http://localhost:5173` in your browser.*

---

## 🦊 Wallet RPC Firewall

Users connect by adding a network to their wallet: the dashboard's **Connect Your Wallet** section shows the values and has an *Add to MetaMask* button.

| Field | Value (defaults) |
|---|---|
| RPC URL | `<API origin>/rpc`, e.g. `https://prithvi28-firewall-api.onrender.com/rpc` or `http://localhost:5173/rpc` locally |
| Chain ID | `11155111` (Sepolia) |
| Currency symbol | `SepoliaETH` |
| Block explorer | `https://sepolia.etherscan.io` |

Each transaction submission gets two independent checks, and the stricter result wins:

| Check | Fires when | Effect |
|---|---|---|
| **Counterparty: flagged wallet** | the recipient, called contract, approved spender / operator or token recipient is one of the 1,763 wallets labelled fraudulent in BCCC-DeFiFraudTrans-2025 | **block** |
| **Counterparty: approval to a plain wallet** | unlimited `approve` / `increaseAllowance`, or `setApprovalForAll(…, true)`, to an address with no contract code | **block** |
| **Counterparty: other approvals** | the same grants to a contract (e.g. a DEX router), or a bounded approval to a plain wallet | warn |
| **DeFiGuard** (pending mode — no receipt exists before broadcast) | probability ≥ the model's pending threshold (0.5948) | warn (block only if `RPC_BLOCK_THRESHOLD` < 1) |

| Verdict | Effect |
|---|---|
| **allowed** | Forwarded to `UPSTREAM_RPC_URL` |
| **warned** | Forwarded, and listed with its findings in the live feed |
| **blocked** | JSON-RPC error `-32003` naming the cause; not broadcast |
| **overridden** | A blocked transaction (same sender, recipient, value, calldata) sent again within `RPC_OVERRIDE_WINDOW_SECONDS`: forwarded |

**Why the model does not block.** DeFiGuard's labels are wallet-level: every transaction sent by a fraud-labelled wallet is "fraud". It therefore learns what *fraudsters' own transactions* look like, while a firewall user is usually the potential victim, harmed by the other side of the transaction. The counterparty checks look at that other side; the model's score is shown as a sender-pattern warning.

Every other method (`eth_chainId`, `eth_getBalance`, `eth_call`, `eth_estimateGas`, …) is forwarded unchanged, batches included. If the model is not loaded or scoring fails, transactions are refused rather than forwarded unscored. Transaction submissions count against `RATE_LIMIT_PER_MINUTE`; other calls against the higher `RPC_RATE_LIMIT_PER_MINUTE`, so normal wallet polling is not throttled. If an `eth_getCode` lookup fails, the approval rules fall back to a warning rather than guessing. `GET /rpc/info` returns the network details, thresholds and denylist size; `GET /rpc/recent` the last 50 verdicts (addresses masked, with counterparty findings and model signals) and totals.

```bash
# Through the firewall like any node
curl -s http://localhost:8001/rpc -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"eth_blockNumber","params":[]}'

# Demo: one signed Sepolia transaction per path (warned / blocked ×2 / warned).
# Uses a throwaway unfunded key, so forwarded ones are rejected by the node.
cd backend && PYTHONPATH=. python scripts/demo_rpc_firewall.py --rpc http://localhost:8001/rpc
```

The denylist is generated from the research dataset (it refuses to write if any address is also labelled legitimate):
```bash
cd backend && python scripts/build_flagged_addresses.py --dataset-dir /path/to/Dataset \
  --out inference_engine/resources/flagged_addresses.json
```

**Limitations — read before demoing.**
- **Testnet only.** At its threshold the pending model flags 16% of legitimate transactions on held-out data; at a realistic 0.1% fraud rate about 99% of its flags would be false alarms. That is why it only warns.
- **Fresh wallets get warnings.** Fraud-labelled wallets tend to be young, so the model leans on the sender's nonce: a plain transfer scores ≈ 0.85 at nonce 0 and stays above the warning threshold for roughly the first 50 transactions.
- **The denylist is narrow.** It holds only the dataset's 1,763 fraud-labelled wallets (Ethereum mainnet addresses, wallet-level labels) and is not updated; new scam addresses are caught only by the approval rules.
- **Off-chain signatures are invisible.** Permit and marketplace-order signatures (`eth_signTypedData`) are produced inside the wallet and never reach an RPC endpoint; drains that use them cannot be seen here.
- **Trust.** All wallet traffic, including addresses and balances, passes through this server; the upstream URL (which may hold a provider key) is never exposed to clients.
- **Free hosting.** While the Render free API is asleep, the wallet's calls fail until it wakes (see [Deployment](#-deployment-free)).

---

## 🧪 API Examples (cURL)

### Scan a pending transaction from user-level parameters — `POST /scan`
The dashboard uses this endpoint. Features are extracted server-side and scored in
*pending* mode; `explain` (default `true`) adds a risk explanation, rule-based
unless `use_llm` is `true` and `GROQ_API_KEY` is set. `nonce` and `data`
(hex calldata) are optional.
```bash
curl -s -X POST http://localhost:8001/scan \
  -H "Content-Type: application/json" \
  -d '{
    "from": "0x3b2a1f88e4c5d6a7890bcdef2345678901bcde22",
    "to":   "0x8c5d7e12f3a4b5c6d7890abcdef3456789012cd3",
    "value_eth": 0.45,
    "gas_price_gwei": 22,
    "gas_limit": 21000,
    "nonce": 57,
    "data": "",
    "use_llm": false
  }' | python3 -m json.tool
```
Response: `{"features": {...31 features...}, "prediction": {"is_fraud", "fraud_probability", "threshold", "mode", "model", "exec_time_ms"}, "explanation": {"risk_level", "reasons", "recommendation", "is_fallback", "exec_time_ms"}}`.

### Score a raw JSON-RPC transaction — `POST /predict`
Send the transaction object (`eth_getTransactionByHash` / `eth_sendTransaction`
params). Add its `receipt` (`eth_getTransactionReceipt`, or the result of
simulating the transaction) to score it in *receipt* mode with the full graph model.
```bash
curl -s -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"tx": {"from": "0x3b2a1f88e4c5d6a7890bcdef2345678901bcde22",
              "to": "0x8c5d7e12f3a4b5c6d7890abcdef3456789012cd3",
              "value": "0xde0b6b3a7640000", "gas": "0x5208",
              "gasPrice": "0x51f4d5c00", "nonce": "0x0", "input": "0x"},
       "receipt": {"gasUsed": "0x5208", "status": "0x1", "logs": []}}' | python3 -m json.tool
```

### Other endpoints
| Endpoint | Purpose |
|---|---|
| `POST /predict/explain` (alias `/api/transactions/explain`) | `/predict` + explanation; response includes `prediction` |
| `POST /extract` | Raw JSON-RPC payload (pending tx, `{tx, receipt}`, or a stored `{result: [receipt, tx]}` pair) → features + mode |
| `GET /health` | Model status and per-mode thresholds (not rate limited) |
| `GET /stats` | Predictions, fraud rate and average latency since startup (not rate limited) |
| `POST /rpc` | Wallet JSON-RPC endpoint — see [Wallet RPC firewall](#-wallet-rpc-firewall) |
| `GET /rpc/info`, `GET /rpc/recent` | Network details + thresholds; recent verdicts with masked addresses |

Invalid input (negative or non-finite numbers, malformed hex calldata, oversized strings, malformed transaction objects) returns `422` with `detail: [{loc, msg, type}]`. More than `RATE_LIMIT_PER_MINUTE` scoring requests from one client in a minute returns `429` with a `Retry-After` header. `503` means the model is not loaded (see `/health`).

---

## 📊 DeFiGuard v2 Inputs

All definitions live in `backend/models/features.py` (identical to the research repository). Nothing describes an address's history, the block a transaction landed in, or the identity of any address — these were the leakage paths in DeFiGuard v1 (see the research paper's leakage audit).

| Group | Features |
|---|---|
| **Pre-execution (16)** — pending *and* receipt mode | `log_value_eth`, `log_gas_limit`, `log_gas_price_gwei`, `log_nonce`, `is_contract_creation`, `is_self_transfer`, `has_calldata`, `log_calldata_bytes`, `calldata_zero_frac`, function-selector groups (`sel_token_transfer`, `sel_approval`, `sel_transfer_from`, `sel_dex_swap`, `sel_multicall`, `sel_weth`, `sel_unknown`) |
| **Receipt (15)** — receipt mode only | `status_failed`, `log_gas_used`, `gas_used_ratio`, log/emitter/transfer/NFT/approval counts, distinct transfer tokens and parties, `sender_sends_tokens`, `sender_receives_tokens`, `has_mint`, `has_burn`, `created_contract` |
| **Receipt ego-graph** | nodes: TX · account · token/emitter contract, described only by their roles and degrees in this transaction; edges: submit · call · emit · token-transfer hops (+ reverse) |

Selectors and event topics are fixed from the ERC-20/721 and common DEX interfaces — nothing is learned from the labelled data.

---

## 📈 Model Evaluation

The deployed model was trained on fold 0 of the research repository's **address-disjoint** protocol; `backend/scripts/evaluate_model.py` scores 2,000 fraud + 2,000 legitimate transactions from that fold's **held-out addresses** (never used for training or threshold selection) through the API code path:

| Mode | Threshold | Precision | Recall | False-positive rate |
|---|---|---|---|---|
| **pending** (what `/scan` scores) | 0.595 | 0.843 | 0.858 | 0.160 |
| **receipt** (tx + receipt graph) | 0.559 | 0.829 | 0.882 | 0.182 |

These are measured on a 50:50 sample. At realistic fraud base rates precision is far lower (about 0.32 at an 8.65% fraud rate, about 0.005 at 0.1%), so a flag should be treated as a warning to review, not proof of fraud — which is how the wallet firewall uses it. Cross-validated results over all folds are in the research paper.

```bash
cd backend
PYTHONPATH=. python scripts/evaluate_model.py --dataset-dir /path/to/Dataset --n 2000 \
  --json-out ../frontend/src/data/modelEvaluation.json
PYTHONPATH=. python scripts/build_scanner_presets.py --dataset-dir /path/to/Dataset \
  --out ../frontend/src/data/scannerPresets.json
```
`--json-out` regenerates the figures the dashboard shows (Hero, How It Works, Features, Analytics read them via `frontend/src/data/modelMetrics.js`); re-run both scripts after changing the model. The Analytics counters (transactions scanned, blocked, latency) are live values from `GET /stats`.

**Known limitations**
- Rate limiting is in-process: limits are per server process, and `X-Forwarded-For` is trusted only from a loopback peer (the Vite proxy). Behind another reverse proxy, every client shares the proxy's address unless that proxy runs on the same host.
- There is no authentication; do not expose the API publicly without putting it behind an authenticating gateway.
- `/scan` has no receipt, so it uses the pending-mode model; the receipt graph is only used when a receipt (or simulation result) is sent to `/predict`.
- Labels in the training data are address-level (every transaction of a flagged wallet is "fraud"), so the model scores how fraud-wallet-like a transaction is, not whether that individual transaction steals funds.
- The scanner's examples are real held-out transactions (all form fields from the original transaction). Only rows whose verdict matches their label are used as examples; across all sampled candidates the scanner blocked 85% of fraud-labelled and passed 85% of legitimate transactions (recorded in `frontend/src/data/scannerPresets.json`). `tests/test_scanner_presets.py` checks every example's verdict against its label.
- Parity with the research repository's reference predictor, in both modes, is enforced by `tests/test_defiguard.py` (`tests/fixtures/defiguard_golden.json`, regenerated with `scripts/generate_defiguard_golden.py`).

---

## ⚙️ Environment Variables

Configure these in `.env`:

| Variable | Default | Description |
|---|---|---|
| `VITE_API_BASE_URL` | *(blank)* | API origin for the frontend. Leave blank to use the Vite dev/preview proxy |
| `DEFIGUARD_MODEL_PATH` | `models/defi_guard_model.joblib` | DeFiGuard model (relative to `backend/`) |
| `FRAUD_THRESHOLD` | unset | Optional override (0–1, exclusive) of the model's per-mode decision thresholds |
| `RATE_LIMIT_PER_MINUTE` | `60` | Scoring requests per client IP per minute (`0` disables) |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins — restrict in production |
| `LOG_LEVEL` | `INFO` | Python logging verbosity |
| `GROQ_API_KEY` | *(empty)* | Enables LLM-written explanations |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Groq chat model for explanations. Groq retires models; if the scanner reports the model unavailable, pick a current one from [console.groq.com/docs/models](https://console.groq.com/docs/models) |
| `LLM_MAX_CALLS_PER_MINUTE` | `20` | Process-wide cap on Groq calls (`0` disables the LLM) |
| `UPSTREAM_RPC_URL` | `https://ethereum-sepolia-rpc.publicnode.com` | Node the wallet RPC forwards to; may contain a provider key (never shown to clients) |
| `RPC_CHAIN_ID`, `RPC_NETWORK_NAME`, `RPC_CURRENCY_SYMBOL`, `RPC_EXPLORER_URL` | Sepolia values | Network details shown to users; must match the upstream node |
| `RPC_BLOCK_THRESHOLD` | `1.0` | DeFiGuard probability from which `/rpc` refuses a transaction on the model alone; `1` = the model only warns and blocks come from counterparty checks |
| `FLAGGED_ADDRESSES_PATH` | `inference_engine/resources/flagged_addresses.json` | Counterparty denylist (`{"addresses": [...]}`), relative to `backend/` |
| `RPC_OVERRIDE_WINDOW_SECONDS` | `120` | Re-sending a blocked transaction within this window forwards it (`0` disables overrides) |
| `RPC_RATE_LIMIT_PER_MINUTE` | `600` | Non-transaction RPC calls per client IP per minute (`0` disables) |

The `.env` file is read from the repository root (and `backend/.env`) regardless of the working directory; real environment variables override it. Invalid values (e.g. a threshold override outside (0, 1)) stop the backend at startup with a clear error. Settings from the removed Random Forest setup (`MODEL_PATH`, `ENSEMBLE_MODE`, `RF_WEIGHT`, `DG_WEIGHT`) are ignored with a startup warning.

---

## 🧪 Running Tests

Backend (unit, real-model integration, DeFiGuard parity, API, settings, explainer):
```bash
cd backend
PYTHONPATH=. pytest tests/ -v
```
Tests pin their own settings and never call the real Groq API. Real-model tests are skipped when the `.joblib` files are absent.

Frontend (API client + form validation, Node's built-in test runner), lint and build:
```bash
cd frontend
npm test
npm run lint
npm run build
```

Or from the repository root: `npm test` (both suites).

---

## 🚀 Deployment (free)

Both services run on **Render's free plan**, defined in [`render.yaml`](render.yaml):

| Service | What it is | Free-plan behaviour |
|---|---|---|
| `prithvi28-firewall-api` | FastAPI inference engine, built from the root `Dockerfile` (CPU-only PyTorch) | Sleeps after 15 min without traffic; the next request waits several minutes while it restarts |
| `prithvi28-firewall` | React dashboard as a static site | Always on |

**Deploy:** push the branch to GitHub, then in the Render dashboard choose **New → Blueprint**, select this repository and apply it. When prompted, paste `GROQ_API_KEY` (optional — enables AI explanations). Every push to the branch redeploys both services.

If Render gives the API a different URL than `https://prithvi28-firewall-api.onrender.com` (it adds a suffix when a name is taken), update `VITE_API_BASE_URL` in `render.yaml` or on the static site's Environment page and redeploy the site.

**Before a demo:** open the API's `/health` URL about 5 minutes early so the backend is awake — a wallet pointed at a sleeping `/rpc` shows network errors until it is. The same `Dockerfile` also runs unchanged on Azure Container Apps or App Service.

---

## 📜 License

This project is developed for educational and research purposes.
