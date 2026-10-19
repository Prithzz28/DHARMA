# Web3 Transaction Firewall 🛡️

Real-time Ethereum transaction firewall. Evaluates transactions before execution, extracts a 20-dimensional DeFiTransLyzer feature vector, scores fraud risk with the DeFiGuard graph neural network, and flags malicious transactions before they are signed.

---

## 🌟 Key Features

- **DeFiGuard Model:** GATv2 graph encoder + tabular encoder with cross-attention fusion, run with the exact forward pass it was trained with. The block threshold was chosen by measuring it on real labelled transactions — see [Model evaluation](#-model-evaluation).
- **Rate Limited:** Scoring endpoints are limited per client IP (`RATE_LIMIT_PER_MINUTE`, default 60) and answer `429` with `Retry-After` when exceeded.
- **Fast Inference:** In-memory scoring API (`:8001`); well under 100 ms per scan on CPU.
- **Server-side Feature Extraction:** `POST /scan` turns user-level transaction parameters into the 20-feature DeFiTransLyzer vector on the server, using the same definitions as the training data.
- **Configurable Blocking:** Transactions whose fraud probability exceeds `FRAUD_THRESHOLD` (default `0.80`) are reported as BLOCKED.
- **Explainable Security (XAI):** Plain-language breakdown of risk factors (e.g., self-transfers, abnormal gas-to-base-fee ratios, drainer patterns) for wallet users.
- **Interactive UI Dashboard:** Modern React + Vite + Tailwind CSS frontend (`:5173`) featuring an interactive transaction scanner, attack simulation presets, risk analytics, and real-time engine health indicators.
- **1-Click Demo Launcher:** Automated shell script (`demo.sh`) to launch backend and frontend services with a single command.

---

## 🏗️ Architecture

```
 User Request / Web Dashboard / Transaction Payload
                       │
                       │ JSON Payload (From, To, Value, Gas)
                       ▼
 ┌─────────────────────────────────────────────────────────────┐
 │           Interactive Web Dashboard (:5173)                 │
 └─────────────────────────────┬───────────────────────────────┘
                               │
                               │ POST /scan  (or /predict, /predict/explain)
                               ▼
 ┌─────────────────────────────────────────────────────────────┐
 │           FastAPI Inference Engine (:8001)                  │
 ├─────────────────────────────────────────────────────────────┤
 │   Server-side Feature Extraction (20 DeFiTransLyzer feats)  │
 ├─────────────────────────────────────────────────────────────┤
 │   Per-client rate limit (RATE_LIMIT_PER_MINUTE)             │
 ├─────────────────────────────┬───────────────────────────────┤
 │                             ▼                               │
 │            ┌─────────────────────────────────┐              │
 │            │ DeFiGuard (.joblib)             │              │
 │            │ GATv2 graph + tabular encoders  │              │
 │            │ → cross-attention fusion        │              │
 │            └────────────────┬────────────────┘              │
 ├─────────────────────────────────────────────────────────────┤
 │   Explainer Engine (Deterministic Heuristics + Groq API)    │
 └─────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
               ┌───────────────────────────────┐
               │                               │
            [BLOCKED]                       [SAFE]
     (prob > FRAUD_THRESHOLD)      (prob <= FRAUD_THRESHOLD)
               │                               │
               ▼                               ▼
        Threat Detected                 Safe to Sign
        + Feature Breakdown             + Normal Parameters
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
│   ├── feature_extractor/             # DeFiTransLyzer feature extraction pipeline
│   │   ├── extractor.py               # Raw RPC & tx payload parser
│   │   └── metrics.py                 # Feature computation helper utilities
│   ├── inference_engine/              # FastAPI inference & explainability service
│   │   ├── api/                       # API routes (main.py, schemas.py, dependencies.py)
│   │   ├── core/                      # Settings (config.py) & rate limiting (rate_limit.py)
│   │   └── services/                  # DeFiGuard prediction service & risk explainer
│   ├── scripts/                       # evaluate_model.py, generate_defiguard_golden.py
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

## 🧪 API Examples (cURL)

### Scan a transaction from user-level parameters — `POST /scan`
The dashboard uses this endpoint. Features are extracted server-side; `explain`
(default `true`) adds a risk explanation, rule-based unless `use_llm` is `true`
and `GROQ_API_KEY` is set.
```bash
curl -s -X POST http://localhost:8001/scan \
  -H "Content-Type: application/json" \
  -d '{
    "from": "0x3b2a1f88e4c5d6a7890bcdef2345678901bcde22",
    "to":   "0x8c5d7e12f3a4b5c6d7890abcdef3456789012cd3",
    "value_eth": 0.45,
    "gas_price_gwei": 22,
    "gas_limit": 21000,
    "block_number": 19000000,
    "log_count": 0,
    "use_llm": false
  }' | python3 -m json.tool
```
Response: `{"features": {...20 features...}, "prediction": {"is_fraud", "fraud_probability", "threshold", "model", "exec_time_ms"}, "explanation": {"risk_level", "reasons", "recommendation", "is_fallback", "exec_time_ms"}}`.

### Score a pre-extracted feature vector — `POST /predict`
Omitted features default to the values a pending transaction has in the
training data (mostly `0`, `gas_price_ratio = 1`).
```bash
curl -s -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"length_transaction_hash": 66, "length_to": 42, "length_from": 42,
       "block_number": 4228383, "chain_id": 1, "gas_used": 21000,
       "effective_gas_price": 60000000000, "total_gas_cost": 1260000000000000,
       "value": 9000000000000000000}' | python3 -m json.tool
```

### Other endpoints
| Endpoint | Purpose |
|---|---|
| `POST /predict/explain` (alias `/api/transactions/explain`) | `/predict` + explanation; response includes `prediction` |
| `POST /extract` | Raw JSON-RPC transaction / receipt → 20-feature vector |
| `GET /health` | Model status and threshold (not rate limited) |
| `GET /stats` | Predictions, fraud rate and average latency since startup (not rate limited) |

Invalid input (negative or non-finite numbers, oversized strings) returns `422` with `detail: [{loc, msg, type}]`. More than `RATE_LIMIT_PER_MINUTE` scoring requests from one client in a minute returns `429` with a `Retry-After` header. `503` means the model is not loaded (see `/health`).

---

## 📊 20-Feature DeFiTransLyzer Schema

The models consume 20 structural and behavioural metrics. Definitions follow the **DeFiTransLyzer training data** (measured from its CSVs and the scaler statistics stored with DeFiGuard), which differs from the column names in a few places — sending the "textbook" meaning puts inputs far outside what the models were trained on:

| Category | Feature Name | Description |
|---|---|---|
| **Identity & Structure** | `length_transaction_hash` | Character length of the transaction hash string |
| | `length_to` | Length of destination address string |
| | `length_from` | Length of sender address string |
| | `is_same_address` | Binary flag (1.0 if `from == to`, common in drainers) |
| | `index` | Transaction position/index within block |
| **Block Context** | `block_number` | Target block height |
| | `chain_id` | Network chain ID (1 = Mainnet, 11155111 = Sepolia) |
| | `cumulative_gas_used` | Cumulative gas consumed in the block |
| **Gas Dynamics** | `gas_used` | Estimated / actual gas units consumed |
| | `gas_efficiency` | Cumulative block gas ÷ this transaction's gas (0 before the tx is mined) |
| | `effective_gas_price` | Gas price paid per gas unit (wei) |
| | `total_gas_cost` | Total fee: `gas_used × effective_gas_price` |
| | `gas_price_ratio` | Paid ÷ offered gas price (1.0 for pending transactions) |
| | `gas_per_log_event` | Constant 0 in the training data |
| **Value Transfer** | `value` | Native ETH amount transferred (in wei) |
| | `normalized_token_transfer`| Constant 0 in the training data |
| **Event Activity** | `log_count` | Constant 0 in the training data (the count is in `length_log`) |
| | `length_log` | Number of log entries emitted |
| | `event_activity_flag` | Binary indicator (1.0 if logs are emitted) |
| | `log_removed` | Flag indicating chain reorganization log removal |

---

## 📈 Model Evaluation

`backend/scripts/evaluate_model.py` scores 2,000 fraud + 2,000 legitimate transactions sampled from the DeFiTransLyzer CSVs (not included in this repo) using only the 20 features the API receives. The **pending** view zeroes the receipt-only fields (`cumulative_gas_used`, `index`, `gas_efficiency`) — what `/scan` can know before a transaction is mined.

| DeFiGuard threshold (pending view) | Precision | Recall | False-positive rate |
|---|---|---|---|
| 0.50 | 0.740 | 0.991 | 0.347 |
| 0.65 | 0.758 | 0.988 | 0.316 |
| **0.80 (default)** | **0.780** | **0.972** | **0.275** |
| 0.90 | 0.809 | 0.945 | 0.223 |

```bash
cd backend
PYTHONPATH=. python scripts/evaluate_model.py --csv-dir /path/to/CSVs --n 2000 \
  --json-out ../frontend/src/data/modelEvaluation.json
```
Regenerate the scanner examples the same way after model or threshold changes:
```bash
PYTHONPATH=. python scripts/build_scanner_presets.py --csv-dir /path/to/CSVs \
  --out ../frontend/src/data/scannerPresets.json
```
`--json-out` regenerates the figures the dashboard shows (Hero, How It Works, Features, Analytics read them via `frontend/src/data/modelMetrics.js`); re-run it after changing the model or `FRAUD_THRESHOLD`. The Analytics counters (transactions scanned, blocked, latency) are live values from `GET /stats`.

A Random Forest (7 features) used to run alongside DeFiGuard. Measured on the same data it performed at chance level (precision 0.47, recall 0.40, FPR 0.46 at 0.65), and no way of combining it with DeFiGuard improved on DeFiGuard alone at comparable recall (e.g. a weighted blend reached precision 0.90 but recall only 0.39), so it was removed.

**Known limitations**
- Rate limiting is in-process: limits are per server process, and `X-Forwarded-For` is trusted only from a loopback peer (the Vite proxy). Behind another reverse proxy, every client shares the proxy's address unless that proxy runs on the same host.
- There is no authentication; do not expose the API publicly without putting it behind an authenticating gateway.
- DeFiGuard's ego-graph is built from the transaction node only. It was trained on receipt graphs that also contained token-holding nodes from a token database that is not available for pending transactions; with sender/recipient nodes but no token nodes its output collapses to ~0.014 for every input.
- With 20 of its 45 training features, DeFiGuard has a much higher false-positive rate than in its paper setting (≈3% FPR with all 45 features on the same data).
- The scanner's examples are real labelled DeFiTransLyzer transactions (sender, hash, value, gas, block and log count from the dataset; the recipient is a placeholder because the dataset has none), built by `backend/scripts/build_scanner_presets.py`. Only rows whose verdict matches their label are used as examples; across all sampled candidates the scanner blocked 99% of real fraud but passed only 64% of real legitimate transactions (recorded in `frontend/src/data/scannerPresets.json`). `tests/test_scanner_presets.py` checks every example's verdict against its label.
- `DeFiGuard` parity with the training repository's reference predictor is enforced by `tests/test_defiguard.py` (`tests/fixtures/defiguard_golden.json`, regenerated with `scripts/generate_defiguard_golden.py`).

---

## ⚙️ Environment Variables

Configure these in `.env`:

| Variable | Default | Description |
|---|---|---|
| `VITE_API_BASE_URL` | *(blank)* | API origin for the frontend. Leave blank to use the Vite dev/preview proxy |
| `DEFIGUARD_MODEL_PATH` | `models/defi_guard_model.joblib` | DeFiGuard model (relative to `backend/`) |
| `FRAUD_THRESHOLD` | `0.80` | Block when the fraud probability exceeds this value (0–1, exclusive) |
| `RATE_LIMIT_PER_MINUTE` | `60` | Scoring requests per client IP per minute (`0` disables) |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins — restrict in production |
| `LOG_LEVEL` | `INFO` | Python logging verbosity |
| `GROQ_API_KEY` | *(empty)* | Enables LLM-written explanations |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Groq chat model for explanations. Groq retires models; if the scanner reports the model unavailable, pick a current one from [console.groq.com/docs/models](https://console.groq.com/docs/models) |
| `LLM_MAX_CALLS_PER_MINUTE` | `20` | Process-wide cap on Groq calls (`0` disables the LLM) |

The `.env` file is read from the repository root (and `backend/.env`) regardless of the working directory; real environment variables override it. Invalid values (e.g. a threshold outside (0, 1)) stop the backend at startup with a clear error. Settings from the removed Random Forest setup (`MODEL_PATH`, `ENSEMBLE_MODE`, `RF_WEIGHT`, `DG_WEIGHT`) are ignored with a startup warning.

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

## 📜 License

This project is developed for educational and research purposes.
