# Web3 Transaction Firewall 🛡️

Real-time Ethereum transaction firewall. Evaluates transactions before execution, extracts a 20-dimensional feature vector using DeFiTransLyzer heuristics, scores fraud risk via a dual-model machine learning ensemble (Random Forest classifier + DeFiGuard neural network), and identifies malicious transactions before they compromise wallet funds.

---

## 🌟 Key Features

- **Dual-Model ML Ensemble:** Combines an optimized 200-tree Random Forest classifier with the DeFiGuard GNN + tabular fusion neural network for maximum recall and precision.
- **Sub-Millisecond Inference:** Fast scoring API (`:8001`) with in-memory model evaluation (< 2 ms median latency).
- **20-Feature DeFiTransLyzer Vector:** Dynamic extraction of structural, gas dynamics, block context, value transfer, and log event metrics.
- **Fail-Closed Security:** Configurable fraud threshold (default `0.65`) with automatic blocking of phishing drainers, honeypots, and reentrancy exploits.
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
                               │ POST /predict or /predict/explain
                               ▼
 ┌─────────────────────────────────────────────────────────────┐
 │           FastAPI Inference Engine (:8001)                  │
 ├─────────────────────────────────────────────────────────────┤
 │   Feature Extraction & Normalization (20 Features)          │
 ├─────────────────────────────┬───────────────────────────────┤
 │                             ▼                               │
 │   ┌─────────────────────────────┐ ┌───────────────────────┐ │
 │   │ Random Forest Model (.joblib)│ │ DeFiGuard Net (.joblib)│ │
 │   │ 200 Estimators (Tabular)    │ │ (GNN + Tabular Fusion)│ │
 │   └──────────────┬──────────────┘ └───────────┬───────────┘ │
 │                  │                            │             │
 │                  └──────────────┬─────────────┘             │
 │                                 ▼                           │
 │                     Dual-Model Ensemble Scorer              │
 │                     (Strategy: Max Recall / Weighted)       │
 ├─────────────────────────────────────────────────────────────┤
 │   Explainer Engine (Deterministic Heuristics + Groq API)    │
 └─────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
               ┌───────────────────────────────┐
               │                               │
            [BLOCKED]                       [SAFE]
        (Fraud Prob > 0.65)           (Fraud Prob <= 0.65)
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
│   ├── download_model.py              # Automated download of trained RF model from GDrive
│   ├── models/                        # Pre-trained models (.joblib) & DeFiGuard stub modules
│   │   ├── random_forest_fraud_model.joblib
│   │   ├── defi_guard_model.joblib
│   │   ├── defiguard.py
│   │   └── predictor.py
│   ├── feature_extractor/             # DeFiTransLyzer feature extraction pipeline
│   │   ├── extractor.py               # Raw RPC & tx payload parser
│   │   └── metrics.py                 # Feature computation helper utilities
│   ├── inference_engine/              # FastAPI dual-model inference & explainability service
│   │   ├── api/                       # API routes (main.py, schemas.py, dependencies.py)
│   │   ├── core/                      # Configuration & settings management (config.py)
│   │   └── services/                  # Prediction service, ensemble scorer & risk explainer
│   ├── torch_geometric/               # Lightweight local unpickling stubs
│   ├── tests/                         # Backend test suite (extractor, inference, ensemble)
│   ├── conftest.py                    # Pytest configuration & fixtures
│   └── requirements.txt               # Backend Python runtime dependencies
├── frontend/                          # React + Vite + Tailwind CSS dashboard
│   ├── src/
│   │   ├── components/                # UI components (AIExplanation, Analytics, Hero, etc.)
│   │   ├── services/api.js            # API client for inference engine
│   │   └── App.jsx                    # Root application layout
│   └── package.json                   # Frontend dependencies and scripts
├── demo.sh                            # 1-Click demo launcher for backend & frontend
├── .env.example                       # Template for environment variables
└── Readme.md                          # Project documentation
```

---

## ⚡ Quick Start

### Prerequisites
- **Python** 3.10+
- **Node.js** 18+ & **npm** 9+
- **Git**

---

### Option 1: 1-Click Demo Launcher (Recommended)

Start all services (Inference Engine on `:8001`, Vite Frontend on `:5173`):

```bash
# 1. Download pre-trained model (if not already downloaded)
cd backend && python download_model.py && cd ..

# 2. Launch all services
bash demo.sh
```

**Helpful Demo Commands:**
```bash
bash demo.sh --curls    # Print ready-to-use attack simulation cURL commands
bash demo.sh --stop     # Gracefully stop all background services
```

Access the Web Dashboard at: **`http://localhost:5173`**

---

### Option 2: Manual Step-by-Step Setup

#### 1. Download the Trained Model
The pre-trained Random Forest model (~41 MB) is hosted on Google Drive:
```bash
cd backend
python download_model.py
cd ..
```

#### 2. Configure Environment Variables
```bash
cp .env.example .env
```

#### 3. Start the Backend Inference Engine
```bash
cd backend
pip install -r requirements.txt
uvicorn inference_engine.api.main:app --host 0.0.0.0 --port 8001 --reload
```
*Interactive Swagger documentation available at: `http://localhost:8001/docs`*

#### 4. Start the Frontend Dashboard (in a new terminal)
```bash
cd frontend
npm install
npm run dev
```
*Open `http://localhost:5173` in your browser.*

---

## 🧪 Attack Simulations & cURL Demos

You can test the firewall inference engine directly using `curl`:

### 1. 🎣 Phishing Drainer (Self-Transfer + Abnormal Gas) → **BLOCKED**
```bash
curl -s -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "length_transaction_hash": 66.0,
    "length_to": 42.0,
    "length_from": 42.0,
    "is_same_address": 1.0,
    "index": 1.0,
    "block_number": 1650000.0,
    "chain_id": 1.0,
    "cumulative_gas_used": 848000.0,
    "gas_used": 848000.0,
    "gas_efficiency": 1.0,
    "effective_gas_price": 204000000000.0,
    "total_gas_cost": 172992000000000000.0,
    "gas_price_ratio": 10.2,
    "gas_per_log_event": 848000.0,
    "value": 0.0,
    "normalized_token_transfer": 0.0,
    "log_count": 0.0,
    "log_removed": 0.0,
    "length_log": 0.0,
    "event_activity_flag": 0.0
  }' | python3 -m json.tool
```

### 2. 🍯 Honeypot Scam (High Value + Inflated Gas Price) → **BLOCKED**
```bash
curl -s -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "length_transaction_hash": 66.0,
    "length_to": 42.0,
    "length_from": 42.0,
    "is_same_address": 0.0,
    "index": 5.0,
    "block_number": 1680319.0,
    "chain_id": 1.0,
    "cumulative_gas_used": 1200000.0,
    "gas_used": 1020000.0,
    "gas_efficiency": 0.85,
    "effective_gas_price": 50000000000.0,
    "total_gas_cost": 51000000000000000.0,
    "gas_price_ratio": 2.5,
    "gas_per_log_event": 1020000.0,
    "value": 10000000000000000000.0,
    "normalized_token_transfer": 0.01,
    "log_count": 0.0,
    "log_removed": 0.0,
    "length_log": 0.0,
    "event_activity_flag": 0.0
  }' | python3 -m json.tool
```

### 3. 🔄 Reentrancy Exploit (Excessive Gas Allocation) → **BLOCKED**
```bash
curl -s -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "length_transaction_hash": 66.0,
    "length_to": 42.0,
    "length_from": 42.0,
    "is_same_address": 0.0,
    "index": 12.0,
    "block_number": 1727488.0,
    "chain_id": 1.0,
    "cumulative_gas_used": 2490000.0,
    "gas_used": 2116500.0,
    "gas_efficiency": 0.85,
    "effective_gas_price": 100000000000.0,
    "total_gas_cost": 211650000000000000.0,
    "gas_price_ratio": 5.0,
    "gas_per_log_event": 2116500.0,
    "value": 2000000000000000000.0,
    "normalized_token_transfer": 0.002,
    "log_count": 0.0,
    "log_removed": 0.0,
    "length_log": 0.0,
    "event_activity_flag": 0.0
  }' | python3 -m json.tool
```

### 4. ✅ Legitimate Transfer (Normal Parameters) → **ALLOWED**
```bash
curl -s -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "length_transaction_hash": 66.0,
    "length_to": 42.0,
    "length_from": 42.0,
    "is_same_address": 0.0,
    "index": 2.0,
    "block_number": 100000.0,
    "chain_id": 1.0,
    "cumulative_gas_used": 52500.0,
    "gas_used": 21000.0,
    "gas_efficiency": 1.0,
    "effective_gas_price": 22000000000.0,
    "total_gas_cost": 462000000000000.0,
    "gas_price_ratio": 1.1,
    "gas_per_log_event": 21000.0,
    "value": 450000000000000000.0,
    "normalized_token_transfer": 0.00045,
    "log_count": 0.0,
    "log_removed": 0.0,
    "length_log": 0.0,
    "event_activity_flag": 0.0
  }' | python3 -m json.tool
```

---

## 📊 20-Feature DeFiTransLyzer Schema

The ensemble evaluates transactions against 20 normalized structural and behavioral metrics:

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
| | `gas_efficiency` | Ratio of gas consumed to gas limit |
| | `effective_gas_price` | Gas price paid per gas unit (wei) |
| | `total_gas_cost` | Total fee: `gas_used × effective_gas_price` |
| | `gas_price_ratio` | Ratio of effective gas price to base fee |
| | `gas_per_log_event` | Gas consumption per emitted event log |
| **Value Transfer** | `value` | Native ETH amount transferred (in wei) |
| | `normalized_token_transfer`| Value normalized against baseline (clamped to [0, 1]) |
| **Event Activity** | `log_count` | Number of logs emitted by contract call |
| | `length_log` | Logarithmic transformation `log(log_count + 1)` |
| | `event_activity_flag` | Binary indicator (1.0 if logs are emitted) |
| | `log_removed` | Flag indicating chain reorganization log removal |

---

## ⚙️ Environment Variables

Configure these in `.env`:

| Variable | Default | Description |
|---|---|---|
| `VITE_API_BASE_URL` | `http://localhost:8001` | FastAPI inference API base URL for the frontend |
| `MODEL_PATH` | `models/random_forest_fraud_model.joblib` | Path to trained Random Forest `.joblib` model file |
| `DEFIGUARD_MODEL_PATH` | `models/defi_guard_model.joblib` | Path to trained DeFiGuard `.joblib` model file |
| `ENSEMBLE_MODE` | `max` | Ensemble mode: `max`, `mean`, `weighted`, `rf_only`, `dg_only` |
| `FRAUD_THRESHOLD` | `0.65` | Probability cutoff above which transactions are flagged |
| `LOG_LEVEL` | `INFO` | Python logging verbosity (`DEBUG`, `INFO`, `WARNING`) |
| `GROQ_API_KEY` | *(optional)* | API key for LLM-powered plain language risk explanations |

---

## 🧪 Running Tests

To run the backend test suite across the feature extractor, inference engine, and ensemble scorer:

```bash
cd backend
pytest tests/ -v
```

To run frontend linting and production build checks:
```bash
cd frontend
npm run lint
npm run build
```

---

## 📜 License

This project is developed for educational and research purposes.
