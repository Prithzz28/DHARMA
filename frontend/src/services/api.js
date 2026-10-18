// src/services/api.js
/**
 * Web3 Transaction Firewall API Client
 * Configurable base URL – falls back to Vite dev proxy (relative URLs)
 * when VITE_API_BASE_URL is not set.
 */

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL !== undefined
    ? import.meta.env.VITE_API_BASE_URL
    : '';

/**
 * Helper to handle fetch responses and errors
 */
async function handleResponse(response) {
  if (!response.ok) {
    let errorMessage = `HTTP error ${response.status}: ${response.statusText}`;
    try {
      const errorData = await response.json();
      if (errorData.detail) {
        errorMessage =
          typeof errorData.detail === 'string'
            ? errorData.detail
            : JSON.stringify(errorData.detail);
      }
    } catch {
      // Ignore JSON parse error on non-JSON error response
    }
    throw new Error(errorMessage);
  }
  return response.json();
}

/**
 * Resilient fetch helper:
 * 1. Tries configured API_BASE_URL (or relative URL via Vite dev proxy)
 * 2. If relative request fails, falls back to direct http://<hostname>:8001
 */
async function apiFetch(endpoint, options = {}) {
  const url = `${API_BASE_URL}${endpoint}`;
  try {
    const res = await fetch(url, options);
    return await handleResponse(res);
  } catch (err) {
    // If relative proxy attempt threw NetworkError and API_BASE_URL is '', try direct backend host
    if (API_BASE_URL === '' && typeof window !== 'undefined') {
      const fallbackHost = window.location.hostname || 'localhost';
      const fallbackUrl = `http://${fallbackHost}:8001${endpoint}`;
      try {
        const fallbackRes = await fetch(fallbackUrl, options);
        return await handleResponse(fallbackRes);
      } catch {
        // Re-throw original helpful error
      }
    }
    throw err;
  }
}

/**
 * Check backend liveness and model status (GET /health)
 * @returns {Promise<{status: string, model_loaded: boolean, model_path: string}>}
 */
export async function checkHealth() {
  return apiFetch('/health', {
    method: 'GET',
    headers: { Accept: 'application/json' },
  });
}

/**
 * Predict fraud for a transaction feature vector (POST /predict)
 * @param {Object} features - Full 20-field TransactionFeatures object
 */
export async function predictTransaction(features) {
  return apiFetch('/predict', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json',
    },
    body: JSON.stringify(features),
  });
}

/**
 * Fetch structured risk explanation (POST /predict/explain)
 * @param {Object} features - Full feature vector
 * @param {Object} formData  - Raw user form data (for address masking etc.)
 */
export async function fetchRiskExplanation(features, formData = {}) {
  const rawAmt = parseFloat(formData.amount);
  const payload = {
    features,
    raw_from: formData.from || null,
    raw_to: formData.to || null,
    amount_eth: !isNaN(rawAmt) && isFinite(rawAmt) ? rawAmt : null,
  };
  return apiFetch('/predict/explain', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json',
    },
    body: JSON.stringify(payload),
  });
}

/**
 * Convert user form inputs into the exact TransactionFeatures schema expected
 * by the FastAPI backend — faithfully passing user inputs without pre-judging
 * fraud signals. The ML model is responsible for classification.
 *
 * @param {Object} formData - Raw form fields from the scanner UI
 * @returns {Object} 20-field feature vector matching TransactionFeatures schema
 */
export function buildFeatureVector(formData) {
  const fromAddr = (formData.from || '').trim();
  const toAddr = (formData.to || '').trim();

  // ── Value ──────────────────────────────────────────────────────────────────
  const rawAmount = parseFloat(formData.amount);
  const amountETH =
    !isNaN(rawAmount) && isFinite(rawAmount) && rawAmount >= 0 ? rawAmount : 0;
  const valueWei = amountETH * 1e18;

  // ── Self-transfer flag ─────────────────────────────────────────────────────
  const isSameAddress =
    fromAddr && toAddr && fromAddr.toLowerCase() === toAddr.toLowerCase()
      ? 1.0
      : 0.0;

  // ── Gas parameters ─────────────────────────────────────────────────────────
  const rawGasPrice = parseFloat(formData.gasPrice);
  const gasPriceGwei =
    !isNaN(rawGasPrice) && isFinite(rawGasPrice) && rawGasPrice >= 0
      ? rawGasPrice
      : 25;
  const effectiveGasPriceWei = gasPriceGwei * 1e9;

  const rawGasLimit = parseFloat(formData.gasLimit);
  const gasLimit =
    !isNaN(rawGasLimit) && isFinite(rawGasLimit) && rawGasLimit > 0
      ? rawGasLimit
      : 21000;

  const gasUsed = gasLimit * 0.85; // estimated consumed gas
  const gasEfficiency = gasLimit > 0 ? Math.min(gasUsed / gasLimit, 1.0) : 0.85;

  // ── Gas cost & ratio ───────────────────────────────────────────────────────
  // NOTE: baseFeeGwei is a fixed reference used purely for display ratio in the UI.
  // The actual base fee is network-dynamic and not available without an RPC call.
  const BASE_FEE_GWEI_REFERENCE = 20;
  const gasPriceRatio = BASE_FEE_GWEI_REFERENCE > 0 ? gasPriceGwei / BASE_FEE_GWEI_REFERENCE : 1.0;
  const totalGasCost = gasUsed * effectiveGasPriceWei;

  // ── Block number ───────────────────────────────────────────────────────────
  // Use user input if provided; default to a reasonable mainnet-era block height.
  const rawBlock = parseFloat(formData.blockNumber);
  const blockNumber =
    !isNaN(rawBlock) && isFinite(rawBlock) && rawBlock >= 0
      ? rawBlock
      : 19_000_000;

  // ── Chain ID ───────────────────────────────────────────────────────────────
  const chainId = parseFloat(formData.chainId) || 1.0;

  // ── Log / event fields ─────────────────────────────────────────────────────
  const rawLogCount = parseFloat(formData.logCount);
  const logCount =
    !isNaN(rawLogCount) && isFinite(rawLogCount) && rawLogCount >= 0
      ? rawLogCount
      : 0;
  const logRemoved = parseFloat(formData.logRemoved) || 0.0;
  const gasPerLogEvent = gasUsed / (logCount + 1);
  const eventActivityFlag = logCount > 0 ? 1.0 : 0.0;
  // length_log = log(log_count + 1) — matches the backend feature_extractor/metrics.py definition
  const lengthLog = Math.log(logCount + 1);

  // ── Transaction hash ───────────────────────────────────────────────────────
  const txHash = (formData.hash || '').trim();
  const lengthTxHash = txHash.length > 0 ? txHash.length : 66.0;

  // ── Normalised transfer value ──────────────────────────────────────────────
  const normalizedTokenTransfer = Math.min(valueWei / 1e21, 1.0);

  // ── Index ──────────────────────────────────────────────────────────────────
  const index = parseFloat(formData.index) || 0.0;

  // ── Cumulative gas ─────────────────────────────────────────────────────────
  const rawCumGas = parseFloat(formData.cumulativeGasUsed);
  const cumulativeGasUsed =
    !isNaN(rawCumGas) && isFinite(rawCumGas) && rawCumGas >= 0
      ? rawCumGas
      : gasUsed * 2.5;

  return {
    length_transaction_hash: lengthTxHash,
    length_to: toAddr ? toAddr.length : 42.0,
    length_from: fromAddr ? fromAddr.length : 42.0,
    is_same_address: isSameAddress,
    index,
    block_number: blockNumber,
    chain_id: chainId,
    cumulative_gas_used: cumulativeGasUsed,
    gas_used: gasUsed,
    gas_efficiency: gasEfficiency,
    effective_gas_price: effectiveGasPriceWei,
    total_gas_cost: totalGasCost,
    gas_price_ratio: gasPriceRatio,
    gas_per_log_event: gasPerLogEvent,
    value: valueWei,
    normalized_token_transfer: normalizedTokenTransfer,
    log_count: logCount,
    log_removed: logRemoved,
    length_log: lengthLog,  // ← corrected: log(logCount + 1) not raw logCount
    event_activity_flag: eventActivityFlag,
  };
}

/**
 * Fetch live inference engine stats (GET /stats)
 * @returns {Promise<{total_predictions, total_fraud, fraud_rate_pct, avg_latency_ms, uptime_seconds}>}
 */
export async function fetchStats() {
  return apiFetch('/stats', {
    method: 'GET',
    headers: { Accept: 'application/json' },
  });
}
