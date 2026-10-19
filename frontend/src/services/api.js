// src/services/api.js
/**
 * Web3 Transaction Firewall API client.
 *
 * Base URL: VITE_API_BASE_URL when set; otherwise relative URLs, which the
 * Vite dev/preview proxy forwards to the inference engine (see vite.config.js).
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '';

export const REQUEST_TIMEOUT_MS = 15_000;

/** Error carrying the HTTP status (0 for network/timeout failures). */
export class ApiError extends Error {
  constructor(message, status = 0) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

/** Turn a FastAPI error body into a readable message. */
export function formatErrorDetail(detail) {
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((err) => {
        const field = (err.loc || []).filter((part) => part !== 'body').join('.');
        return field ? `${field}: ${err.msg}` : err.msg;
      })
      .join('; ');
  }
  return detail ? JSON.stringify(detail) : '';
}

/** Combine the caller's abort signal with a timeout. */
function withTimeout(signal, timeoutMs) {
  const timeout = AbortSignal.timeout(timeoutMs);
  return signal ? AbortSignal.any([signal, timeout]) : timeout;
}

async function apiFetch(endpoint, { signal, timeoutMs = REQUEST_TIMEOUT_MS, ...options } = {}) {
  let response;
  try {
    response = await fetch(`${API_BASE_URL}${endpoint}`, {
      ...options,
      headers: { Accept: 'application/json', ...options.headers },
      signal: withTimeout(signal, timeoutMs),
    });
  } catch (err) {
    if (signal?.aborted) throw err; // cancelled by the caller — propagate as-is
    if (err?.name === 'TimeoutError') {
      throw new ApiError(`Request timed out after ${timeoutMs / 1000}s.`);
    }
    throw new ApiError('Cannot reach the inference engine. Is the backend running on port 8001?');
  }

  if (response.status === 429) {
    const retryAfter = Number.parseInt(response.headers.get('Retry-After') ?? '', 10);
    const wait = Number.isFinite(retryAfter) ? ` — retry in ${retryAfter}s` : '';
    throw new ApiError(`Rate limit reached${wait}.`, 429);
  }
  if (!response.ok) {
    let message = `HTTP ${response.status} ${response.statusText}`.trim();
    try {
      const body = await response.json();
      message = formatErrorDetail(body.detail) || message;
    } catch {
      // Non-JSON error body — keep the status line.
    }
    throw new ApiError(message, response.status);
  }
  return response.json();
}

/**
 * Backend liveness and model status (GET /health).
 * @returns {Promise<{status: string, model_loaded: boolean, model_path: string,
 *   fraud_threshold: number}>}
 */
export function checkHealth({ signal } = {}) {
  return apiFetch('/health', { method: 'GET', signal });
}

/**
 * Live counters since the inference engine started (GET /stats).
 * @returns {Promise<{total_predictions: number, total_fraud: number, fraud_rate_pct: number,
 *   avg_latency_ms: number, uptime_seconds: number}>}
 */
export function fetchStats({ signal } = {}) {
  return apiFetch('/stats', { method: 'GET', signal });
}

/**
 * Score a transaction (POST /scan). Feature extraction happens server-side.
 *
 * @param {Object} form - Scanner fields (strings from the form inputs)
 * @param {{useLlm?: boolean, signal?: AbortSignal}} options
 *   useLlm — allow the Groq LLM explainer (otherwise rule-based; always explained)
 * @returns {Promise<{features: Object, prediction: Object, explanation: Object|null}>}
 */
export function scanTransaction(form, { useLlm = false, signal } = {}) {
  return apiFetch('/scan', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...toScanRequest(form), explain: true, use_llm: useLlm }),
    signal,
  });
}

/** Map scanner form strings to the ScanRequest schema (blank optional fields are omitted). */
export function toScanRequest(form) {
  const request = {
    from: (form.from || '').trim(),
    to: (form.to || '').trim(),
    value_eth: Number(form.amount || 0),
    gas_price_gwei: Number(form.gasPrice),
    gas_limit: Number(form.gasLimit),
  };
  const optional = { blockNumber: 'block_number', logCount: 'log_count' };
  for (const [field, key] of Object.entries(optional)) {
    if (String(form[field] ?? '').trim() !== '') request[key] = Number(form[field]);
  }
  if ((form.hash || '').trim()) request.tx_hash = form.hash.trim();
  return request;
}

const NUMERIC_RULES = {
  amount: { label: 'Amount', integer: false, required: false, positive: false },
  gasPrice: { label: 'Gas price', integer: false, required: true, positive: true },
  gasLimit: { label: 'Gas limit', integer: true, required: true, positive: true },
  blockNumber: { label: 'Block number', integer: true, required: false, positive: false },
  logCount: { label: 'Log count', integer: true, required: false, positive: false },
};

/**
 * Client-side validation mirroring the server's ScanRequest constraints.
 * The server validates again; this only gives immediate feedback.
 * @returns {Object<string, string>} field → error message (empty when valid)
 */
export function validateScanForm(form) {
  const errors = {};
  if (!(form.from || '').trim() && !(form.to || '').trim()) {
    errors.from = 'Enter a From or To address.';
  }
  for (const [field, rule] of Object.entries(NUMERIC_RULES)) {
    const raw = String(form[field] ?? '').trim();
    if (raw === '') {
      if (rule.required) errors[field] = `${rule.label} is required.`;
      continue;
    }
    const value = Number(raw);
    if (!Number.isFinite(value)) errors[field] = `${rule.label} must be a number.`;
    else if (rule.positive ? value <= 0 : value < 0) {
      errors[field] = `${rule.label} must be ${rule.positive ? 'greater than 0' : '0 or more'}.`;
    } else if (rule.integer && !Number.isInteger(value)) errors[field] = `${rule.label} must be a whole number.`;
  }
  for (const field of ['from', 'to', 'hash']) {
    if ((form[field] || '').trim().length > 100) errors[field] = 'Must be at most 100 characters.';
  }
  return errors;
}
