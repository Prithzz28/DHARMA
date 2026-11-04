import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  ApiError,
  formatErrorDetail,
  scanTransaction,
  toScanRequest,
  validateScanForm,
} from './api.js';

const VALID_FORM = {
  from: '0x3b2a1f88e4c5d6a7890bcdef2345678901bcde22',
  to: '0x8c5d7e12f3a4b5c6d7890abcdef3456789012cd3',
  amount: '0.45',
  gasPrice: '22',
  gasLimit: '21000',
  nonce: '',
  data: '',
};

function jsonResponse(status, body, headers = {}) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json', ...headers },
  });
}

function mockFetch(impl) {
  return vi.spyOn(globalThis, 'fetch').mockImplementation(impl);
}

afterEach(() => vi.restoreAllMocks());

describe('formatErrorDetail', () => {
  it('formats FastAPI 422 detail arrays into field messages', () => {
    const detail = [
      { loc: ['body', 'gas_price_gwei'], msg: 'Input should be greater than 0' },
      { loc: ['body', 'to'], msg: 'String should have at most 100 characters' },
    ];
    expect(formatErrorDetail(detail)).toBe(
      'gas_price_gwei: Input should be greater than 0; to: String should have at most 100 characters',
    );
  });

  it('passes string details through', () => {
    expect(formatErrorDetail('Model not loaded')).toBe('Model not loaded');
  });
});

describe('scanTransaction', () => {
  it('sends one POST /scan with the mapped request and returns the body', async () => {
    const fetchMock = mockFetch(async () =>
      jsonResponse(200, { features: {}, prediction: { is_fraud: false }, explanation: null }),
    );
    const body = await scanTransaction(VALID_FORM, { useLlm: true });

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe('/scan');
    expect(init.method).toBe('POST');
    expect(JSON.parse(init.body)).toEqual({ ...toScanRequest(VALID_FORM), explain: true, use_llm: true });
    expect(body.prediction.is_fraud).toBe(false);
  });

  it('does not retry on HTTP errors and surfaces the server message', async () => {
    const fetchMock = mockFetch(async () =>
      jsonResponse(422, { detail: [{ loc: ['body', 'gas_limit'], msg: 'Input should be greater than 0' }] }),
    );
    const error = await scanTransaction(VALID_FORM).catch((err) => err);
    expect(error).toBeInstanceOf(ApiError);
    expect(error.status).toBe(422);
    expect(error.message).toBe('gas_limit: Input should be greater than 0');
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('reports rate limiting with the Retry-After delay', async () => {
    mockFetch(async () => jsonResponse(429, { detail: 'Rate limit exceeded' }, { 'Retry-After': '42' }));
    const error = await scanTransaction(VALID_FORM).catch((err) => err);
    expect(error.status).toBe(429);
    expect(error.message).toBe('Rate limit reached — retry in 42s.');
  });

  it('reports network failures as ApiError with status 0', async () => {
    mockFetch(async () => {
      throw new TypeError('fetch failed');
    });
    const error = await scanTransaction(VALID_FORM).catch((err) => err);
    expect(error).toBeInstanceOf(ApiError);
    expect(error.status).toBe(0);
  });

  it('propagates caller aborts unchanged', async () => {
    mockFetch(
      (_url, init) =>
        new Promise((_resolve, reject) => {
          init.signal.addEventListener('abort', () => reject(init.signal.reason));
        }),
    );
    const controller = new AbortController();
    const pending = scanTransaction(VALID_FORM, { signal: controller.signal });
    controller.abort();
    await expect(pending).rejects.toMatchObject({ name: 'AbortError' });
  });
});

describe('toScanRequest', () => {
  it('converts strings to numbers and omits blank optional fields', () => {
    expect(toScanRequest(VALID_FORM)).toEqual({
      from: VALID_FORM.from,
      to: VALID_FORM.to,
      value_eth: 0.45,
      gas_price_gwei: 22,
      gas_limit: 21000,
    });
  });

  it('includes optional fields when provided', () => {
    const req = toScanRequest({ ...VALID_FORM, nonce: '7', data: ' 0xa9059cbb ' });
    expect(req.nonce).toBe(7);
    expect(req.data).toBe('0xa9059cbb');
  });
});

describe('validateScanForm', () => {
  it('accepts a valid form', () => {
    expect(validateScanForm(VALID_FORM)).toEqual({});
  });

  it('rejects non-numeric, negative and fractional inputs instead of defaulting', () => {
    const errors = validateScanForm({
      ...VALID_FORM,
      amount: '-1',
      gasPrice: 'abc',
      gasLimit: '21000.5',
      nonce: '-3',
      data: '0xzz',
    });
    expect(Object.keys(errors).sort()).toEqual(['amount', 'data', 'gasLimit', 'gasPrice', 'nonce']);
  });

  it('requires at least one address and the gas fields', () => {
    const errors = validateScanForm({ ...VALID_FORM, from: '', to: '', gasPrice: '', gasLimit: '' });
    expect(errors.from).toBeTruthy();
    expect(errors.gasPrice).toBeTruthy();
    expect(errors.gasLimit).toBeTruthy();
  });
});

describe('scanner presets', async () => {
  const { default: presets } = await import('../data/scannerPresets.json');

  it('cover both dataset labels', () => {
    expect(new Set(presets.presets.map((p) => p.expected))).toEqual(new Set(['BLOCKED', 'SAFE']));
  });

  it.each(presets.presets.map((p) => [p.label, p]))('%s passes form validation and maps to /scan', (_label, preset) => {
    expect(validateScanForm(preset.data)).toEqual({});
    // Must equal the request backend/tests/test_scanner_presets.py sends.
    expect(toScanRequest(preset.data)).toEqual({
      from: preset.data.from,
      to: preset.data.to,
      value_eth: Number(preset.data.amount),
      gas_price_gwei: Number(preset.data.gasPrice),
      gas_limit: Number(preset.data.gasLimit),
      nonce: Number(preset.data.nonce),
      ...(preset.data.data ? { data: preset.data.data } : {}),
    });
  });
});

describe('wallet RPC helpers', () => {
  it('builds the RPC URL from the API base or the page origin', async () => {
    const { rpcUrl } = await import('./api.js');
    expect(rpcUrl('https://api.example.com/', 'http://localhost:5173')).toBe('https://api.example.com/rpc');
    expect(rpcUrl('', 'http://localhost:5173')).toBe('http://localhost:5173/rpc');
  });

  it('maps /rpc/info to wallet_addEthereumChain params', async () => {
    const { addChainParams } = await import('./api.js');
    const info = {
      chain_id_hex: '0xaa36a7', network_name: 'Sepolia', currency_symbol: 'SepoliaETH',
      explorer_url: 'https://sepolia.etherscan.io',
    };
    expect(addChainParams(info, 'https://api.example.com/rpc')).toEqual({
      chainId: '0xaa36a7',
      chainName: 'Sepolia (DeFiGuard)',
      rpcUrls: ['https://api.example.com/rpc'],
      nativeCurrency: { name: 'SepoliaETH', symbol: 'SepoliaETH', decimals: 18 },
      blockExplorerUrls: ['https://sepolia.etherscan.io'],
    });
    expect(addChainParams({ ...info, explorer_url: '' }, 'x').blockExplorerUrls).toEqual([]);
  });

  it('requests the recent-interceptions feed with a limit', async () => {
    const { fetchRecentInterceptions } = await import('./api.js');
    const fetchSpy = mockFetch(async () => jsonResponse(200, { items: [], totals: {} }));
    await expect(fetchRecentInterceptions({ limit: 5 })).resolves.toEqual({ items: [], totals: {} });
    expect(fetchSpy.mock.calls[0][0]).toBe('/rpc/recent?limit=5');
  });
});

describe('normalizeInterception', () => {
  it('maps an older API item (model signals as `reasons`, no counterparty field)', async () => {
    const { normalizeInterception } = await import('./api.js');
    const old = {
      id: 4, verdict: 'blocked', fraud_probability: 0.73, value_eth: 0,
      from_address: '0xee0e...6c79', to_address: '0xfff9...6b14', reasons: ['Calls a token approval'],
    };
    const item = normalizeInterception(old);
    expect(item.counterparty).toEqual([]);
    expect(item.model_signals).toEqual(['Calls a token approval']);
    expect(item.tx_hash).toBeNull();
  });

  it('fills safe defaults for missing or malformed fields', async () => {
    const { normalizeInterception } = await import('./api.js');
    const item = normalizeInterception({ verdict: 'exploded', value_eth: null, counterparty: 'x' });
    expect(item).toMatchObject({
      verdict: 'allowed', value_eth: 0, fraud_probability: 0, counterparty: [], model_signals: [],
      from_address: '—', to_address: '—',
    });
  });

  it('is applied to every feed item and tolerates a malformed body', async () => {
    const { fetchRecentInterceptions } = await import('./api.js');
    mockFetch(async () => jsonResponse(200, { items: [{ verdict: 'warned', reasons: ['r'] }] }));
    const feed = await fetchRecentInterceptions();
    expect(feed.items[0].model_signals).toEqual(['r']);
    expect(feed.totals).toEqual({});
    vi.restoreAllMocks();
    mockFetch(async () => jsonResponse(200, { unexpected: true }));
    await expect(fetchRecentInterceptions()).resolves.toEqual({ items: [], totals: {} });
  });
});
