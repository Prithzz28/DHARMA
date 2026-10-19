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
  blockNumber: '',
  hash: '',
  logCount: '0',
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
      log_count: 0,
    });
  });

  it('includes optional fields when provided', () => {
    const req = toScanRequest({ ...VALID_FORM, blockNumber: '19000000', hash: ' 0xabc ' });
    expect(req.block_number).toBe(19000000);
    expect(req.tx_hash).toBe('0xabc');
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
      logCount: '-3',
    });
    expect(Object.keys(errors).sort()).toEqual(['amount', 'gasLimit', 'gasPrice', 'logCount']);
  });

  it('requires at least one address and the gas fields', () => {
    const errors = validateScanForm({ ...VALID_FORM, from: '', to: '', gasPrice: '', gasLimit: '' });
    expect(errors.from).toBeTruthy();
    expect(errors.gasPrice).toBeTruthy();
    expect(errors.gasLimit).toBeTruthy();
  });
});
