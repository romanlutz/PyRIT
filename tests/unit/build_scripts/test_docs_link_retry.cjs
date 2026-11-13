// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

const assert = require('node:assert/strict');
const { test } = require('node:test');

const modulePromise = import('../../../build_scripts/docs_link_check/retry.mjs');
const URL = 'https://example.test/page';

function response(status, retryAfter = null) {
  return { status, headers: new Headers(retryAfter === null ? {} : { 'Retry-After': retryAfter }),
    body: { destroy() {} } };
}

async function exercise(sequence, options = {}) {
  const { withRetries } = await modulePromise;
  const delays = [];
  let calls = 0;
  const fetch = withRetries(async () => {
    const result = sequence[Math.min(calls++, sequence.length - 1)];
    if (result instanceof Error) throw result;
    return result;
  }, { wait: async (delay) => delays.push(delay), log() {}, ...options });
  return { fetch, delays, calls: () => calls };
}

test('transient HTTP errors recover with bounded exponential backoff', async () => {
  for (const status of [408, 429, 500, 502, 503, 504, 599]) {
    const state = await exercise([response(status), response(status), response(200)]);
    assert.equal((await state.fetch(URL)).status, 200);
    assert.equal(state.calls(), 3);
    assert.deepEqual(state.delays, [2000, 4000]);
  }
});

test('exhausted HTTP retries return failure unchanged', async () => {
  const failed = response(504);
  const state = await exercise([failed]);
  assert.equal(await state.fetch(URL), failed);
  assert.equal(state.calls(), 4);
  assert.deepEqual(state.delays, [2000, 4000, 8000]);
});

test('permanent responses including broken links and auth fail immediately', async () => {
  for (const status of [200, 301, 400, 401, 403, 404, 410, 422]) {
    const result = response(status);
    const state = await exercise([result]);
    assert.equal(await state.fetch(URL), result);
    assert.equal(state.calls(), 1);
    assert.deepEqual(state.delays, []);
  }
});

test('network reset and temporary DNS failure recover', async () => {
  for (const code of ['ECONNRESET', 'EAI_AGAIN', 'ETIMEDOUT']) {
    const error = Object.assign(new Error(code), { code });
    const state = await exercise([error, response(200)]);
    assert.equal((await state.fetch(URL)).status, 200);
    assert.deepEqual(state.delays, [2000]);
  }
});

test('exhausted network retries preserve original error', async () => {
  const error = Object.assign(new Error('reset'), { code: 'ECONNRESET' });
  const state = await exercise([error]);
  await assert.rejects(state.fetch(URL), (actual) => actual === error);
  assert.equal(state.calls(), 4);
});

test('certificate, permanent DNS, and programming errors are never retried', async () => {
  for (const code of ['CERT_HAS_EXPIRED', 'ENOTFOUND', undefined]) {
    const error = Object.assign(new Error('permanent'), { code });
    const state = await exercise([error]);
    await assert.rejects(state.fetch(URL), (actual) => actual === error);
    assert.equal(state.calls(), 1);
  }
});

test('Retry-After accepts seconds and HTTP dates without shortening backoff', async () => {
  const now = Date.parse('Mon, 09 Nov 2026 12:00:00 GMT');
  for (const [header, expected] of [
    ['10', 10_000], ['Mon, 09 Nov 2026 12:00:12 GMT', 12_000],
    ['0', 2000], ['invalid', 2000], ['Mon, 09 Nov 2026 11:00:00 GMT', 2000],
  ]) {
    const state = await exercise([response(429, header), response(200)], { now: () => now });
    await state.fetch(URL);
    assert.deepEqual(state.delays, [expected]);
  }
});

test('Retry-After beyond the wait budget fails instead of retrying early', async () => {
  const failed = response(503, '120');
  const state = await exercise([failed]);
  assert.equal(await state.fetch(URL), failed);
  assert.equal(state.calls(), 1);
  assert.deepEqual(state.delays, []);
});

test('local requests and non-idempotent methods do not retry', async () => {
  for (const [url, init] of [
    ['http://localhost/page', undefined], ['http://127.0.0.1/page', undefined],
    ['http://[::1]/page', undefined], [URL, { method: 'POST' }],
  ]) {
    const state = await exercise([response(503)]);
    assert.equal((await state.fetch(url, init)).status, 503);
    assert.equal(state.calls(), 1);
  }
});

test('caller cancellation never triggers a retry', async () => {
  const controller = new AbortController();
  controller.abort();
  const error = Object.assign(new Error('cancelled'), { name: 'AbortError' });
  const state = await exercise([error]);
  await assert.rejects(state.fetch(URL, { signal: controller.signal }), (actual) => actual === error);
  assert.equal(state.calls(), 1);
});

test('attempt timeout is bounded and transient', async () => {
  const { withRetries } = await modulePromise;
  let calls = 0;
  const fetch = withRetries(async (_url, { signal }) => {
    calls++;
    if (calls === 2) return response(200);
    return new Promise((_resolve, reject) => {
      signal.addEventListener('abort', () => reject(new Error('timeout')), { once: true });
    });
  }, { timeoutMilliseconds: 1, wait: async () => {}, log() {} });
  assert.equal((await fetch(URL)).status, 200);
  assert.equal(calls, 2);
});

test('cloned MyST sessions retain request-level retries', async () => {
  const { enableRetries } = await modulePromise;
  function session() {
    let calls = 0;
    return { fetch: async () => response(++calls === 1 ? 503 : 200),
      clone: async () => session(), log: { warn() {} } };
  }
  const configured = enableRetries(session(), { wait: async () => {} });
  const clone = await configured.clone();
  const nested = await clone.clone();
  for (const target of [configured, clone, nested]) {
    assert.equal((await target.fetch(URL)).status, 200);
  }
});
