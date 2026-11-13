// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import { setTimeout as sleep } from 'node:timers/promises';

const NETWORK_ERRORS = new Set([
  'ECONNABORTED', 'ECONNREFUSED', 'ECONNRESET', 'EHOSTUNREACH',
  'ENETUNREACH', 'EAI_AGAIN', 'ETIMEDOUT',
]);

function retryAfterMilliseconds(value, now) {
  if (value === null) return 0;
  if (/^\d+$/.test(value.trim())) return Number(value) * 1000;
  const date = Date.parse(value);
  return Number.isNaN(date) ? 0 : Math.max(0, date - now());
}

function externalGet(url, init) {
  const method = init?.method ?? url?.method ?? 'GET';
  if (method.toUpperCase() !== 'GET') return false;
  const parsed = new URL(url?.url ?? url);
  return ['http:', 'https:'].includes(parsed.protocol) &&
    !['localhost', '127.0.0.1', '[::1]'].includes(parsed.hostname);
}

export function withRetries(fetch, {
  wait = sleep,
  now = Date.now,
  log = console.warn,
  timeoutMilliseconds = 20_000,
} = {}) {
  return async (url, init) => {
    if (!externalGet(url, init)) return fetch(url, init);
    for (let attempt = 1; attempt <= 4; attempt++) {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), timeoutMilliseconds);
      let response;
      let reason;
      try {
        response = await fetch(url, {
          ...init,
          signal: init?.signal
            ? AbortSignal.any([init.signal, controller.signal])
            : controller.signal,
        });
        if (response.status !== 408 && response.status !== 429 &&
            !(response.status >= 500 && response.status <= 599)) return response;
        reason = `HTTP ${response.status}`;
      } catch (error) {
        if (init?.signal?.aborted || attempt === 4 ||
            !(controller.signal.aborted || NETWORK_ERRORS.has(error.code ?? error.cause?.code))) throw error;
        reason = error.message;
      } finally {
        clearTimeout(timer);
      }
      if (attempt === 4) return response;
      const delay = Math.max(
        2000 * 2 ** (attempt - 1),
        retryAfterMilliseconds(response?.headers.get('retry-after') ?? null, now),
      );
      // Do not retry sooner than requested when the server exceeds our bounded wait budget.
      if (delay > 60_000) {
        log(`Not retrying ${url}: Retry-After exceeds 60 seconds (${reason}).`);
        return response;
      }
      response?.body?.destroy();
      log(`Retrying ${url} after ${reason}; attempt ${attempt + 1}/4 in ${delay / 1000}s.`);
      await wait(delay, undefined, init?.signal ? { signal: init.signal } : undefined);
    }
    throw new Error('Unreachable retry state');
  };
}

export function enableRetries(session, options) {
  const originalFetch = session.fetch.bind(session);
  session.fetch = withRetries(originalFetch, { log: (message) => session.log.warn(message), ...options });
  const originalClone = session.clone.bind(session);
  session.clone = async () => enableRetries(await originalClone(), options);
  return session;
}
