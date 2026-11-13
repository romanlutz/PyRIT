// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import assert from 'node:assert/strict';
import { readdirSync, readFileSync } from 'node:fs';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

import plugin from '../../../doc/_plugins/notebook_identifiers.mjs';

const docRoot = fileURLToPath(new URL('../../../doc/', import.meta.url));
const transform = plugin.transforms[0].plugin();
const file = (name) => ({ path: path.join(docRoot, name) });
const cell = (identifier = '0', extra = {}) => ({
  type: 'block',
  kind: 'notebook-code',
  identifier,
  html_id: identifier,
  children: [
    { type: 'code', value: 'print(1)' },
    { type: 'outputs', children: [{ type: 'output', data: [{ 'text/plain': '1' }] }] },
  ],
  ...extra,
});
const root = (...children) => ({ type: 'root', children });

test('namespaces numeric notebook IDs, preserves HTML fragments and outputs, and is idempotent', () => {
  assert.equal(plugin.transforms[0].stage, 'document');
  const left = root(cell());
  const right = structuredClone(left);
  const original = structuredClone(left.children[0]);
  transform(left, file('left.ipynb'));
  transform(right, file('right.ipynb'));
  assert.notEqual(left.children[0].identifier, right.children[0].identifier);
  assert.match(left.children[0].identifier, /^notebook-[a-f0-9]{64}-0$/);
  assert.deepEqual({ ...left.children[0], identifier: '0' }, original);
  const once = structuredClone(left);
  transform(left, file('left.ipynb'));
  assert.deepEqual(left, once);
});

test('preserves explicit labels, nonnumeric cell IDs, headings, and non-notebook content', () => {
  const tree = root(
    cell('0', { label: '0' }),
    cell('author-id'),
    { type: 'heading', identifier: '0', html_id: '0', children: [] },
    { type: 'block', identifier: '1', html_id: 'author-block', children: [] },
    cell(undefined, { identifier: undefined, html_id: undefined }),
  );
  const original = structuredClone(tree);
  transform(tree, file('example.ipynb'));
  assert.deepEqual(tree, original);
  const markdown = root(cell());
  const originalMarkdown = structuredClone(markdown);
  transform(markdown, file('example.md'));
  assert.deepEqual(markdown, originalMarkdown);
});

test('handles notebook content and cells with an author-customized kind', () => {
  const tree = root(cell('0', { kind: 'notebook-content' }), cell('1', { kind: 'custom-kind' }));
  transform(tree, file('custom.ipynb'));
  assert.match(tree.children[0].identifier, /^notebook-[a-f0-9]{64}-0$/);
  assert.match(tree.children[1].identifier, /^notebook-[a-f0-9]{64}-1$/);
});

test('namespaces fallback identifiers moved to caption containers, but not explicit figure labels', () => {
  const tree = root(
    cell('0', {
      identifier: undefined,
      children: [{ type: 'container', identifier: '0', children: [{ type: 'text', value: 'caption' }] }],
    }),
    cell('1', {
      identifier: undefined,
      children: [{ type: 'container', identifier: '1', label: '1', children: [] }],
    }),
  );
  transform(tree, file('figures.ipynb'));
  assert.match(tree.children[0].children[0].identifier, /^notebook-[a-f0-9]{64}-0$/);
  assert.equal(tree.children[0].html_id, '0');
  assert.equal(tree.children[1].children[0].identifier, '1');
});

test('rewrites local references to the namespaced target without changing other URLs', () => {
  const tree = root(
    cell(),
    { type: 'crossReference', identifier: '0', children: [] },
    { type: 'link', url: '#0', children: [] },
    { type: 'link', url: '#heading', children: [] },
    { type: 'link', url: 'https://example.com/left.ipynb#0', children: [] },
  );
  transform(tree, file('left.ipynb'));
  assert.equal(tree.children[1].identifier, tree.children[0].identifier);
  assert.equal(tree.children[2].url, `#${tree.children[0].identifier}`);
  assert.equal(tree.children[3].url, '#heading');
  assert.equal(tree.children[4].url, 'https://example.com/left.ipynb#0');

  const source = root({ type: 'link', url: '../left.ipynb#0', children: [] });
  transform(source, file('nested/source.md'));
  assert.equal(source.children[0].url, '../left.ipynb#0');
});

test('all committed documentation notebook fallback identifiers are project-unique after normalization', () => {
  const notebooks = [];
  function collect(directory) {
    for (const entry of readdirSync(directory, { withFileTypes: true })) {
      const name = path.join(directory, entry.name);
      if (entry.isDirectory() && entry.name !== '_build') collect(name);
      else if (entry.isFile() && name.endsWith('.ipynb')) notebooks.push(name);
    }
  }
  collect(docRoot);
  assert.ok(notebooks.length > 0);
  const identifiers = new Map();
  for (const notebook of notebooks) {
    const data = JSON.parse(readFileSync(notebook, 'utf8'));
    const tree = root(...data.cells.filter((entry) => entry.id).map((entry) => cell(entry.id)));
    transform(tree, { path: notebook });
    for (const node of tree.children) {
      assert.ok(!identifiers.has(node.identifier), `${notebook} duplicates ${identifiers.get(node.identifier)}`);
      identifiers.set(node.identifier, notebook);
    }
  }
  assert.ok(identifiers.size > notebooks.length);
});
