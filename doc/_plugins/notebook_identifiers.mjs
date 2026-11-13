// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import { createHash } from 'node:crypto';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const docRoot = fileURLToPath(new URL('../', import.meta.url));

function cellIdentifier(file, identifier) {
  const relativePath = path.relative(docRoot, file).split(path.sep).join('/').toLowerCase();
  const namespace = createHash('sha256').update(relativePath).digest('hex');
  return `notebook-${namespace}-${identifier}`;
}

function visit(node, callback) {
  callback(node);
  node.children?.forEach((child) => visit(child, callback));
}

function notebookIdentifiers() {
  return (tree, file) => {
    const replacements = new Map();
    if (path.extname(file.path).toLowerCase() === '.ipynb') {
      visit(tree, (block) => {
        if (block.type !== 'block' || !/^\d+$/.test(block.html_id)) return;
        // MyST moves a captioned cell's identifier from its block to a container.
        const targets = [block, ...(block.children ?? []).filter((node) => node.type === 'container')];
        targets.forEach((node) => {
          if (node.label != null || node.identifier !== block.html_id) return;
          const identifier = cellIdentifier(file.path, node.identifier);
          replacements.set(node.identifier, identifier);
          node.identifier = identifier;
        });
      });
    }

    visit(tree, (node) => {
      if (node.type === 'crossReference' && replacements.has(node.identifier)) {
        node.identifier = replacements.get(node.identifier);
      }
      if (node.type === 'link' && node.url?.startsWith('#')) {
        const identifier = replacements.get(node.url.slice(1));
        if (identifier) node.url = `#${identifier}`;
      }
    });
  };
}

export default {
  name: 'notebook-cell-identifiers',
  transforms: [
    { name: 'notebook-cell-identifiers', stage: 'document', plugin: notebookIdentifiers },
  ],
};
