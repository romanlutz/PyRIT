// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import { fileURLToPath } from 'node:url';
import { build, Session } from 'myst-cli';
import { enableRetries } from './retry.mjs';

process.chdir(process.argv[2] ?? fileURLToPath(new URL('../../doc/', import.meta.url)));
const session = enableRetries(new Session());
try {
  await session.reload();
  await build(session, [], { site: true, strict: true, checkLinks: true });
  // Match the upstream CLI: version-check timers must not keep a finished build alive.
  process.exit(0);
} catch (error) {
  session.log.error(error.message);
  process.exit(1);
}
