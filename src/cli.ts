#!/usr/bin/env node
/**
 * Entry point. Kept trivially small so that everything testable lives in
 * `src/cli/index.ts` and can be invoked in-process by the test suite.
 */

import { main } from './cli/index.ts';

const code = await main(process.argv.slice(2));
process.exit(code);
