<!--
Copyright (c) 2026 The TIRx Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
-->

# Optimized re-implementations

Native Kern kernels that re-implement an entry already in the registry with a
different schedule and are accepted only when they beat it. Each module keeps
the original kernel's functional contract (inputs, dtypes, tolerances, config
labels) so the two can be compared row for row, and its `run_bench` times the
original TIRx kernel as an in-repo reference (`tirx_<original>`) next to the
external one.

| Registry name | Re-implements | Schedule change |
| --- | --- | --- |
| `rmsnorm_opt` | `rmsnorm` (basic) | one global read per row: input and weight are held in registers across the reduction instead of being staged in shared memory or re-read; one CTA per row (or 16 rows per CTA for `hidden_size <= 256`) instead of a 152-CTA persistent loop; warp shuffle plus one shared-memory exchange instead of two barriers |

Acceptance for a kernel in this directory is the bench-suite ratio
`original_tirx_time / optimized_time > 1.0` on every config of the original
kernel, measured with `--with-references`.
