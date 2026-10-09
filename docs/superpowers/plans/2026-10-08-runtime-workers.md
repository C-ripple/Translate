# Runtime workers implementation and verification

1. Add failing execution tests for real pthread scheduling with explicit/caller
   SIMD blocks, uneven grid ownership, excess workers, bad IDs/counts, out-of-rank queries and overflow.
2. Generate optional worker launchers using the official opaque Ripple ID/size
   query ABI, cyclic flattened 3D grid assignment and checked arithmetic.
3. Expose threaded selection through Python, CLI, HTTP and the Python web UI.
4. Compile unmodified output with native Ripple clang and run the Hexagon
   simulator against independent numerical references and guard regions.
5. Retain native failures and isolate compiler issues with integer and handwritten
   Ripple controls. Record vendor concurrency as unverified if its runtime is absent.
6. Run the full suite, document the launch contract and report remaining native
   limits without relaxing numerical or guard checks.
