# Native SDK hardware-worker verification

The previous target worker driver ran logical workers sequentially. Add real
Hexagon SDK hardware threads, preserving the generated kernel output unchanged.
Use SDK thread_create/stop/join and lockMutex for an all-worker rendezvous, with
IDs taken from thread_get_tnum rather than fabricated per-worker IDs.

Configure 128-byte vector mode once before fork: set_double_vector_mode changes
global SYSCFG and belongs outside worker callbacks. SDK 19.0.07 release
publishes the allocator entry as free and unlocks it before clearing the old
owner's SSR permissions. Disassembly with relocations identifies unlockMutex
at offset 0x68 and the permission clear at 0x90-0xa0. Protect the complete
allocation/release transition with an outer handoff mutex. Use NO_WAIT
acquisition and retry outside the mutex so owners can release: waiting while
holding the handoff lock would deadlock. Each callback synchronizes stores,
releases its context under that mutex and stops. Kernel bodies execute outside
the handoff lock, so the mutex does not serialize them. The main hardware thread participates as worker zero and
joins all secondary threads. Normal driver loops disable automatic vectorization
so they cannot accidentally issue HVX instructions outside context ownership.

Test float and integer additive output over a 7x3x2 grid, shape64, 1/2/3/4/6
workers when supported, zero/partial/full lengths and both guard regions.
Require all participating workers to reach the rendezvous and report distinct
hardware IDs. Output must equal a single addition: duplicate or missing block
work fails. Independent simulator instruction counts must show execution by
every secondary hardware thread tested.

Retain red missing-query linker output and six-worker guard corruption from
concurrent global mode changes. The corrected mode setup must pass fresh native
execution. Keep separate report fields for actual SDK hardware concurrency and
QuRT/QHPI integration; standalone proof cannot certify either OS runtime.
