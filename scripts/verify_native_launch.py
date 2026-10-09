#!/usr/bin/env python3
"""Verify unmodified generated CUDA-to-Ripple code with real Hexagon tools.

Use the cached native compiler/SDK/linker/simulator adapters from a toolchain
directory. This verifies target SIMD, reference worker assignment and real SDK
standalone hardware-thread concurrency, not QuRT/QHPI integration. The official
Ripple query declarations bind either to the labeled reference SPMD driver or
to actual hardware thread IDs in the standalone concurrency driver.
"""
import argparse
import hashlib
import json
import os
import re
import signal
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cuda2ripple import translate

VECADD = """__global__ void vecadd(float *x, float *y, float *z, int n) {
    int i = 64 * blockIdx.x + threadIdx.x;
    if (i < n) z[i] = x[i] + y[i];
}"""
HARDWARE_WORKERS = """__global__ void hardware_add(float *x, float *y, float *z, int n) {
    int b = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
    int i = b * blockDim.x + threadIdx.x;
    if (i < n) z[i] += x[i] + y[i];
}"""
COORDS = """__global__ void coords(int *out) {
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    out[gy * gridDim.x * blockDim.x + gx] = gy * 1000 + gx;
}"""

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def run_tool(command, timeout, env=None):
    """Capture a tool result and stop its entire process group on timeout."""
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, env=env, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        def stop_group(sig):
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass
        stop_group(signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            stop_group(signal.SIGKILL)
            stdout, stderr = process.communicate()
        raise subprocess.TimeoutExpired(command, timeout, output=stdout, stderr=stderr)
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toolchain-root", type=Path, required=True,
                        help="Directory containing artifacts/ripple-build, hexagon-sdk and scripts/run_hexagon_*")
    parser.add_argument("--compiler", type=Path, help="Override clang without replacing the cached toolchain")
    parser.add_argument("--thread-header", type=Path,
                        help="Official Ripple thread.h, required to verify the worker launcher on target")
    parser.add_argument("--output", type=Path, default=ROOT / "output/verification")
    parser.add_argument("--compile-timeout", type=float, default=300,
                        help="Compilation deadline in seconds (debug builds can be slow)")
    parser.add_argument("--optimization", choices=["0", "1", "2", "3"], default="2")
    parser.add_argument("--float-mode", choices=["auto", "default", "disabled", "strict-ieee", "ieee"], default="auto")
    parser.add_argument("--case", choices=["vecadd64", "vecadd_int64", "vecadd_float_ieee64", "coords16x16", "workers_vecadd64", "workers_vecadd_int64", "sdk_hardware_workers_float", "sdk_hardware_workers_int", "float_tail_baseline"],
                        help="Run one case for compiler diagnostics")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "native-launch-verification.json"
    # Invalidate earlier success before touching tools or starting subprocesses.
    report_path.write_text(json.dumps({"passed": False, "state": "running"}) + "\n")
    report = {"passed": False, "errors": [], "cases": [],
              "scope": "real Hexagon SIMD, reference SPMD assignment and SDK standalone hardware-thread concurrency",
              "native_thread_concurrency_verified": False,
              "vendor_runtime_concurrency_verified": False,
              "vendor_runtime_limitation": "QuRT/QHPI thread runtime is absent from the available SDK"}
    try:
        base = args.toolchain_root.resolve()
        artifacts = base / "artifacts"
        clang = args.compiler.resolve() if args.compiler else artifacts / "ripple-build/bin/clang"
        sdk = artifacts / "hexagon-sdk"
        linker = base / "scripts/run_hexagon_linker.py"
        simulator = base / "scripts/run_hexagon_simulator.py"
        for p in (clang, sdk, linker, simulator):
            if not p.exists():
                raise RuntimeError(f"Native tool prerequisite missing: {p}")
        version = subprocess.run([str(clang), "--version"], capture_output=True, text=True, timeout=15)
        if version.returncode:
            raise RuntimeError(version.stdout + version.stderr)
        report["compiler"] = {"path": str(clang.resolve()), "sha256": digest(clang),
                              "version": version.stdout.strip()}
        # This verifier targets v68. Modern QFloat strict-ieee lowering runs
        # only on v79+, so accepting that flag does not establish v68 precision.
        mode = "ieee" if args.float_mode == "auto" else args.float_mode
        precision_flag = {"default": "-mhvx-qfloat", "disabled": "-mno-hvx-qfloat",
                          "strict-ieee": "-mhvx-qfloat=strict-ieee",
                          "ieee": "-mhvx-ieee-fp"}[mode]
        precision_flags = (["-mno-hvx-qfloat", precision_flag]
                           if mode == "ieee" else [precision_flag])
        report["floating_point_policy"] = {
            "flags": precision_flags,
            "reference": "exact IEEE values; no tolerance added",
            "reason": "v68 IEEE arithmetic requires HVX IEEE enabled and QFloat disabled; strict-ieee QFloat lowering is v79+"
        }
        report["sdk_manifest_sha256"] = digest(sdk / "extraction-manifest.json")
        report["simulator_image"] = json.loads((artifacts / "simulator-image.json").read_text())
        cases = [
            ("vecadd64", VECADD, (64,), False, r"""
#include <stdio.h>
int main(void) {
    _Alignas(128) float x[256], y[256], z[256];
    int sizes[]={0,1,31,32,33,63,64,65,127,128};
    for (unsigned t=0; t<sizeof(sizes)/sizeof(sizes[0]); ++t) {
        int n=sizes[t];
        for (int i=0; i<256; ++i) { x[i]=i; y[i]=2*i; z[i]=-999; }
        vecadd_ripple_launch((ripple_dim3_t){(n+63)/64,1,1}, x+32,y+32,z+32,n);
        for (int i=0; i<256; ++i) {
            float expected=(i>=32 && i<32+n) ? 3*i : -999;
            if (z[i]!=expected) {
                printf("vecadd mismatch n=%d i=%d actual=%f expected=%f\n",n,i,z[i],expected); return 1;
            }
        }
    }
    puts("CUDA2RIPPLE_NATIVE_PASS"); return 0;
}
"""),
            ("coords16x16", COORDS, (16,16), False, r"""
#include <stdio.h>
int main(void) {
    _Alignas(128) int out[1600];
    for (int i=0; i<1600; ++i) out[i]=-999;
    coords_ripple_launch((ripple_dim3_t){2,3,1}, out+32);
    for (int i=0; i<1600; ++i) {
        int local=i-32;
        int expected=(i>=32 && i<1568) ? (local/32)*1000+local%32 : -999;
        if (out[i]!=expected) { printf("coordinate mismatch i=%d\n",i); return 2; }
    }
    puts("CUDA2RIPPLE_NATIVE_PASS"); return 0;
}
"""),
        ]
        # Integer tail probe isolates predicated stores from floating-point lowering.
        name, source, shape, threaded, driver = cases[0]
        cases.insert(1, ("vecadd_int64", source.replace("float", "int"), shape, threaded,
                         driver.replace("float", "int").replace("%f", "%d")))

        cases.append(("vecadd_float_ieee64", VECADD, (64,), False, r"""
#include <stdio.h>
static float from_bits(uint32_t bits) { union { uint32_t bits; float value; } v={.bits=bits}; return v.value; }
static uint32_t to_bits(float value) { union { uint32_t bits; float value; } v={.value=value}; return v.bits; }
__attribute__((noinline,optnone))
static float scalar_reference(float x, float y) { return x+y; }
int main(void) {
    _Alignas(128) float x[256], y[256], z[256];
    const uint32_t pairs[][2]={
        {0,0x80000000}, {0x80000000,0x80000000}, {0x3f800000,0x33800000},
        {0x3f800001,0x33800000}, {0x7f7fffff,0x7f7fffff}, {1,1},
        {0x007fffff,1}, {0x00800000,0x807fffff}, {0x3f800000,0xbf800000},
        {0x7f800000,0xff800000}, {0x7fc12345,0x3f800000}, {0x7f812345,0},
        {0x3eaaaaab,0x3eaaaaab}, {0x80000001,0x80000001}
    };
    const int sizes[]={0,1,31,32,33,63,64,65,127,128};
    for (unsigned t=0; t<sizeof(sizes)/sizeof(sizes[0]); ++t) {
        int n=sizes[t];
        for (int i=0; i<256; ++i) {
            unsigned k=i%(sizeof(pairs)/sizeof(pairs[0]));
            x[i]=from_bits(pairs[k][0]); y[i]=from_bits(pairs[k][1]); z[i]=-999;
        }
        vecadd_ripple_launch((ripple_dim3_t){(n+63)/64,1,1},x+32,y+32,z+32,n);
        for (int i=0; i<256; ++i) {
            uint32_t expected=to_bits(i>=32 && i<32+n ? scalar_reference(x[i],y[i]) : -999);
            uint32_t actual=to_bits(z[i]);
            int expected_nan=(expected&0x7fffffff)>0x7f800000;
            int actual_nan=(actual&0x7fffffff)>0x7f800000;
            if (expected_nan ? !actual_nan : actual!=expected) {
                printf("float edge mismatch n=%d i=%d actual=%08x expected=%08x\n",n,i,(unsigned)actual,(unsigned)expected); return 5;
            }
        }
    }
    puts("CUDA2RIPPLE_NATIVE_PASS"); return 0;
}
"""))

        if args.thread_header:
            header = args.thread_header.resolve()
            report["thread_header"] = {"path": str(header), "sha256": digest(header)}
            cases.append(("workers_vecadd64", VECADD, (64,), True, r"""
#include <stdio.h>
struct ripple_thread_block { size_t id, count; };
size_t ripple_thd_id(ripple_thd_block_t b, int dim) { if (dim != 0) __builtin_trap(); return b->id; }
size_t ripple_thd_get_block_size(ripple_thd_block_t b, int dim) { if (dim != 0) __builtin_trap(); return b->count; }
int main(void) {
    _Alignas(128) float x[512], y[512], z[512];
    const unsigned counts[]={1,2,4,7,16};
    const int sizes[]={0,1,31,32,33,63,64,65,127,128,447,448};
    for (unsigned t=0; t<sizeof(counts)/sizeof(counts[0]); ++t) {
        for (unsigned k=0; k<sizeof(sizes)/sizeof(sizes[0]); ++k) {
            int n=sizes[k];
            for (int i=0; i<512; ++i) { x[i]=i; y[i]=2*i; z[i]=-999; }
            for (unsigned w=0; w<counts[t]; ++w) {
                struct ripple_thread_block b={w,counts[t]};
                if (vecadd_ripple_launch_workers(&b,(ripple_dim3_t){(n+63)/64,1,1},x+32,y+32,z+32,n)) return 3;
            }
            for (int i=0; i<512; ++i) {
                float expected=(i>=32 && i<32+n) ? 3*i : -999;
                if (z[i]!=expected) { printf("worker mismatch count=%u n=%d i=%d\n",counts[t],n,i); return 4; }
            }
        }
    }
    puts("CUDA2RIPPLE_NATIVE_PASS"); return 0;
}
"""))
            name, source, shape, threaded, driver = cases[-1]
            cases.append(("workers_vecadd_int64", source.replace("float", "int"), shape, threaded,
                          driver.replace("float", "int")))
            driver = (ROOT / "tests/native/standalone_workers.c").read_text()
            cases.append(("sdk_hardware_workers_float", HARDWARE_WORKERS, (64,), True, driver))
            cases.append(("sdk_hardware_workers_int", HARDWARE_WORKERS.replace("float", "int"),
                          (64,), True, driver.replace("float", "int")))
        if args.case == "float_tail_baseline":
            # Independent handwritten Ripple baseline, bypassing the translator.
            baseline = """#include <ripple.h>
#include <stddef.h>
typedef struct { size_t x, y, z; } ripple_dim3_t;
void vecadd_ripple_launch(ripple_dim3_t grid, float *x, float *y, float *z, int n) {
    ripple_block_t block = ripple_set_block_shape(0,64,1,1);
    for (size_t b=0; b<grid.x; ++b) {
        int i = 64*b + ripple_id(block,0);
        if (i<n) z[i] = x[i]+y[i];
    }
}
"""
            cases.append(("float_tail_baseline", baseline, (64,), False, cases[0][4]))
        if args.case:
            cases = [case for case in cases if case[0] == args.case]
            if not cases:
                raise RuntimeError("Worker case requires --thread-header")
        for name, source, shape, threaded, driver in cases:
            hardware_threads = name.startswith("sdk_hardware_workers_")
            runtime = ("SDK standalone hardware threads" if hardware_threads
                       else "reference SPMD queries" if threaded else "standalone")
            case = {"name": name, "passed": False, "runtime": runtime,
                    "native_simd": True, "hardware_thread_concurrency": hardware_threads}
            report["cases"].append(case)
            for suffix in ("compile.log", "run.log"):
                (args.output / f"{name}.{suffix}").unlink(missing_ok=True)
            with tempfile.TemporaryDirectory(prefix="ripple-target-cuda-", dir=artifacts) as temporary:
                tmp = Path(temporary)
                if threaded:
                    (tmp / "ripple").mkdir()
                    (tmp / "ripple/thread.h").write_bytes(args.thread_header.read_bytes())
                generated = (source if name == "float_tail_baseline"
                             else translate(source, block_shape=shape, threaded=threaded))
                case["translator_used"] = name != "float_tail_baseline"
                src = tmp / "probe.c"
                src.write_text(generated + "\n" + driver)
                (args.output / f"{name}.c").write_text(src.read_text())
                case["source_sha256"] = digest(src)
                binary = tmp / f"{name}.elf"
                command = [str(clang), "--target=hexagon-unknown-unknown-elf", "-fenable-ripple",
                           "-O" + args.optimization, "-mv68", "-mhvx", "-mhvx-length=128B", *precision_flags, "-G0", "-std=c11",
                           "-I", str(tmp), str(src), "-B" + str(sdk / "target"),
                           "-fuse-ld=" + str(linker), "-o", str(binary)]
                if hardware_threads:
                    command[1:1] = ["-fno-vectorize", "-fno-slp-vectorize"]
                case["compile_command"] = command
                environment = os.environ.copy()
                environment["TMPDIR"] = str(tmp)
                log = args.output / f"{name}.compile.log"
                try:
                    result = run_tool(command, timeout=args.compile_timeout, env=environment)
                except subprocess.TimeoutExpired as error:
                    log.write_text("Compilation timed out; process group stopped.\n" +
                                   (error.stdout or "") + (error.stderr or ""))
                    report["errors"].append(f"{name}: compilation timed out; see {log}")
                    continue
                log.write_text(result.stdout + result.stderr)
                if result.returncode:
                    report["errors"].append(f"{name}: native compilation failed; see {log}")
                    continue
                case["elf_sha256"] = digest(binary)
                command = [str(simulator), "--simulated_returnval", "--", str(binary)]
                case["run_command"] = command
                log = args.output / f"{name}.run.log"
                try:
                    result = run_tool(command, timeout=70)
                except subprocess.TimeoutExpired as error:
                    log.write_text("Execution timed out; process group stopped.\n" +
                                   (error.stdout or "") + (error.stderr or ""))
                    report["errors"].append(f"{name}: execution timed out; see {log}")
                    continue
                log.write_text(result.stdout + result.stderr)
                case["passed"] = result.returncode == 0 and "CUDA2RIPPLE_NATIVE_PASS" in result.stdout
                if hardware_threads:
                    # Independent simulator statistics must show another actual
                    # hardware thread executing, not just a printed success marker.
                    active = [int(t) for t, insns in re.findall(r"T(\d+): Insns=(\d+)", result.stdout + result.stderr)
                              if int(t)>0 and int(insns)>0]
                    counts = [int(n) for n in re.findall(
                        r"CUDA2RIPPLE_HARDWARE_THREADS workers=(\d+)", result.stdout)]
                    case["active_secondary_hardware_threads"] = active
                    case["tested_worker_counts"] = counts
                    required = set(range(1, max(counts, default=1)))
                    case["passed"] = (case["passed"] and 2 in counts
                                      and required.issubset(active)
                                      and "CUDA2RIPPLE_HARDWARE_THREADS workers=2 mask=3" in result.stdout)
                if not case["passed"]:
                    report["errors"].append(f"{name}: native execution failed; see {log}")
        hardware_cases = [c for c in report["cases"] if c["hardware_thread_concurrency"]]
        report["native_thread_concurrency_verified"] = bool(hardware_cases) and all(
            c["passed"] for c in hardware_cases)
        report["passed"] = bool(report["cases"]) and all(c["passed"] for c in report["cases"])
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as e:
        report["errors"].append(str(e))
    finally:
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print("PASS" if report["passed"] else "FAILED", report_path)
        for error in report["errors"]:
            print(error)
        print("SDK hardware-thread concurrency:",
              "verified" if report["native_thread_concurrency_verified"] else "not verified")
        print("QuRT/QHPI integration:", "not verified")
    return 0 if report["passed"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
