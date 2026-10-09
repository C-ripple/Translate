"""Logical launch regression tests; scalar harness does not emulate SIMD codegen."""
import re
import shutil
import subprocess

import pytest

from cuda2ripple import translate
from core.semantic_model import CUDADim3, TranslationContext, TranslationError
from core.translation_rules import infer_block_shape
from frontends.source.cuda_frontend import CUDAToRIPPLETransformer

VECADD = """__global__ void vecadd(float *x, float *y, float *z) {
    int i = 64 * blockIdx.x + threadIdx.x;
    z[i] = x[i] + y[i];
}"""

def test_explicit_launch_shape_is_preserved():
    assert infer_block_shape(VECADD, CUDADim3(64, 1, 1)).dimensions == [64, 1, 1]

def test_unknown_shape_is_never_inferred_from_hardware():
    with pytest.raises(TranslationError, match="shape"):
        infer_block_shape(VECADD)

def test_omitted_shape_requires_caller_block():
    ctx = TranslationContext()
    output = CUDAToRIPPLETransformer(ctx).transform(VECADD)
    assert any("caller" in w and "shape" in w for w in ctx.warnings)
    assert "ripple_set_block_shape(" not in output
    assert re.search(r"void vecadd_ripple_launch\(\s*ripple_block_t", output)

@pytest.mark.parametrize("shape", [(0,), (-1, 2), (1, 2, 3, 4), ("n",), (True,), (1.5,)])
def test_invalid_simd_shape_is_rejected(shape):
    with pytest.raises(TranslationError, match="shape"):
        translate(VECADD, block_shape=shape)

def test_unknown_kernel_in_shape_mapping_is_rejected():
    with pytest.raises(TranslationError, match="missing"):
        translate(VECADD, block_shape={"missing": (64,)})

def test_host_launch_syntax_is_rejected():
    with pytest.raises(TranslationError, match="launch"):
        translate(VECADD + " void host() { vecadd<<<2, 64>>>(x, y, z); }")

HEADER = """
#include <stddef.h>
#include <stdarg.h>
typedef struct { size_t dims[3]; } ripple_block_t;
extern _Thread_local size_t test_lane[3];
static ripple_block_t ripple_set_block_shape(int pe, ...) {
    va_list args; va_start(args, pe);
    ripple_block_t b;
    for (int d = 0; d < 3; ++d) b.dims[d] = (size_t)va_arg(args, int);
    va_end(args); return b;
}
static size_t ripple_id(ripple_block_t b, int d) { return test_lane[d]; }
static size_t ripple_get_block_size(ripple_block_t b, int d) { return b.dims[d]; }
"""

def run_generated(tmp_path, output, kernels, main, thread_header=None, record_blocks=False):
    """Execute generated grid loops; explicitly scalarize only SIMD invocation."""
    if not shutil.which("clang"):
        pytest.skip("clang not installed")
    for name, params, args in kernels:
        output = output.replace("void " + name + "_ripple(",
                                "void " + name + "_ripple_scalar(")
        record = "test_record(ripple_ctx.block_idx.x, ripple_ctx.block_idx.y, ripple_ctx.block_idx.z);" if record_blocks else ""
        shim = f"""
void {name}_ripple(ripple_block_t ripple_block,
                  ripple_launch_context_t ripple_ctx{params}) {{
    {record}
    for (test_lane[2] = 0; test_lane[2] < ripple_block.dims[2]; ++test_lane[2])
      for (test_lane[1] = 0; test_lane[1] < ripple_block.dims[1]; ++test_lane[1])
        for (test_lane[0] = 0; test_lane[0] < ripple_block.dims[0]; ++test_lane[0])
          {name}_ripple_scalar(ripple_block, ripple_ctx{args});
}}
"""
        output = output.replace("void " + name + "_ripple_launch(",
                                shim + "\nvoid " + name + "_ripple_launch(")
    (tmp_path / "ripple.h").write_text(HEADER)
    if thread_header is not None:
        (tmp_path / "ripple").mkdir()
        (tmp_path / "ripple/thread.h").write_text(thread_header)
    source = tmp_path / "test.c"
    source.write_text(output + "\n_Thread_local size_t test_lane[3];\n" + main)
    binary = tmp_path / "test"
    built = subprocess.run(["clang", "-std=c11", "-pthread", "-I" + str(tmp_path),
                            str(source), "-o", str(binary)],
                           capture_output=True, text=True, timeout=30)
    assert built.returncode == 0, built.stderr
    ran = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
    assert ran.returncode == 0, ran.stdout + ran.stderr

def test_vecadd_covers_all_64_threads_per_block(tmp_path):
    output = translate(VECADD, block_shape=(64,))
    run_generated(tmp_path, output,
                  [("vecadd", ", float *x, float *y, float *z", ", x, y, z")],
                  """
int main(void) {
    float x[128], y[128], z[128];
    for (int i=0; i<128; ++i) { x[i]=i; y[i]=2*i; z[i]=-1; }
    vecadd_ripple_launch((ripple_dim3_t){2,1,1}, x,y,z);
    for (int i=0; i<128; ++i) if (z[i] != 3*i) return 1;
    return 0;
}
""")

def test_independent_shapes_preserve_2d_and_3d_coordinates(tmp_path):
    source = """
__global__ void tile(int *out) {
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    out[gy * (gridDim.x * blockDim.x) + gx] += 1;
}
__global__ void volume(int *out) {
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    int gz = blockIdx.z * blockDim.z + threadIdx.z;
    out[(gz * gridDim.y * blockDim.y + gy) * gridDim.x * blockDim.x + gx] += 1;
}
"""
    output = translate(source, block_shape={"tile": (16,16), "volume": (8,4,2)})
    run_generated(tmp_path, output,
                  [("tile", ", int *out", ", out"), ("volume", ", int *out", ", out")],
                  """
int main(void) {
    int tile[1536] = {0}, volume[1536] = {0};
    tile_ripple_launch((ripple_dim3_t){2,3,1}, tile);
    volume_ripple_launch((ripple_dim3_t){2,3,4}, volume);
    for (int i=0; i<1536; ++i) if (tile[i] != 1 || volume[i] != 1) return 1;
    return 0;
}
""")

def test_generic_launcher_uses_supplied_shape(tmp_path):
    output = translate(VECADD)
    run_generated(tmp_path, output,
                  [("vecadd", ", float *x, float *y, float *z", ", x, y, z")],
                  """
int main(void) {
    float x[128], y[128], z[128];
    for (int i=0; i<128; ++i) { x[i]=i; y[i]=2*i; z[i]=-1; }
    ripple_block_t b = ripple_set_block_shape(0,64,1,1);
    vecadd_ripple_launch(b, (ripple_dim3_t){2,1,1}, x,y,z);
    for (int i=0; i<128; ++i) if (z[i] != 3*i) return 1;
    return 0;
}
""")

def test_cli_accepts_explicit_shape(tmp_path):
    import sys
    source = tmp_path / "vecadd.cu"
    output = tmp_path / "vecadd.c"
    source.write_text(VECADD)
    ran = subprocess.run([sys.executable, "-m", "interfaces.cli.cuda2ripple",
                          "source", str(source), "-o", str(output),
                          "--block-shape", "64,1,1"],
                         capture_output=True, text=True, timeout=10)
    assert ran.returncode == 0, ran.stdout + ran.stderr
    assert "ripple_set_block_shape(HVX_PE, 64, 1, 1)" in output.read_text()

@pytest.mark.parametrize("module, payload, output_key", [
    ("server", {"source": VECADD, "block_shape": [64]}, "translated"),
    ("interfaces.web.server", {"code": VECADD, "block_shape": [64]}, "output"),
])
def test_http_accepts_explicit_shape(module, payload, output_key):
    pytest.importorskip("flask")
    import importlib
    app = importlib.import_module(module).app
    response = app.test_client().post("/translate", json=payload)
    assert response.status_code == 200
    result = response.get_json()
    assert output_key in result, result
    assert "ripple_set_block_shape(HVX_PE, 64, 1, 1)" in result[output_key]

@pytest.mark.parametrize("module, source_key", [
    ("server", "source"), ("interfaces.web.server", "code"),
])
def test_http_rejects_dynamic_simd_shape(module, source_key):
    pytest.importorskip("flask")
    import importlib
    app = importlib.import_module(module).app
    result = app.test_client().post("/translate",
        json={source_key: VECADD, "block_shape": ["runtime_n"]}).get_json()
    assert "shape" in result.get("error", ""), result

def test_zero_argument_and_array_parameter_launchers_compile():
    from tests.compile_verify import verify_ripple_syntax
    if not shutil.which("clang"):
        pytest.skip("clang not installed")
    output = translate("__global__ void empty(void) {} "
                       "__global__ void array(float a[64]) { a[threadIdx.x] = 1; }",
                       block_shape=(64,))
    success, diagnostics = verify_ripple_syntax(output)
    assert success, diagnostics

def test_kernel_parameter_collision_is_diagnosed():
    with pytest.raises(TranslationError, match="conflicts"):
        translate("__global__ void k(int ripple_grid) {}", block_shape=(64,))
