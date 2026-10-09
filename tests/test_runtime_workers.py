"""Run the generated scheduler with real POSIX workers and scalar SIMD lanes."""
import pytest
from cuda2ripple import translate
from tests.test_launch_shapes import run_generated

# Reference runtime: the opaque type and query ABI match the official compiler
# header; these queries implement a real pthread pool, not QuRT/QHPI.
THREAD_HEADER = """
#include <stddef.h>
typedef struct ripple_thread_block *ripple_thd_block_t;
size_t ripple_thd_id(ripple_thd_block_t, int);
size_t ripple_thd_get_block_size(ripple_thd_block_t, int);
void test_record(size_t, size_t, size_t);
"""
SOURCE = """__global__ void fill(int *out) {
    int block = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
    int i = block * blockDim.x + threadIdx.x;
    out[i] = i + 17;
}"""

@pytest.mark.parametrize("workers", [1, 2, 4, 7, 64])
@pytest.mark.parametrize("shape", [(64,), None])
def test_real_workers_cover_every_block_once(tmp_path, workers, shape):
    output = translate(SOURCE, block_shape=shape, threaded=True)
    launch_args = "" if shape is not None else "ripple_set_block_shape(0,64,1,1), "
    main = """
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
struct ripple_thread_block { size_t id, count; };
static _Thread_local size_t current_worker;
static atomic_uint block_hits[42], worker_hits[64];
static int out[2688];
size_t ripple_thd_id(ripple_thd_block_t b, int dim) { if (dim != 0) __builtin_trap(); return b->id; }
size_t ripple_thd_get_block_size(ripple_thd_block_t b, int dim) { if (dim != 0) __builtin_trap(); return b->count; }
void test_record(size_t x, size_t y, size_t z) {
    if (x >= 7 || y >= 3 || z >= 2) { fprintf(stderr, "bad coordinate"); return; }
    atomic_fetch_add(&block_hits[(z*3+y)*7+x], 1);
    atomic_fetch_add(&worker_hits[current_worker], 1);
}
static void *work(void *args) {
    ripple_thd_block_t b = args; current_worker = b->id;
    int status = fill_ripple_launch_workers(LAUNCH_ARGS b, (ripple_dim3_t){7,3,2}, out);
    return (void *)(size_t)status;
}
int main(void) {
    pthread_t threads[WORKERS];
    struct ripple_thread_block blocks[WORKERS];
    for (int i=0; i<2688; ++i) out[i]=-1;
    for (int w=0; w<WORKERS; ++w) {
        blocks[w]=(struct ripple_thread_block){w,WORKERS};
        if (pthread_create(&threads[w],0,work,&blocks[w])) return 1;
    }
    for (int w=0; w<WORKERS; ++w) {
        void *status; if (pthread_join(threads[w], &status) || status) return 2;
        unsigned expected = 42/WORKERS + (w < 42%WORKERS);
        if (atomic_load(&worker_hits[w]) != expected) return 3;
    }
    for (int b=0; b<42; ++b) if (atomic_load(&block_hits[b]) != 1) return 4;
    for (int i=0; i<2688; ++i) if (out[i] != i+17) return 5;
    return 0;
}
""".replace("LAUNCH_ARGS", launch_args).replace("WORKERS", str(workers))
    run_generated(tmp_path, output, [("fill", ", int *out", ", out")], main,
                  thread_header=THREAD_HEADER, record_blocks=True)

def test_worker_launcher_rejects_bad_runtime_and_grid_overflow(tmp_path):
    output = translate(SOURCE, block_shape=(64,), threaded=True)
    main = """
struct ripple_thread_block { size_t id, count; };
size_t ripple_thd_id(ripple_thd_block_t b, int dim) { if (dim != 0) __builtin_trap(); return b->id; }
size_t ripple_thd_get_block_size(ripple_thd_block_t b, int dim) { if (dim != 0) __builtin_trap(); return b->count; }
int main(void) {
    int out[64]; for (int i=0; i<64; ++i) out[i]=-1;
    struct ripple_thread_block b={0,0};
    if (fill_ripple_launch_workers(&b,(ripple_dim3_t){1,1,1},out) != CUDA2RIPPLE_BAD_WORKERS) return 1;
    b=(struct ripple_thread_block){4,4};
    if (fill_ripple_launch_workers(&b,(ripple_dim3_t){1,1,1},out) != CUDA2RIPPLE_BAD_WORKERS) return 2;
    b=(struct ripple_thread_block){0,4};
    if (fill_ripple_launch_workers(&b,(ripple_dim3_t){1,0,1},out) != CUDA2RIPPLE_LAUNCH_OK) return 3;
    if (fill_ripple_launch_workers(&b,(ripple_dim3_t){SIZE_MAX,SIZE_MAX,SIZE_MAX},out) != CUDA2RIPPLE_BAD_GRID) return 4;
    for (int i=0; i<64; ++i) if (out[i] != -1) return 5;
    return 0;
}
"""
    run_generated(tmp_path, output, [("fill", ", int *out", ", out")], main,
                  thread_header=THREAD_HEADER)

def test_threaded_option_is_source_only():
    with pytest.raises(ValueError, match="source"):
        translate("", mode="ir", threaded=True)

def test_cli_exposes_worker_launcher(tmp_path):
    import subprocess, sys
    source = tmp_path / "fill.cu"
    result = tmp_path / "fill.c"
    source.write_text(SOURCE)
    ran = subprocess.run([sys.executable,"-m","interfaces.cli.cuda2ripple","source",
                          str(source),"-o",str(result),"--block-shape","64","--threaded"],
                         capture_output=True,text=True,timeout=10)
    assert ran.returncode == 0, ran.stdout + ran.stderr
    assert "fill_ripple_launch_workers" in result.read_text()

@pytest.mark.parametrize("module,source_key,output_key", [
    ("server","source","translated"), ("interfaces.web.server","code","output"),
])
def test_http_exposes_worker_launcher(module,source_key,output_key):
    import importlib
    app = importlib.import_module(module).app
    result = app.test_client().post("/translate",
             json={source_key:SOURCE,"block_shape":[64],"threaded":True}).get_json()
    assert output_key in result, result
    assert "fill_ripple_launch_workers" in result[output_key]
