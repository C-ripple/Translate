/* Execute unmodified generated SIMD kernels on real SDK hardware threads.
 * The standalone backend is distinct from QuRT and QHPI. */
#include <hexagon_standalone.h>
#include <stdlib.h>
#include <stdio.h>

enum { MAX_THREADS=6, STACK_BYTES=32768, ELEMENTS=2688, STORAGE=2752 };
struct ripple_thread_block { size_t count; };
/* Bind Ripple's opaque queries to the SDK's actual hardware-thread register.
 * This test backend has one dimension and immutable count for each fork/join. */
size_t ripple_thd_id(ripple_thd_block_t b, int dim) {
    if (!b || dim!=0) __builtin_trap();
    return thread_get_tnum();
}
size_t ripple_thd_get_block_size(ripple_thd_block_t b, int dim) {
    if (!b || dim!=0) __builtin_trap();
    return b->count;
}
static struct ripple_thread_block block;
static _Alignas(128) float x[STORAGE], y[STORAGE], z[STORAGE];
static _Alignas(128) unsigned char stacks[MAX_THREADS][STACK_BYTES];
static int mutex, hvx_mutex;
static volatile unsigned arrived, seen;
static volatile int statuses[MAX_THREADS];
static int length;
static ripple_dim3_t grid;

static void worker(void *unused) {
    unsigned id=thread_get_tnum();
    if (id>=block.count) __builtin_trap();
    lockMutex(&mutex);
    seen |= 1u<<id;
    ++arrived;
    unlockMutex(&mutex);
    /* A rendezvous requires all hardware workers to be alive together. */
    while (arrived!=block.count) __asm__ volatile("nop");
    /* SDK 19.0.07 publishes a free context before disabling the old owner's
     * hardware permission. Serialize only allocation/release, never kernels.
     * Nonblocking acquisition avoids holding this lock while owners must free. */
    for (;;) {
        lockMutex(&hvx_mutex);
        int acquired=acquire_vector_unit(HEXAGON_VECTOR_NO_WAIT);
        unlockMutex(&hvx_mutex);
        if (acquired) break;
    }
    statuses[id]=hardware_add_ripple_launch_workers(&block,grid,x+32,y+32,z+32,length);
    __asm__ volatile("syncht" ::: "memory");
    lockMutex(&hvx_mutex);
    release_vector_unit();
    unlockMutex(&hvx_mutex);
    if (id) thread_stop();
}

int main(void) {
    unsigned available=get_thread_count();
    if (thread_get_tnum()!=0 || available<2) return 6;
    const unsigned counts[]={1,2,3,4,6};
    const int sizes[]={0,1,31,32,33,63,64,65,127,128,2687,2688};
    /* Initial startup may own an HVX unit. Acquire/release explicitly per worker.
     * Normal driver loops are compiled with automatic vectorization disabled. */
    /* This changes global SYSCFG: configure once before any workers run. */
    set_double_vector_mode();
    release_vector_unit();
    printf("SDK hardware threads=%u free_hvx=%d\n",available,acquire_vector_unit(HEXAGON_VECTOR_CHECK));
    for (unsigned c=0;c<sizeof(counts)/sizeof(counts[0]);++c) {
        unsigned count=counts[c];
        if (count>available) continue;
        for (unsigned s=0;s<sizeof(sizes)/sizeof(sizes[0]);++s) {
            length=sizes[s]; block.count=count;
            grid=(ripple_dim3_t){length?7:0,3,2};
            mutex=hvx_mutex=0; arrived=seen=0;
            for (int i=0;i<STORAGE;++i) {
                x[i]=i+1; y[i]=2*i+1;
                z[i]=(i>=32 && i<32+length)?0:-999;
            }
            for (unsigned id=0;id<count;++id) statuses[id]=-99;
            for (unsigned id=1;id<count;++id)
                thread_create(worker,stacks[id]+STACK_BYTES,id,0);
            worker(0);
            thread_join(((1u<<count)-1u)&~1u);
            if (arrived!=count || seen!=((1u<<count)-1u)) return 7;
            for (unsigned id=0;id<count;++id) if (statuses[id]) return 8;
            for (int i=0;i<STORAGE;++i) {
                float expected=(i>=32 && i<32+length)?3*i+2:-999;
                if (z[i]!=expected) {
                    printf("hardware worker mismatch workers=%u n=%d i=%d actual=%f expected=%f\n",count,length,i,(double)z[i],(double)expected);
                    return 9;
                }
            }
        }
        printf("CUDA2RIPPLE_HARDWARE_THREADS workers=%u mask=%x\n",count,seen);
    }
    puts("CUDA2RIPPLE_NATIVE_PASS"); return 0;
}
