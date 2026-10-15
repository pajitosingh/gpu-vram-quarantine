// vrampill2 - vrampill for a card whose VRAM fault is already known: fast start, small holder.
//
// Based on vrampill from JoshBolding/gpu-vram-quarantine (commit c5b6105): allocate the whole card, sweep it with
// test patterns, keep the allocations that fail and free the rest. A held allocation cannot be handed to any other
// process, so everything else on the card is safe to use. Written for an RTX 3090 with one leaky bit; see
// docs/vrampill2.md for the case and the measurements.
//
// That cell is a retention fault: it holds a 1 for a few ms, then reads 0. Read back at once it passes; after
// 30 ms or more it fails every time. Two things follow, both easy to get wrong:
// - The search waits --dwell-ms (50) between filling the card and checking it. Without that, a word waits only as
//   long as a pass takes (~31 ms on a 3090) and a cell on the edge passes some of the time.
// - The held pieces fit in L2 (6 MiB on a 3090). A re-check that fills and reads them back is served from the
//   cache and never touches DRAM, so it reports the piece clean. The watchdog pushes them out first and waits.
//
// - Memory comes from the CUDA virtual memory API (cuMemCreate) in pieces of the card's allocation granularity
//   (2 MiB on Ampere), all mapped into one contiguous range. One kernel launch sweeps the whole card per pass, and
//   the held piece is 2 MiB instead of 8. Each piece is an explicit physical allocation handle.
// - --expect N stops the search as soon as exactly N pieces are confirmed and nothing else has failed, instead of
//   waiting --quiet-seconds. The patterns start with all ones, all zeros, 0xAAAAAAAA and 0x55555555, so a stuck
//   or leaky bit of either polarity fails at least twice in the first 4 passes. Anything beyond the expected count
//   falls back to the stock quiet-period rule.
// - --uuid picks the card by UUID, so moving the card to another slot needs no change.
// - --hold fd exports the faulty pieces as file descriptors and then destroys the CUDA context; the descriptors
//   alone keep the pieces allocated. Before reporting ready, it re-opens them through the descriptors in a fresh
//   context and checks the fault is still inside; if not, it restarts itself with --hold context. SIGUSR1
//   repeats that check on demand. No periodic polling: each check costs a context for about a second.
// - --hold context (default) keeps the context and re-checks the held pieces every --verify-every seconds with
//   32 passes of alternating ones and zeros, pushed out of L2 to DRAM before each read. A re-check with no
//   error at all is a warning.
//
// Build: make vrampill2   (or: nvcc -O3 -arch=sm_86 -o vrampill2 src/vrampill2.cu -lcuda)
//
// Exit codes: 0 active, 2 CUDA error, 3 insufficient VRAM, 4 no fault found (NOT a pass), 5 too many bad pieces,
//             6 no GPU with the --uuid given (nothing touched; tells systemd READY first, so a unit with
//               SuccessExitStatus=6 just goes inactive).

#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <strings.h>
#include <unistd.h>
#include <csignal>
#include <sys/socket.h>
#include <sys/un.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CK(x) do {                                                            \
    cudaError_t _e = (x);                                                     \
    if (_e != cudaSuccess) {                                                  \
        printf("[CUDA ERROR] %s:%d %s -> %s\n", __FILE__, __LINE__, #x,       \
               cudaGetErrorString(_e));                                       \
        fflush(stdout);                                                       \
        return 2;                                                             \
    }                                                                         \
} while (0)

#define CU(x) do {                                                            \
    CUresult _r = (x);                                                        \
    if (_r != CUDA_SUCCESS) {                                                 \
        const char* _s = NULL;                                                \
        cuGetErrorString(_r, &_s);                                            \
        printf("[CUDA ERROR] %s:%d %s -> %s\n", __FILE__, __LINE__, #x,       \
               _s ? _s : "?");                                                \
        fflush(stdout);                                                       \
        return 2;                                                             \
    }                                                                         \
} while (0)

#define NO_CAND 0xFFFFFFFFFFFFFFFFULL

__device__ __forceinline__
unsigned int genval(unsigned long long gidx, unsigned int pat, int mode)
{
    if (mode == 1) return (unsigned int)(gidx * 2654435761ULL) ^ pat;
    if (mode == 2) return (gidx & 1ULL) ? ~pat : pat;
    return pat;
}

__global__ void kfill(unsigned int* p, unsigned long long n,
                      unsigned long long base, unsigned int pat, int mode)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (; i < n; i += stride) p[i] = genval(base + i, pat, mode);
}

// Same bookkeeping as the stock kcheck, but one launch covers the whole card: the piece comes from the element
// index. The first error in a piece nominates a candidate offset; a piece is confirmed once that same offset has
// failed --confirm-hits times.
__global__ void kcheck(const unsigned int* p, unsigned long long n, unsigned long long base,
                       unsigned long long epp, unsigned int pat, int mode,
                       unsigned int* hits, unsigned long long* idx, unsigned int* xr,
                       unsigned int* same, unsigned int* mism)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        unsigned long long g = base + i;
        unsigned int want = genval(g, pat, mode);
        unsigned int got  = p[i];
        if (got != want) {
            unsigned long long c = g / epp, li = g - c * epp;
            atomicAdd(&hits[c], 1u);
            unsigned long long prev = atomicCAS(&idx[c], NO_CAND, li);
            if (prev == NO_CAND || prev == li) {
                atomicAdd(&same[c], 1u);
                atomicOr(&xr[c], want ^ got);
            } else {
                atomicAdd(&mism[c], 1u);
            }
        }
    }
}

__global__ void kverify(const unsigned int* p, unsigned long long n,
                        unsigned long long base, unsigned int pat, int mode,
                        unsigned int* nerr)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (; i < n; i += stride)
        if (p[i] != genval(base + i, pat, mode)) atomicAdd(nerr, 1u);
}

static volatile sig_atomic_t g_stop = 0;
static volatile sig_atomic_t g_check = 0;   // SIGUSR1: re-check the quarantine now
static void on_sig(int) { g_stop = 1; }
static void on_usr1(int) { g_check = 1; }

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

// systemd readiness (Type=notify) without linking libsystemd: one datagram to $NOTIFY_SOCKET, if set.
static void sd_notify_msg(const char* msg)
{
    const char* path = getenv("NOTIFY_SOCKET");
    if (!path || !path[0] || strlen(path) >= sizeof(((struct sockaddr_un*)0)->sun_path)) return;
    struct sockaddr_un sa;
    memset(&sa, 0, sizeof(sa));
    sa.sun_family = AF_UNIX;
    strcpy(sa.sun_path, path);
    if (sa.sun_path[0] == '@') sa.sun_path[0] = '\0';   // abstract namespace
    int s = socket(AF_UNIX, SOCK_DGRAM | SOCK_CLOEXEC, 0);
    if (s < 0) return;
    sendto(s, msg, strlen(msg), 0, (struct sockaddr*)&sa,
           (socklen_t)(offsetof(struct sockaddr_un, sun_path) + strlen(path)));
    close(s);
}

// Re-open the pieces held by file descriptor in a fresh context and count errors in them, with the same L2
// push-out and delay as the context-mode watchdog. Proves that the descriptors still reference the defective
// memory after the original context is gone. Costs a CUDA context (~130 MiB on a 3090) for about a second.
static int fd_verify_body(int dev, const int* fds, int nfd, size_t gran, int blocks, int threads,
                          size_t scr_elems, int delay_ms, int passes, unsigned int* errs)
{
    CK(cudaSetDevice(dev));
    CK(cudaDeviceSetLimit(cudaLimitStackSize, 0));
    CK(cudaFree(0));
    CUdevice cudev;
    CU(cuDeviceGet(&cudev, dev));
    CUdeviceptr va = 0;
    CU(cuMemAddressReserve(&va, nfd * gran, 0, 0, 0));
    for (int i = 0; i < nfd; i++) {
        CUmemGenericAllocationHandle hh;
        CU(cuMemImportFromShareableHandle(&hh, (void*)(uintptr_t)fds[i], CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
        CUresult r = cuMemMap(va + i * gran, gran, 0, hh, 0);
        cuMemRelease(hh);   // the mapping, and then the descriptor, keep the piece
        CU(r);
    }
    CUmemAccessDesc ad;
    memset(&ad, 0, sizeof(ad));
    ad.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    ad.location.id   = cudev;
    ad.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CU(cuMemSetAccess(va, nfd * gran, &ad, 1));

    unsigned int *scr = NULL, *d_err = NULL;
    if (scr_elems) CK(cudaMalloc(&scr, scr_elems * 4ULL));
    CK(cudaMalloc(&d_err, sizeof(unsigned int)));
    CK(cudaMemset(d_err, 0, sizeof(unsigned int)));
    unsigned int* d = (unsigned int*)va;
    unsigned long long n = nfd * (gran / 4ULL);
    for (int it = 0; it < passes; it++) {
        unsigned int pat = (it & 1) ? 0x00000000u : 0xFFFFFFFFu;
        kfill<<<blocks, threads>>>(d, n, 0, pat, 0);
        if (scr_elems) kfill<<<blocks, threads>>>(scr, scr_elems, 0, (unsigned int)it, 1);
        CK(cudaDeviceSynchronize());
        if (delay_ms > 0) usleep((useconds_t)delay_ms * 1000);
        kverify<<<blocks, threads>>>(d, n, 0, pat, 0, d_err);
    }
    CK(cudaMemcpy(errs, d_err, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CU(cuMemUnmap(va, nfd * gran));
    CU(cuMemAddressFree(va, nfd * gran));
    return 0;
}

// Returns 0 if the check ran (errors in *errs), nonzero if it could not run. Never leaves a context behind.
static int fd_verify(int dev, const int* fds, int nfd, size_t gran, int blocks, int threads,
                     size_t scr_elems, int delay_ms, int passes, unsigned int* errs)
{
    *errs = 0;
    int rc = fd_verify_body(dev, fds, nfd, gran, blocks, threads, scr_elems, delay_ms, passes, errs);
    cudaDeviceReset();
    return rc;
}

static void uuid_str(const CUuuid& u, char* out)
{
    const unsigned char* b = (const unsigned char*)u.bytes;
    sprintf(out, "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
            b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]);
}

int main(int argc, char** argv)
{
    int    dev          = 0;
    const char* want_uuid = NULL;
    int    expect       = 0;       // 0: stock behaviour (stop after --quiet-seconds with nothing new)
    int    min_passes   = 4;
    double find_secs    = 1800.0;
    double quiet_secs   = 180.0;
    int    confirm_hits = 2;
    int    max_bad      = 64;
    size_t reserve_mib  = 32;      // left free while searching
    int    hold_fd      = 0;
    double verify_every = 300.0;
    int    verify_delay_ms = 30;   // between pushing the held pieces to DRAM and reading them back
    int    dwell_ms     = 50;      // search: wait between filling the card and checking it
    int    flush_mib    = -1;      // re-check: scratch written to push held pieces out of L2 (-1: 4x L2)
    int    allow_clean  = 0;
    const char* ready_file = NULL;
    const char* alert_cmd  = NULL;   // run (via sh) when a re-check says the quarantine no longer holds

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--device") && i + 1 < argc) dev = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--uuid") && i + 1 < argc) want_uuid = argv[++i];
        else if (!strcmp(argv[i], "--expect") && i + 1 < argc) expect = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--min-passes") && i + 1 < argc) min_passes = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dwell-ms") && i + 1 < argc) dwell_ms = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--find-seconds") && i + 1 < argc) find_secs = atof(argv[++i]);
        else if (!strcmp(argv[i], "--quiet-seconds") && i + 1 < argc) quiet_secs = atof(argv[++i]);
        else if (!strcmp(argv[i], "--confirm-hits") && i + 1 < argc) confirm_hits = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--max-bad") && i + 1 < argc) max_bad = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reserve-mib") && i + 1 < argc) reserve_mib = (size_t)atoll(argv[++i]);
        else if (!strcmp(argv[i], "--hold") && i + 1 < argc) {
            const char* m = argv[++i];
            if (!strcmp(m, "fd")) hold_fd = 1;
            else if (!strcmp(m, "context")) hold_fd = 0;
            else { printf("--hold takes context or fd, not %s\n", m); return 1; }
        }
        else if (!strcmp(argv[i], "--verify-every") && i + 1 < argc) verify_every = atof(argv[++i]);
        else if (!strcmp(argv[i], "--verify-delay-ms") && i + 1 < argc) verify_delay_ms = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--verify-flush-mib") && i + 1 < argc) flush_mib = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--allow-clean")) allow_clean = 1;
        else if (!strcmp(argv[i], "--ready-file") && i + 1 < argc) ready_file = argv[++i];
        else if (!strcmp(argv[i], "--alert-cmd") && i + 1 < argc) alert_cmd = argv[++i];
        else {
            // Reject rather than ignore, as the stock tool does: a dropped flag here means silent corruption.
            if (strcmp(argv[i], "--help") && strcmp(argv[i], "-h"))
                printf("unrecognised or incomplete argument: %s\n\n", argv[i]);
            printf(
                "vrampill2 - quarantine defective VRAM so the rest of the card stays usable\n\n"
                "  --uuid GPU-...      pick the card by UUID (overrides --device)\n"
                "  --device N          CUDA device index (default 0)\n"
                "  --expect N          stop as soon as exactly N pieces are confirmed (default 0: off)\n"
                "  --min-passes N      sweep at least this many passes (default 4)\n"
                "  --dwell-ms N        wait between fill and check in each search pass (default 50)\n"
                "  --find-seconds N    hard ceiling on the search (default 1800)\n"
                "  --quiet-seconds N   without --expect, stop after this long with nothing new (default 180)\n"
                "  --confirm-hits N    repeats at one offset before confirming (default 2)\n"
                "  --max-bad N         refuse to continue past this many bad pieces (default 64)\n"
                "  --reserve-mib N     VRAM left free during the search (default 32)\n"
                "  --hold context|fd   keep the CUDA context (default), or only file descriptors\n"
                "  --verify-every N    context mode: re-check held pieces every N s, 0 disables (default 300)\n"
                "  --verify-delay-ms N wait between write-back and read in a re-check (default 30)\n"
                "  --verify-flush-mib N scratch written to push held pieces out of L2, 0 disables (default 4x L2)\n"
                "  --allow-clean       exit 0 instead of 4 when no fault is found\n"
                "  --ready-file PATH   write a marker once the quarantine is active\n"
                "  --alert-cmd CMD     run CMD (sh -c) if a re-check finds the fault gone from the held pieces\n\n"
                "SIGUSR1 re-checks the quarantine now (fd mode: costs a CUDA context for about a second).\n\n"
                "Exit codes: 0 active, 2 CUDA error, 3 insufficient VRAM,\n"
                "            4 no fault found (NOT a pass), 5 too many bad pieces,\n"
                "            6 no GPU with the --uuid given (nothing touched)\n");
            return (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) ? 0 : 1;
        }
    }
    if (ready_file) unlink(ready_file);

    signal(SIGINT, on_sig);
    signal(SIGTERM, on_sig);
    signal(SIGUSR1, on_usr1);

    double T0 = now_s();
    // With --uuid, let the driver see only that card. Otherwise cuInit opens every GPU and the holder keeps a few
    // MiB of driver state on each of them for as long as it runs.
    if (want_uuid) setenv("CUDA_VISIBLE_DEVICES", want_uuid, 1);
    {
        CUresult r = cuInit(0);
        if (r == CUDA_ERROR_NO_DEVICE && want_uuid) {
            printf("no visible GPU has UUID %s; nothing to hold, no card touched\n", want_uuid);
            sd_notify_msg("READY=1\nSTATUS=card not present; nothing to hold");
            return 6;
        }
        CU(r);
    }
    char ustr[48] = "";
    if (want_uuid) {
        int ndev = 0;
        CU(cuDeviceGetCount(&ndev));
        dev = -1;
        for (int d = 0; d < ndev && dev < 0; d++) {
            CUdevice cd;
            CUuuid u;
            CU(cuDeviceGet(&cd, d));
            CU(cuDeviceGetUuid(&u, cd));
            uuid_str(u, ustr);
            if (!strcasecmp(ustr, want_uuid)) dev = d;
        }
        // Never fall back to another card: no context is created and nothing is touched.
        if (dev < 0) {
            printf("no visible GPU has UUID %s; nothing to hold, no card touched\n", want_uuid);
            sd_notify_msg("READY=1\nSTATUS=card not present; nothing to hold");
            return 6;
        }
    }
    CUdevice cudev;
    CU(cuDeviceGet(&cudev, dev));
    { CUuuid u; CU(cuDeviceGetUuid(&u, cudev)); uuid_str(u, ustr); }

    CK(cudaSetDevice(dev));
    // Stack 0 as in the stock tool (no kernel here uses one; the default reservation is ~124 MiB on a 3090).
    // The printf FIFO and device malloc heap are not used either; a refusal to shrink them is harmless.
    CK(cudaDeviceSetLimit(cudaLimitStackSize, 0));
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 0);
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 0);
    cudaGetLastError();
    CK(cudaFree(0));
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, dev));

    CUmemAllocationProp ap;
    memset(&ap, 0, sizeof(ap));
    ap.type          = CU_MEM_ALLOCATION_TYPE_PINNED;
    ap.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    ap.location.id   = cudev;
    if (hold_fd) ap.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    size_t gran = 0;
    CU(cuMemGetAllocationGranularity(&gran, &ap, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    unsigned long long epp = gran / 4ULL;   // 32-bit elements per piece

    size_t freeB = 0, totalB = 0;
    CK(cudaMemGetInfo(&freeB, &totalB));
    size_t maxp = freeB / gran;

    // Counters, and the kernels loaded, before the big allocation, so the margin left free can stay small.
    unsigned int       *d_hits, *d_xor, *d_same, *d_mism, *d_verr;
    unsigned long long *d_idx;
    CK(cudaMalloc(&d_hits, sizeof(unsigned int) * maxp));
    CK(cudaMalloc(&d_xor,  sizeof(unsigned int) * maxp));
    CK(cudaMalloc(&d_same, sizeof(unsigned int) * maxp));
    CK(cudaMalloc(&d_mism, sizeof(unsigned int) * maxp));
    CK(cudaMalloc(&d_idx,  sizeof(unsigned long long) * maxp));
    CK(cudaMalloc(&d_verr, sizeof(unsigned int)));
    CK(cudaMemset(d_hits, 0, sizeof(unsigned int) * maxp));
    CK(cudaMemset(d_xor,  0, sizeof(unsigned int) * maxp));
    CK(cudaMemset(d_same, 0, sizeof(unsigned int) * maxp));
    CK(cudaMemset(d_mism, 0, sizeof(unsigned int) * maxp));
    CK(cudaMemset(d_idx, 0xFF, sizeof(unsigned long long) * maxp));
    kfill<<<1, 32>>>(d_hits, 0, 0, 0, 0);
    kcheck<<<1, 32>>>(d_hits, 0, 0, epp, 0, 0, d_hits, d_idx, d_xor, d_same, d_mism);
    kverify<<<1, 32>>>(d_hits, 0, 0, 0, 0, d_verr);
    CK(cudaDeviceSynchronize());
    CK(cudaMemGetInfo(&freeB, &totalB));
    double t_init = now_s() - T0;

    printf("=== vrampill2 ===\n");
    printf("device       : %d (%s, %s)\n", dev, prop.name, ustr);
    printf("total VRAM   : %.2f GiB, %.0f MiB in use before the search (this context + anyone else)\n",
           totalB / 1073741824.0, (totalB - freeB) / 1048576.0);
    printf("piece size   : %.0f MiB (allocation granularity)\n", gran / 1048576.0);
    if (expect > 0)
        printf("search       : stop once exactly %d piece(s) confirmed (min %d passes, max %.0fs)\n",
               expect, min_passes, find_secs);
    else
        printf("search       : up to %.0fs, stop after %.0fs with nothing new\n", find_secs, quiet_secs);
    fflush(stdout);

    // Refuse before allocating anything if the card is already in use. The search would miss whatever the other
    // process holds, and taking the rest of the card even for a second could make that process fail.
    if ((double)freeB < 0.80 * (double)totalB) {
        printf("\nABORT: only %.1f%% of VRAM is free; something else is using the card. Nothing allocated.\n",
               100.0 * freeB / totalB);
        return 3;
    }

    // ---- allocate the card: one address range, one physical handle per piece --------------------------------
    double ta = now_s();
    size_t reserve = reserve_mib << 20;
    size_t npmax = (freeB > reserve) ? (freeB - reserve) / gran : 0;
    if (npmax > maxp) npmax = maxp;
    CUdeviceptr va = 0;
    if (npmax) CU(cuMemAddressReserve(&va, npmax * gran, 0, 0, 0));
    CUmemGenericAllocationHandle* h =
        (CUmemGenericAllocationHandle*)calloc(npmax ? npmax : 1, sizeof(CUmemGenericAllocationHandle));
    unsigned char* seen  = (unsigned char*)calloc(npmax ? npmax : 1, 1);   // errored at least once
    unsigned char* isbad = (unsigned char*)calloc(npmax ? npmax : 1, 1);   // confirmed at one offset
    if (!h || !seen || !isbad) { printf("ABORT: host alloc failed\n"); return 2; }
    size_t np = 0;
    for (; np < npmax; np++) {
        if (cuMemCreate(&h[np], gran, &ap, 0) != CUDA_SUCCESS) break;
        if (cuMemMap(va + np * gran, gran, 0, h[np], 0) != CUDA_SUCCESS) { cuMemRelease(h[np]); break; }
    }
    cudaGetLastError();
    CUmemAccessDesc ad;
    memset(&ad, 0, sizeof(ad));
    ad.location = ap.location;
    ad.flags    = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    if (np) CU(cuMemSetAccess(va, np * gran, &ad, 1));
    double t_alloc = now_s() - ta;

    double cover = (double)(np * gran) / (double)totalB;
    printf("allocated    : %.2f GiB in %zu pieces, %.1f%% of total VRAM (%.2fs)\n",
           np * gran / 1073741824.0, np, cover * 100.0, t_alloc);
    if (cover < 0.80) {
        printf("\nABORT: only %.1f%% of VRAM could be allocated. Free the card first;\n"
               "anything holding VRAM hides the region it occupies from this search.\n", cover * 100.0);
        return 3;
    }
    fflush(stdout);

    // ---- search ------------------------------------------------------------------------------------------------
    // Ones, zeros, then the alternating patterns: a hard stuck bit fails in at least two of the first four passes
    // whichever bit and polarity it is. Then the stock list: more fixed patterns and walking ones and zeros.
    unsigned int pats[8 + 64];
    int npat = 0;
    const unsigned int base_pats[] = {
        0xFFFFFFFFu, 0x00000000u, 0xAAAAAAAAu, 0x55555555u,
        0xA5A5A5A5u, 0x5A5A5A5Au, 0xDEADBEEFu, 0x00FF00FFu
    };
    for (int i = 0; i < 8; i++) pats[npat++] = base_pats[i];
    for (int b = 0; b < 32; b++) pats[npat++] = (1u << b);
    for (int b = 0; b < 32; b++) pats[npat++] = ~(1u << b);

    unsigned int*       h_hits = (unsigned int*)calloc(np, sizeof(unsigned int));
    unsigned int*       h_xor  = (unsigned int*)calloc(np, sizeof(unsigned int));
    unsigned int*       h_same = (unsigned int*)calloc(np, sizeof(unsigned int));
    unsigned int*       h_mism = (unsigned int*)calloc(np, sizeof(unsigned int));
    unsigned long long* h_idx  = (unsigned long long*)calloc(np, sizeof(unsigned long long));
    if (!h_hits || !h_xor || !h_same || !h_mism || !h_idx) { printf("ABORT: host alloc failed\n"); return 2; }

    int threads = 256;
    int blocks  = prop.multiProcessorCount * 16;
    unsigned int* d = (unsigned int*)va;
    unsigned long long N = np * epp;

    double ts = now_s(), last_new = ts;
    int nbad = 0, nseen = 0;
    unsigned int pass = 0;
    const char* why = "time limit";
    while (!g_stop) {
        double now = now_s();
        if (now - ts >= find_secs) break;
        if ((int)pass >= min_passes) {
            if (expect > 0 && nseen == expect && nbad == expect) { why = "expected fault(s) confirmed"; break; }
            if (nbad > 0 && now - last_new >= quiet_secs) { why = "quiet period"; break; }
        }

        unsigned int pat  = pats[pass % npat];
        int          mode = (pass / npat) % 3;
        kfill<<<blocks, threads>>>(d, N, 0, pat, mode);
        if (dwell_ms > 0) { CK(cudaDeviceSynchronize()); usleep((useconds_t)dwell_ms * 1000); }
        kcheck<<<blocks, threads>>>(d, N, 0, epp, pat, mode, d_hits, d_idx, d_xor, d_same, d_mism);
        CK(cudaMemcpy(h_hits, d_hits, sizeof(unsigned int) * np, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(h_same, d_same, sizeof(unsigned int) * np, cudaMemcpyDeviceToHost));
        pass++;

        for (size_t c = 0; c < np; c++) {
            if (!h_hits[c]) continue;
            if (!seen[c]) {
                seen[c] = 1;
                nseen++;
                CK(cudaMemcpy(h_idx, d_idx, sizeof(unsigned long long) * np, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(h_xor, d_xor, sizeof(unsigned int) * np, cudaMemcpyDeviceToHost));
                unsigned long long off = (c * epp + h_idx[c]) * 4ULL;
                printf("  [pass %u, %.2fs] candidate: piece %zu, offset 0x%llX (%.2f GiB), xor 0x%08X\n",
                       pass, now_s() - ts, c, off, off / 1073741824.0, h_xor[c]);
                fflush(stdout);
            }
            if (!isbad[c] && (int)h_same[c] >= confirm_hits) {
                isbad[c] = 1;
                nbad++;
                last_new = now_s();
                CK(cudaMemcpy(h_xor, d_xor, sizeof(unsigned int) * np, cudaMemcpyDeviceToHost));
                printf("  [pass %u, %.2fs] CONFIRMED #%d: piece %zu, xor 0x%08X, %u hits at that offset\n",
                       pass, now_s() - ts, nbad, c, h_xor[c], h_same[c]);
                fflush(stdout);
                if (nbad > max_bad) {
                    printf("\nABORT: more than --max-bad (%d) confirmed failing pieces.\n"
                           "This card is too damaged for quarantine to be sensible.\n", max_bad);
                    return 5;
                }
            }
        }
    }
    double searched = now_s() - ts;

    CK(cudaMemcpy(h_idx,  d_idx,  sizeof(unsigned long long) * np, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_xor,  d_xor,  sizeof(unsigned int) * np, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_same, d_same, sizeof(unsigned int) * np, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_mism, d_mism, sizeof(unsigned int) * np, cudaMemcpyDeviceToHost));
    printf("searched     : %u passes in %.2fs (%.0f ms per pass), stopped on %s\n",
           pass, searched, pass ? 1000.0 * searched / pass : 0.0, g_stop ? "signal" : why);

    if (nseen == 0) {
        printf("\nNo fault found in %.0fs / %u passes. This is NOT a clean bill of health: nothing was\n"
               "quarantined, and a fault outside the allocated region cannot be seen at all.\n", searched, pass);
        if (!allow_clean) return 4;
        if (ready_file) {
            FILE* f = fopen(ready_file, "w");
            if (f) { fprintf(f, "status=no-fault-found\nuuid=%s\n", ustr); fclose(f); }
        }
        return 0;
    }
    if (expect > 0 && nbad != expect)
        printf("WARNING: expected %d confirmed piece(s), have %d (%d errored). Holding all of them.\n",
               expect, nbad, nseen);

    // ---- release everything that never failed ------------------------------------------------------------------
    // As in the stock tool, anything that errored is held, confirmed or not.
    // Unmap each run of clean pieces in one call where the driver allows it, else piece by piece.
    double tr = now_s();
    size_t held = 0;
    int run_unmap = 1;
    for (size_t c = 0; c < np; ) {
        if (seen[c]) { held += gran; c++; continue; }
        size_t e = c;
        while (e < np && !seen[e]) e++;
        if (!run_unmap || cuMemUnmap(va + c * gran, (e - c) * gran) != CUDA_SUCCESS) {
            run_unmap = 0;
            for (size_t k = c; k < e; k++) CU(cuMemUnmap(va + k * gran, gran));
        }
        for (size_t k = c; k < e; k++) CU(cuMemRelease(h[k]));
        c = e;
    }
    cudaFree(d_hits); cudaFree(d_xor); cudaFree(d_idx); cudaFree(d_same); cudaFree(d_mism);

    int* fds = (int*)calloc(nseen, sizeof(int));
    int nfd = 0;
    if (hold_fd) {
        // The exported descriptors are references to the physical allocations. Once CUDA's own handles,
        // mappings and context are gone, they are the only thing keeping the faulty pieces allocated.
        for (size_t c = 0; c < np; c++) {
            if (!seen[c]) continue;
            int fd = -1;
            CU(cuMemExportToShareableHandle(&fd, h[c], CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));
            fds[nfd++] = fd;
        }
        for (size_t c = 0; c < np; c++) {
            if (!seen[c]) continue;
            CU(cuMemUnmap(va + c * gran, gran));
            CU(cuMemRelease(h[c]));
        }
        CU(cuMemAddressFree(va, npmax * gran));
        cudaFree(d_verr);
        CK(cudaDeviceReset());
    }
    double t_release = now_s() - tr;

    // fd mode: prove that the descriptors still hold the defective memory now that the context is gone. If they
    // do not, this driver does not keep descriptor-held allocations alive, so start over keeping the context.
    size_t scr_elems = (flush_mib < 0) ? (size_t)prop.l2CacheSize : ((size_t)flush_mib << 20) / 4ULL;
    double t_verify = 0.0;
    unsigned int boot_errs = 0;
    if (hold_fd) {
        double tv = now_s();
        int rc = fd_verify(dev, fds, nfd, gran, blocks, threads, scr_elems, verify_delay_ms, 8, &boot_errs);
        t_verify = now_s() - tv;
        if (rc != 0 || boot_errs == 0) {
            printf("\n<2>WARNING: the pieces held by file descriptor did not show the fault once the context was gone\n"
                   "(check %s, %u errors). Not trusting --hold fd on this driver; restarting with --hold context.\n",
                   rc ? "could not run" : "ran", boot_errs);
            fflush(stdout);
            for (int i = 0; i < nfd; i++) close(fds[i]);
            char** nv = (char**)calloc(argc + 1, sizeof(char*));
            for (int i = 0; i < argc; i++) nv[i] = argv[i];
            for (int i = 1; i + 1 < argc; i++)
                if (!strcmp(nv[i], "--hold")) nv[i + 1] = (char*)"context";
            execv("/proc/self/exe", nv);
            printf("ABORT: could not restart in context mode\n");
            return 2;
        }
    }
    double t_total = now_s() - T0;

    printf("\n=== QUARANTINE ACTIVE (%s) ===\n", hold_fd ? "held by file descriptors, no CUDA context" : "context");
    printf("held         : %.0f MiB in %d piece(s), %d confirmed\n", held / 1048576.0, nseen, nbad);
    if (hold_fd)
        printf("verified     : fault reproduced through the descriptors after the context was gone"
               " (%u errors in 8 passes)\n", boot_errs);
    printf("timing       : ready %.2fs after start (init %.2f, alloc %.2f, search %.2f, release %.2f, verify %.2f;"
           " %s)\n", t_total, t_init, t_alloc, searched, t_release, t_verify,
           run_unmap ? "unmapped in runs" : "unmapped per piece");
    printf("\n%-8s %-20s %-12s %-8s %-9s %s\n", "piece", "offset", "xor", "hits", "elsewhere", "state");
    for (size_t c = 0; c < np; c++) {
        if (!seen[c]) continue;
        printf("%-8zu 0x%016llX  0x%08X   %-8u %-9u %s\n", c,
               (unsigned long long)((c * epp + h_idx[c]) * 4ULL), h_xor[c], h_same[c], h_mism[c],
               isbad[c] ? "CONFIRMED" : "unconfirmed");
    }

    unsigned long long verify_passes = 0, verify_clean = 0;
    const char* last_check = hold_fd ? "ok (at start)" : "none yet";
    auto write_marker = [&]() {
        if (!ready_file) return;
        FILE* f = fopen(ready_file, "w");
        if (!f) { printf("WARNING: could not write ready marker %s\n", ready_file); return; }
        fprintf(f, "status=active\nuuid=%s\nhold=%s\npiece_mib=%.0f\nheld_mib=%.0f\n"
                   "confirmed_pieces=%d\nheld_pieces=%d\nexpected=%d\nready_s=%.2f\n"
                   "verify_passes=%llu\nverify_clean=%llu\nlast_check=%s\n",
                ustr, hold_fd ? "fd" : "context", gran / 1048576.0, held / 1048576.0,
                nbad, nseen, expect, t_total, verify_passes, verify_clean, last_check);
        for (size_t c = 0; c < np; c++)
            if (seen[c])
                fprintf(f, "bad_offset=0x%llX xor=0x%08X hits=%u state=%s\n",
                        (unsigned long long)((c * epp + h_idx[c]) * 4ULL), h_xor[c], h_same[c],
                        isbad[c] ? "confirmed" : "unconfirmed");
        fclose(f);
    };
    write_marker();
    if (ready_file) printf("ready marker : %s\n", ready_file);
    {
        char st[200];
        snprintf(st, sizeof(st), "READY=1\nSTATUS=%squarantine active: %.0f MiB in %d piece(s), %d confirmed, %s hold%s",
                 (expect > 0 && (nbad != expect || nseen != expect)) ? "WARNING, not the expected fault count: " : "",
                 held / 1048576.0, nseen, nbad, hold_fd ? "fd" : "context", hold_fd ? ", verified" : "");
        sd_notify_msg(st);
    }
    if (!hold_fd && verify_every > 0)
        printf("watchdog     : re-checking held pieces every %.0fs\n", verify_every);
    if (hold_fd)
        printf("re-check     : on SIGUSR1 (costs a CUDA context for about a second)\n");
    printf("\nHolding until killed.\n");
    fflush(stdout);

    // A failed re-check never releases anything: a false alarm must not drop a quarantine that may still be
    // needed. It is logged at critical priority (journald reads the <2> prefix), shown as the systemd status and
    // handed to --alert-cmd.
    auto alarm = [&](const char* what) {
        printf("<2>%s\n", what);
        fflush(stdout);
        char st[300];
        snprintf(st, sizeof(st), "STATUS=ALERT: %s", what);
        sd_notify_msg(st);
        if (alert_cmd) { int r = system(alert_cmd); (void)r; }
    };

    double next_verify = now_s() + verify_every;
    while (!g_stop) {
        sleep(1);
        int forced = g_check;
        g_check = 0;

        if (hold_fd) {
            if (!forced) continue;
            unsigned int e = 0;
            int rc = fd_verify(dev, fds, nfd, gran, blocks, threads, scr_elems, verify_delay_ms, 16, &e);
            verify_passes++;
            if (rc) {
                last_check = "could not run";
                printf("[check] could not run (no room for a CUDA context?); try again when the card is less busy\n");
            } else if (e) {
                last_check = "ok";
                printf("[check] fault still inside the held pieces (%u errors in 16 passes)\n", e);
                sd_notify_msg("STATUS=quarantine active, fd hold, last re-check OK");
            } else {
                last_check = "FAILED";
                verify_clean++;
                alarm("re-check FAILED: the held pieces no longer show the fault, so the quarantine is not known to"
                      " hold. Still holding; run a full search when the card is free.");
            }
            write_marker();
            fflush(stdout);
            continue;
        }

        if (!forced && (verify_every <= 0 || now_s() < next_verify)) continue;
        next_verify = now_s() + verify_every;

        // Alternating ones and zeros; a defect that stops failing all of them has left the held pieces or changed
        // behaviour, and either way the quarantine is no longer known good. The held pieces fit in L2, so after
        // each fill, write a scratch buffer of 4x L2 to push them out to DRAM, then wait --verify-delay-ms before
        // reading. Measured on the test card: 0 ms -> never fails (the first version of this watchdog), 30 or
        // 100 ms -> every all-ones pass fails.
        const int vpasses = 32;
        unsigned int* scr = NULL;
        if (scr_elems && cudaMalloc(&scr, scr_elems * 4ULL) != cudaSuccess) {
            cudaGetLastError();
            printf("[watchdog] could not allocate the L2 scratch buffer, skipping this check\n");
            fflush(stdout);
            continue;
        }
        unsigned int total = 0;
        CK(cudaMemset(d_verr, 0, sizeof(unsigned int)));
        for (int it = 0; it < vpasses && !g_stop; it++) {
            unsigned int pat = (it & 1) ? 0x00000000u : 0xFFFFFFFFu;
            for (size_t c = 0; c < np; c++)
                if (seen[c]) kfill<<<blocks, threads>>>(d + c * epp, epp, c * epp, pat, 0);
            if (scr_elems) kfill<<<blocks, threads>>>(scr, scr_elems, 0, (unsigned int)it, 1);
            CK(cudaDeviceSynchronize());
            if (verify_delay_ms > 0) usleep((useconds_t)verify_delay_ms * 1000);
            for (size_t c = 0; c < np; c++)
                if (seen[c]) kverify<<<blocks, threads>>>(d + c * epp, epp, c * epp, pat, 0, d_verr);
        }
        CK(cudaMemcpy(&total, d_verr, sizeof(total), cudaMemcpyDeviceToHost));
        if (scr) cudaFree(scr);
        verify_passes++;
        if (total) {
            last_check = "ok";
            printf("[watchdog] fault still inside the held pieces (%u errors in %d passes)\n", total, vpasses);
        } else {
            last_check = "FAILED";
            verify_clean++;
            alarm("watchdog: the held pieces passed every check, so the known fault is no longer reproducing inside"
                  " the quarantine. Still holding; run a full search when the card is free.");
        }
        write_marker();
        fflush(stdout);
    }

    printf("\nsignal received, releasing quarantine.\n");
    if (ready_file) unlink(ready_file);
    for (int i = 0; i < nfd; i++) close(fds[i]);
    return 0;
}
