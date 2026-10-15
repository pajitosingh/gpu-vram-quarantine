# vrampill2

`vrampill2` is a variant of `vrampill` for a card whose fault you have already found and characterised. It
holds the bad memory within about 3 seconds of starting, and in its file-descriptor mode the holder costs
about 5 MiB of VRAM instead of about 140 MiB.

It was written for one RTX 3090 with one leaky bit. The fault behaves differently from the temperature-gated
fault the main README describes, so this page covers that case, the measurements, and two traps that affect any
tool of this kind.

## The case: a leaky bit that fails cold

- **What fails:** one bit (bit 14) of one 32-bit word. When a 1 is stored there, it reads back as 0. It never
  goes the other way.
- **memtest_vulkan:** errors at the same address in every 2.5-minute run, from the first pass, in about 50% of
  passes (random data puts a 1 in that bit about half the time).
- **What makes no difference:**
  - temperature (cold or warm);
  - memory clock (stock or overclocked);
  - power limit (every setting tried).
- **Retention:** it is a retention fault. Read back immediately, the cell passes. Read back 30 ms or more later,
  it fails every time.

If memtest_vulkan reports the same address in every run, from the first minutes, with a single bit in the xor,
you probably have this kind of fault. If yours only appears after minutes of heavy load, it is the
temperature-gated kind: run the stock search first, with its long quiet period, before reaching for `--expect`.

## Results on that card

RTX 3090 on Linux, driver 595.99.02, CUDA 13.0.

|                               | `vrampill`                                   | `vrampill2 --expect 1 --hold fd` |
|-------------------------------|----------------------------------------------|----------------------------------|
| Time until the quarantine holds | ~3.5 min at the defaults (31 s with `--quiet-seconds 30`) | ~3 s, including a self-check |
| VRAM the holder costs (nvidia-smi) | 142 MiB                               | ~5 MiB                           |
| Size of the held piece        | 8 MiB                                        | 2 MiB                            |
| Share of VRAM searched        | 96.7%                                        | 99.2%                            |

The ~3 s breaks down as:

| Step                                 | Time        |
|--------------------------------------|-------------|
| Driver and context init              | 0.2-0.7 s   |
| Allocating the card                  | 0.8 s       |
| Search: 4-6 passes of ~110 ms, including the 50 ms wait | 0.45-0.7 s |
| Release                              | 0.8 s       |
| Self-check through the descriptor (fd mode) | 0.5 s |

How the hold was checked:

- **memtest_vulkan, quarantine held:** 0 errors in every mode tried (stock `vrampill`, `vrampill2` context mode,
  and `vrampill2` fd mode).
- **Same session, no holder:** 1,031 errors at the usual address in 2.5 minutes.

## What is different from `vrampill`

- **Physical allocation handles.** Memory comes from the CUDA VMM API (`cuMemCreate`), in pieces of the card's
  allocation granularity (2 MiB on Ampere), all mapped into one contiguous range.
  - One kernel launch sweeps the whole card per pass.
  - The held piece is an explicit physical allocation, which is the fix the main README proposes for the
    relocation risk. I have not tested whether a driver could still move it.
- **`--expect N`.** The search stops as soon as exactly N pieces are confirmed and nothing else has failed,
  instead of waiting out `--quiet-seconds`.
  - The patterns start with all ones, all zeros, 0xAAAAAAAA and 0x55555555, so a stuck or leaky bit of either
    polarity fails at least twice in the first four passes.
  - If more pieces fail than expected, the stock quiet-period rule takes over.
  - `--find-seconds` is still a hard ceiling.
- **`--dwell-ms` (default 50).** The search waits this long between filling the card and checking it. See the
  traps below.
- **`--hold fd`.** Exports the bad pieces as POSIX file descriptors, then releases everything else and destroys
  the CUDA context. The descriptors alone keep the pieces allocated: nvidia-smi showed the card at 13 MiB against
  8 MiB idle, and memtest_vulkan stayed clean.
  - CUDA documents this: an allocation is freed only when all mappings and all references to its handle,
    "including its shareable counterparts", are released.
  - It is still driver behaviour, so before reporting ready the program checks it: it re-opens the pieces through
    the descriptors in a fresh context and confirms the fault is still inside. If not, it restarts itself with
    `--hold context`.
- **`--hold context`** (the default) keeps the context and runs a watchdog that works for this kind of fault
  (see below).
- **`--uuid GPU-...`** selects the card by UUID. The program also sets `CUDA_VISIBLE_DEVICES` to that UUID before
  initialising CUDA, so no other card is opened.
  - Without that, a long-running holder keeps about 3 MiB of driver state on every GPU in the machine.
  - If no card has that UUID, it exits 6 and touches nothing.
- **systemd `Type=notify`.** The program sends `READY=1` with a status line once the quarantine is active, so
  other services can be ordered after it.
- **Safe to restart.** A card that is already in use (less than 80% free) is refused before anything is
  allocated (exit 3). A restart while a job runs can't take the rest of the card away from it.
- **A missing card is not an error.** With `--uuid` and no such card, the program tells systemd it is ready and
  exits 6. A unit with `SuccessExitStatus=6` just goes inactive.

## Watchdog without polling

After it has started, the holder does no GPU work at all. A periodic re-check in fd mode would need a CUDA context
every time, and taking ~130 MiB at an unlucky moment could make a job on the card fail an allocation. What covers
the ways the quarantine can stop holding:

| Risk                                                   | Covered by |
|--------------------------------------------------------|------------|
| The holder dies (crash, kill, OOM)                     | systemd: `OnFailure=` runs `vrampill2-alert` (critical log + desktop notification); `OOMScoreAdjust=-900` |
| A driver that frees descriptor-held memory             | the self-check at every start; falls back to `--hold context` |
| The driver moving the held memory                      | CUDA documents `PINNED` allocations as unable to migrate while in use; `SIGUSR1` re-checks on demand |
| Suspend or hibernate (the driver can save and restore VRAM) | `vrampill2-resume.service` sends `SIGUSR1` after resume |
| New faults elsewhere on the card                       | every start re-sweeps the whole card; rerun the full stock search now and then when the card is idle |

**A failed re-check never releases anything.** A false alarm must not drop a quarantine that may still be needed.
It is logged at critical priority, shown as the systemd status, and handed to `--alert-cmd`.
`systemctl kill -s USR1 --kill-whom=main vrampill2` runs a re-check by hand.

In context mode the stock-style periodic watchdog is still there (`--verify-every`). The context already exists,
so it costs nothing extra.

## Two traps for any tool of this kind

### 1. A re-check of a small held region reads from L2, not DRAM

A watchdog that fills the held memory and reads it straight back is served from the L2 cache. A 3090 has 6 MiB of
L2, so a 2 MiB piece stays entirely in cache, and most of an 8 MiB chunk does too. Cards with more L2 (64 MiB on
an RTX 4080, 72 MiB on a 4090) cache even larger regions. The read never reaches the defective cell.

### 2. A retention fault needs time between write and read

Even from DRAM, this cell passes if it is read back within microseconds.

### Measurements

All on the test card. Each `vrampill2` re-check is 32 passes of alternating all-ones and all-zeros over the held
2 MiB piece:

| Re-check                                                     | Result                              |
|--------------------------------------------------------------|-------------------------------------|
| Fill, read back at once                                      | clean                               |
| Fill, wait 30 ms, read (still in L2)                         | clean                               |
| Fill, write 4x L2 of scratch to push it out, read at once    | clean                               |
| Fill, push it out of L2, wait 30 ms, read                    | fails every all-ones pass (16/16)   |
| Fill, push it out of L2, wait 100 ms, read                   | 16/16                               |
| Stock `vrampill` watchdog (`--verify-every 5 --verify-iters 200`) | "no reproduction", 5 of 5 checks |

**The full-card search is mostly safe from both traps.** Each pass pushes the whole card through L2, and a word
waits about one pass (~31 ms on a 3090) between write and read. That is right at the edge for this cell, though:
0x55555555 sometimes passed. The 50 ms dwell makes the cell fail on every pass that stores a 1 there.

**The `vrampill2` watchdog avoids both.** It pushes the held pieces out of L2 with a scratch buffer of 4x
`l2CacheSize`, then waits `--verify-delay-ms` (default 30) before reading.

## Usage

```bash
make vrampill2                        # or: make ARCH=sm_86 vrampill2

# First, find out what you have: the stock search, as long as you like.
sudo ./vrampill --quiet-seconds 600

# Once you know the card has exactly one bad piece:
sudo ./vrampill2 --uuid GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx --expect 1 --hold fd \
     --ready-file /run/vrampill2.ready

# Re-check a running holder (fd mode: costs a CUDA context for about a second):
sudo kill -USR1 $(pgrep -x vrampill2)
```

Get the UUID from `nvidia-smi -L`. Options not listed here behave as in `vrampill`; see `vrampill2 --help`.

## Running it at boot

`examples/systemd/vrampill2.service` is a system unit. It holds the quarantine before anything else can put work
on the card:

- **Ordering:** it runs before `systemd-user-sessions.service`. That covers logins, the display manager, and
  user services started by lingering (`loginctl enable-linger`), which start at boot, not at login. It also runs
  before `docker.service`.
- **Bounded boot cost:** each attempt is capped by `--find-seconds 3`, and the fault normally confirms in under a
  second. The worst case is the fault not being found (exit 4, fails closed): 3 attempts, about 20 s.
- **Card removed:** a missing card exits 6 in about 0.1 s and the unit goes inactive without an error.
- **Alerts and resume:** `vrampill2-alert` / `vrampill2-alert.service` and `vrampill2-resume.service` are in the
  same folder.

## Caveats

- **One card, one cell, one driver version.**
- **`--expect` is only for a fault you have already characterised.** A first diagnosis should use the stock
  search with a long quiet period.
- **About 0.8% of the card is not searched:** the program's own context and a small reserve. A fault there is not
  found (exit 4).
- **In fd mode re-checks run only at start, on `SIGUSR1` and after resume.** For a periodic watchdog, use
  `--hold context`, which keeps the context (~130 MiB on a 3090).
- **As with `vrampill`, this routes around the damage; it does not repair the card.**
