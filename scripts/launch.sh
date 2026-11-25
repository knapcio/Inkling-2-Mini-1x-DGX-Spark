#!/usr/bin/env bash
# launch.sh up|down|status|logs: Inkling-2-Mini on one DGX Spark (GB10), one container.
#
#   scripts/launch.sh up                 # default profile: SGLang + the DFlash2 drafter (see README)
#   ENGINE=vllm scripts/launch.sh up     # alternative profile: vLLM + DSpark
#   SPEC=dspark scripts/launch.sh up     # fallback: the model's own DSpark draft (the v2.7.2 profile + the v2.8 exact fixes)
#   scripts/launch.sh up --some-flag x   # extra engine flags are appended to the command line
#   scripts/launch.sh status | logs | down
#
# Every setting is an environment variable with the measured default (README, "Every flag explained").
# The server binds to 127.0.0.1 by default; reach it from another machine with `ssh -L 8000:127.0.0.1:8000 spark`
# or set HOST=0.0.0.0 on a trusted LAN. Never expose it through the router.
set -euo pipefail

ENGINE=${ENGINE:-sglang}
NAME=${NAME:-inkling-2-mini}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}
MODEL_DIR=${MODEL_DIR:-$HOME/models/Inkling-2-Mini}
CACHE_DIR=${CACHE_DIR:-$HOME/.cache/inkling-2-mini-$ENGINE}   # JIT / Triton / FlashInfer caches: faster restarts
SERVED=${SERVED:-inkling-2-mini}
CTX=${CTX:-262144}
MIN_AVAIL_GIB=${MIN_AVAIL_GIB:-3.5}      # memory guard: stop the container below this MemAvailable during start-up
GUARD_SECONDS=${GUARD_SECONDS:-1200}   # how long the guard watches after `up`

# Pinned images (multi-arch index digests; Docker picks linux/arm64 on the Spark) and model revision.
SGLANG_IMAGE=${SGLANG_IMAGE:-lmsysorg/sglang@sha256:efac853b46811ccb760545b7b29e7119bdf907537ccf6e863fac3011fe8b98de}
VLLM_IMAGE=${VLLM_IMAGE:-vllm/vllm-openai@sha256:c2dc759f5984f69c1e2418c6af7c26eb756f5ed2a8934c768ae1253645d1449e}
MODEL_REVISION=ed40e7da625f594fc9c4990f5f66be24873c22d7

# SGLang profile
MEM_FRACTION_USER=${MEM_FRACTION:-}
MEM_FRACTION=${MEM_FRACTION:-0.93}
SWA_RATIO_USER=${SWA_RATIO:-}
SWA_RATIO=${SWA_RATIO:-0.05}                                 # sliding-window pool / full pool tokens (image default 0.1; 48 requests need ~72K)
LOAD_NOMMAP=${LOAD_NOMMAP:-1}                               # read the checkpoint in one pass instead of mmap page faults (start-up ~265 -> ~130 s)
# Speed overlays (overlay/, see overlay/README.md). OVERLAY=0 runs the stock image exactly as the cookbook does.
OVERLAY=${OVERLAY:-1}
OVERLAY_DIR=${OVERLAY_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/overlay}
VERIFY_CAP=${VERIFY_CAP:-policy:/opt/inkling-overlay/inkling_vcap/policies/r2-theta004.json,bsmax=16}  # verify-width policy (off = stock)
DRAFT_BYTES=${DRAFT_BYTES:-gemv,fold8,head8,draftmx8,hotvocab:65536,hotbase,wide8,w8tile,rnnhoist}  # draft-side byte cuts (empty = stock)
HOTVOCAB_FILE=${HOTVOCAB_FILE:-/opt/inkling-overlay/inkling_draftbytes/hotvocab/inkling2-mini-hot-r2.json}  # ranked draft vocabulary
KERNELS=${KERNELS:-moe_tiles,qkvr_gemv}                     # target-side kernel choices (empty = stock)
PREFILL=${PREFILL:-qkvr_split,bias_layout,quant_once,partial_bf16}  # exact prefill levers (empty = stock)
PICO12=${PICO12:-dense,lmhead,stack1}                              # pico packed 12-bit BF16 GEMVs (empty = stock)
CERT_HEAD=${CERT_HEAD:-1}                                   # certified greedy LM-head argmax (0 = off; needs PICO12 lmhead)
WIDE=${WIDE:-dense,lmhead}                                  # packed 12-bit GEMM for 17-256-row verify batches (empty = stock)
WIDE_SCOPE=${WIDE_SCOPE:-verify}                            # wide only on decode/verify forwards
VATTN=${VATTN:-1}                                           # verify attention on the split-KV relative-bias kernel at all batch sizes
ATTN=${ATTN:-prefill_swa,prefill_global}                    # fused relative-bias prefill attention (empty = stock FA4 path)
MOE2=${MOE2:-tiles}                                          # GB10 tile tables for the routed-expert GEMMs (9-128 tokens NVFP4, MXFP8 tactic)
VCAP_BV=${VCAP_BV:-1}                                       # block verification under the verify cap (exact; INKLING_VERIFY_CAP_BV=mask)
VCAP_FASTPROBS=${VCAP_FASTPROBS:-1}                         # draft statistics / sampled accept q from one BF16 read (exact)
VCAP_FUSED=${VCAP_FUSED:-1}                                 # sampled accept straight from the logits, >= 3 requests (exact)
RSFIX=${RSFIX:-1}                                           # exact residual / bonus token rule in sampled verification (+ SGLang PR #51681 / #51692 fixes)
NOEXP=${NOEXP:-1}                                           # dead verify rows get no experts under the verify cap (exact for live rows; inactive when the cap or its remap is off)
KV12=${KV12:-1}                                             # global-layer KV also kept as packed 12-bit pages (exact; side pool)
CERT_WIDE=${CERT_WIDE:-1}                                   # certified greedy LM head for 17-128-row verify batches (exact; CERT_WIDE_MAX_ROWS)
CERT_WIDE_MAX_ROWS=${CERT_WIDE_MAX_ROWS:-}                  # largest verify batch (rows) the certified wide head serves (empty = overlay default)
CERT_WIDE_STATS=${CERT_WIDE_STATS:-0}                       # 1: log wide certified-head outcomes and union sizes per batch size (diagnostic)
RNNHOIST_MAX_BS=${RNNHOIST_MAX_BS:-32}                      # draft rnnhoist row limit (v2.7.2: 16; 32 changes c17-c32 drafts by FP32 order)
DFLASH_BYTES_USER=${DFLASH_BYTES+set}                       # SPEC=dflash2 drafter byte cuts: mx8 (MXFP8 layers), hot8 / hot16
DFLASH_BYTES=${DFLASH_BYTES:-}                               # (hot-vocab candidates); unset = mx8,mx8w,hot8 with SPEC=dflash2 + OVERLAY=1
DFLASH_FUSED=${DFLASH_FUSED:-}                               # SPEC=dflash2: sampled accept from the logits with sparse q (inkling_dfaccept); unset = 1 with SPEC=dflash2 + OVERLAY=1
PADX=${PADX:-1}                                              # CUDA-graph padding rows -> phantom expert (inkling_padx, exact for live rows); 0: off
DFLASH_HOT_N=${DFLASH_HOT_N:-}                               # hot-vocab size for hot8 / hot16 (empty = 65536)
DFLASH_PREFIX_TAIL=${DFLASH_PREFIX_TAIL:-2048}                 # SPEC=dflash2: on a prefix-cache hit, prefill the last N prompt tokens again so the drafter sees them (0 = off)
DFLASH_PREFIX_TAIL_MAX_INPUT=${DFLASH_PREFIX_TAIL_MAX_INPUT:-4096}  # ... only for prompts of at most this many tokens (longer prompts keep the cache as before)
DFLASH_TAIL_SNAPSHOTS=${DFLASH_TAIL_SNAPSHOTS:-8}                # SPEC=dflash2: prompt-end draft windows kept for later prefix hits (40 MiB each, charged to the KV budget; 0 = off)
DFLASH_TAIL_DEFER=${DFLASH_TAIL_DEFER:-1}                        # ... the prefix tail defers to a snapshot that already covers the drafter's window (the target keeps its cache hit; 0 = v2.9.3: tail and snapshots independent)
DFLASH_PCOPY=${DFLASH_PCOPY:-12}                               # SPEC=dflash2: prompt-copy forced drafts for long prompts (12 = on, the measured rule; 0 = off)
DFLASH_PCOPY_MIN_CTX=${DFLASH_PCOPY_MIN_CTX:-8192}             # ... only for requests whose prompt has at least this many tokens (shorter prompts: no copy)
PROMPT_TOUCH=${PROMPT_TOUCH:-1}                              # OVERLAY=1: after a request finishes, refresh its prompt-end cache checkpoint so the next turn of a long conversation keeps its prefix hit (cache LRU order only; 0 = off)
[[ $ENGINE == vllm && -z ${SPEC:-} ]] && SPEC=dspark   # the vLLM profile keeps the DSpark draft
SPEC=${SPEC:-dflash2}                  # dflash2 (default: third-party DFlash2 drafter) | dspark (the model's own draft, v2.7.2 profile) | none
DFLASH_BV=${DFLASH_BV:-1}                                    # SPEC=dflash2 + OVERLAY=1: block verification on sampled steps (exact with RSFIX)
DFCAP_USER=${DFCAP+set}
DFCAP=${DFCAP:-}                                             # SPEC=dflash2 verify-width cap; unset = the v2.8.2 policy (df-SLDF2x) with OVERLAY=1, off = full width
DFLASH2_DIR=${DFLASH2_DIR:-$HOME/models/Inkling-2-Mini-DFlash2}   # scripts/download.sh --dflash2
DFLASH2_REVISION=4f9088ea66c21f2b99ed657d1941c306b03e84d2           # incoai/Inkling-2-Mini-DFlash2
DFLASH2_SHA256="14223ae2f3aa1d661da67cf0143ec33070d952a6e28a119e6cec64fcc0bebe50 sglang_inkling_dflash2.py
d094e435eb939e01fca6297f637c5af35e3da4b552f8e1ab0a181cd278a1cefa config.json
217980ab00d116693fa5946df7bbfd967caf5fac1f529afd541fbd470ddb54ee model.safetensors"
# DFLASH_FUSED default (ABBA 2026-11-24, cap off: sampled c24 per-request decode +2.3 %, c8 unproven so batches below
# 16 requests keep the stock accept (INKLING_DFLASH_FUSED_MIN_BS); calls with a verify-cap cutoff always do)
[[ -n $DFLASH_FUSED ]] || { [[ $SPEC == dflash2 && $OVERLAY == 1 ]] && DFLASH_FUSED=1 || DFLASH_FUSED=0; }
[[ $DFLASH_FUSED == 0 || ( $SPEC == dflash2 && $OVERLAY == 1 ) ]] || {
  echo "DFLASH_FUSED applies to SPEC=dflash2 with OVERLAY=1 only" >&2; exit 2; }
[[ -z $DFLASH_BYTES || ( $SPEC == dflash2 && $OVERLAY == 1 ) ]] || {
  echo "DFLASH_BYTES applies to SPEC=dflash2 with OVERLAY=1 only" >&2; exit 2; }
if [[ $SPEC == dflash2 ]]; then
  # The DFLASH worker never calls the DSpark draft / verify overlays and the certified heads are armed by the
  # DSpark verify epilogue: all off. The target-side kernels and RSFIX stay as configured. The draft KV is a
  # ring over its 2,048-token window; the target's global KV is the packed store (KV12S=1, static pools) or,
  # with KV12S=0, the stock BF16 pool (the KV12 side pool does not fit next to the draft at this fraction).
  [[ $ENGINE == sglang ]] || { echo "SPEC=dflash2 needs ENGINE=sglang" >&2; exit 2; }
  [[ ${KV12S_POOL:-static} == static ]] || { echo "SPEC=dflash2 needs KV12S_POOL=static" >&2; exit 2; }
  KV12=0
  VERIFY_CAP=off; DRAFT_BYTES=; VCAP_BV=0; VCAP_FASTPROBS=0; VCAP_FUSED=0; NOEXP=0;
  # CERT_DFLASH=1: the certified head (<= 16 rows) on DFLASH greedy verify steps (inkling_cert/dflash.py); the wide
  # head writes no logits on certified steps and stays off
  # Defaults with the overlay (v2.8 ABBA 2026-11-24: mx8,hot8 speedbench quick +8.4 %, prose c1 +6.5 %;
  # CERT_DFLASH +2.8 %, check mode 0 mismatches in 34,808 rows): DFLASH_BYTES=mx8,hot8, CERT_DFLASH=1
  # v2.8.4 adds mx8w (b12x MXFP8 GEMM on the eligible drafter linears, ABBA speedbench full +1.2 %, greedy identical)
  if [[ $OVERLAY == 1 ]]; then
    [[ -n $DFLASH_BYTES_USER ]] || DFLASH_BYTES=mx8,mx8w,hot8
    CERT_HEAD=${CERT_DFLASH:-1}
  else
    CERT_HEAD=${CERT_DFLASH:-0}
  fi
  # CERT_WIDE_DFLASH=1: also the wide certified head (17..CERT_WIDE_MAX_ROWS rows); its IDs reach the DFLASH accept
  # as target_predict (inkling_cert/dflash.py). On with the certified head (greedy hashes 30/30, code c8 +3 %,
  # sampled verify graph unchanged: the unarmed pass replaces the stock softcap at the same bytes).
  [[ $CERT_HEAD == 1 ]] && CERT_WIDE=${CERT_WIDE_DFLASH:-1} || CERT_WIDE=${CERT_WIDE_DFLASH:-0}
  [[ $CERT_WIDE == 0 || $CERT_HEAD == 1 ]] || { echo "CERT_WIDE_DFLASH=1 needs CERT_DFLASH=1" >&2; exit 2; }
  [[ $CERT_HEAD == 0 || $OVERLAY == 1 ]] || { echo "CERT_DFLASH needs OVERLAY=1" >&2; exit 2; }
  [[ $PICO12 == *drafthead* ]] && { echo "PICO12 drafthead patches the DSpark draft head; drop it with SPEC=dflash2" >&2; exit 2; }
  [[ -n $MEM_FRACTION_USER ]] || MEM_FRACTION=0.85
  # Full-width DFLASH capture (inkling_vcap/dfsteplog.py): STEPLOG=<label> writes
  # $CACHE_DIR/steplog/<label>/dflash1-*.npz; measurement only (verification stays full width)
  if [[ -n ${STEPLOG:-} ]]; then
    [[ $OVERLAY == 1 ]] || { echo "STEPLOG needs OVERLAY=1 (the capture is an overlay)" >&2; exit 2; }
    VERIFY_CAP=steplog
  fi
  # The verify cap on DFLASH. DFCAP = fixed:K | conf:T | dfpolicy:/opt/inkling-overlay/...json (+ ,bsmax=N);
  # DFCAP_NOEXP=1: dead rows get no experts (inkling_noexp); DFCAP_BV=1: block verification (sparse-q mask).
  if [[ -z $DFCAP_USER && $OVERLAY == 1 && -z ${STEPLOG:-} ]]; then
    DFCAP=dfpolicy:/opt/inkling-overlay/inkling_vcap/policies/df-SLDF2x.json,bsmax=8
  fi
  [[ $DFCAP == off ]] && DFCAP=
  if [[ -n ${DFCAP:-} ]]; then
    [[ $OVERLAY == 1 ]] || { echo "DFCAP needs OVERLAY=1" >&2; exit 2; }
    [[ -z ${STEPLOG:-} ]] || { echo "DFCAP and STEPLOG are exclusive (the capture is full width)" >&2; exit 2; }
    VERIFY_CAP=$DFCAP; NOEXP=${DFCAP_NOEXP:-1}; VCAP_BV=${DFCAP_BV:-$DFLASH_BV}
  fi
fi
KV12S_DEFAULT=${KV12S_DEFAULT:-1}                           # packed global KV store default for the SGLang overlay profile
if [[ -z ${KV12S:-} ]]; then                                # vLLM and OVERLAY=0 never default to KV12S
  if [[ $ENGINE == sglang && $OVERLAY == 1 ]]; then KV12S=$KV12S_DEFAULT; else KV12S=0; fi
fi                                                          # KV12S=1: packed global target KV, raw BF16 draft lane
case $(printf %s "$KV12S" | tr '[:upper:]' '[:lower:]') in   # one spelling for the shell checks and the overlay switch
  1|true|yes|on) KV12S=1 ;;
  0|false|no|off|'') KV12S=0 ;;
  *) echo "KV12S must be 0 or 1 (got '$KV12S')" >&2; exit 2 ;;
esac
case " ${SGLANG_ENV:-} " in                                   # the prefill levers are sized here (MEM_FRACTION): no env overrides
  *INKLING_PFUSE_HOIST*|*INKLING_ATTN_GATE*|*INKLING_PROLOGUE*)
    echo "set PFUSE_HOIST / ATTN_GATE / PROLOGUE instead of passing INKLING_PFUSE_HOIST / INKLING_ATTN_GATE / INKLING_PROLOGUE in SGLANG_ENV" >&2; exit 2 ;;
esac
case " ${SGLANG_ENV:-} ${SGLANG_ARGS:-} $* " in                # the store is chosen here, so engine flags and overlay agree
  *INKLING_KV12*)                                            # any INKLING_KV12* name, with or without a value
    echo "set KV12S / KV12S_POOL / KV12S_SETTLE / KV12 / KV12_SLOTS instead of passing the KV store's variables in SGLANG_ENV" >&2; exit 2 ;;
esac
case " ${SGLANG_ENV:-} ${SGLANG_ARGS:-} $* " in                # the prefix tail is DFlash2-only: passed by the launcher alone
  *INKLING_DFLASH_PREFIX_TAIL*|*INKLING_DFLASH_TAIL_SNAPSHOTS*|*INKLING_DFLASH_TAIL_DEFER*|*INKLING_DFLASH_PCOPY*)  # the tail, its _MAX_INPUT, the snapshots, the deferral, the prompt copy and its _MIN_CTX
    echo "set DFLASH_PREFIX_TAIL / DFLASH_PREFIX_TAIL_MAX_INPUT / DFLASH_TAIL_SNAPSHOTS / DFLASH_TAIL_DEFER / DFLASH_PCOPY / DFLASH_PCOPY_MIN_CTX instead of passing INKLING_DFLASH_* variables in SGLANG_ENV" >&2; exit 2 ;;
esac
case " ${SGLANG_ENV:-} ${SGLANG_ARGS:-} $* " in                # the prompt-end refresh is switched by the launcher alone
  *INKLING_PROMPT_TOUCH*)
    echo "set PROMPT_TOUCH instead of passing INKLING_PROMPT_TOUCH in SGLANG_ENV" >&2; exit 2 ;;
esac
case $PROMPT_TOUCH in 0|1) ;; *) echo "PROMPT_TOUCH must be 0 or 1 (got '$PROMPT_TOUCH')" >&2; exit 2 ;; esac
[[ $KV12S == 1 ]] && case " ${SGLANG_ENV:-} ${SGLANG_ARGS:-} $* " in
  *enable-unified-memory*|*cuda-graph-backend-prefill*)
    echo "with KV12S=1, choose the pool with KV12S_POOL instead of passing its engine flags" >&2; exit 2 ;;
esac
KV12S_POOL=${KV12S_POOL:-static}                           # static: stock (non-unified) pools; unified: --enable-unified-memory (v2.7.1)
case $KV12S_POOL in static|unified) ;; *) echo "KV12S_POOL must be static or unified (got '$KV12S_POOL')" >&2; exit 2 ;; esac
KV12S_OVERFLOW_HEADS=${KV12S_OVERFLOW_HEADS:-2048}          # exact page-head overflow slots per target visit (32 KiB each)
KV12S_FALLBACK_PAGES=${KV12S_FALLBACK_PAGES:-64}            # raw fallback pages per visit if that arena is exhausted (512 KiB each)
KV12S_SETTLE=${KV12S_SETTLE:-shared}                        # SPEC=dflash2: the 8 packed visits' verify settle builds its page candidates once (shared, exact; 1 = also one launch per settle kernel for all visits, opt-in: slower at c4/c8 in the v2.9.5 ABBA; 0 = v2.9.4: per visit)
case $KV12S_SETTLE in 0|1|shared) ;; *) echo "KV12S_SETTLE must be 1, shared or 0 (got '$KV12S_SETTLE')" >&2; exit 2 ;; esac
KV12_SLOTS=${KV12_SLOTS:-2048}                              # packed side-pool pages per global layer (2048 = 6.1 GiB, one 256K request)
SCODES=${SCODES:-1}                                          # NVFP4 expert scale codes: 5-bit codes + row base, expanded in the GEMM (exact)
PFUSE=${PFUSE:-all}                                         # prefill glue fusion (empty = stock)
PFUSE_HOIST=${PFUSE_HOIST:-1}                               # prefill loop-bank hoist over pfuse loop_mix (1: changes FP32 summation order, not the math; quality bracketed equal; exact: bit-identical, slower; 0: off)
ATTN_GATE=${ATTN_GATE:-all}                                 # gate_sink in the prefill attention epilogue, sliding-window and global layers (bit-identical; 1: sliding-window only; 0: off)
PROLOGUE=${PROLOGUE:-1}                                     # SWA prefill: K conv + L2 clip and V conv + value norm in one kernel each (bit-identical: the CuTe row-sum order; 0: off)
[[ ",$PFUSE," == *,all,* || ",$PFUSE," == *,loopmix,* ]] || PFUSE_HOIST=0   # the hoist wraps pfuse loopmix
_onoff() { case $(printf %s "$1" | tr '[:upper:]' '[:lower:]') in ''|0|off|false|no|none) echo 0 ;; *) echo 1 ;; esac; }
[[ $(_onoff "$PFUSE_HOIST") == 0 ]] && PFUSE_HOIST=0
[[ $(_onoff "$ATTN_GATE") == 0 ]] && ATTN_GATE=0
[[ $(_onoff "$PROLOGUE") == 0 ]] && PROLOGUE=0
PREFILL_LEVERS=0; [[ $PFUSE_HOIST != 0 || $ATTN_GATE != 0 || $PROLOGUE != 0 ]] && PREFILL_LEVERS=1
[[ $PREFILL_LEVERS == 1 ]] && case " ${SGLANG_ARGS:-} $* " in
  *--enable-mixed-chunk*)  # not lossless: the prefill budget shrinks by the running count (chunk boundaries move) and
                           # running requests drop to plain 1-token decode (no speculation); untested with the levers
    echo "--enable-mixed-chunk is refused with PFUSE_HOIST / ATTN_GATE / PROLOGUE on: it moves the long prompts' chunk boundaries (not lossless) and turns running requests into plain 1-token decode without speculation; use --prefill-decode-interval for prefill / decode interleaving" >&2; exit 2 ;;
esac
MOE_TILES=${MOE_TILES:-1@2304:swap_ab=1,tile_n=32,tile_k=256;1@1152:swap_ab=1,tile_n=32,tile_k=256;8@2304:swap_ab=1,tile_n=32,tile_k=256;8@1152:swap_ab=1,tile_n=32,tile_k=256;64@2304:swap_ab=0,tile_n=64,tile_k=256;64@1152:swap_ab=1,tile_n=32,tile_k=256}
SGLANG_ARGS=${SGLANG_ARGS:-}
# vLLM profile
GPU_UTIL=${GPU_UTIL:-0.80}
VLLM_ARGS=${VLLM_ARGS:-}

cmd=${1:-}; shift || true

case $cmd in
  down)
    docker stop "$NAME" >/dev/null 2>&1 || true
    docker rm "$NAME" >/dev/null 2>&1 || true
    for _ in $(seq 60); do docker ps -aq --filter "name=^${NAME}$" | grep -q . || break; sleep 1; done  # wait for --rm removal
    pkill -f "memguard.sh $NAME " 2>/dev/null || true
    echo "stopped $NAME"; exit 0 ;;
  status)
    docker ps -a --filter "name=^${NAME}$" --format '{{.Names}}  {{.Image}}  {{.Status}}'
    curl -fsS --max-time 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && echo "API: ready on :$PORT" || echo "API: not ready"
    awk '/MemAvailable/ {printf "MemAvailable: %.1f GiB\n", $2/1048576}' /proc/meminfo
    exit 0 ;;
  logs)
    exec docker logs -f "$NAME" ;;
  up) ;;
  *) echo "usage: $0 up|down|status|logs [extra engine flags]" >&2; exit 2 ;;
esac

kv12s_args=(); kv12s_env=()
if [[ $KV12S == 1 ]]; then
  [[ $ENGINE == sglang && $OVERLAY == 1 ]] || {
    echo "KV12S=1 requires ENGINE=sglang and OVERLAY=1" >&2; exit 2; }
  KV12=0
  if [[ $KV12S_POOL == static ]]; then
    # The packed store replaces the global BF16 pool and the KV12 side copy; with the draft's
    # window lanes priced per window token (static.DRAFT_WINDOW_LANES), 0.89 sizes ~2.19M
    # tokens and keeps the kvstress worst case above the 3.5 GiB guard (prefill graphs kept).
    # With SPEC=dflash2 the draft is a fixed 2.16 GB ring: 0.89 sizes ~2.38M tokens (measured: 0.90 gave 2.42M
    # with the kvstress worst case at 4.6 GiB, guard 3.5); contexts above 256K keep 0.85 (their prefix scratch
    # is larger) until measured.
    # v2.8.2: the prefill levers (PFUSE_HOIST, ATTN_GATE, PROLOGUE) deepen the long-prefill MemAvailable dip by ~0.3-0.7 GB,
    # so SPEC=dflash2 up to 256K sizes at 0.88 with them on (-1 % KV tokens).
    if [[ $SPEC == dflash2 ]]; then
      if (( CTX > 262144 )); then [[ -n $MEM_FRACTION_USER ]] || MEM_FRACTION=0.85
      elif [[ $PREFILL_LEVERS == 1 ]]; then [[ -n $MEM_FRACTION_USER ]] || MEM_FRACTION=0.88
      else [[ -n $MEM_FRACTION_USER ]] || MEM_FRACTION=0.89; fi
    else [[ -n $MEM_FRACTION_USER ]] || MEM_FRACTION=0.89; fi
    # ~79K window tokens, as v2.7 and v2.7.1 (48 requests' windows plus an 8K chunk).
    [[ -n $SWA_RATIO_USER ]] || SWA_RATIO=0.037
    kv12s_args=(--max-running-requests 48 --cuda-graph-max-bs-decode=48)
  else
    # The unified pool leaves less post-capture headroom than the stock pool at the same
    # fraction; 0.91 keeps MemAvailable above the guard during 8K prefill chunks.
    [[ -n $MEM_FRACTION_USER ]] || MEM_FRACTION=0.91
    # 3.5 % of 2.26M FULL tokens still covers 48 requests' windows (think48 + kvstress measured at 0.035).
    [[ -n $SWA_RATIO_USER ]] || SWA_RATIO=0.035
    kv12s_args=(--enable-unified-memory --cuda-graph-backend-prefill=disabled --max-running-requests 48
                --cuda-graph-max-bs-decode=48)
  fi
  kv12s_env=(-e SGLANG_RAGGED_VERIFY_MODE=static -e INKLING_KV12S_POOL="$KV12S_POOL"
             -e INKLING_KV12S_DFLASH_SETTLE="$KV12S_SETTLE")
fi

[[ -f $MODEL_DIR/config.json && -f $MODEL_DIR/mtp.safetensors ]] || {
  echo "no model in $MODEL_DIR (run scripts/download.sh first; mtp.safetensors is the DSpark draft)" >&2; exit 1; }
if [[ -f $MODEL_DIR/.revision && $(cat "$MODEL_DIR/.revision") != "$MODEL_REVISION" ]]; then
  echo "warning: $MODEL_DIR is revision $(cat "$MODEL_DIR/.revision"), this recipe was measured on $MODEL_REVISION" >&2
fi
if [[ $SPEC == dflash2 ]]; then
  # Every file SGLang reads from the drafter directory is checked on every launch, and nothing else may be there
  # (stock DFLASH would load an extra mask_embedding.pt into the live target embedding).
  [[ $(cat "$DFLASH2_DIR/.revision" 2>/dev/null) == "$DFLASH2_REVISION" ]] || {
    echo "no DFlash2 drafter at revision $DFLASH2_REVISION in $DFLASH2_DIR (scripts/download.sh --dflash2)" >&2; exit 1; }
  extra=$(cd "$DFLASH2_DIR" && ls -A | grep -vxE 'config\.json|model\.safetensors|sglang_inkling_dflash2\.py|README\.md|\.revision' || true)
  [[ -z $extra ]] || { echo "unexpected files in $DFLASH2_DIR: $extra" >&2; exit 1; }
  (cd "$DFLASH2_DIR" && printf '%s\n' "$DFLASH2_SHA256" | sed 's/ /  /' | sha256sum -c --quiet -) || {
    echo "DFlash2 drafter files in $DFLASH2_DIR do not match the pinned sha256" >&2; exit 1; }
fi
if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "$NAME is already running (scripts/launch.sh down first)" >&2; exit 1
fi
# One model server per Spark: unified memory is shared by the GPU and the OS.
others=$(docker ps --format '{{.Names}} {{.Image}}' | grep -Ei 'sglang|vllm|tensorrt|llama|ollama' || true)
[[ -z $others ]] || { echo "another model server is running, stop it first:"$'\n'"$others" >&2; exit 1; }
avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
(( avail > ${MIN_AVAIL_GIB%.*} + 8 )) || {
  echo "MemAvailable is ${avail} GiB: free memory first (another process, the page cache of a download)" >&2; exit 1; }
docker rm "$NAME" >/dev/null 2>&1 || true
mkdir -p "$CACHE_DIR"

common=(docker run -d --name "$NAME" --gpus all --ipc=host --network host --shm-size 16g --restart no
        -v "$MODEL_DIR":/model:ro -v "$CACHE_DIR":/root/.cache
        -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1)

if [[ $ENGINE == sglang ]]; then
  nommap_arg=(); nommap_env=()
  # one-pass checkpoint read; large frees go straight back to the OS so the KV pool is sized on the real free memory
  [[ $LOAD_NOMMAP == 1 ]] && { nommap_arg=(--weight-loader-disable-mmap); nommap_env=(-e MALLOC_MMAP_THRESHOLD_=1048576); }
  spec=()
  [[ $SPEC == dspark ]] && spec=(--speculative-algorithm DSPARK --speculative-num-draft-tokens 8)
  [[ $SPEC == dspark && $OVERLAY == 1 && $VCAP_BV == 1 ]] && spec+=(--speculative-use-block-verification)
  entry=(--entrypoint python3 "$SGLANG_IMAGE")
  if [[ $SPEC == dflash2 ]]; then
    spec=(--speculative-algorithm DFLASH --speculative-draft-model-path /draft --speculative-num-draft-tokens 8
          --speculative-draft-attention-backend fa4)
    # block verification: the stock flag; under the cap the overlay's sparse-q mask (INKLING_VERIFY_CAP_BV)
    # (exact only with the RSFIX reject fix: without it block verification stays off)
    rsfix_reject=0; [[ $OVERLAY == 1 && ( $RSFIX == 1 || $RSFIX == *reject* ) ]] && rsfix_reject=1
    if [[ -n ${DFCAP:-} ]]; then [[ $VCAP_BV == 1 && $rsfix_reject == 1 ]] && spec+=(--speculative-use-block-verification)
    else [[ $DFLASH_BV == 1 && $rsfix_reject == 1 ]] && spec+=(--speculative-use-block-verification); fi
    [[ $rsfix_reject == 1 ]] || VCAP_BV=0
    # CTX and DFLASH2_DIR are the inputs the memory budget and the sha256 check were made for
    for a in "$@"; do case $a in --context-length*|--speculative-draft-model-path*)
      echo "with SPEC=dflash2 set CTX / DFLASH2_DIR instead of $a" >&2; exit 2;; esac; done
    # The drafter's SGLang compatibility patch rewrites installed sglang files, so it runs inside this container
    # only (--rm: it never outlives the run), before the server starts; the server log is also kept in CACHE_DIR.
    # The pinned sha256 is checked again inside the container, on the read-only mount the patch and weights load from.
    entry=(--rm -e SGLANG_ENABLE_DFLASH_DRAFT_KV_RING=1 -e DFLASH2_SHA256="$DFLASH2_SHA256" -v "$DFLASH2_DIR":/draft:ro
           --entrypoint bash "$SGLANG_IMAGE"
           -c 'mkdir -p /root/.cache/dflash2-logs; exec > >(tee -a /root/.cache/dflash2-logs/server-$(date -u +%Y%m%dT%H%M%SZ).log) 2>&1
               (cd /draft && printf "%s\n" "$DFLASH2_SHA256" | sed "s/ /  /" | sha256sum -c --quiet -) || {
                 echo "DFlash2 drafter files do not match the pinned sha256" >&2; exit 1; }
               python3 -I /draft/sglang_inkling_dflash2.py && exec python3 "$@"' dflash2)
  fi
  dftail=; [[ $SPEC == dflash2 && $DFLASH_PREFIX_TAIL != 0 ]] && dftail=$DFLASH_PREFIX_TAIL
  [[ $DFLASH_TAIL_SNAPSHOTS =~ ^[0-9]+$ ]] || { echo "DFLASH_TAIL_SNAPSHOTS must be a whole number (got '$DFLASH_TAIL_SNAPSHOTS')" >&2; exit 2; }
  dfsnap=; [[ $SPEC == dflash2 && $OVERLAY == 1 && $DFLASH_TAIL_SNAPSHOTS != 0 ]] && dfsnap=$DFLASH_TAIL_SNAPSHOTS
  case $DFLASH_TAIL_DEFER in 0|1) ;; *) echo "DFLASH_TAIL_DEFER must be 0 or 1 (got '$DFLASH_TAIL_DEFER')" >&2; exit 2 ;; esac
  dfdefer=; [[ -n $dftail && -n $dfsnap && $DFLASH_TAIL_DEFER == 1 ]] && dfdefer=1   # needs both the tail and the snapshots
  case $DFLASH_PCOPY in 0|12) ;; *) echo "DFLASH_PCOPY must be 12 or 0 (got '$DFLASH_PCOPY')" >&2; exit 2 ;; esac
  dfpcopy=; dfpcopymin=; [[ $SPEC == dflash2 && $OVERLAY == 1 && $DFLASH_PCOPY != 0 ]] && dfpcopy=$DFLASH_PCOPY
  if [[ -n $dfpcopy ]]; then                                 # the threshold is checked and passed only when the copy is on
    [[ $DFLASH_PCOPY_MIN_CTX =~ ^[0-9]+$ ]] || { echo "DFLASH_PCOPY_MIN_CTX must be a whole number of tokens (got '$DFLASH_PCOPY_MIN_CTX')" >&2; exit 2; }
    dfpcopymin=$DFLASH_PCOPY_MIN_CTX
  fi
  ptouch=0; [[ $OVERLAY == 1 && $PROMPT_TOUCH == 1 ]] && ptouch=1
  ov=()
  if [[ $OVERLAY == 1 ]]; then
    [[ -f $OVERLAY_DIR/sitecustomize.py ]] || { echo "OVERLAY=1 but no overlay at $OVERLAY_DIR" >&2; exit 1; }
    ov=(-v "$OVERLAY_DIR":/opt/inkling-overlay:ro -e PYTHONPATH=/opt/inkling-overlay
        -e INKLING_VERIFY_CAP="$VERIFY_CAP" -e INKLING_VERIFY_CAP_LOG_EVERY="${VERIFY_CAP_LOG_EVERY:-0}"
        -e INKLING_VERIFY_CAP_STEPLOG="${STEPLOG:+/root/.cache/steplog/$STEPLOG}" -e INKLING_VERIFY_CAP_REMAP="${DFCAP_REMAP:-1}"
        -e INKLING_DRAFT_BYTES="$DRAFT_BYTES" -e INKLING_DRAFT_HOTVOCAB_FILE="$HOTVOCAB_FILE" -e INKLING_KERNELS="$KERNELS" -e INKLING_MOE_TILES="$MOE_TILES"
        -e INKLING_PICO12="$PICO12" -e INKLING_PREFILL="$PREFILL" -e INKLING_PREFILL_QKVR_OUT=strided
        -e INKLING_CERT_HEAD="$CERT_HEAD" -e INKLING_WIDE="$WIDE" -e INKLING_WIDE_SCOPE="$WIDE_SCOPE"
        -e INKLING_VATTN="$VATTN" -e INKLING_ATTN="$ATTN" -e INKLING_PFUSE="$PFUSE" -e INKLING_PFUSE_HOIST="$PFUSE_HOIST" -e INKLING_ATTN_GATE="$ATTN_GATE" -e INKLING_PROLOGUE="$PROLOGUE" -e INKLING_MOE2="$MOE2" -e INKLING_SCODES="$SCODES"
        -e INKLING_VERIFY_CAP_FASTPROBS="$VCAP_FASTPROBS" -e INKLING_VERIFY_CAP_FUSED_ACCEPT="$VCAP_FUSED" -e INKLING_NOEXP="$NOEXP" -e INKLING_PADX="$PADX" -e INKLING_DFLASH_BYTES="$DFLASH_BYTES" -e INKLING_DFLASH_HOT_N="$DFLASH_HOT_N" -e INKLING_DFLASH_PREFIX_TAIL="$dftail" -e INKLING_DFLASH_PREFIX_TAIL_MAX_INPUT="$DFLASH_PREFIX_TAIL_MAX_INPUT" -e INKLING_DFLASH_TAIL_SNAPSHOTS="$dfsnap" -e INKLING_DFLASH_TAIL_DEFER="$dfdefer" -e INKLING_DFLASH_PCOPY="$dfpcopy" -e INKLING_DFLASH_PCOPY_MIN_CTX="$dfpcopymin" -e INKLING_PROMPT_TOUCH="$ptouch" -e INKLING_DFLASH_FUSED_ACCEPT="$DFLASH_FUSED" -e INKLING_RSFIX="$RSFIX"
        -e INKLING_KV12="$KV12" -e INKLING_KV12_MAX_PAGES="$KV12_SLOTS"
        -e INKLING_CERT_WIDE="$CERT_WIDE" -e INKLING_CERT_WIDE_MAX_ROWS="$CERT_WIDE_MAX_ROWS" -e INKLING_CERT_WIDE_STATS="$CERT_WIDE_STATS"
        -e INKLING_DRAFT_RNNHOIST_MAX_BS="$RNNHOIST_MAX_BS"
        -e INKLING_KV12S="$KV12S" -e INKLING_KV12S_OVERFLOW_HEADS="$KV12S_OVERFLOW_HEADS"
        -e INKLING_KV12S_FALLBACK_PAGES="$KV12S_FALLBACK_PAGES" ${kv12s_env[@]+"${kv12s_env[@]}"})
  fi
  run=("${common[@]}" ${ov[@]+"${ov[@]}"}
       -e MAX_JOBS=4
       -e SGLANG_OPT_INKLING2_VERIFY_REL_BIAS_KERNEL=1
       -e SGLANG_OPT_DSV41_DSPARK_MID_FOLD=1
       -e SGLANG_OPT_DSV41_DSPARK_PREPARE_INPUTS_ONCE=1
       -e SGLANG_OPT_DSV41_DSPARK_SKIP_PREFILLED_INPUTS=1
       ${SGLANG_ENV:-}
       ${nommap_env[@]+"${nommap_env[@]}"}
       "${entry[@]}"
       -m sglang.launch_server --model-path /model --served-model-name "$SERVED"
       --tp 1 --ep 1 --attention-backend fa4 --dtype bfloat16 --page-size 128
       ${spec[@]+"${spec[@]}"} ${kv12s_args[@]+"${kv12s_args[@]}"}
       --mem-fraction-static "$MEM_FRACTION" --context-length "$CTX"
       --grammar-backend xgrammar --reasoning-parser inkling --tool-call-parser inkling2
       --enable-metrics --host "$HOST" --port "$PORT" ${nommap_arg[@]+"${nommap_arg[@]}"} --swa-full-tokens-ratio "$SWA_RATIO"
       --max-mamba-cache-size 512 --max-prefill-tokens 8192 $SGLANG_ARGS "$@")
elif [[ $ENGINE == vllm ]]; then
  specj='{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}'
  spec=(--speculative-config "$specj")
  [[ $SPEC == none ]] && spec=()
  run=("${common[@]}" ${VLLM_ENV:-}
       "$VLLM_IMAGE"
       /model --served-model-name "$SERVED" --tensor-parallel-size 1
       --max-model-len "$CTX" --gpu-memory-utilization "$GPU_UTIL"
       --reasoning-parser inkling --enable-auto-tool-choice --tool-call-parser inkling
       ${spec[@]+"${spec[@]}"}
       --host "$HOST" --port "$PORT" $VLLM_ARGS "$@")
else
  echo "ENGINE must be sglang or vllm" >&2; exit 2
fi

echo "+ ${run[*]}" >&2
"${run[@]}" >/dev/null
echo "started $NAME ($ENGINE) on $HOST:$PORT; first start ~6-8 min (weights + JIT + CUDA graphs), later starts faster"

# Memory guard: on GB10 an out-of-memory during start-up can hang the whole box until a power cycle.
# scripts/memguard.sh stops the container if MemAvailable stays below MIN_AVAIL_GIB for two samples (2 s apart).
nohup "$(dirname "${BASH_SOURCE[0]}")/memguard.sh" "$NAME" "$MIN_AVAIL_GIB" "$GUARD_SECONDS" >"$CACHE_DIR/memguard.log" 2>&1 &
echo "memory guard: stops $NAME if MemAvailable < ${MIN_AVAIL_GIB} GiB in the next ${GUARD_SECONDS} s (log $CACHE_DIR/memguard.log)"
