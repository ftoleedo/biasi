# Biasi-ROCm — llama.cpp with custom HIP kernels for AMD Strix Halo (gfx1151)

Prebuilt binaries (see **Releases**) of **Biasi-ROCm**, a llama.cpp fork tuned for the Radeon 8060S iGPU in the Ryzen AI Max+ 395 ("Strix Halo", gfx1151). Binaries only, no source.

It runs the usual `llama-server` / `llama-bench` / `llama-perplexity`. The prefill and decode paths of the models below use custom kernels: MoE experts on WMMA, sparse attention (QSA), Gated DeltaNet, hyper-connections, fused norms, and an MTP speculative decoding path.

| Model | Prefill (tok/s) | Decode | Decode with MTP |
|---|---:|---:|---:|
| Qwen3.8-Flash-Next (unsloth UD-IQ4_XS), ~8.5K prompt | 1602 | 29.7 | 50.5 |
| Qwen3.6-35B-A3B (unsloth UD-Q4_K_XL), ~7.6K prompt | 2884 | 63.5 | 93.6 |
| gpt-oss-120b (MXFP4), ~7.4K prompt | 1392 | 51.2 | n/a |

These were measured on an ASUS ProArt PX13 (128 GB) on AC power, performance profile, 2026-10-09. The Flash-Next row is the best of 5 sessions; session-to-session spread was ~10% for prefill. The full comparison against llama.cpp upstream and halogen is in the release notes.

## Requirements

- **GPU**: AMD gfx1151 (Ryzen AI Max / Max+ 395 / 390 / 385). The binaries are built for gfx1151 only.
- **ROCm 10.0.0 runtime libraries for gfx1151** (TheRock build). Point `ROCM_LIB` at its `lib` directory; the default is `~/rocm-10.0.0/lib`.
- **Linux with glibc ≥ 2.43 and libstdc++ with GLIBCXX_3.4.32** (built on CachyOS; a current Arch-based distro works).
- **Enough GPU-visible memory for the model.** For the 87 GiB Flash-Next on a 128 GB machine we boot with `amdgpu.gttsize=126976 ttm.pages_limit=32505856 ttm.page_pool_size=32505856`.

## Usage

```bash
# download biasi-rocm-gfx1151-20261009.tar.gz from Releases (check it against SHA256SUMS), then:
tar xf biasi-rocm-gfx1151-20261009.tar.gz && cd biasi-rocm-gfx1151-20261009
./biasi-server.sh --version

# one model, OpenAI-compatible server
./biasi-server.sh -m /path/to/model.gguf -ngl 999 -fa on -c 65536 --jinja --host 127.0.0.1 --port 8080

# Qwen3.8-Flash-Next + MTP speculative decoding (as we run it daily)
./biasi-server.sh -m /path/to/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf \
  -md /path/to/mtp-Qwen3.8-Flash-Next-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 3 -ngld 99 \
  -ngl 999 -fa on -c 262144 -b 8192 -ub 4096 --parallel 3 --kv-unified -bs --jinja \
  --override-tensor per_layer_token_embd=CPU --host 127.0.0.1 --port 8080

# router: several models, one loaded at a time (edit the paths in examples/models.ini first)
./biasi-server.sh --models-preset examples/models.ini --models-max 1 --models-default next --host 127.0.0.1 --port 8080

./biasi-bench.sh -m /path/to/model.gguf -p 4096 -n 128 -fa on -ngl 999
```

The `biasi-*.sh` wrappers set the library path and the defaults we run in production: `GGML_CUDA_BLOCKING_SYNC=1`, `LLAMA_HEAD_2STAGE=256` and `LLAMA_RS_REPLAY=1`. Override any of them in the environment.

## Switches

Each of these environment variables turns off one optimization:

| Variable | Effect |
|---|---|
| `GGML_CUDA_FA_GATHER_WMMA9=0` | sparse-attention (QSA) WMMA kernel off; uses the scalar kernel |
| `GGML_CUDA_HC_RES16=0` | f32 hyper-connection residual instead of f16 |
| `GGML_CUDA_CONV_QKN_RAW=1` | GDN conv writes the raw q/k channels again |
| `LLAMA_HEAD_2STAGE=0` | full LM head every token |

## Files

| Path | Contents |
|---|---|
| `bin/` | `llama-server`, `llama-bench`, `llama-perplexity` and their libraries (RUNPATH `$ORIGIN`) |
| `biasi-server.sh`, `biasi-bench.sh`, `biasi-perplexity.sh` | the wrappers above |
| `examples/models.ini` | the router preset we use (Flash-Next + Qwen3.6-35B) |
| `BUILD.txt` | build details |
| `LICENSE` | llama.cpp (MIT) |

## License and disclaimer

Biasi-ROCm is a derivative of [llama.cpp](https://github.com/ggml-org/llama.cpp), MIT License (see `LICENSE`). Binaries are provided as-is, without warranty. This is not affiliated with AMD, ggml.ai or halogen. ROCm is not included.
