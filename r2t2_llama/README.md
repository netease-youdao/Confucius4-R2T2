# r2t2_llama — llama.cpp decoding for Confucius4-R2T2

This folder adds llama.cpp decoding support to Confucius4-R2T2.

Point `GGUF_DIR` at a directory holding both the projector (`mmproj*`) and the language model (`*.gguf`). Both filenames are auto-discovered.

## Decode routes

| mode | audio encoder | decoder | notes |
|---|---|---|---|
| `stream_llama_hybrid` | vLLM-native encoder (PyTorch, GPU) | llama.cpp | recommended; best accuracy |
| `stream_llama` | llama.cpp (mel + encoder) | llama.cpp | fully in-process, no PyTorch encoder |
| `onetime_llama` | llama.cpp | llama.cpp | one-shot, no streaming state |

The hybrid route exists because the two encoders are not equally robust on
quiet audio onsets. It keeps the PyTorch encoder and swaps only the decoder,
which isolates decoding cost without changing encoding quality.

## Usage

### CLI

```bash
# hybrid streaming (recommended)
GGUF_DIR=/path/to/gguf python -m r2t2_llama.example_llama \
    --audio resources/test.wav \
    --model_path /path/to/Confucius4-R2T2 \
    --infer_mode stream_llama_hybrid \
    --language Chinese

# in-process llama.cpp streaming, end to end
GGUF_DIR=/path/to/gguf python -m r2t2_llama.example_llama \
    --audio resources/test.wav \
    --model_path /path/to/Confucius4-R2T2 \
    --infer_mode stream_llama

# one-shot; --model_path is the GGUF dir, no HF assets needed
python -m r2t2_llama.example_llama \
    --audio resources/test.wav \
    --model_path /path/to/gguf \
    --infer_mode onetime_llama
```

Other flags: `--chunk_size_ms` (default 160), `--lookahead_ms` (160),
`--unfixed_token_num` (1), `--context` (hotword / context hint).
Set `DEBUG_PRINT=1` for per-chunk decode traces.

The demo reuses `example.py` from the repository root, so it exercises the same chunk schedule as the vLLM demo.

`--unfixed_token_num` sets the provisional-token budget `U`: use `1` for
quality priority or `0` for latency priority. Only the finalized prefix
should be consumed as stable text. The paper's results have not been
established for these GGUF routes; cache reuse and suffix verification
remain future work.

### Python

```python
from r2t2_llama import R2T2LlamaASRModel

asr = R2T2LlamaASRModel.LlamaHybrid(
    processor_path="/path/to/Confucius4-R2T2",   # HF dir
    gguf_dir="/path/to/gguf",
)

state = asr.init_streaming_state(
    language="Chinese",
    unfixed_chunk_num=0,
    unfixed_token_num=1,
    chunk_size_sec=0.16,
)
for chunk in audio_chunks:                        # float32, 16 kHz, mono
    text, fixed_text = asr.streaming_transcribe(chunk, state, max_new_tokens=4)
final_text = asr.finish_streaming_transcribe(state, max_new_tokens=4)
```

## Prebuilt native artifacts

The llama.cpp source is not committed to this repository. Only build products
are shipped, pinned to the following build configuration:

| | |
|---|---|
| llama.cpp | 0.4.0 (build `b10950`) |
| commit | `ad6c66839af3c5646fba8c6c2e2087a1e4e38948` (2026-09-14) |
| ggml | 0.23.0 |
| compiler | GNU 14.4.0 |
| platform | Linux x86_64, CUDA |
| Python ABI | CPython 3.12 (`cp312`) |

Shipped libraries in `bin/`: `libllama.so.0.4.0`, `libmtmd.so.0.4.0`,
`libggml.so.0.23.0`, `libggml-base.so.0.23.0`, `libggml-cpu.so.0.23.0`,
`libggml-cuda.so.0.23.0`, plus their soname symlinks.

The extension is built for CPython 3.12. Other Python versions need a rebuild.

## Rebuilding the native extension

You only need to rebuild if you change the pinned llama.cpp version, target a different Python ABI, or build for a non-CUDA platform.

```bash
# 1. fetch the pinned llama.cpp
git clone https://github.com/ggml-org/llama.cpp third_party/llama.cpp
git -C third_party/llama.cpp checkout ad6c66839af3c5646fba8c6c2e2087a1e4e38948

# 2. configure (use -DGGML_CUDA=OFF for a CPU-only build)
cmake -S r2t2_llama -B build \
    -DLLAMA_CPP_DIR="$PWD/third_party/llama.cpp" \
    -DGGML_CUDA=ON \
    -DCMAKE_BUILD_TYPE=Release

# 3. build; the extension lands in r2t2_llama/native/
cmake --build build -j

# 4. refresh the shipped libraries
cp -a build/bin/lib*.so* r2t2_llama/bin/
```

Verify with `objdump -x <so> | grep RUNPATH`.

**Do not rename the extension.** The module name `qwen3asr_native` is baked into `native_ext.cpp`'s `PYBIND11_MODULE` declaration and into the compiled
`.so`; renaming the file without rebuilding breaks the import.

### Environment variables read by the extension

All are diagnostic and default to off.

| variable | effect |
|---|---|
| `QWEN3ASR_STOP_BIAS` | bias toward stop tokens; unset or `<= 0` is plain greedy argmax |
| `QWEN3ASR_EMBED_BF16` | keep audio embeddings in bf16 |
| `QWEN3ASR_LOGITS_TOPK` | dump top-k logits |
| `QWEN3ASR_EMBED_DUMP_PATH` | dump prefill audio embeddings |
| `QWEN3ASR_EMBED_DECODE_DUMP_PATH` | dump decode-step embeddings |

For `stream_llama_hybrid`, a gloo process group is created on port `29577`. Running several hybrid processes in parallel requires distinct ports — set
`HYBRID_MASTER_PORT` per process, otherwise the group fails with `EADDRINUSE`.
