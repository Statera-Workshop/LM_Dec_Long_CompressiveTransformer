<!-- PAI-SG-LINKS:BEGIN -->
## Project Links

- [Project Portal](https://statera-workshop.github.io/#repo-lm_dec_long_compressivetransformer)
- [Statera Workshop Portal](https://statera-workshop.github.io/)
<!-- PAI-SG-LINKS:END -->

# LM_Dec_Long_CompressiveTransformer

CompressiveTransformer - Long family, decoder-only byte-level language model, standardised in the Statera Working Group Language Model library.

## Status

- Verification level: **Benchmarked**
- Failure class: Note: failed all memory tests (recall, update, state tracking) in this setting

## Provenance

- Source repository: (new) lucidrains/compressive-transformer-pytorch
- Upstream / license: lucidrains/compressive-transformer-pytorch (MIT)

## Differences from the source ([Unified] protocol)

- segments of 128, memory 128 and compressed memory 32 (ratio 4) carried across segments

All models share one protocol: 256-byte vocabulary, dim 512, 6 layers, 8 heads x 64, Adam 2e-4 with 200 warm-up steps, gradient clip 0.5, bf16 autocast (automatic fp32 fallback). Numbers rank structures under one budget on one GPU (RTX 3070 Ti 8 GB); they do not reproduce paper results.

## Results

- Parameters: 44.52 M
- Causality check: pass (largest change of earlier outputs 0.0e+00)
- Overfit check (4 x 256 bytes): 5.556 -> 0.003 nats, pass
- LM-02 enwik8 (200M tokens): valid 1.748 / test 1.749 bits per byte
- LM-02 enwik8 (1B tokens): valid 1.504 / test 1.514 bits per byte
- Peak GPU memory at length 2048 (batch 1): train 958 MB, inference 101 MB
- Peak GPU memory at length 8192 (batch 1): train 3537 MB, inference 104 MB
- LM-05 recall `lm05v3`: 4 pairs @ 512: 0.10 (keys removed 0.09); 4 pairs @ 2048: 0.03 (keys removed 0.03); 16 pairs @ 512: 0.03 (keys removed 0.02); 16 pairs @ 2048: 0.03 (keys removed 0.03)
- LM-05 recall `lm05v3long`: 4 pairs @ 512: 0.27 (keys removed 0.26); 4 pairs @ 2048: 0.02 (keys removed 0.01); 4 pairs @ 8192: 0.01 (keys removed 0.01); 16 pairs @ 512: 0.05 (keys removed 0.04); 16 pairs @ 2048: 0.02 (keys removed 0.01); 16 pairs @ 8192: 0.02 (keys removed 0.01)
- LM-06 memory update `lm06`: @512: latest 0.01, unchanged 0.01, stale 0.02; @2048: latest 0.02, unchanged 0.01, stale 0.01
- LM-06 memory update `lm06long`: @512: latest 0.08, unchanged 0.02, stale 0.02; @2048: latest 0.02, unchanged 0.01, stale 0.01; @8192: latest 0.01, unchanged 0.01, stale 0.01

## Tests

- check: data counts, causality (no look-ahead, outputs use the current byte), parameters
- scaling: peak GPU memory and time vs length 512-8192
- overfit: memorise 4 x 256 bytes below 0.5 nats
- LM-02: enwik8 next-byte prediction (bits per byte)
- LM-05: long-horizon key-value recall with key-removed control (v3: distinct queries)
- LM-06: memory update - latest value vs stale value after a key is reassigned

## Run

```bash
pip install torch==2.5.1 einops numpy compressive-transformer-pytorch==0.4.0 mogrifier
python run.py check compressive
python run.py lm02 compressive      # enwik8 is downloaded to data/ on first use
```
