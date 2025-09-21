# Spleeter Python Pipeline Reference

End‑to‑end description of how the original Python implementation converts an input audio file into model inputs, performs source separation, and converts model outputs back to audio waveforms.

---

## 1. High‑Level Flow

```
Audio File → Decode (ffmpeg) → Waveform (T × C float32)
  → (Prediction path) STFT inside model graph → Magnitude Spectrogram (time × freq × C) → Model (U‑Net / BLSTM) → Estimated Source Magnitudes
  → Mask Computation (+ optional multichannel Wiener filtering) → Estimated Complex STFTs per Source → Inverse STFT → Separated Waveforms (T' × C) → (Optional) Resave (ffmpeg)
```

(Training adds: dataset CSV → multiple instruments loaded → spectrogram preprocessing, cropping, augmentation, uint8 log‑db compression cache → batching.)

---

## 2. Audio Ingestion

- Loader: `FFMPEGProcessAudioAdapter` (`spleeter/audio/ffmpeg.py`).
- Uses `ffmpeg.probe` to read stream metadata (channels, sample rate). If caller supplies a `sample_rate`, resampling is requested via ffmpeg `-ar`.
- Output format requested from ffmpeg: raw 32‑bit float little‑endian (`f32le`).
- Buffer is reshaped to `(num_samples, n_channels)` and cast to requested dtype (default `float32`).
- During inference (`Separator.separate_to_file`):
  - `AudioAdapter.load(audio_path, offset, duration, sample_rate)` yields `(waveform, sample_rate)`.
  - If waveform is mono it is converted to stereo via `to_stereo` (simple channel duplication) before feeding model.

Waveform Tensor Characteristics:

- Shape: `(T, C)` with `C = 2` enforced.
- dtype: `float32`.
- Value range: raw linear PCM floats from ffmpeg (typically within [-1, 1]).

---

## 3. Spectrogram Computation (Two Contexts)

### 3.1 Training / Dataset Pipeline (`dataset.py`)

For each instrument (including synthetic "mix"):

1. Load waveforms per instrument path (central or enumerated segments) using the adapter’s `load_waveform` wrapper which wraps Python I/O via `tf.py_function` and returns `{waveform, waveform_error}`.
2. Compute magnitude spectrogram with `compute_spectrogram_tf` (`audio/spectrogram.py`).
   - Uses `tf.signal.stft` with Hann window (periodic=True) raised to `window_exponent`.
   - Parameters from config (example default MUSDB configs):
     - `frame_length` (e.g., 4096 for BLSTM, 2048 for U‑Net 16kHz models)
     - `frame_step` (hop, e.g., 1024 or 512).
   - Returns `abs(STFT) ** spec_exponent` (usually magnitude since `spec_exponent = 1`).
3. Frequency cropping: keep only lowest `F` bins (config param), ensuring `F ≤ frame_length/2 + 1`.
4. (Optional) Convert to uint8 compressed log‑dB space for caching:
   - `spectrogram_to_db_uint` → stores per‑sample min/max dB values to allow reversible scaling.
5. Cache (if enabled) stores the reduced, uint spectrograms (space efficient).
6. After cache reload: filter infinities, maybe repeat, harmonize lengths across instruments (truncate to min time length), ensure ≥ T frames.
7. Time cropping:
   - Random crop (training) or central crop (validation) to exactly `T` frames; ensures uniform shape `(T, F, C)`.
8. (Optional) Data augmentation (synchronized across sources): random time stretch and pitch shift performed by resizing spectrogram tensors (approximate in magnitude domain).
9. Convert back to float magnitudes if uint compression was used.
10. Shape check + explicit set tensor shape.
11. Mapping to model I/O:
    - Inputs dict: `{mix_spectrogram: (B, T, F, C)}`.
    - Labels dict: `{instrument_spectrogram: (B, T, F, C) for each instrument except mix}`.

### 3.2 Inference Path

- The raw waveform is provided directly to the estimator as feature `waveform` plus `audio_id`.
- Inside model graph (see `model/__init__.py` EstimatorSpecBuilder):
  - STFT computed (same `frame_length`/`frame_step`) for the mixture only, producing complex tensor `(N_frames, frame_length/2+1, C)`.
  - Magnitude (or power if `spec_exponent!=1`) used as model input.
  - Model outputs per instrument reduced-frequency magnitudes shaped `(N_frames, F, C)` (batch dimension absent in predict mode because `yield_single_examples=False`).

---

## 4. Model Inputs and Architectures

Two core architectures supplied: U‑Net (`model/functions/unet.py`) and BLSTM (`model/functions/blstm.py`). Selection handled by configuration JSON.

Common Input Tensor Concept:

- Reduced magnitude spectrogram of the mixture with shape `(T, F, C)`.
- Batch dimension added during training: `(B, T, F, C)` (TensorFlow often channels-last; for convolutions the input gets an extra singleton instrument/mask stack dimension as operations proceed).

### 4.1 U‑Net

- Treats spectrogram as a 2D image (time × frequency) with `C` channels.
- Downsampling path: series of strided `Conv2D` layers (kernel 5×5, stride 2×2) with batch norm + activation (configurable: ReLU/ELU/LeakyReLU) expanding filters list `conv_n_filters` (default `[16, 32, 64, 128, 256, 512]`).
- Upsampling path: `Conv2DTranspose` layers mirroring down path, concatenating skip connections, with dropout (0.5) in first three decoder stages.
- Final mask estimation stack:
  - If `output_mask_logit=False`: produce a 2‑channel mask with sigmoid, multiply (elementwise) with input mixture magnitude to yield estimated instrument magnitude (one network per instrument) OR
  - If softmax variant: build logits for each instrument, stack, apply softmax across instrument axis so masks sum to 1, then elementwise multiply by mixture magnitude.
- Output per instrument key: `<instrument>_spectrogram` (predicted magnitude, reduced F bins).

### 4.2 BLSTM

- Flattens (time-distributed) the 2D freq×channel plane per time frame.
- 3 stacked bidirectional CuDNNLSTM layers (units configurable, default 250 each direction) with `return_sequences=True`.
- TimeDistributed dense → reshape back to `(T, F, C)` magnitude for each instrument.

---

## 5. Mask Construction and Separation Logic

Implemented in `model/__init__.py` (EstimatorSpecBuilder):

1. Collect model output magnitudes per instrument (already reduced to `F`).
2. Compute energy‑ratio masks (if not softmax version) with `separation_exponent` (config, often 2.0 for powered ratio):
   `mask_i = (mag_i ** separation_exponent + epsilon / N) / (Σ_j mag_j ** separation_exponent + epsilon)`.
3. Extend masks from reduced `F` bins to full STFT bin count (`frame_length/2 + 1`) by either:
   - Average: replicate the mean value of existing frequency bins into the high-frequency extension region, or
   - Zeros: pad with zeros (config `mask_extension`).
4. Reshape & truncate to match actual STFT time dimension (handles padding in framing during STFT).
5. Multiply complex mixture STFT (`stft_feature`) by each instrument mask to obtain complex estimated STFT per source.
6. Optional Multichannel Wiener Filtering (MWF) if `MWF=True`:
   - Stack estimated magnitudes, call `norbert.wiener(v, x)` via `tf.py_function` where:
     - `v`: stacked per-source (padded) magnitude spectrograms shaped to match complex STFT dims.
     - `x`: original complex mixture STFT.
   - Produces refined complex STFT per source (iterative EM refinement in multi-channel domain).
7. Else: direct masked STFTs are used (ratio masking).

---

## 6. Inverse STFT and Waveform Reconstruction

- Function `_inverse_stft` in `EstimatorSpecBuilder`:
  - Applies `tf.signal.inverse_stft` with same `frame_length`, `frame_step`, Hann window.
  - Multiplies by `WINDOW_COMPENSATION_FACTOR` (constant in builder; compensates Hann window overlap so forward+inverse approximates identity for exponent choices used).
  - Transposes back to `(time, channels)` waveform.
  - Crops leading and trailing padding: returns slice `[frame_length : frame_length + time_crop]` to remove initial synthesis delay and limit to original mixture length.
- Output waveforms stored in prediction dict under instrument names (e.g., `vocals`, `accompaniment`).

Waveform Output Characteristics:

- Shape: `(T_original, C)` (same channel count as input, enforced stereo).
- dtype: float32.
- Amplitude: mixture-masked reconstruction (not peak-normalized unless downstream user normalizes). MWF can slightly modify amplitude distribution.

---

## 7. Export

- `Separator.save_to_file` iterates instrument→waveform mapping and calls `AudioAdapter.save`:
  - ffmpeg writes using raw float32 pipe in, with sample rate = model config sample rate.
  - Codec & bitrate optional (default WAV / PCM). For compressed formats, selects appropriate ffmpeg encoder (`aac`, `libvorbis`, etc.).
- Filenames parameterized by `{filename}/{instrument}.{codec}` format string (customizable; conflict detection included).

---

## 8. Configuration Parameters (Embedded JSON)

Key relevant fields (names inferred from code & standard Spleeter configs):

- `sample_rate`: 44100 or 16000.
- `frame_length`, `frame_step`.
- `F`: number of retained frequency bins after cropping (≤ frame_length/2 + 1).
- `T`: number of time frames per training crop.
- `instrument_list`: e.g. `["vocals","accompaniment"]`, or for 4/5 stems adds `drums`, `bass`, `other`, `piano`.
- `n_channels`: enforced 2.
- Model section: filter counts, activations, separation exponent, optimizer parameters.
- `mask_extension`: `average` or `zeros`.
- `separation_exponent`: exponent for mask energy computation (1 → magnitude, 2 → power).
- `MWF`: runtime flag (not baked into JSON) enabling Wiener refinement.

---

## 9. Data Compression (Training Cache)

- Conversion: magnitude → dB → scaled to uint8 (0–255) with per-sample min/max retained to allow linear rescaling back to dB then linear gain.
- Benefit: drastically reduces disk footprint and I/O pressure for repeated dataset epochs.
- Lossless relative to stored min/max precision (since it stores extremes per example and uses uniform quantization across that span).

---

## 10. Shapes Summary

| Stage                       | Shape (single example, stereo)                    |
| --------------------------- | ------------------------------------------------- |
| Waveform                    | `(T_samples, 2)`                                  |
| STFT (complex)              | `(N_frames, frame_length/2+1, 2)` before cropping |
| Magnitude (reduced)         | `(N_frames, F, 2)`                                |
| Training crop               | `(T, F, 2)`                                       |
| Batch                       | `(B, T, F, 2)`                                    |
| Model output per instrument | `(T, F, 2)`                                       |
| Mask (extended & flattened) | `(N_frames, frame_length/2+1, 2)`                 |
| Estimated complex STFT      | `(N_frames, frame_length/2+1, 2)`                 |
| Reconstructed waveform      | `(T_original, 2)`                                 |

---

## 11. Key Implementation Files

- Loading & Saving: `audio/ffmpeg.py`, `audio/adapter.py`.
- Spectrogram: `audio/spectrogram.py` (training path); inline STFT logic in `model/__init__.py` for inference.
- Dataset pipeline: `dataset.py`.
- Models: `model/functions/unet.py`, `model/functions/blstm.py`.
- Separation Orchestration: `separator.py`.
- Configuration: `utils/configuration.py` + embedded JSON in `spleeter/resources`.
- Postprocessing & Masks: `model/__init__.py` (mask extension, inverse STFT, MWF).

---

## 12. Potential Edge/Design Notes

- Mono inputs duplicated → could degrade separation vs native mono training; ensure pre-mix is stereo if possible.
- High frequency band (above retained `F`) is reconstructed via average or zeros extension; choosing `average` preserves energy but may smear high-frequency detail; `zeros` can cause energy loss.
- MWF introduces CPU overhead due to Python `py_function` and NumPy-based `norbert` operations; can be disabled for speed.
- Window compensation factor ensures approximate perfect reconstruction; mismatch between `window_exponent` and compensation would cause loudness bias.
- Augmentations (time/pitch stretch) operate purely on magnitude, ignoring phase — acceptable during training for invariance but approximate physically.

---

## 13. End‑to‑End Example (Inference)

1. User calls `Separator('spleeter:2stems').separate_to_file('track.wav', 'out_dir')`.
2. Load 30s (default duration 600s so effectively full) segment at configured sample rate (e.g., 44100) via ffmpeg.
3. Ensure stereo, feed `waveform` (Tensor shape `(T,2)`) to estimator `predict`.
4. Graph computes STFT, builds mixture magnitude, feeds through chosen model producing per‑instrument magnitudes.
5. Compute ratio masks (or softmax masks) with exponent; extend to full frequency.
6. Apply masks to complex mixture STFT (or refine with MWF) → per-instrument complex STFTs.
7. Inverse STFT and crop → separated waveforms.
8. Save each waveform with ffmpeg using requested codec.

---

## 14. End‑to‑End Example (Training Sample)

1. CSV lists relative paths for mix and each instrument (e.g., `mixture/vocals.wav`, `mixture/bass.wav`, ...).
2. Build dataset -> expand paths -> compute song duration & segmentation positions.
3. For each segment/instrument: load waveform, compute magnitude spectrogram, crop F bins.
4. Convert to uint8 log-dB and cache; repeat for other instruments.
5. After cache: harmonize lengths, filter, crop time window of length `T` frames.
6. (Optional) Apply synchronized augmentation (time-stretch + pitch-shift) to each spectrogram.
7. Convert back to float magnitudes; batch.
8. Feed to model as input `mix_spectrogram` with targets `<instrument>_spectrogram`.
9. Train loss built over predicted vs ground truth magnitudes (details in loss builder, not shown in extracted snippets but standard L1/L2 or composite).

---

## 15. Glossary

- STFT: Short-Time Fourier Transform producing complex matrix (time frames × frequency bins × channels).
- Magnitude Spectrogram: Absolute value of STFT (optionally exponentiated) used as model feature space.
- Mask: Elementwise weighting applied to mixture complex STFT to extract estimated source STFT.
- MWF: Multichannel Wiener Filtering iterative refinement algorithm using per-source variance estimates.
- `F`: Retained frequency bins subset feeding model, reduces compute & memory.
- `T`: Time frames per training crop enabling fixed-size network inputs.

---

## 16. References

- Jansson et al., ISMIR 2017 (U‑Net for singing voice separation).
- Uhlich et al., ICASSP 2017 (BLSTM with data augmentation / blending, multi-channel Wiener filtering).
- Norbert library for multichannel Wiener filtering implementation.

---

This document can be updated as the JS port evolves; keep as authoritative reference for parity.

---

## 17. C++ Inference Implementation (spleeterpp) Notes

The `spleeterpp` directory offers a lightweight TensorFlow C API based inference layer. It does NOT re‑implement the DSP steps (STFT, masking, MWF); instead it loads an already-exported TensorFlow graph whose output nodes are the separated waveforms directly. Key observations:

### 17.1 Entry Points and Data Structures

- Public API (`spleeterpp/spleeter.h`) exposes overloaded `Split` functions for 2, 4, or 5 stems.
- `Waveform` type (from `spleeter_common`) behaves like a 2×T (rows=channels, cols=frames) Eigen matrix (inference code uses `input.cols()` as frame count and `input.rows()` as channel count when building the TF tensor dims).
- Input TF tensor dims are set as `{input.cols(), input.rows()}` (i.e. `{frames, channels}`) rather than `{channels, frames}`; matches how the exported graph expects the placeholder (likely a 2D time-major tensor). This is a subtle ordering difference relative to Python training code which uses `(T, C)` internally but transposes inside STFT calculations—ensure consistency when exporting models for C++.

### 17.2 Model Invocation

- `RunModel` (`model.cc`):
  - Resolves a `(session, graph)` bundle by `SeparationType` through a registry.
  - Locates input placeholder by the fixed operation name `"Placeholder"`.
  - Discovers output operation names manually (hardcoded) via `GetOutputNames` using strings like `strided_slice_13`, etc. These correspond to graph nodes that already yield per‑stem waveform tensors.
  - Builds a single input `TF_Tensor` that directly wraps existing waveform memory (no copy) and uses a no‑op deleter; caller must ensure the input memory outlives the session run.
  - Creates output handles and runs `TF_SessionRun`.

### 17.3 Output Extraction

- `SetOutput` allocates each destination `Waveform` with shape `2 × frame_count` (channels first dimension) and memcpy/copies returned float buffer sequentially into that matrix.
- Output tensors are assumed to already be stereo waveforms of identical length to input (`frame_count = input.cols()`). No explicit inverse STFT or mask application occurs in C++ layer.

### 17.4 Graph Export Implications

- Compared to Python pipeline (which performs STFT → masking → iSTFT dynamically), the exported graph encapsulates those steps (and possibly mask extension & optional Wiener filtering if baked at export). Thus the observable I/O contract for C++ inference is simplified:
  - Input: 2‑D float32 tensor `[num_frames, 2]` (time major) representing stereo mixture.
  - Outputs: N tensors (N = number of stems) each `[num_frames, 2]` float32 stereo waveforms.
- Any change to internal preprocessing (e.g., different STFT parameters, extra normalization) requires re‑exporting the graph; C++ layer has no adaptive logic.

### 17.5 Hardcoded Output Node Names

- Output operation names (e.g., `strided_slice_13`) are fragile and version‑dependent. Recommendation: in future exports, add identity nodes with stable semantic names (e.g., `vocals_waveform`, `bass_waveform`) and update `GetOutputNames` accordingly to reduce breakage risk.

### 17.6 Memory Layout & Performance Considerations

- Zero‑copy wrapping of input buffer avoids allocation but risks use‑after‑free if caller scope ends prematurely; ensure lifetime management or provide a deleter capturing a retained buffer.
- Copy on outputs is unavoidable unless downstream consumers can operate directly on `TF_Tensor` memory; a potential optimization path.
- Input dimension ordering `{frames, channels}` must be mirrored when preparing data from other languages; mismatched shape ordering would silently produce shape errors or transposed audio.

### 17.7 Differences vs Python Pipeline Summary

| Aspect             | Python Separator                            | C++ (`spleeterpp`)                                |
| ------------------ | ------------------------------------------- | ------------------------------------------------- |
| STFT/Mask/iSTFT    | Computed at runtime inside Estimator graph  | Already embedded in exported graph nodes          |
| Input Provided     | Raw waveform `(T,2)` → graph builds STFT    | Raw waveform `[frames,2]` directly to placeholder |
| Outputs            | Dict of separated waveforms (post iSTFT)    | Vector of waveforms from named slice nodes        |
| Config Flexibility | Controlled via JSON & flags (MWF optional)  | Fixed at export (MWF status baked)                |
| Node Naming        | Semantically named outputs (`vocals`, etc.) | Obscure TF op names (`strided_slice_*`)           |
| Memory Handling    | NumPy arrays / TF tensors owned by runtime  | Manual TF tensor creation, no‑copy input          |

### 17.8 Actionable Notes for JS / Future Ports

- When reproducing the C++ inference style (e.g., for WebAssembly), prefer exporting a graph (or ONNX) with stable, semantic I/O node names.
- Document the required input shape explicitly to avoid confusion over `[channels, frames]` vs `[frames, channels]` ordering.
- Provide a lightweight shape validator before session run to produce clear errors.
- Consider embedding model metadata (sample rate, stem names, expected normalization) in a sidecar JSON rather than inferring from build type.

---
