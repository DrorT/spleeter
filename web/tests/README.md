# Browser Testing Instructions

## Testing the Model Loading Fix

1. **Open the test page in your browser:**

   - Navigate to `/home/chester/dev/music/spleeter/web/tests/test-browser.html`
   - Or serve the files using a local web server

2. **Test the model loading:**

   - Click "Load 2stems Model"
   - Check the browser console for debug output
   - Look for the success message and debug info

3. **Expected behavior:**
   - Model should load without shape mismatch errors
   - Console should show successful warmup completion
   - Debug info should display TensorFlow.js version and backend

## What was fixed:

- **Input shape mismatch**: Changed from `[1, 512, 1024, 2]` to `[1, 2]` for the Placeholder input
- **Model signature compliance**: Now matches the actual Spleeter model signature
- **Dynamic ops handling**: Switched from `execute()` to `executeAsync()` to handle TensorFlow dynamic ops (Merge nodes)
- **Correct input names**: Updated to use proper input names from model signature:
  - `audio_id` (string) instead of `Placeholder_1`
  - `mix_stft` (complex64, [-1,2049,2]) instead of `transpose_1`
  - `mix_spectrogram` (float32, [-1,512,1024,2]) instead of `strided_slice_3`
- **Proper warmup**: Uses correct tensor shapes and async execution for model initialization

## Model Signature:

```
Inputs:
- audio_id: DT_STRING []
- mix_stft: DT_COMPLEX64 [-1,2049,2]
- mix_spectrogram: DT_FLOAT [-1,512,1024,2]

Outputs:
- strided_slice_13: DT_FLOAT [-1,2] (vocals)
- strided_slice_23: DT_FLOAT [-1,2] (accompaniment)
```

## Debug Output:

The test page now includes detailed console logging to help verify:

- ModelLoader initialization
- Available models list
- Model loading progress
- Successful warmup completion (using executeAsync)
- TensorFlow.js version and backend info
