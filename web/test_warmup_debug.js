/**
 * Test script to debug the Spleeter model warmup process.
 * This script will load the model, attempt to warm it up,
 * and report any errors encountered.
 */

const path = require('path');
const tf = require('@tensorflow/tfjs'); // Assuming TensorFlow.js is used
const ModelLoader = require('./src/ModelLoader'); // Adjust path if necessary

async function testWarmup() {
    console.log('Starting Spleeter warmup test...');

    // 1. Model Loading
    console.log('Step 1: Loading the model...');
    let modelLoader;
    try {
        modelLoader = new ModelLoader();
        await modelLoader.loadModel('2stems');
        console.log('Model loaded successfully.');
    } catch (error) {
        console.error('Error loading model:', error);
        return; // Stop if model loading fails
    }

    // 2. Warmup Process
    console.log('Step 2: Attempting model warmup...');
    try {
        // Create proper inputs for the Spleeter model based on its signature
        // The model expects: audio_id, mix_stft, mix_spectrogram
        
        // 1. audio_id: string tensor
        const audioId = tf.fill([1], '');
        
        // 2. mix_stft: complex STFT tensor [1, 2049, 2]
        const stftShape = [1, 2049, 2];
        const stftReal = new Float32Array(2049);
        const stftImag = new Float32Array(2049);
        const mixStft = tf.complex(stftReal, stftImag).expandDims(0);
        
        // 3. mix_spectrogram: magnitude spectrogram [1, 512, 1024, 2]
        const spectrogramShape = [1, 512, 1024, 2];
        const spectrogramData = new Float32Array(512 * 1024 * 2);
        const mixSpectrogram = tf.tensor4d(spectrogramData, spectrogramShape);
        
        const inputs = {
            audio_id: audioId,
            mix_stft: mixStft,
            mix_spectrogram: mixSpectrogram
        };

        console.log('Using model inputs with shapes:', {
            audio_id: audioId.shape,
            mix_stft: mixStft.shape,
            mix_spectrogram: mixSpectrogram.shape
        });
        
        // Run inference using the updated predict method
        const predictions = await modelLoader.predict(inputs);

        console.log('Warmup completed successfully.');
        console.log('Warmup output shape(s):');
        if (Array.isArray(predictions)) {
            predictions.forEach((tensor, i) => console.log(`  Output ${i}: ${tensor.shape}`));
        } else if (predictions && predictions.shape) {
            console.log(`  Output: ${predictions.shape}`);
        }

        // Dispose tensors to free memory
        audioId.dispose();
        mixStft.dispose();
        mixSpectrogram.dispose();
        if (Array.isArray(predictions)) {
            predictions.forEach(tensor => tensor.dispose());
        } else if (predictions) {
            predictions.dispose();
        }

    } catch (error) {
        console.error('Error during model warmup:', error);
        console.log('--- Error Details ---');
        console.error(error.message);
        if (error.stack) {
            console.error(error.stack);
        }
        console.log('--- End Error Details ---');
        return; // Stop if warmup fails
    }

    console.log('Spleeter warmup test finished successfully.');
}

// Run the test
testWarmup().catch(err => {
    console.error('An unexpected error occurred during the test setup:', err);
});
