/**
 * Test script to debug the Spleeter model warmup process.
 * This script will load the model, attempt to warm it up,
 * and report any errors encountered.
 */

const path = require('path');
const tf = require('@tensorflow/tfjs-node'); // Use tfjs-node for Node.js environment

// Make TensorFlow.js available globally for the ModelLoader
global.tf = tf;

const ModelLoader = require('./src/ModelLoader'); // Adjust path if necessary

async function testWarmup() {
    console.log('Starting Spleeter warmup test...');

    // 1. Model Loading
    console.log('Step 1: Loading the model...');
    let modelLoader;
    try {
        modelLoader = new ModelLoader();
        // Use file:// protocol for Node.js
        const modelPath = 'file://' + path.resolve(__dirname, 'models/2stems/model.json');
        await modelLoader.loadModel('2stems', { path: modelPath });
        console.log('Model loaded successfully.');
    } catch (error) {
        console.error('Error loading model:', error);
        return; // Stop if model loading fails
    }

    // 2. Warmup Process
    console.log('Step 2: Attempting model warmup...');
    try {
        // Create proper inputs for the Spleeter model based on actual model expectations
        // The model expects: Placeholder (raw audio waveform) with shape [-1, 2]
        
        // 1. audio_id: string tensor
        const audioId = tf.fill([1], '');
        
        // 2. Placeholder: raw audio waveform with shape [-1, 2]
        // This is the actual input the model expects - raw audio samples
        const audioLength = 44100; // 1 second of audio at 44100 Hz
        const audioData = new Float32Array(audioLength * 2); // Stereo audio
        const waveform = tf.tensor2d(audioData, [audioLength, 2]); // Shape: [samples, channels]
        
        const inputs = {
            audio_id: audioId,
            Placeholder: waveform // Use the actual input name the model expects
        };

        console.log('Using model inputs with shapes:', {
            audio_id: audioId.shape,
            Placeholder: waveform.shape
        });
        
        // Run inference using model.executeAsync() as suggested by the error message
        const model = modelLoader.getModel('2stems');
        
        // The model expects conv2d_13/SpaceToBatchND as input, which is likely the spectrogram
        // The error suggests the model expects batch size divisible by 4 (2*2 block shape)
        const batchSize = 4; // Use batch size divisible by 4
        const T = 512; // Fixed time dimension
        const F = 1024; // Fixed frequency dimension
        const nChannels = 1; // Use mono (single channel) as expected by the model
        const spectrogramData = new Float32Array(batchSize * T * F * nChannels);
        const spectrogram = tf.tensor4d(spectrogramData, [batchSize, T, F, nChannels]);
        
        const modelInputs = {
            'conv2d_13/SpaceToBatchND': spectrogram  // Use the exact input name the model expects
        };
        
        const predictions = await model.executeAsync(modelInputs);

        console.log('Warmup completed successfully.');
        console.log('Warmup output shape(s):');
        if (Array.isArray(predictions)) {
            predictions.forEach((tensor, i) => console.log(`  Output ${i}: ${tensor.shape}`));
        } else if (predictions && predictions.shape) {
            console.log(`  Output: ${predictions.shape}`);
        }

        // Dispose tensors to free memory
        audioId.dispose();
        waveform.dispose();
        spectrogram.dispose();
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
