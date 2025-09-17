/**
 * Node.js test script to debug the Spleeter model inference issue.
 * This script tests the core model loading and inference logic.
 */

const path = require('path');
const tf = require('@tensorflow/tfjs');

// Mock window object for Node.js environment
global.window = {
    SpleeterJS: {},
    // Make TensorFlow.js available on window object for ModelLoader
    tf: tf
};

// Make TensorFlow.js available globally for ModelLoader
global.tf = tf;

// Mock logger
class MockLogger {
    constructor() {
        this.logs = [];
    }

    info(component, message, data = null) {
        this.log('info', component, message, data);
    }

    debug(component, message, data = null) {
        this.log('debug', component, message, data);
    }

    warn(component, message, data = null) {
        this.log('warn', component, message, data);
    }

    error(component, message, error = null) {
        this.log('error', component, message, error);
    }

    log(level, component, message, data = null) {
        const logEntry = {
            timestamp: new Date().toISOString(),
            level,
            component,
            message,
            data
        };
        this.logs.push(logEntry);
        console.log(`[${level}] [${component}] ${message}`, data || '');
    }
}

// Set up mock SpleeterJS namespace
window.SpleeterJS.Logger = MockLogger;

// Load ModelLoader after setting up mocks
const ModelLoader = require('./src/ModelLoader');

async function testModelInference() {
    console.log('Starting Spleeter model inference test...');

    let modelLoader;
    try {
        // 1. Initialize ModelLoader
        console.log('Step 1: Initializing ModelLoader...');
        const logger = new MockLogger();
        modelLoader = new ModelLoader({ logger });
        console.log('ModelLoader initialized successfully.');

        // 2. Load model with absolute path
        console.log('Step 2: Loading model...');
        const modelPath = 'file://' + path.resolve(__dirname, 'models/2stems/model.json');
        await modelLoader.loadModel('2stems', { path: modelPath });
        console.log('Model loaded successfully.');

        // 3. Test model inference with proper inputs
        console.log('Step 3: Testing model inference...');
        
        // Create proper inputs for the Spleeter model based on its signature
        // The model expects: audio_id, mix_stft, mix_spectrogram
        
        // 1. audio_id: string tensor
        const audioId = tf.fill([1], '');
        
        // 2. mix_stft: complex STFT tensor [1, 2049, 2]
        const stftReal = new Float32Array(2049);
        const stftImag = new Float32Array(2049);
        const mixStft = tf.complex(stftReal, stftImag).expandDims(0);
        
        // 3. mix_spectrogram: magnitude spectrogram [1, 512, 1024, 2]
        const spectrogramData = new Float32Array(512 * 1024 * 2);
        const mixSpectrogram = tf.tensor4d(spectrogramData, [1, 512, 1024, 2]);
        
        const inputs = {
            audio_id: audioId,
            mix_stft: mixStft,
            mix_spectrogram: mixSpectrogram
        };

        console.log('Model inputs created with shapes:', {
            audio_id: audioId.shape,
            mix_stft: mixStft.shape,
            mix_spectrogram: mixSpectrogram.shape
        });
        
        // Run inference using the updated predict method
        const predictions = await modelLoader.predict(inputs);

        console.log('Model inference completed successfully!');
        console.log('Output shape(s):');
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

        console.log('Tensor cleanup completed.');

    } catch (error) {
        console.error('Error during model inference test:', error);
        console.log('--- Error Details ---');
        console.error('Message:', error.message);
        if (error.stack) {
            console.error('Stack:', error.stack);
        }
        console.log('--- End Error Details ---');
        
        // Print logs for debugging
        if (modelLoader && modelLoader.logger) {
            console.log('\n--- Logs ---');
            modelLoader.logger.logs.forEach(log => {
                console.log(`[${log.level}] [${log.component}] ${log.message}`, log.data || '');
            });
            console.log('--- End Logs ---');
        }
        
        throw error;
    }

    console.log('Spleeter model inference test completed successfully!');
}

// Run the test
testModelInference().catch(err => {
    console.error('Test failed:', err);
    process.exit(1);
});
