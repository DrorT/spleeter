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
    const modelPath = path.resolve(__dirname, 'models/2stems/model.json'); // Example with 2stems model
    let model;
    try {
        const modelLoader = new ModelLoader();
        model = await modelLoader.load(modelPath);
        console.log('Model loaded successfully.');
    } catch (error) {
        console.error('Error loading model:', error);
        return; // Stop if model loading fails
    }

    // 2. Warmup Process
    console.log('Step 2: Attempting model warmup...');
    try {
        // We need to determine the expected input shape and type for the warmup.
        // This will likely involve inspecting the model's input signature or
        // referring to the Python implementation.
        // For now, let's assume a common input shape for audio models.
        // This is a placeholder and will likely need adjustment.
        const warmupInputShape = [1, 512, 2, 1]; // Example: [batchSize, frames, channels, ?]
        // Or it could be [batchSize, frequencyBins, timeSteps, channels] for spectrograms
        
        // The Python implementation might use dummy data or a specific audio snippet.
        // Let's try with zeros first.
        const warmupInput = tf.zeros(warmupInputShape);

        console.log(`Using warmup input shape: ${warmupInputShape}`);
        
        // The model might have a specific method for warmup, or we might just run a prediction.
        // Let's assume a prediction for now.
        const warmupResult = model.predict(warmupInput);
        
        // Ensure the prediction completes by awaiting its data (if it's a promise)
        // or by synchronously getting data.
        if (warmupResult && warmupResult.dataSync) {
            warmupResult.dataSync(); // Force execution
        } else if (warmupResult && warmupResult.then) {
            await warmupResult; // If predict returns a promise
        }

        console.log('Warmup completed successfully.');
        console.log('Warmup output shape(s):');
        if (Array.isArray(warmupResult)) {
            warmupResult.forEach((tensor, i) => console.log(`  Output ${i}: ${tensor.shape}`));
        } else if (warmupResult && warmupResult.shape) {
            console.log(`  Output: ${warmupResult.shape}`);
        }

        // Dispose tensors to free memory
        warmupInput.dispose();
        if (Array.isArray(warmupResult)) {
            warmupResult.forEach(tensor => tensor.dispose());
        } else if (warmupResult) {
            warmupResult.dispose();
        }

    } catch (error) {
        console.error('Error during model warmup:', error);
        console.log('--- Error Details ---');
        console.error(error.message);
        if (error.stack) {
            console.error(error.stack);
        }
        console.log('--- End Error Details ---');
        // Potentially, we might want to inspect the model's input/output layers here
        // to understand the expected tensor shapes.
        if (model && model.inputs) {
            console.log('Model input signature:');
            model.inputs.forEach(input => console.log(`  Name: ${input.name}, Shape: ${input.shape}, Dtype: ${input.dtype}`));
        }
        if (model && model.outputs) {
            console.log('Model output signature:');
            model.outputs.forEach(output => console.log(`  Name: ${output.name}, Shape: ${output.shape}, Dtype: ${output.dtype}`));
        }
        return; // Stop if warmup fails
    }

    console.log('Spleeter warmup test finished successfully.');
}

// Run the test
testWarmup().catch(err => {
    console.error('An unexpected error occurred during the test setup:', err);
});
