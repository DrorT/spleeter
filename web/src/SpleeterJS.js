/**
 * SpleeterJS - Main class for audio source separation in the browser
 * Based on Deezer's Spleeter library, ported to JavaScript with TensorFlow.js
 */
class SpleeterJS {
    constructor(options = {}) {
        // Initialize logger first
        this.logger = options.logger || new window.SpleeterJS.Logger({
            enabled: options.enableLogs !== false,
            level: options.logLevel || 'INFO'
        });

        // Configuration
        this.config = {
            sampleRate: 44100,
            frameLength: 4096,
            frameStep: 1024,
            nChannels: 2,
            chunkSize: options.chunkSize || 10, // seconds
            separationExponent: 2,
            maskExtension: 'zeros',
            enableLogs: options.enableLogs !== false,
            ...options.config
        };

        // Initialize components
        this.audioProcessor = new window.SpleeterJS.AudioProcessor({
            logger: this.logger,
            sampleRate: this.config.sampleRate,
            chunkSize: this.config.chunkSize
        });

        this.stftProcessor = new window.SpleeterJS.STFTProcessor({
            logger: this.logger,
            sampleRate: this.config.sampleRate,
            frameLength: this.config.frameLength,
            frameStep: this.config.frameStep,
            nChannels: this.config.nChannels
        });

        this.modelLoader = new window.SpleeterJS.ModelLoader({
            logger: this.logger
        });

        // State management
        this.isProcessing = false;
        this.isCancelled = false;
        this.separatedStems = null;
        this.processingStats = null;

        this.logger.info('SpleeterJS', 'SpleeterJS initialized', {
            config: this.config,
            components: ['AudioProcessor', 'STFTProcessor', 'ModelLoader']
        });
    }

    /**
     * Load a separation model
     * @param {string} modelType - Type of model to load ('2stems', '4stems', '5stems')
     * @param {Object} options - Model loading options
     * @returns {Promise<void>}
     */
    async loadModel(modelType = '2stems', options = {}) {
        try {
            this.logger.info('SpleeterJS', `Loading model: ${modelType}`);
            
            await this.modelLoader.loadModel(modelType, options);
            
            // Update config with model-specific parameters
            const modelConfig = this.modelLoader.getModelConfig(modelType);
            if (modelConfig) {
                this.config = { ...this.config, ...modelConfig };
                this.logger.debug('SpleeterJS', 'Configuration updated with model parameters', {
                    modelConfig
                });
            }
            
        } catch (error) {
            this.logger.error('SpleeterJS', `Failed to load model ${modelType}`, error);
            throw error;
        }
    }

    /**
     * Separate audio file into stems
     * @param {File} audioFile - Audio file to separate
     * @param {Object} options - Separation options
     * @returns {Promise<Object>} Separated stems
     */
    async separate(audioFile, options = {}) {
        const startTime = performance.now();
        
        try {
            if (this.isProcessing) {
                throw new Error('Already processing audio. Please wait or cancel current operation.');
            }

            this.isProcessing = true;
            this.isCancelled = false;
            
            this.logger.info('SpleeterJS', 'Starting audio separation', {
                fileName: audioFile.name,
                fileSize: audioFile.size,
                fileType: audioFile.type,
                options
            });

            // Load and process audio
            const audioBuffer = await this.audioProcessor.loadAudio(audioFile);
            
            // Perform separation
            const stems = await this.performSeparation(audioBuffer, options);
            
            // Store results
            this.separatedStems = stems;
            
            const endTime = performance.now();
            this.processingStats = {
                totalDuration: `${(endTime - startTime).toFixed(2)}ms`,
                audioDuration: audioBuffer.duration,
                processingSpeed: `${(audioBuffer.duration / ((endTime - startTime) / 1000)).toFixed(2)}x realtime`,
                modelType: this.modelLoader.getCurrentModel(),
                memoryUsage: this.modelLoader.getMemoryUsage()
            };

            this.logger.info('SpleeterJS', 'Audio separation completed successfully', {
                stems: Object.keys(stems),
                stats: this.processingStats
            });

            return stems;

        } catch (error) {
            this.logger.error('SpleeterJS', 'Audio separation failed', error);
            throw error;
        } finally {
            this.isProcessing = false;
            this.isCancelled = false;
        }
    }

    /**
     * Perform the actual separation process
     * @param {AudioBuffer} audioBuffer - Audio buffer to separate
     * @param {Object} options - Separation options
     * @returns {Promise<Object>} Separated stems
     */
    async performSeparation(audioBuffer, options = {}) {
        const modelType = this.modelLoader.getCurrentModel();
        if (!modelType) {
            throw new Error('No model loaded. Please load a model first.');
        }

        const modelConfig = this.modelLoader.getModelConfig(modelType);
        const chunkSize = options.chunkSize || this.config.chunkSize;
        
        // Get audio chunks for processing
        const chunks = this.audioProcessor.getChunks(chunkSize);
        this.logger.info('SpleeterJS', `Audio split into ${chunks.length} chunks for processing`);

        // Process chunks
        const separatedChunks = [];
        for (let i = 0; i < chunks.length; i++) {
            if (this.isCancelled) {
                this.logger.info('SpleeterJS', 'Separation cancelled by user');
                break;
            }

            this.logger.debug('SpleeterJS', `Processing chunk ${i + 1}/${chunks.length}`);
            
            const chunkResult = await this.processChunk(chunks[i], modelConfig);
            separatedChunks.push(chunkResult);
        }

        if (this.isCancelled) {
            throw new Error('Separation was cancelled');
        }

        // Combine chunks into final stems
        return this.combineChunks(separatedChunks, modelConfig.instruments);
    }

    /**
     * Process a single audio chunk
     * @param {AudioBuffer} chunk - Audio chunk to process
     * @param {Object} modelConfig - Model configuration
     * @returns {Promise<Object>} Separated chunk stems
     */
    async processChunk(chunk, modelConfig) {
        const startTime = performance.now();
        
        try {
            // Convert to spectrogram
            const stftResult = this.stftProcessor.computeSTFT(chunk);
            
            // Prepare input tensor for model
            const inputTensor = this.prepareModelInput(stftResult, modelConfig);
            
            // Run model inference
            const predictions = await this.modelLoader.predict(inputTensor);
            
            // Process predictions to get separated stems
            const separatedStems = this.processPredictions(predictions, stftResult, modelConfig);
            
            // Cleanup tensors
            inputTensor.dispose();
            if (Array.isArray(predictions)) {
                predictions.forEach(pred => pred.dispose());
            } else {
                predictions.dispose();
            }
            
            const endTime = performance.now();
            this.logger.debug('SpleeterJS', 'Chunk processing completed', {
                duration: `${(endTime - startTime).toFixed(2)}ms`,
                chunkDuration: chunk.duration
            });
            
            return separatedStems;
            
        } catch (error) {
            this.logger.error('SpleeterJS', 'Chunk processing failed', error);
            throw error;
        }
    }

    /**
     * Prepare model input tensor from STFT result
     * @param {Object} stftResult - STFT computation result
     * @param {Object} modelConfig - Model configuration
     * @returns {tf.Tensor} Input tensor for model
     */
    prepareModelInput(stftResult, modelConfig) {
        const { magnitude, nFrames, nFreqBins } = stftResult;
        
        // Reshape magnitude for model input
        // Expected shape: [batch, frames, freq_bins, channels]
        const inputArray = new Float32Array(nFrames * nFreqBins * 2); // 2 channels
        
        // Copy magnitude data to both channels (stereo input)
        for (let frame = 0; frame < nFrames; frame++) {
            for (let freq = 0; freq < nFreqBins; freq++) {
                const srcIndex = frame * nFreqBins + freq;
                const dstIndex = (frame * nFreqBins + freq) * 2;
                
                const value = magnitude[srcIndex];
                inputArray[dstIndex] = value;     // Left channel
                inputArray[dstIndex + 1] = value; // Right channel
            }
        }
        
        // Create tensor and reshape
        const inputTensor = tf.tensor(inputArray, [1, nFrames, nFreqBins, 2]);
        
        return inputTensor;
    }

    /**
     * Process model predictions to get separated stems
     * @param {tf.Tensor|Array<tf.Tensor>} predictions - Model predictions
     * @param {Object} stftResult - Original STFT result
     * @param {Object} modelConfig - Model configuration
     * @returns {Object} Separated stems
     */
    processPredictions(predictions, stftResult, modelConfig) {
        const { phase, nFrames, nFreqBins, fftSize, hopSize, sampleRate } = stftResult;
        const instruments = modelConfig.instruments;
        
        // Convert predictions to array if needed
        const predArray = Array.isArray(predictions) ? predictions : [predictions];
        
        const separatedStems = {};
        
        // Process each instrument prediction
        for (let i = 0; i < instruments.length; i++) {
            const instrument = instruments[i];
            const prediction = predArray[i];
            
            // Get prediction data
            const predData = prediction.dataSync();
            
            // Create magnitude spectrogram for this instrument
            const instrumentMagnitude = new Float32Array(nFrames * nFreqBins);
            
            // Copy prediction data (assuming output is same shape as input)
            for (let j = 0; j < instrumentMagnitude.length; j++) {
                instrumentMagnitude[j] = predData[j];
            }
            
            // Reconstruct waveform using ISTFT
            const instrumentStftData = {
                magnitude: instrumentMagnitude,
                phase: phase,
                nFrames,
                nFreqBins,
                fftSize,
                hopSize,
                sampleRate
            };
            
            const waveform = this.stftProcessor.computeISTFT(instrumentStftData);
            
            // Convert to AudioBuffer
            const audioBuffer = this.audioProcessor.float32ArrayToAudioBuffer({
                channels: [waveform],
                sampleRate,
                length: waveform.length,
                duration: waveform.length / sampleRate
            });
            
            separatedStems[instrument] = audioBuffer;
        }
        
        return separatedStems;
    }

    /**
     * Combine processed chunks into final stems
     * @param {Array<Object>} separatedChunks - Array of separated chunk results
     * @param {Array<string>} instruments - List of instruments
     * @returns {Object} Combined stems
     */
    combineChunks(separatedChunks, instruments) {
        const combinedStems = {};
        
        // Initialize combined stems
        for (const instrument of instruments) {
            const totalLength = separatedChunks.reduce((sum, chunk) => 
                sum + chunk[instrument].length, 0
            );
            
            combinedStems[instrument] = {
                channels: [new Float32Array(totalLength)],
                sampleRate: separatedChunks[0][instrument].sampleRate,
                length: totalLength,
                duration: totalLength / separatedChunks[0][instrument].sampleRate
            };
        }
        
        // Combine chunks
        let offset = 0;
        for (const chunk of separatedChunks) {
            for (const instrument of instruments) {
                const chunkData = chunk[instrument].getChannelData(0);
                combinedStems[instrument].channels[0].set(chunkData, offset);
            }
            offset += separatedChunks[0][instruments[0]].length;
        }
        
        // Convert to AudioBuffers
        const finalStems = {};
        for (const instrument of instruments) {
            finalStems[instrument] = this.audioProcessor.float32ArrayToAudioBuffer(
                combinedStems[instrument]
            );
        }
        
        return finalStems;
    }

    /**
     * Cancel current separation process
     */
    cancelSeparation() {
        if (this.isProcessing) {
            this.isCancelled = true;
            this.logger.info('SpleeterJS', 'Separation cancellation requested');
        }
    }

    /**
     * Get separated stems
     * @returns {Object|null} Separated stems or null if not available
     */
    getSeparatedStems() {
        return this.separatedStems;
    }

    /**
     * Get processing statistics
     * @returns {Object|null} Processing statistics or null if not available
     */
    getProcessingStats() {
        return this.processingStats;
    }

    /**
     * Enable or disable logging
     * @param {boolean} enabled - Whether to enable logging
     */
    enableLogs(enabled) {
        this.config.enableLogs = enabled;
        this.logger.enableLogs(enabled);
    }

    /**
     * Get logs
     * @param {string} level - Optional filter by log level
     * @returns {Array} Array of log entries
     */
    getLogs(level = null) {
        return this.logger.getLogs(level);
    }

    /**
     * Clear logs
     */
    clearLogs() {
        this.logger.clearLogs();
    }

    /**
     * Update configuration
     * @param {Object} newConfig - New configuration
     */
    updateConfig(newConfig) {
        this.config = { ...this.config, ...newConfig };
        
        // Update component configurations
        this.stftProcessor.updateConfig(this.config);
        
        this.logger.info('SpleeterJS', 'Configuration updated', {
            config: this.config
        });
    }

    /**
     * Get current configuration
     * @returns {Object} Current configuration
     */
    getCurrentConfig() {
        return { ...this.config };
    }

    /**
     * Get available models
     * @returns {Array<string>} Available model types
     */
    getAvailableModels() {
        return this.modelLoader.getAvailableModels();
    }

    /**
     * Get current model type
     * @returns {string|null} Current model type
     */
    getCurrentModel() {
        return this.modelLoader.getCurrentModel();
    }

    /**
     * Get model statistics
     * @returns {Object} Model statistics
     */
    getModelStats() {
        return this.modelLoader.getModelStats();
    }

    /**
     * Get memory usage information
     * @returns {Object} Memory usage data
     */
    getMemoryUsage() {
        return this.modelLoader.getMemoryUsage();
    }

    /**
     * Check if currently processing
     * @returns {boolean} True if processing
     */
    isProcessingAudio() {
        return this.isProcessing;
    }

    /**
     * Get system information
     * @returns {Object} System information
     */
    getSystemInfo() {
        return {
            userAgent: navigator.userAgent,
            platform: navigator.platform,
            memory: navigator.deviceMemory,
            hardwareConcurrency: navigator.hardwareConcurrency,
            webAudioAPI: !!(window.AudioContext || window.webkitAudioContext),
            webWorkers: typeof Worker !== 'undefined',
            offscreenCanvas: typeof OffscreenCanvas !== 'undefined',
            tensorflow: {
                version: tf?.version?.tfjs || 'not available',
                backend: tf?.engine()?.backendState?.backend || 'unknown'
            }
        };
    }

    /**
     * Dispose and cleanup resources
     */
    dispose() {
        this.modelLoader.dispose();
        this.audioProcessor.clear();
        this.separatedStems = null;
        this.processingStats = null;
        
        this.logger.info('SpleeterJS', 'SpleeterJS disposed');
    }
}

// Export for Node.js and browser
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SpleeterJS;
} else if (typeof window !== 'undefined') {
    window.SpleeterJS = window.SpleeterJS || {};
    window.SpleeterJS.SpleeterJS = SpleeterJS;
    
    // Create global instance for convenience
    window.Spleeter = SpleeterJS;
}
