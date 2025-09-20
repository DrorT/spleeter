/**
 * ModelLoader class for SpleeterJS - handles TensorFlow.js model loading and management
 */
class ModelLoader {
  constructor(options = {}) {
    // Handle both browser and Node.js environments
    let Logger;
    if (typeof window !== 'undefined' && window.SpleeterJS && window.SpleeterJS.Logger) {
      Logger = window.SpleeterJS.Logger;
    } else {
      Logger = require('./Logger.js');
    }
    
    this.logger = options.logger || new Logger();
    this.models = new Map();
    this.modelConfigs = this.initializeModelConfigs();
    this.currentModel = null;

    // Check if TensorFlow.js is available
    if (typeof tf === "undefined") {
      this.logger.error(
        "ModelLoader",
        "TensorFlow.js is not loaded. Please include TensorFlow.js in your project."
      );
      throw new Error("TensorFlow.js is required but not loaded");
    }

    this.logger.info("ModelLoader", "ModelLoader initialized", {
      availableModels: Object.keys(this.modelConfigs),
      tfVersion: tf.version.tfjs,
    });
  }

  /**
   * Initialize model configurations based on Spleeter's structure
   * @returns {Object} Model configurations
   */
  initializeModelConfigs() {
    const basePath =
      typeof window === "undefined" ? __dirname + "/../models/" : "../models/";
    return {
      "2stems": {
        path: basePath + "2stems/model.json",
        instruments: ["vocals", "accompaniment"],
        sampleRate: 44100,
        frameLength: 4096,
        frameStep: 1024,
        nChannels: 2,
        description: "Separates audio into vocals and accompaniment",
      },
      "4stems": {
        path: basePath + "4stems/model.json",
        instruments: ["vocals", "drums", "bass", "other"],
        sampleRate: 44100,
        frameLength: 4096,
        frameStep: 1024,
        nChannels: 2,
        description: "Separates audio into vocals, drums, bass, and other",
      },
      "5stems": {
        path: basePath + "5stems/model.json",
        instruments: ["vocals", "drums", "bass", "piano", "other"],
        sampleRate: 44100,
        frameLength: 4096,
        frameStep: 1024,
        nChannels: 2,
        description:
          "Separates audio into vocals, drums, bass, piano, and other",
      },
    };
  }

  /**
   * Load a model
   * @param {string} modelType - Type of model to load ('2stems', '4stems', '5stems')
   * @param {Object} options - Loading options
   * @returns {Promise<tf.LayersModel>} Loaded TensorFlow.js model
   */
  async loadModel(modelType, options = {}) {
    const startTime = performance.now();

    try {
      if (!this.modelConfigs[modelType]) {
        throw new Error(`Unknown model type: ${modelType}`);
      }

      // Check if model is already loaded
      if (this.models.has(modelType)) {
        this.logger.info("ModelLoader", `Model ${modelType} already loaded`);
        this.currentModel = modelType;
        return this.models.get(modelType);
      }

      const config = this.modelConfigs[modelType];
      const modelPath = options.path || config.path;

      this.logger.info("ModelLoader", `Loading model: ${modelType}`, {
        path: modelPath,
        instruments: config.instruments,
      });

      // Load TensorFlow.js model
      // Use loadGraphModel for GraphDef format models
      const model = await tf.loadGraphModel(modelPath);

      // Store the model
      this.models.set(modelType, model);
      this.currentModel = modelType;

      // Warm up the model (run inference with dummy data)
      await this.warmupModel(model, config);

      const endTime = performance.now();
      this.logger.info(
        "ModelLoader",
        `Model ${modelType} loaded successfully`,
        {
          duration: `${(endTime - startTime).toFixed(2)}ms`,
          memoryUsage: this.getMemoryUsage(),
        }
      );

      return model;
    } catch (error) {
      this.logger.error(
        "ModelLoader",
        `Failed to load model ${modelType}`,
        error
      );
      throw error;
    }
  }

  /**
   * Warm up the model with dummy data to prevent first-inference lag
   * @param {tf.GraphModel} model - TensorFlow.js model
   * @param {Object} config - Model configuration
   */
  async warmupModel(model, config) {
    try {
      // Create dummy inputs that match the exact format the model expects
      // Based on Python Spleeter: mix_stft and mix_spectrogram with proper shapes
      
      // 1. Create mix_stft (complex STFT)
      // Shape should be: [batch, frames, freq_bins, channels] = [1, frames, 2049, 2]
      // For warmup, we'll use a reasonable number of frames
      const nFrames = 10; // Small number for warmup
      const stftFreqBins = 2049; // frame_length // 2 + 1 = 4096 // 2 + 1
      const nChannels = 2;
      
      // Create real and imaginary parts
      const stftReal = tf.zeros([nFrames, stftFreqBins, nChannels]);
      const stftImag = tf.zeros([nFrames, stftFreqBins, nChannels]);
      
      // Create complex tensor - no batch dimension as expected by model
      const mixStft = tf.complex(stftReal, stftImag); // Shape: [frames, freq_bins, channels]

      // 2. Create mix_spectrogram (magnitude spectrogram)
      // Shape should be: [batch, time, freq, channels] = [1, 512, 1024, 2]
      const T = 512; // Fixed time dimension
      const F = 1024; // Fixed frequency dimension
      const mixSpectrogram = tf.zeros([1, T, F, nChannels]);

      // 3. Create audio_id input
      const audioId = tf.fill([1], "");

      const inputs = {
        audio_id: audioId,
        mix_stft: mixStft,
        mix_spectrogram: mixSpectrogram,
      };

      this.logger.debug("ModelLoader", "Warming up model with inputs", {
        audio_id: audioId.shape,
        mix_stft: mixStft.shape,
        mix_spectrogram: mixSpectrogram.shape,
      });

      // Check model signature and use appropriate execution method for warmup too
      const modelInputs = model.inputs;
      this.logger.debug("ModelLoader", "Model input signature for warmup", {
        inputNames: modelInputs.map(input => input.name),
        inputShapes: modelInputs.map(input => input.shape)
      });

      // Try to map our named inputs to the model's expected input names
      const executionInputs = {};
      if (modelInputs.length === 1) {
        // Single input model - use the first input
        executionInputs[modelInputs[0].name] = mixSpectrogram; // Use spectrogram as primary input
      } else {
        // Multiple inputs - try to map them based on dtype and name patterns
        modelInputs.forEach((input, index) => {
          const inputName = input.name;
          const inputDtype = input.dtype;
          
          this.logger.debug("ModelLoader", `Processing warmup model input`, {
            name: inputName,
            dtype: inputDtype,
            shape: input.shape
          });
          
          // Map based on dtype first, then name patterns
          if (inputDtype === 'string') {
            // String input - must be audio_id
            if (audioId) {
              executionInputs[inputName] = audioId;
              this.logger.debug("ModelLoader", `Mapped warmup string input`, {
                inputName,
                source: 'audio_id',
                dtype: audioId.dtype
              });
            } else {
              throw new Error(`Model expects string input '${inputName}' but no audio_id provided`);
            }
          } else if (inputDtype === 'float32' || inputDtype === 'float64') {
            // Float input - likely spectrogram or audio data
            if (inputName.includes('spectrogram') || inputName.includes('conv2d') || inputName.includes('Placeholder')) {
              executionInputs[inputName] = mixSpectrogram;
              this.logger.debug("ModelLoader", `Mapped warmup float input to spectrogram`, {
                inputName,
                source: 'mix_spectrogram',
                dtype: mixSpectrogram.dtype
              });
            } else {
              // Default to spectrogram for float inputs
              executionInputs[inputName] = mixSpectrogram;
              this.logger.debug("ModelLoader", `Mapped warmup float input to spectrogram (default)`, {
                inputName,
                source: 'mix_spectrogram',
                dtype: mixSpectrogram.dtype
              });
            }
          } else if (inputDtype === 'complex64' || inputDtype === 'complex128') {
            // Complex input - must be STFT
            if (mixStft) {
              executionInputs[inputName] = mixStft;
              this.logger.debug("ModelLoader", `Mapped warmup complex input to STFT`, {
                inputName,
                source: 'mix_stft',
                dtype: mixStft.dtype
              });
            } else {
              throw new Error(`Model expects complex input '${inputName}' but no mix_stft provided`);
            }
          } else {
            // Unknown dtype - try name-based mapping as fallback
            this.logger.warn("ModelLoader", `Unknown warmup input dtype: ${inputDtype}, using name-based mapping`, {
              inputName,
              dtype: inputDtype
            });
            
            if (audioId && inputName.includes('audio_id')) {
              executionInputs[inputName] = audioId;
            } else if (mixStft && inputName.includes('stft')) {
              executionInputs[inputName] = mixStft;
            } else if (mixSpectrogram && (inputName.includes('spectrogram') || inputName.includes('conv2d'))) {
              executionInputs[inputName] = mixSpectrogram;
            } else {
              // Default to spectrogram
              executionInputs[inputName] = mixSpectrogram;
            }
          }
        });
      }

      this.logger.debug("ModelLoader", "Mapped warmup execution inputs", {
        mappedInputs: Object.keys(executionInputs)
      });

      // Use execute() instead of executeAsync() to avoid dynamic ops issues
      const predictions = await model.execute(executionInputs);

      this.logger.debug("ModelLoader", "Model warmup completed", {
        outputShape: Array.isArray(predictions)
          ? predictions.map((p) => p.shape)
          : predictions.shape,
      });

      // Cleanup tensors
      audioId.dispose();
      mixStft.dispose();
      mixSpectrogram.dispose();
      if (Array.isArray(predictions)) {
        predictions.forEach((pred) => pred.dispose());
      } else {
        predictions.dispose();
      }
    } catch (error) {
      this.logger.error("ModelLoader", "Model warmup failed", error);
      // Don't throw here - warmup failure shouldn't prevent model usage
    }
  }

  /**
   * Get expected input shape for model
   * @param {Object} config - Model configuration
   * @returns {Array} Input shape
   */
  getInputShape(config) {
    // Based on the actual model signature, Placeholder expects raw waveform: [-1, 2]
    // This is different from the spectrogram shape [1, 512, 1024, 2]
    return [-1, 2]; // [batch, channels] for raw waveform input
  }

  /**
   * Get loaded model
   * @param {string} modelType - Type of model
   * @returns {tf.LayersModel|null} TensorFlow.js model or null if not loaded
   */
  getModel(modelType = null) {
    const type = modelType || this.currentModel;
    return this.models.get(type) || null;
  }

  /**
   * Get currently loaded model type
   * @returns {string|null} Current model type
   */
  getCurrentModel() {
    return this.currentModel;
  }

  /**
   * Get available model types
   * @returns {Array<string>} Available model types
   */
  getAvailableModels() {
    return Object.keys(this.modelConfigs);
  }

  /**
   * Get model configuration
   * @param {string} modelType - Model type
   * @returns {Object|null} Model configuration
   */
  getModelConfig(modelType) {
    return this.modelConfigs[modelType] || null;
  }

  /**
   * Unload a model
   * @param {string} modelType - Type of model to unload
   */
  unloadModel(modelType) {
    try {
      if (this.models.has(modelType)) {
        const model = this.models.get(modelType);

        // Dispose model to free memory
        model.dispose();
        this.models.delete(modelType);

        if (this.currentModel === modelType) {
          this.currentModel = null;
        }

        this.logger.info("ModelLoader", `Model ${modelType} unloaded`, {
          memoryUsage: this.getMemoryUsage(),
        });
      }
    } catch (error) {
      this.logger.error(
        "ModelLoader",
        `Failed to unload model ${modelType}`,
        error
      );
    }
  }

  /**
   * Unload all models
   */
  unloadAllModels() {
    const modelTypes = Array.from(this.models.keys());
    modelTypes.forEach((type) => this.unloadModel(type));
    this.logger.info("ModelLoader", "All models unloaded");
  }

  /**
   * Run inference with loaded model
   * @param {Object} inputs - Input tensors object with named inputs
   * @param {string} modelType - Model type to use
   * @returns {Promise<tf.Tensor|Array<tf.Tensor>>} Model predictions
   */
  async predict(inputs, modelType = null) {
    const type = modelType || this.currentModel;
    const model = this.getModel(type);

    if (!model) {
      throw new Error(`Model ${type} is not loaded`);
    }

    try {
      const startTime = performance.now();

      // Validate inputs
      if (!inputs || typeof inputs !== 'object') {
        throw new Error('Inputs must be an object with named input tensors');
      }

      // Check for required inputs based on model signature
      const requiredInputs = ['audio_id', 'mix_stft', 'mix_spectrogram'];
      for (const inputName of requiredInputs) {
        if (!inputs[inputName]) {
          throw new Error(`Missing required input: ${inputName}`);
        }
      }

      this.logger.debug("ModelLoader", "Running model inference", {
        modelType: type,
        inputNames: Object.keys(inputs),
        inputShapes: Object.entries(inputs).reduce((acc, [name, tensor]) => {
          acc[name] = tensor.shape;
          return acc;
        }, {})
      });

      // Check model signature and use appropriate execution method
      const modelInputs = model.inputs;
      this.logger.debug("ModelLoader", "Model input signature", {
        inputNames: modelInputs.map(input => input.name),
        inputShapes: modelInputs.map(input => input.shape)
      });

      // Try to map our named inputs to the model's expected input names
      const executionInputs = {};
      if (modelInputs.length === 1) {
        // Single input model - use the first input
        executionInputs[modelInputs[0].name] = inputs.mix_spectrogram; // Use spectrogram as primary input
      } else {
        // Multiple inputs - try to map them based on dtype and name patterns
        modelInputs.forEach((input, index) => {
          const inputName = input.name;
          const inputDtype = input.dtype;
          
          this.logger.debug("ModelLoader", `Processing model input`, {
            name: inputName,
            dtype: inputDtype,
            shape: input.shape
          });
          
          // Map based on dtype first, then name patterns
          if (inputDtype === 'string') {
            // String input - must be audio_id
            if (inputs.audio_id) {
              executionInputs[inputName] = inputs.audio_id;
              this.logger.debug("ModelLoader", `Mapped string input`, {
                inputName,
                source: 'audio_id',
                dtype: inputs.audio_id.dtype
              });
            } else {
              throw new Error(`Model expects string input '${inputName}' but no audio_id provided`);
            }
          } else if (inputDtype === 'float32' || inputDtype === 'float64') {
            // Float input - likely spectrogram or audio data
            if (inputName.includes('spectrogram') || inputName.includes('conv2d') || inputName.includes('Placeholder')) {
              executionInputs[inputName] = inputs.mix_spectrogram;
              this.logger.debug("ModelLoader", `Mapped float input to spectrogram`, {
                inputName,
                source: 'mix_spectrogram',
                dtype: inputs.mix_spectrogram.dtype
              });
            } else {
              // Default to spectrogram for float inputs
              executionInputs[inputName] = inputs.mix_spectrogram;
              this.logger.debug("ModelLoader", `Mapped float input to spectrogram (default)`, {
                inputName,
                source: 'mix_spectrogram',
                dtype: inputs.mix_spectrogram.dtype
              });
            }
          } else if (inputDtype === 'complex64' || inputDtype === 'complex128') {
            // Complex input - must be STFT
            if (inputs.mix_stft) {
              executionInputs[inputName] = inputs.mix_stft;
              this.logger.debug("ModelLoader", `Mapped complex input to STFT`, {
                inputName,
                source: 'mix_stft',
                dtype: inputs.mix_stft.dtype
              });
            } else {
              throw new Error(`Model expects complex input '${inputName}' but no mix_stft provided`);
            }
          } else {
            // Unknown dtype - try name-based mapping as fallback
            this.logger.warn("ModelLoader", `Unknown input dtype: ${inputDtype}, using name-based mapping`, {
              inputName,
              dtype: inputDtype
            });
            
            if (inputs.audio_id && inputName.includes('audio_id')) {
              executionInputs[inputName] = inputs.audio_id;
            } else if (inputs.mix_stft && inputName.includes('stft')) {
              executionInputs[inputName] = inputs.mix_stft;
            } else if (inputs.mix_spectrogram && (inputName.includes('spectrogram') || inputName.includes('conv2d'))) {
              executionInputs[inputName] = inputs.mix_spectrogram;
            } else {
              // Default to spectrogram
              executionInputs[inputName] = inputs.mix_spectrogram;
            }
          }
        });
      }

      this.logger.debug("ModelLoader", "Mapped execution inputs", {
        mappedInputs: Object.keys(executionInputs),
        originalInputs: Object.keys(inputs)
      });

      // Use execute() instead of executeAsync() to avoid dynamic ops issues
      const predictions = await model.execute(executionInputs);

      const endTime = performance.now();
      this.logger.debug("ModelLoader", "Model inference completed", {
        modelType: type,
        duration: `${(endTime - startTime).toFixed(2)}ms`,
        outputShape: Array.isArray(predictions)
          ? predictions.map((p) => p.shape)
          : predictions.shape,
      });

      return predictions;
    } catch (error) {
      this.logger.error("ModelLoader", "Model inference failed", error);
      throw error;
    }
  }

  /**
   * Get memory usage information
   * @returns {Object} Memory usage data
   */
  getMemoryUsage() {
    if (typeof tf === "undefined") {
      return { error: "TensorFlow.js not available" };
    }

    const memory = tf.memory();
    return {
      numTensors: memory.numTensors,
      numDataBuffers: memory.numDataBuffers,
      bytes: memory.numBytes,
      bytesFormatted: this.formatBytes(memory.numBytes),
      unreliable: memory.unreliable,
    };
  }

  /**
   * Format bytes to human readable format
   * @param {number} bytes - Number of bytes
   * @returns {string} Formatted string
   */
  formatBytes(bytes) {
    if (bytes === 0) return "0 Bytes";

    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));

    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  }

  /**
   * Check if a model is loaded
   * @param {string} modelType - Model type
   * @returns {boolean} True if model is loaded
   */
  isModelLoaded(modelType) {
    return this.models.has(modelType);
  }

  /**
   * Get model statistics
   * @returns {Object} Model statistics
   */
  getModelStats() {
    const stats = {
      loadedModels: Array.from(this.models.keys()),
      currentModel: this.currentModel,
      availableModels: this.getAvailableModels(),
      memoryUsage: this.getMemoryUsage(),
      totalModels: this.models.size,
    };

    // Add individual model info
    this.models.forEach((model, type) => {
      stats[type] = {
        trainableParams: model.countParams(),
        config: this.getModelConfig(type),
      };
    });

    return stats;
  }

  /**
   * Add custom model configuration
   * @param {string} modelType - Model type name
   * @param {Object} config - Model configuration
   */
  addModelConfig(modelType, config) {
    this.modelConfigs[modelType] = {
      instruments: ["vocals", "accompaniment"],
      sampleRate: 44100,
      frameLength: 4096,
      frameStep: 1024,
      nChannels: 2,
      ...config,
    };

    this.logger.info(
      "ModelLoader",
      `Added custom model configuration: ${modelType}`,
      {
        config: this.modelConfigs[modelType],
      }
    );
  }

  /**
   * Set TensorFlow.js backend
   * @param {string} backend - Backend name ('webgl', 'cpu', 'wasm')
   * @returns {Promise<boolean>} True if successful
   */
  async setBackend(backend) {
    try {
      this.logger.info(
        "ModelLoader",
        `Setting TensorFlow.js backend to: ${backend}`
      );

      const success = await tf.setBackend(backend);
      if (success) {
        await tf.ready();
        this.logger.info(
          "ModelLoader",
          `Backend set successfully to: ${backend}`
        );
      } else {
        this.logger.warn("ModelLoader", `Failed to set backend to: ${backend}`);
      }

      return success;
    } catch (error) {
      this.logger.error(
        "ModelLoader",
        `Error setting backend to ${backend}`,
        error
      );
      return false;
    }
  }

  /**
   * Get available backends
   * @returns {Array<string>} Available backends
   */
  getAvailableBackends() {
    return tf.engine().registryFactory;
  }

  /**
   * Cleanup and dispose all resources
   */
  dispose() {
    this.unloadAllModels();
    this.logger.info("ModelLoader", "ModelLoader disposed");
  }
}

// Export for Node.js and browser
if (typeof module !== "undefined" && module.exports) {
  module.exports = ModelLoader;
} else if (typeof window !== "undefined") {
  window.SpleeterJS = window.SpleeterJS || {};
  window.SpleeterJS.ModelLoader = ModelLoader;
}
