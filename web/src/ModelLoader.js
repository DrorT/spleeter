/**
 * ModelLoader class for SpleeterJS - handles TensorFlow.js model loading and management
 */
class ModelLoader {
  constructor(options = {}) {
    this.logger = options.logger || new window.SpleeterJS.Logger();
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
      // Create dummy input based on expected shape
      const dummyShape = this.getInputShape(config);
      const dummyInput = tf.zeros(dummyShape);

      // For GraphModel, we need to provide inputs as an object with named inputs
      const inputs = {
        Placeholder: dummyInput,
        Placeholder_1: tf.fill([1], ""), // Empty string for the string input
      };

      // Run inference using execute for GraphModel
      const predictions = await model.execute(inputs);

      // Cleanup tensors
      dummyInput.dispose();
      if (Array.isArray(predictions)) {
        predictions.forEach((pred) => pred.dispose());
      } else {
        predictions.dispose();
      }

      this.logger.debug("ModelLoader", "Model warmup completed");
    } catch (error) {
      this.logger.warn("ModelLoader", "Model warmup failed", error);
      // Don't throw here - warmup failure shouldn't prevent model usage
    }
  }

  /**
   * Get expected input shape for model
   * @param {Object} config - Model configuration
   * @returns {Array} Input shape
   */
  getInputShape(config) {
    // Based on Spleeter's input shape: [batch, time, frequency, channels]
    // For real-time processing, we use batch size 1
    return [1, 512, 1024, 2]; // [batch, frames, freq_bins, channels]
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
   * @param {tf.Tensor} input - Input tensor
   * @param {string} modelType - Model type to use
   * @returns {Promise<tf.Tensor|Array<tf.Tensor>>} Model predictions
   */
  async predict(input, modelType = null) {
    const type = modelType || this.currentModel;
    const model = this.getModel(type);

    if (!model) {
      throw new Error(`Model ${type} is not loaded`);
    }

    try {
      const startTime = performance.now();

      // For GraphModel, use execute with named inputs
      const inputs = {
        Placeholder: input,
        Placeholder_1: tf.fill([1], ""), // Empty string for the string input
      };

      // Run inference using execute for GraphModel
      const predictions = await model.execute(inputs);

      const endTime = performance.now();
      this.logger.debug("ModelLoader", "Model inference completed", {
        modelType: type,
        duration: `${(endTime - startTime).toFixed(2)}ms`,
        inputShape: input.shape,
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
