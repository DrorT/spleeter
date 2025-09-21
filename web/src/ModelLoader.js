/**
 * ModelLoader class for SpleeterJS - handles TensorFlow.js model loading and management
 */
class ModelLoader {
  constructor(options = {}) {
    // Handle both browser and Node.js environments
    let Logger;
    if (
      typeof window !== "undefined" &&
      window.SpleeterJS &&
      window.SpleeterJS.Logger
    ) {
      Logger = window.SpleeterJS.Logger;
    } else {
      Logger = require("./Logger.js");
    }

    this.logger = options.logger || new Logger();
    // Async execution control
    this.forceAsync = !!options.forceAsync; // user-forced async
    this._requiresAsync = this.forceAsync; // detected dynamic graph requiring async
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
      const modelInputs = model.inputs.map((i) => i.name);
      // Create minimal tensors matching each input type
      const execInputs = {};
      const disposables = [];
      model.inputs.forEach((inp) => {
        if (inp.dtype === "string") {
          const t = tf.fill([1], "");
          execInputs[inp.name] = t;
          disposables.push(t);
        } else if (inp.dtype === "complex64") {
          // Assume shape [-1,2049,2]; use 4 frames for warmup
          const real = tf.zeros([4, 2049, 2]);
          const imag = tf.zeros([4, 2049, 2]);
          const c = tf.complex(real, imag);
          real.dispose();
          imag.dispose();
          execInputs[inp.name] = c;
          disposables.push(c);
        } else if (inp.shape.length === 4) {
          // spectrogram-like
          const shape = [1, 512, 1024, 2];
          const t = tf.zeros(shape);
          execInputs[inp.name] = t;
          disposables.push(t);
        } else if (inp.shape.length === 2) {
          // waveform [samples,channels]
          const t = tf.zeros([4096, 2]);
          execInputs[inp.name] = t;
          disposables.push(t);
        } else {
          const t = tf.zeros([1]);
          execInputs[inp.name] = t;
          disposables.push(t);
        }
      });
      let out;
      if (this.forceAsync || this._requiresAsync) {
        out = await model.executeAsync(execInputs);
      } else {
        try {
          out = await model.execute(execInputs);
        } catch (e) {
          if (/dynamic op|Merge/.test(e.message)) {
            this._requiresAsync = true;
            this.logger.warn(
              "ModelLoader",
              "Warmup detected dynamic graph; switching to executeAsync"
            );
            out = await model.executeAsync(execInputs);
          } else {
            throw e;
          }
        }
      }
      if (Array.isArray(out)) out.forEach((o) => o.dispose());
      else out.dispose();
      disposables.forEach((d) => d.dispose());
    } catch (e) {
      this.logger.warn("ModelLoader", "Warmup skipped", { error: e.message });
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
      if (!inputs || typeof inputs !== "object") {
        throw new Error("Inputs must be an object with named input tensors");
      }

      // No fixed required input list now; will map provided names to signature.

      this.logger.debug("ModelLoader", "Running model inference", {
        modelType: type,
        inputNames: Object.keys(inputs),
        inputShapes: Object.entries(inputs).reduce((acc, [name, tensor]) => {
          acc[name] = tensor.shape;
          return acc;
        }, {}),
      });

      // Check model signature and use appropriate execution method
      const modelInputs = model.inputs;
      this.logger.debug("ModelLoader", "Model input signature", {
        inputNames: modelInputs.map((input) => input.name),
        inputShapes: modelInputs.map((input) => input.shape),
      });

      const executionInputs = {};
      model.inputs.forEach((inp) => {
        if (inputs[inp.name]) {
          executionInputs[inp.name] = inputs[inp.name];
        } else {
          // Try legacy alias mapping
          if (inp.name === "Placeholder" && inputs.Placeholder)
            executionInputs[inp.name] = inputs.Placeholder;
          else if (
            inp.name === "transpose_1" &&
            (inputs.transpose_1 || inputs.mix_stft)
          )
            executionInputs[inp.name] = inputs.transpose_1 || inputs.mix_stft;
          else if (
            inp.name === "strided_slice_3" &&
            (inputs.strided_slice_3 || inputs.mix_spectrogram)
          )
            executionInputs[inp.name] =
              inputs.strided_slice_3 || inputs.mix_spectrogram;
          else if (inp.dtype === "string" && inputs.Placeholder_1)
            executionInputs[inp.name] = inputs.Placeholder_1;
        }
      });

      this.logger.debug("ModelLoader", "Mapped execution inputs", {
        mappedInputs: Object.keys(executionInputs),
        originalInputs: Object.keys(inputs),
      });

      const config = this.getModelConfig(type);
      const outputNames =
        config && config.outputNames ? config.outputNames : undefined;
      let predictions;
      if (this.forceAsync || this._requiresAsync) {
        predictions = await model.executeAsync(executionInputs, outputNames);
      } else {
        try {
          predictions = await model.execute(executionInputs, outputNames);
        } catch (e) {
          if (/dynamic op|Merge/.test(e.message)) {
            this._requiresAsync = true;
            this.logger.warn(
              "ModelLoader",
              "Detected dynamic ops; switching permanently to executeAsync",
              { error: e.message }
            );
            predictions = await model.executeAsync(
              executionInputs,
              outputNames
            );
          } else {
            throw e;
          }
        }
      }

      const endTime = performance.now();
      this.logger.debug("ModelLoader", "Model inference completed", {
        modelType: type,
        duration: `${(endTime - startTime).toFixed(2)}ms`,
        outputShape: Array.isArray(predictions)
          ? predictions.map((p) => p.shape)
          : predictions.shape,
        asyncMode: this.forceAsync || this._requiresAsync,
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
