/**
 * Test file for ModelLoader warmup functionality
 */

console.log("Starting test...");

// Simple test to check if ModelLoader can be loaded
try {
  const ModelLoader = require("../src/ModelLoader.js");
  console.log("✅ ModelLoader loaded successfully"); // Create a simple logger
  class MockLogger {
    info(source, message, data = {}) {
      console.log(`[${source}] [INFO] ${message}`, data);
    }
    warn(source, message, data = {}) {
      console.warn(`[${source}] [WARN] ${message}`, data);
    }
    error(source, message, data = {}) {
      console.error(`[${source}] [ERROR] ${message}`, data);
    }
    debug(source, message, data = {}) {
      console.debug(`[${source}] [DEBUG] ${message}`, data);
    }
  }

  // Mock global window and tf
  global.window = {
    SpleeterJS: {
      Logger: MockLogger,
    },
  };

  global.tf = {
    version: { tfjs: "4.0.0" },
    loadGraphModel: async (path) => {
      console.log("Mock loading graph model from:", path);
      return {
        signature: {
          inputs: {
            audio_id: {
              dtype: "string",
              tensorShape: { dim: [] },
            },
            mix_stft: {
              dtype: "complex64",
              tensorShape: { dim: [{ size: -1 }, { size: 2049 }, { size: 2 }] },
            },
            mix_spectrogram: {
              dtype: "float32",
              tensorShape: {
                dim: [{ size: -1 }, { size: 512 }, { size: 1024 }, { size: 2 }],
              },
            },
          },
        },
        execute: async (inputs) => {
          console.log("Mock execute called with inputs:", Object.keys(inputs));
          return { dispose: () => {} };
        },
        executeAsync: async (inputs) => {
          console.log(
            "Mock executeAsync called with inputs:",
            Object.keys(inputs)
          );
          return { dispose: () => {} };
        },
        dispose: () => {},
        countParams: () => 1000000,
        engine: () => ({ registryFactory: ["webgl", "cpu"] }),
      };
    },
    zeros: (shape, dtype = "float32") => {
      console.log("Creating zeros tensor:", shape, dtype);
      return { shape, dtype, dispose: () => {} };
    },
    fill: (shape, value) => {
      console.log("Creating fill tensor:", shape, value);
      return { shape, dtype: "string", dispose: () => {} };
    },
    memory: () => ({
      numTensors: 10,
      numDataBuffers: 5,
      numBytes: 1024000,
      unreliable: false,
    }),
    setBackend: async () => true,
    ready: async () => {},
  };

  const logger = new MockLogger();
  const modelLoader = new ModelLoader({ logger });

  console.log("Testing model loading...");
  modelLoader
    .loadModel("2stems")
    .then(() => {
      console.log("✅ Test completed successfully");
      modelLoader.dispose();
    })
    .catch((error) => {
      console.error("❌ Test failed:", error.message);
      console.error("Stack:", error.stack);
    });
} catch (error) {
  console.error("❌ Failed to load ModelLoader:", error.message);
}
