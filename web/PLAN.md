# Spleeter JavaScript Porting Plan

## Project Overview
This plan outlines the porting of Spleeter (Deezer's source separation library) from Python/TensorFlow to JavaScript/TensorFlow.js for browser-based audio separation.

## Project Structure
```
web/
├── src/                    # Core library code
│   ├── SpleeterJS.js      # Main library class
│   ├── AudioProcessor.js  # Web Audio API integration
│   ├── STFTProcessor.js   # Short-Time Fourier Transform
│   ├── ModelLoader.js     # TensorFlow.js model management
│   ├── SeparationPipeline.js # End-to-end processing
│   └── Logger.js          # Logging system
├── models/                # TensorFlow.js converted models
├── examples/              # Simple HTML demos
├── visualizers/           # External visualization modules
│   ├── WaveformVisualizer.js
│   └── SpectrogramVisualizer.js
├── utils/                 # Audio processing utilities
└── PLAN.md               # This plan document
```

## Core Architecture

### Phase 1: Foundation Setup

#### 1.1 Project Structure
- **Location**: `/web/` folder separate from original Spleeter project
- **Package Focus**: Reusable library for other projects
- **UI Approach**: Simple HTML/CSS only, no build stages
- **Audio Handling**: Web Audio API or HTML Audio Element

#### 1.2 Core Library Components
- `SpleeterJS` - Main class for audio separation
- `AudioProcessor` - Web Audio API integration
- `STFTProcessor` - Short-Time Fourier Transform implementation
- `ModelLoader` - TensorFlow.js model management
- `SeparationPipeline` - End-to-end processing pipeline
- `Logger` - Comprehensive logging system

### Phase 2: Audio Processing Implementation

#### 2.1 Audio Loading & Processing
- Use Web Audio API for audio file loading and decoding
- Implement audio buffer management for large files
- Create chunked processing system for memory efficiency
- Support for various audio formats (MP3, WAV, etc.)

#### 2.2 STFT Implementation
- Port Python STFT to JavaScript with optimal window sizes
- **Parameters from Spleeter config**:
  - Frame length: 4096
  - Frame step: 1024
  - Sample rate: 44100Hz (configurable)
  - Hann window implementation
- Implement inverse STFT for waveform reconstruction

#### 2.3 Performance Optimization
- Benchmark different window sizes for optimal browser performance
- Implement parallel processing using Web Workers
- Memory-efficient buffer management
- Process audio in chunks to prevent browser freezing

### Phase 3: Machine Learning Integration

#### 3.1 TensorFlow.js Integration
- Load pre-converted TensorFlow.js models (already done)
- Implement model inference pipeline
- Handle different model types (2stems, 4stems, 5stems)
- Model caching and management

#### 3.2 Neural Network Processing
- Port BLSTM architecture logic to TensorFlow.js
- Implement mask application for source separation
- Handle multi-channel audio processing
- Support for different separation algorithms

### Phase 4: Separation Pipeline

#### 4.1 End-to-End Processing
- **Pipeline**: Audio → Spectrogram → Model Inference → Mask Application → Waveform
- Implement real-time progress tracking
- Handle errors gracefully with detailed logging
- Support for cancellation of long-running processes

#### 4.2 Output Generation
- Generate separated audio stems
- Create HTML Audio Elements for playback
- Support for different audio formats (WAV, MP3 if needed)
- Metadata preservation (when possible)

### Phase 5: Visualization & Debugging (External Modules)

#### 5.1 Visualization Components
- **WaveformVisualizer.js** - External module for waveform display
- **SpectrogramVisualizer.js** - External module for spectrogram display
- Comparison tools for original vs separated audio
- Real-time visualization during processing

#### 5.2 Logging System
- Detailed process logging for learning/debugging
- Toggle-able logs to prevent bottleneck in production
- Performance metrics tracking
- Structured logging with different levels

## Technical Specifications

### Audio Processing Parameters (Based on Spleeter Config)
```javascript
const DEFAULT_CONFIG = {
  sampleRate: 44100,
  frameLength: 4096,
  frameStep: 1024,
  nChannels: 2,
  separationExponent: 2,
  maskExtension: 'zeros',
  instruments: ['vocals', 'accompaniment'] // for 2stems model
};
```

### Performance Optimization Strategies

#### 1. Window Size Optimization
- Test different FFT sizes (2048, 4096, 8192)
- Measure processing time vs quality trade-offs
- Find optimal settings for browser environment
- Configurable parameters for different use cases

#### 2. Parallel Processing
- Use Web Workers for STFT computation
- Parallelize model inference across available cores
- Implement worker pool management
- Load balancing for optimal performance

#### 3. Memory Management
- Process audio in chunks (e.g., 10-second segments)
- Reuse audio buffers to minimize garbage collection
- Implement memory usage monitoring
- Graceful degradation for low-memory devices

## Library API Design

### Main API
```javascript
class SpleeterJS {
  constructor(options = {});
  
  // Model management
  async loadModel(modelType = '2stems');
  async unloadModel();
  getAvailableModels();
  
  // Audio processing
  async separate(audioFile);
  async separateChunk(audioChunk);
  cancelSeparation();
  
  // Results
  getSeparatedStems();
  getProcessingStats();
  
  // Logging
  enableLogs(enabled);
  getLogs();
  clearLogs();
  
  // Configuration
  updateConfig(newConfig);
  getCurrentConfig();
}

// Usage example
const spleeter = new SpleeterJS({
  enableLogs: true,
  chunkSize: 10 // seconds
});

await spleeter.loadModel('2stems');
const stems = await spleeter.separate(audioFile);
// stems contains { vocals: AudioBuffer, accompaniment: AudioBuffer }
```

### AudioProcessor API
```javascript
class AudioProcessor {
  async loadAudio(file);
  decodeAudioData(arrayBuffer);
  getAudioBuffer();
  getDuration();
  getSampleRate();
  extractChunk(startTime, duration);
}
```

### STFTProcessor API
```javascript
class STFTProcessor {
  computeSTFT(audioBuffer, config);
  computeISTFT(spectrogram, config);
  getWindowFunction(type, size);
  getOptimalParameters();
}
```

### ModelLoader API
```javascript
class ModelLoader {
  async loadModel(modelType);
  getModel(modelType);
  getAvailableModels();
  unloadModel(modelType);
}
```

## Learning & Comparison Features

### Process Visualization
1. **Before/After Comparison**
   - Side-by-side waveform displays
   - Spectrogram comparisons
   - Audio playback synchronization

2. **Process Transparency**
   - Real-time processing progress
   - Performance metrics (processing time, memory usage)
   - Model confidence scores

### Educational Logging
1. **Detailed Process Logs**
   - STFT parameters and results
   - Model inference details
   - Mask application statistics
   - Performance benchmarks

2. **Toggle-able Debug Mode**
   - `spleeter.enableLogs(false)` for production
   - `spleeter.enableLogs(true)` for development/learning
   - Log levels: ERROR, WARN, INFO, DEBUG

## Implementation Milestones

### Milestone 1: Basic Audio Processing
- [ ] Web Audio API integration
- [ ] STFT implementation
- [ ] Audio chunk processing
- [ ] Basic waveform visualization

### Milestone 2: Model Integration
- [ ] TensorFlow.js model loading
- [ ] Model inference pipeline
- [ ] Basic separation functionality
- [ ] HTML Audio Element output

### Milestone 3: Performance Optimization
- [ ] Web Worker implementation
- [ ] Window size optimization
- [ ] Memory management improvements
- [ ] Performance benchmarking

### Milestone 4: Visualization & Learning
- [ ] External visualization modules
- [ ] Process logging system
- [ ] Comparison tools
- [ ] Educational features

### Milestone 5: Package Finalization
- [ ] API documentation
- [ ] Example implementations
- [ ] Performance testing
- [ ] Library optimization

## Success Criteria
1. **Functional**: Successfully separate audio into stems with comparable quality to original Spleeter
2. **Performance**: Process audio efficiently without browser freezing
3. **Educational**: Provide clear insights into the separation process
4. **Reusable**: Clean API that can be easily integrated into other projects
5. **Optimized**: Balance between quality and performance for browser environment

## Key Technical Challenges & Solutions

### Challenge 1: Model Size and Loading
- **Problem**: TensorFlow.js models can be large (100MB+)
- **Solution**: Implement progressive loading and model caching

### Challenge 2: Real-time Processing
- **Problem**: Audio separation is computationally intensive
- **Solution**: Web Workers, chunked processing, optimized algorithms

### Challenge 3: Memory Management
- **Problem**: Large audio files can exceed browser memory limits
- **Solution**: Stream processing, chunked handling, buffer reuse

### Challenge 4: Cross-browser Compatibility
- **Problem**: Different browsers have varying Web Audio API support
- **Solution**: Feature detection, graceful degradation, polyfills

## Testing Strategy
1. **Unit Testing**: Test individual components (STFT, model loading, etc.)
2. **Integration Testing**: Test end-to-end separation pipeline
3. **Performance Testing**: Benchmark processing speed and memory usage
4. **Quality Testing**: Compare separation quality with original Spleeter
5. **Browser Testing**: Test across different browsers and devices

## Dependencies
- **TensorFlow.js**: For machine learning inference
- **Web Audio API**: For audio processing (built into browsers)
- **Web Workers**: For parallel processing (built into browsers)
- **Optional**: Visualization libraries for external modules

## Next Steps
1. Set up basic project structure
2. Implement core audio processing components
3. Integrate TensorFlow.js models
4. Build separation pipeline
5. Add optimization and visualization features
6. Test and refine the implementation

This plan focuses on creating a clean, optimized library while providing excellent learning opportunities through visualization and logging. The external visualization modules ensure the core library remains lightweight and performant.
