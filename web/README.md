# SpleeterJS - Audio Source Separation in the Browser

![SpleeterJS Logo](https://img.shields.io/badge/SpleeterJS-Browser%20Audio%20Separation-blue.svg)

A JavaScript port of Deezer's Spleeter library for real-time audio source separation in the browser using TensorFlow.js and Web Audio API.

## 🎯 Overview

SpleeterJS brings the power of professional audio source separation to web browsers, allowing developers to separate music into individual stems (vocals, drums, bass, etc.) directly in the browser without server-side processing.

### Key Features

- 🚀 **Browser-based**: Runs entirely in the browser using Web Audio API and TensorFlow.js
- 🎵 **Multiple Models**: Support for 2, 4, and 5 stem separation models
- ⚡ **Real-time Processing**: Optimized for performance with chunked processing
- 🔧 **Easy Integration**: Simple API that works with any web project
- 📊 **Comprehensive Logging**: Detailed logging system for debugging and learning
- 🎛️ **Configurable**: Flexible parameters for different use cases
- 💻 **No Build Required**: Pure JavaScript/HTML/CSS implementation

## 🏗️ Project Structure

```
web/
├── src/                    # Core library code
│   ├── SpleeterJS.js      # Main library class
│   ├── AudioProcessor.js  # Web Audio API integration
│   ├── STFTProcessor.js   # Short-Time Fourier Transform
│   ├── ModelLoader.js     # TensorFlow.js model management
│   └── Logger.js          # Logging system
├── models/                # TensorFlow.js converted models
├── examples/              # Demo implementations
│   └── basic-demo.html    # Basic usage demo
├── visualizers/           # External visualization modules
├── utils/                 # Audio processing utilities
├── PLAN.md               # Detailed implementation plan
└── README.md             # This file
```

## 🚀 Quick Start

### Basic Usage

```html
<!-- Include required libraries -->
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest"></script>
<script src="src/Logger.js"></script>
<script src="src/AudioProcessor.js"></script>
<script src="src/STFTProcessor.js"></script>
<script src="src/ModelLoader.js"></script>
<script src="src/SpleeterJS.js"></script>

<script>
// Initialize SpleeterJS
const spleeter = new SpleeterJS({
    enableLogs: true,
    chunkSize: 10 // Process in 10-second chunks
});

// Load a model
await spleeter.loadModel('2stems');

// Separate audio file
const audioFile = document.getElementById('audioFile').files[0];
const stems = await spleeter.separate(audioFile);

// stems contains: { vocals: AudioBuffer, accompaniment: AudioBuffer }
</script>
```

### API Reference

#### Constructor

```javascript
const spleeter = new SpleeterJS(options);
```

**Options:**
- `enableLogs` (boolean): Enable/disable logging (default: true)
- `chunkSize` (number): Audio chunk size in seconds (default: 10)
- `sampleRate` (number): Audio sample rate (default: 44100)
- `logLevel` (string): Log level ('ERROR', 'WARN', 'INFO', 'DEBUG') (default: 'INFO')

#### Methods

##### `async loadModel(modelType, options)`

Load a separation model.

**Parameters:**
- `modelType` (string): Model type ('2stems', '4stems', '5stems')
- `options` (object): Model loading options

**Returns:** `Promise<void>`

##### `async separate(audioFile, options)`

Separate audio file into stems.

**Parameters:**
- `audioFile` (File): Audio file to separate
- `options` (object): Separation options

**Returns:** `Promise<Object>` - Separated stems as AudioBuffers

##### `getSeparatedStems()`

Get the last separated stems.

**Returns:** `Object|null` - Separated stems or null

##### `getProcessingStats()`

Get processing statistics.

**Returns:** `Object|null` - Processing statistics

##### `enableLogs(enabled)`

Enable or disable logging.

**Parameters:**
- `enabled` (boolean): Whether to enable logging

##### `getLogs(level)`

Get logs.

**Parameters:**
- `level` (string): Optional filter by log level

**Returns:** `Array` - Log entries

##### `cancelSeparation()`

Cancel current separation process.

##### `dispose()`

Cleanup and dispose resources.

## 🎵 Available Models

### 2 Stems Model
- **Instruments**: Vocals, Accompaniment
- **Use Case**: Karaoke, backing track creation
- **Performance**: Fastest processing

### 4 Stems Model
- **Instruments**: Vocals, Drums, Bass, Other
- **Use Case**: Music production, remixing
- **Performance**: Moderate speed

### 5 Stems Model
- **Instruments**: Vocals, Drums, Bass, Piano, Other
- **Use Case**: Detailed music analysis, production
- **Performance**: Slowest but most detailed

## 🔧 Configuration

### Audio Processing Parameters

```javascript
const config = {
    sampleRate: 44100,        // Audio sample rate
    frameLength: 4096,       // FFT window size
    frameStep: 1024,         // Hop size between frames
    nChannels: 2,            // Number of audio channels
    chunkSize: 10,           // Processing chunk size (seconds)
    separationExponent: 2,   // Spectrogram exponent
    maskExtension: 'zeros'   // Mask extension method
};
```

### Performance Optimization

- **Chunk Size**: Smaller chunks use less memory but may reduce quality
- **Sample Rate**: Lower sample rates process faster but reduce quality
- **Frame Length**: Larger frames provide better frequency resolution but slower processing

## 📊 Demo

Try the basic demo by opening `examples/basic-demo.html` in a web browser. The demo provides:

- Model selection (2, 4, or 5 stems)
- Audio file upload
- Real-time separation progress
- Audio playback for original and separated stems
- Processing statistics and system information
- Detailed logging for debugging

## 🔍 Browser Compatibility

### Required Features
- **Web Audio API**: For audio processing
- **Web Workers**: For parallel processing (optional but recommended)
- **TensorFlow.js**: For machine learning inference
- **File API**: For audio file loading

### Supported Browsers
- ✅ Chrome 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Edge 90+

### Mobile Support
- ✅ iOS Safari 14+
- ✅ Android Chrome 90+

## 🚧 Performance Considerations

### Memory Usage
- Large audio files can consume significant memory
- Use chunked processing for files longer than 5 minutes
- Monitor memory usage with `getMemoryUsage()`

### Processing Speed
- Processing speed depends on:
  - Audio length and complexity
  - Model complexity (2stems < 4stems < 5stems)
  - Device CPU/GPU performance
  - Browser optimization

### Optimization Tips
1. Use appropriate chunk sizes (10-30 seconds)
2. Disable logging in production
3. Use WebGL backend for TensorFlow.js when available
4. Process shorter audio segments when possible

## 🐛 Troubleshooting

### Common Issues

#### Model Loading Failed
```javascript
// Check if TensorFlow.js is loaded
if (typeof tf === 'undefined') {
    console.error('TensorFlow.js not loaded');
}
```

#### Audio Processing Failed
```javascript
// Check Web Audio API support
if (!window.AudioContext && !window.webkitAudioContext) {
    console.error('Web Audio API not supported');
}
```

#### Memory Issues
```javascript
// Check memory usage
const memory = spleeter.getMemoryUsage();
console.log('Memory usage:', memory);

// Use smaller chunks
spleeter.updateConfig({ chunkSize: 5 });
```

### Debug Mode
Enable detailed logging for troubleshooting:

```javascript
spleeter.enableLogs(true);
spleeter.logger.setLevel('DEBUG');

// View logs
const logs = spleeter.getLogs();
console.log(logs);
```

## 📚 Advanced Usage

### Custom Model Integration
```javascript
// Add custom model configuration
spleeter.modelLoader.addModelConfig('custom', {
    path: 'models/custom/model.json',
    instruments: ['vocals', 'guitar', 'drums'],
    sampleRate: 44100,
    frameLength: 4096,
    frameStep: 1024
});

// Load custom model
await spleeter.loadModel('custom');
```

### Progress Monitoring
```javascript
// Monitor processing progress
const checkProgress = setInterval(() => {
    if (spleeter.isProcessingAudio()) {
        console.log('Still processing...');
    } else {
        clearInterval(checkProgress);
        console.log('Processing complete!');
    }
}, 1000);
```

### Error Handling
```javascript
try {
    const stems = await spleeter.separate(audioFile);
    // Process stems...
} catch (error) {
    if (error.message.includes('cancelled')) {
        console.log('Separation was cancelled');
    } else {
        console.error('Separation failed:', error);
    }
}
```

## 🔮 Future Enhancements

### Planned Features
- [ ] Web Workers for parallel processing
- [ ] WASM acceleration for performance-critical components
- [ ] Real-time streaming audio separation
- [ ] Additional model architectures (U-Net, etc.)
- [ ] Advanced visualization modules
- [ ] Node.js support for server-side processing

### Performance Optimizations
- [ ] Model quantization for reduced memory usage
- [ ] GPU acceleration improvements
- [ ] Memory pool management
- [ ] Adaptive chunk sizing

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
1. Clone the repository
2. Ensure you have a modern web browser
3. Place TensorFlow.js models in the `models/` directory
4. Open `examples/basic-demo.html` to test

### Guidelines
- Follow the existing code style
- Add comprehensive logging for new features
- Include error handling for all async operations
- Update documentation for API changes

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## 🙏 Acknowledgments

- **Deezer Research** for the original Spleeter library
- **TensorFlow.js team** for the excellent machine learning framework
- **Web Audio API** developers for powerful browser audio processing

## 📞 Support

For issues, questions, or contributions, please:
1. Check the [troubleshooting section](#-troubleshooting)
2. Review the [demo implementation](#-demo)
3. Open an issue on GitHub

---

**Happy Audio Separating! 🎵**
