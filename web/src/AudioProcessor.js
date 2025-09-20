/**
 * AudioProcessor class for SpleeterJS - handles Web Audio API integration and audio loading
 */
class AudioProcessor {
    constructor(options = {}) {
        // Handle both browser and Node.js environments
        let Logger;
        if (typeof window !== 'undefined' && window.SpleeterJS && window.SpleeterJS.Logger) {
            Logger = window.SpleeterJS.Logger;
        } else {
            Logger = require('./Logger.js');
        }
        
        this.logger = options.logger || new Logger();
        this.audioContext = null;
        this.audioBuffer = null;
        this.sampleRate = options.sampleRate || 44100;
        this.chunkSize = options.chunkSize || 10; // seconds
        
        this.logger.info('AudioProcessor', 'AudioProcessor initialized', {
            sampleRate: this.sampleRate,
            chunkSize: this.chunkSize
        });
    }

    /**
     * Initialize AudioContext
     * @returns {Promise<AudioContext>} Initialized AudioContext
     */
    async initAudioContext() {
        if (this.audioContext) {
            return this.audioContext;
        }

        try {
            // Create audio context
            const AudioContext = window.AudioContext || window.webkitAudioContext;
            this.audioContext = new AudioContext();
            
            // Set sample rate if specified
            if (this.sampleRate && this.audioContext.sampleRate !== this.sampleRate) {
                this.logger.warn('AudioProcessor', 
                    `Requested sample rate ${this.sampleRate} differs from context rate ${this.audioContext.sampleRate}`);
            }
            
            this.logger.info('AudioProcessor', 'AudioContext initialized', {
                sampleRate: this.audioContext.sampleRate,
                state: this.audioContext.state
            });
            
            return this.audioContext;
        } catch (error) {
            this.logger.error('AudioProcessor', 'Failed to initialize AudioContext', error);
            throw new Error('AudioContext not supported');
        }
    }

    /**
     * Load audio from file
     * @param {File} file - Audio file to load
     * @returns {Promise<AudioBuffer>} Loaded audio buffer
     */
    async loadAudio(file) {
        try {
            this.logger.info('AudioProcessor', 'Loading audio file', {
                name: file.name,
                size: file.size,
                type: file.type
            });

            // Initialize audio context if needed
            await this.initAudioContext();

            // Read file as array buffer
            const arrayBuffer = await this.readFileAsArrayBuffer(file);
            
            // Decode audio data
            const audioBuffer = await this.decodeAudioData(arrayBuffer);
            
            this.audioBuffer = audioBuffer;
            
            this.logger.info('AudioProcessor', 'Audio loaded successfully', {
                duration: audioBuffer.duration,
                sampleRate: audioBuffer.sampleRate,
                numberOfChannels: audioBuffer.numberOfChannels,
                length: audioBuffer.length
            });
            
            return audioBuffer;
        } catch (error) {
            this.logger.error('AudioProcessor', 'Failed to load audio', error);
            throw error;
        }
    }

    /**
     * Read file as ArrayBuffer
     * @param {File} file - File to read
     * @returns {Promise<ArrayBuffer>} File data as ArrayBuffer
     */
    readFileAsArrayBuffer(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            
            reader.onload = () => {
                this.logger.debug('AudioProcessor', 'File read successfully', {
                    size: reader.result.byteLength
                });
                resolve(reader.result);
            };
            
            reader.onerror = () => {
                this.logger.error('AudioProcessor', 'Failed to read file');
                reject(new Error('Failed to read file'));
            };
            
            reader.readAsArrayBuffer(file);
        });
    }

    /**
     * Decode audio data
     * @param {ArrayBuffer} arrayBuffer - Audio data as ArrayBuffer
     * @returns {Promise<AudioBuffer>} Decoded audio buffer
     */
    async decodeAudioData(arrayBuffer) {
        try {
            const startTime = performance.now();
            
            const audioBuffer = await this.audioContext.decodeAudioData(arrayBuffer);
            
            const endTime = performance.now();
            this.logger.debug('AudioProcessor', 'Audio decoded successfully', {
                duration: `${(endTime - startTime).toFixed(2)}ms`
            });
            
            return audioBuffer;
        } catch (error) {
            this.logger.error('AudioProcessor', 'Failed to decode audio data', error);
            throw new Error('Failed to decode audio data');
        }
    }

    /**
     * Get audio buffer
     * @returns {AudioBuffer|null} Current audio buffer
     */
    getAudioBuffer() {
        return this.audioBuffer;
    }

    /**
     * Get audio duration
     * @returns {number} Duration in seconds
     */
    getDuration() {
        return this.audioBuffer ? this.audioBuffer.duration : 0;
    }

    /**
     * Get sample rate
     * @returns {number} Sample rate
     */
    getSampleRate() {
        return this.audioBuffer ? this.audioBuffer.sampleRate : this.sampleRate;
    }

    /**
     * Get number of channels
     * @returns {number} Number of audio channels
     */
    getNumberOfChannels() {
        return this.audioBuffer ? this.audioBuffer.numberOfChannels : 0;
    }

    /**
     * Extract chunk of audio data
     * @param {number} startTime - Start time in seconds
     * @param {number} duration - Duration in seconds
     * @returns {AudioBuffer} Audio buffer chunk
     */
    extractChunk(startTime, duration) {
        if (!this.audioBuffer) {
            throw new Error('No audio buffer loaded');
        }

        const sampleRate = this.audioBuffer.sampleRate;
        const startSample = Math.floor(startTime * sampleRate);
        const endSample = Math.floor((startTime + duration) * sampleRate);
        const chunkLength = endSample - startSample;

        // Create new audio buffer for the chunk
        const chunkBuffer = this.audioContext.createBuffer(
            this.audioBuffer.numberOfChannels,
            chunkLength,
            sampleRate
        );

        // Copy channel data
        for (let channel = 0; channel < this.audioBuffer.numberOfChannels; channel++) {
            const channelData = this.audioBuffer.getChannelData(channel);
            const chunkData = chunkBuffer.getChannelData(channel);
            
            for (let i = 0; i < chunkLength; i++) {
                chunkData[i] = channelData[startSample + i];
            }
        }

        this.logger.debug('AudioProcessor', 'Chunk extracted', {
            startTime,
            duration,
            startSample,
            endSample,
            chunkLength
        });

        return chunkBuffer;
    }

    /**
     * Get all chunks for processing
     * @param {number} chunkSize - Chunk size in seconds
     * @returns {Array<ArrayBuffer>> Array of audio chunks
     */
    getChunks(chunkSize = null) {
        if (!this.audioBuffer) {
            return [];
        }

        const size = chunkSize || this.chunkSize;
        const totalDuration = this.audioBuffer.duration;
        const chunks = [];

        for (let startTime = 0; startTime < totalDuration; startTime += size) {
            const chunkDuration = Math.min(size, totalDuration - startTime);
            chunks.push(this.extractChunk(startTime, chunkDuration));
        }

        this.logger.info('AudioProcessor', 'Audio split into chunks', {
            totalChunks: chunks.length,
            chunkSize: size,
            totalDuration: totalDuration
        });

        return chunks;
    }

    /**
     * Convert AudioBuffer to Float32Array data
     * @param {AudioBuffer} buffer - Audio buffer to convert
     * @returns {Object} Object with channel data
     */
    audioBufferToFloat32Array(buffer) {
        const channels = [];
        
        for (let i = 0; i < buffer.numberOfChannels; i++) {
            channels.push(buffer.getChannelData(i));
        }

        return {
            channels,
            sampleRate: buffer.sampleRate,
            duration: buffer.duration,
            length: buffer.length
        };
    }

    /**
     * Convert Float32Array data back to AudioBuffer
     * @param {Object} data - Audio data object
     * @returns {AudioBuffer} Audio buffer
     */
    float32ArrayToAudioBuffer(data) {
        const buffer = this.audioContext.createBuffer(
            data.channels.length,
            data.length,
            data.sampleRate
        );

        for (let i = 0; i < data.channels.length; i++) {
            const channelData = buffer.getChannelData(i);
            channelData.set(data.channels[i]);
        }

        return buffer;
    }

    /**
     * Clear loaded audio
     */
    clear() {
        this.audioBuffer = null;
        this.logger.info('AudioProcessor', 'Audio buffer cleared');
    }

    /**
     * Check if audio is loaded
     * @returns {boolean} True if audio is loaded
     */
    isAudioLoaded() {
        return this.audioBuffer !== null;
    }

    /**
     * Get audio info
     * @returns {Object} Audio information
     */
    getAudioInfo() {
        if (!this.audioBuffer) {
            return null;
        }

        return {
            duration: this.audioBuffer.duration,
            sampleRate: this.audioBuffer.sampleRate,
            numberOfChannels: this.audioBuffer.numberOfChannels,
            length: this.audioBuffer.length,
            durationFormatted: this.formatDuration(this.audioBuffer.duration)
        };
    }

    /**
     * Format duration in human readable format
     * @param {number} seconds - Duration in seconds
     * @returns {string} Formatted duration
     */
    formatDuration(seconds) {
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = Math.floor(seconds % 60);
        return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
    }
}

// Export for Node.js and browser
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AudioProcessor;
} else if (typeof window !== 'undefined') {
    window.SpleeterJS = window.SpleeterJS || {};
    window.SpleeterJS.AudioProcessor = AudioProcessor;
}
