/**
 * STFTProcessor class for SpleeterJS - handles Short-Time Fourier Transform operations
 * Based on Spleeter's Python implementation with optimal parameters for browser performance
 */
class STFTProcessor {
    constructor(options = {}) {
        this.logger = options.logger || new window.SpleeterJS.Logger();
        
        // Default configuration based on Spleeter's 2stems config
        this.config = {
            sampleRate: options.sampleRate || 44100,
            frameLength: options.frameLength || 4096,
            frameStep: options.frameStep || 1024,
            nChannels: options.nChannels || 2,
            specExponent: options.specExponent || 1.0,
            windowExponent: options.windowExponent || 1.0,
            ...options.config
        };

        // Pre-compute window function
        this.windowFunction = this.createWindowFunction();
        
        // Performance optimization parameters
        this.fftSize = this.frameLength;
        this.hopSize = this.frameStep;
        this.nFrames = 0;
        this.nFreqBins = Math.floor(this.fftSize / 2) + 1;

        this.logger.info('STFTProcessor', 'STFTProcessor initialized', {
            config: this.config,
            fftSize: this.fftSize,
            hopSize: this.hopSize,
            nFreqBins: this.nFreqBins
        });
    }

    /**
     * Create Hann window function
     * @returns {Float32Array} Window function
     */
    createWindowFunction() {
        const window = new Float32Array(this.fftSize);
        
        for (let i = 0; i < this.fftSize; i++) {
            // Hann window: 0.5 * (1 - cos(2π * n / (N-1)))
            window[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / (this.fftSize - 1)));
            // Apply window exponent if needed
            if (this.config.windowExponent !== 1.0) {
                window[i] = Math.pow(window[i], this.config.windowExponent);
            }
        }
        
        return window;
    }

    /**
     * Compute STFT from audio buffer
     * @param {AudioBuffer|Object} audioData - Audio buffer or audio data object
     * @returns {Object} STFT result with magnitude and phase
     */
    computeSTFT(audioData) {
        const startTime = performance.now();
        
        // Extract audio data
        let channels;
        let sampleRate;
        
        if (audioData instanceof AudioBuffer) {
            channels = [];
            for (let i = 0; i < audioData.numberOfChannels; i++) {
                channels.push(audioData.getChannelData(i));
            }
            sampleRate = audioData.sampleRate;
        } else if (audioData.channels) {
            channels = audioData.channels;
            sampleRate = audioData.sampleRate;
        } else {
            throw new Error('Invalid audio data format');
        }

        // Use first channel for mono processing, or average channels
        const signal = this.prepareSignal(channels);
        
        // Compute number of frames
        this.nFrames = Math.ceil((signal.length - this.fftSize) / this.hopSize) + 1;
        
        // Initialize output arrays
        const magnitude = new Float32Array(this.nFrames * this.nFreqBins);
        const phase = new Float32Array(this.nFrames * this.nFreqBins);
        
        // Process each frame
        for (let frame = 0; frame < this.nFrames; frame++) {
            const startSample = frame * this.hopSize;
            const frameSignal = new Float32Array(this.fftSize);
            
            // Extract frame and apply window
            for (let i = 0; i < this.fftSize; i++) {
                const sampleIndex = startSample + i;
                if (sampleIndex < signal.length) {
                    frameSignal[i] = signal[sampleIndex] * this.windowFunction[i];
                } else {
                    frameSignal[i] = 0; // Zero-padding for end of signal
                }
            }
            
            // Compute FFT
            const fftResult = this.computeFFT(frameSignal);
            
            // Store magnitude and phase
            const frameOffset = frame * this.nFreqBins;
            for (let i = 0; i < this.nFreqBins; i++) {
                const real = fftResult[i * 2];
                const imag = fftResult[i * 2 + 1];
                
                // Compute magnitude: sqrt(real^2 + imag^2)
                magnitude[frameOffset + i] = Math.sqrt(real * real + imag * imag);
                
                // Apply spectrogram exponent
                if (this.config.specExponent !== 1.0) {
                    magnitude[frameOffset + i] = Math.pow(magnitude[frameOffset + i], this.config.specExponent);
                }
                
                // Compute phase: atan2(imag, real)
                phase[frameOffset + i] = Math.atan2(imag, real);
            }
        }
        
        const endTime = performance.now();
        this.logger.debug('STFTProcessor', 'STFT computed successfully', {
            duration: `${(endTime - startTime).toFixed(2)}ms`,
            nFrames: this.nFrames,
            nFreqBins: this.nFreqBins,
            signalLength: signal.length
        });
        
        return {
            magnitude,
            phase,
            nFrames: this.nFrames,
            nFreqBins: this.nFreqBins,
            fftSize: this.fftSize,
            hopSize: this.hopSize,
            sampleRate: sampleRate
        };
    }

    /**
     * Prepare signal for processing (mono conversion)
     * @param {Array<Float32Array>} channels - Audio channels
     * @returns {Float32Array} Mono signal
     */
    prepareSignal(channels) {
        if (channels.length === 1) {
            return channels[0];
        }
        
        // Average channels for stereo to mono conversion
        const signal = new Float32Array(channels[0].length);
        for (let i = 0; i < signal.length; i++) {
            let sum = 0;
            for (let channel = 0; channel < channels.length; channel++) {
                sum += channels[channel][i];
            }
            signal[i] = sum / channels.length;
        }
        
        return signal;
    }

    /**
     * Compute FFT using Cooley-Tukey algorithm
     * @param {Float32Array} signal - Input signal
     * @returns {Float32Array} FFT result [real0, imag0, real1, imag1, ...]
     */
    computeFFT(signal) {
        const N = signal.length;
        
        // Pad to power of 2 if needed
        const paddedN = Math.pow(2, Math.ceil(Math.log2(N)));
        const paddedSignal = new Float32Array(paddedN);
        paddedSignal.set(signal);
        
        // Initialize output array (interleaved real and imaginary)
        const fftResult = new Float32Array(paddedN * 2);
        
        // Copy signal to real part
        for (let i = 0; i < paddedN; i++) {
            fftResult[i * 2] = paddedSignal[i];
            fftResult[i * 2 + 1] = 0; // Imaginary part is 0 for real signal
        }
        
        // Apply FFT
        this.fft(fftResult, paddedN);
        
        // Return only the positive frequencies (first half)
        const resultSize = paddedN / 2 + 1;
        const result = new Float32Array(resultSize * 2);
        result.set(fftResult.subarray(0, resultSize * 2));
        
        return result;
    }

    /**
     * In-place FFT implementation using Cooley-Tukey algorithm
     * @param {Float32Array} data - Interleaved real and imaginary data
     * @param {number} N - Size of the FFT
     */
    fft(data, N) {
        // Bit-reversal permutation
        const bits = Math.log2(N);
        for (let i = 0; i < N; i++) {
            const j = this.reverseBits(i, bits);
            if (i < j) {
                // Swap real parts
                const tempReal = data[i * 2];
                data[i * 2] = data[j * 2];
                data[j * 2] = tempReal;
                
                // Swap imaginary parts
                const tempImag = data[i * 2 + 1];
                data[i * 2 + 1] = data[j * 2 + 1];
                data[j * 2 + 1] = tempImag;
            }
        }
        
        // Cooley-Tukey FFT
        for (let len = 2; len <= N; len *= 2) {
            const angle = -2 * Math.PI / len;
            const wReal = Math.cos(angle);
            const wImag = Math.sin(angle);
            
            for (let i = 0; i < N; i += len) {
                let wRealCurrent = 1;
                let wImagCurrent = 0;
                
                for (let j = 0; j < len / 2; j++) {
                    const u = i + j;
                    const v = i + j + len / 2;
                    
                    // Butterfly operation
                    const tReal = wRealCurrent * data[v * 2] - wImagCurrent * data[v * 2 + 1];
                    const tImag = wRealCurrent * data[v * 2 + 1] + wImagCurrent * data[v * 2];
                    
                    data[v * 2] = data[u * 2] - tReal;
                    data[v * 2 + 1] = data[u * 2 + 1] - tImag;
                    data[u * 2] = data[u * 2] + tReal;
                    data[u * 2 + 1] = data[u * 2 + 1] + tImag;
                    
                    // Update twiddle factor
                    const nextWReal = wRealCurrent * wReal - wImagCurrent * wImag;
                    const nextWImag = wRealCurrent * wImag + wImagCurrent * wReal;
                    wRealCurrent = nextWReal;
                    wImagCurrent = nextWImag;
                }
            }
        }
    }

    /**
     * Reverse bits of a number
     * @param {number} num - Number to reverse
     * @param {number} bits - Number of bits
     * @returns {number} Reversed number
     */
    reverseBits(num, bits) {
        let reversed = 0;
        for (let i = 0; i < bits; i++) {
            reversed = (reversed << 1) | (num & 1);
            num >>= 1;
        }
        return reversed;
    }

    /**
     * Compute inverse STFT
     * @param {Object} stftData - STFT data with magnitude and phase
     * @returns {Float32Array} Reconstructed signal
     */
    computeISTFT(stftData) {
        const startTime = performance.now();
        
        const { magnitude, phase, nFrames, nFreqBins, fftSize, hopSize } = stftData;
        
        // Initialize output signal
        const signalLength = (nFrames - 1) * hopSize + fftSize;
        const signal = new Float32Array(signalLength);
        const windowSum = new Float32Array(signalLength);
        
        // Process each frame
        for (let frame = 0; frame < nFrames; frame++) {
            const frameOffset = frame * nFreqBins;
            
            // Reconstruct complex spectrum
            const spectrum = new Float32Array(fftSize * 2);
            for (let i = 0; i < nFreqBins; i++) {
                const mag = magnitude[frameOffset + i];
                const ph = phase[frameOffset + i];
                
                spectrum[i * 2] = mag * Math.cos(ph); // Real part
                spectrum[i * 2 + 1] = mag * Math.sin(ph); // Imaginary part
                
                // Fill negative frequencies (conjugate symmetry)
                if (i > 0 && i < nFreqBins - 1) {
                    const negIndex = fftSize - i;
                    spectrum[negIndex * 2] = mag * Math.cos(ph); // Real part
                    spectrum[negIndex * 2 + 1] = -mag * Math.sin(ph); // Imaginary part (negative)
                }
            }
            
            // Compute inverse FFT
            const frameSignal = this.computeIFFT(spectrum);
            
            // Apply window and overlap-add
            const startSample = frame * hopSize;
            for (let i = 0; i < fftSize; i++) {
                const sampleIndex = startSample + i;
                if (sampleIndex < signalLength) {
                    signal[sampleIndex] += frameSignal[i] * this.windowFunction[i];
                    windowSum[sampleIndex] += this.windowFunction[i] * this.windowFunction[i];
                }
            }
        }
        
        // Normalize by window sum
        for (let i = 0; i < signalLength; i++) {
            if (windowSum[i] > 0) {
                signal[i] /= windowSum[i];
            }
        }
        
        const endTime = performance.now();
        this.logger.debug('STFTProcessor', 'ISTFT computed successfully', {
            duration: `${(endTime - startTime).toFixed(2)}ms`,
            signalLength: signalLength
        });
        
        return signal;
    }

    /**
     * Compute inverse FFT
     * @param {Float32Array} spectrum - Frequency domain data
     * @returns {Float32Array} Time domain signal
     */
    computeIFFT(spectrum) {
        const N = spectrum.length / 2;
        
        // Conjugate the input
        for (let i = 0; i < N; i++) {
            spectrum[i * 2 + 1] = -spectrum[i * 2 + 1];
        }
        
        // Compute FFT
        this.fft(spectrum, N);
        
        // Conjugate the output and scale
        const signal = new Float32Array(N);
        for (let i = 0; i < N; i++) {
            signal[i] = spectrum[i * 2] / N;
            spectrum[i * 2 + 1] = -spectrum[i * 2 + 1]; // Restore original
        }
        
        return signal;
    }

    /**
     * Get optimal parameters for performance
     * @returns {Object} Optimal parameters
     */
    getOptimalParameters() {
        return {
            fftSize: this.fftSize,
            hopSize: this.hopSize,
            nFreqBins: this.nFreqBins,
            windowFunction: 'hann',
            recommendedChunkSize: Math.ceil(this.fftSize / this.hopSize) * this.hopSize / this.config.sampleRate
        };
    }

    /**
     * Update configuration
     * @param {Object} newConfig - New configuration
     */
    updateConfig(newConfig) {
        this.config = { ...this.config, ...newConfig };
        this.windowFunction = this.createWindowFunction();
        this.fftSize = this.frameLength;
        this.hopSize = this.frameStep;
        this.nFreqBins = Math.floor(this.fftSize / 2) + 1;
        
        this.logger.info('STFTProcessor', 'Configuration updated', {
            config: this.config,
            fftSize: this.fftSize,
            hopSize: this.hopSize,
            nFreqBins: this.nFreqBins
        });
    }
}

// Export for Node.js and browser
if (typeof module !== 'undefined' && module.exports) {
    module.exports = STFTProcessor;
} else if (typeof window !== 'undefined') {
    window.SpleeterJS = window.SpleeterJS || {};
    window.SpleeterJS.STFTProcessor = STFTProcessor;
}
