/**
 * Logger class for SpleeterJS - provides comprehensive logging with toggle capability
 */
class Logger {
    constructor(options = {}) {
        this.enabled = options.enabled !== false;
        this.level = options.level || 'INFO';
        this.logs = [];
        this.maxLogs = options.maxLogs || 1000;
        
        this.levels = {
            'ERROR': 0,
            'WARN': 1,
            'INFO': 2,
            'DEBUG': 3
        };
    }

    /**
     * Log a message with specified level
     * @param {string} level - Log level (ERROR, WARN, INFO, DEBUG)
     * @param {string} component - Component generating the log
     * @param {string} message - Log message
     * @param {any} data - Additional data to log
     */
    log(level, component, message, data = null) {
        if (!this.enabled || this.levels[level] > this.levels[this.level]) {
            return;
        }

        const logEntry = {
            timestamp: new Date().toISOString(),
            level,
            component,
            message,
            data
        };

        this.logs.push(logEntry);

        // Keep only the most recent logs
        if (this.logs.length > this.maxLogs) {
            this.logs.shift();
        }

        // Also log to console for development
        if (typeof console !== 'undefined') {
            const consoleMethod = level.toLowerCase();
            if (console[consoleMethod]) {
                console[consoleMethod](`[${level}] [${component}] ${message}`, data || '');
            }
        }
    }

    /**
     * Log error message
     * @param {string} component - Component generating the error
     * @param {string} message - Error message
     * @param {any} error - Error object or additional data
     */
    error(component, message, error = null) {
        this.log('ERROR', component, message, error);
    }

    /**
     * Log warning message
     * @param {string} component - Component generating the warning
     * @param {string} message - Warning message
     * @param {any} data - Additional data
     */
    warn(component, message, data = null) {
        this.log('WARN', component, message, data);
    }

    /**
     * Log info message
     * @param {string} component - Component generating the info
     * @param {string} message - Info message
     * @param {any} data - Additional data
     */
    info(component, message, data = null) {
        this.log('INFO', component, message, data);
    }

    /**
     * Log debug message
     * @param {string} component - Component generating the debug info
     * @param {string} message - Debug message
     * @param {any} data - Additional data
     */
    debug(component, message, data = null) {
        this.log('DEBUG', component, message, data);
    }

    /**
     * Enable or disable logging
     * @param {boolean} enabled - Whether to enable logging
     */
    enableLogs(enabled) {
        this.enabled = enabled;
        this.info('Logger', `Logging ${enabled ? 'enabled' : 'disabled'}`);
    }

    /**
     * Set log level
     * @param {string} level - Log level (ERROR, WARN, INFO, DEBUG)
     */
    setLevel(level) {
        if (this.levels.hasOwnProperty(level)) {
            this.level = level;
            this.info('Logger', `Log level set to ${level}`);
        } else {
            this.warn('Logger', `Invalid log level: ${level}`);
        }
    }

    /**
     * Get all logs
     * @param {string} level - Optional filter by level
     * @returns {Array} Array of log entries
     */
    getLogs(level = null) {
        if (level) {
            return this.logs.filter(log => log.level === level);
        }
        return [...this.logs];
    }

    /**
     * Clear all logs
     */
    clearLogs() {
        this.logs = [];
        this.info('Logger', 'Logs cleared');
    }

    /**
     * Get logs for a specific component
     * @param {string} component - Component name
     * @returns {Array} Array of log entries for the component
     */
    getLogsByComponent(component) {
        return this.logs.filter(log => log.component === component);
    }

    /**
     * Get performance metrics from logs
     * @returns {Object} Performance metrics
     */
    getPerformanceMetrics() {
        const performanceLogs = this.logs.filter(log => 
            log.component === 'Performance' || log.message.includes('ms') || log.message.includes('seconds')
        );
        
        return {
            totalLogs: this.logs.length,
            performanceLogs: performanceLogs.length,
            errorCount: this.logs.filter(log => log.level === 'ERROR').length,
            warningCount: this.logs.filter(log => log.level === 'WARN').length
        };
    }
}

// Export for Node.js and browser
if (typeof module !== 'undefined' && module.exports) {
    module.exports = Logger;
} else if (typeof window !== 'undefined') {
    window.SpleeterJS = window.SpleeterJS || {};
    window.SpleeterJS.Logger = Logger;
}
