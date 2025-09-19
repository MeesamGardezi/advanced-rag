/**
 * Utility Functions
 * Common helper functions used throughout the Construction Graph RAG system
 */

import crypto from 'crypto';
import { performance } from 'perf_hooks';

// =============================================================================
// STRING UTILITIES
// =============================================================================

/**
 * Clean and normalize text for processing
 */
export function normalizeText(text) {
  if (!text || typeof text !== 'string') return '';
  
  return text
    .trim()
    .replace(/\s+/g, ' ')  // Multiple spaces to single space
    .replace(/[\r\n]+/g, ' ')  // Line breaks to spaces
    .replace(/[^\w\s\-.,()$]/g, '')  // Remove special chars except essential ones
    .toLowerCase();
}

/**
 * Extract cost codes from text
 */
export function extractCostCodes(text) {
  if (!text) return [];
  
  const patterns = [
    /\b\d{2,3}[A-Z]\b/g,  // 114L, 203M, etc.
    /\b[A-Z]{2,4}\d{2,4}\b/g,  // AB123, ELEC101, etc.
    /\b\d{3}[-\s][A-Z\s]+/g  // 114 - LABOR, etc.
  ];
  
  const matches = new Set();
  patterns.forEach(pattern => {
    const found = text.match(pattern);
    if (found) {
      found.forEach(match => matches.add(match.trim()));
    }
  });
  
  return Array.from(matches);
}

/**
 * Extract monetary amounts from text
 */
export function extractMonetaryAmounts(text) {
  if (!text) return [];
  
  const patterns = [
    /\$[\d,]+\.?\d*/g,  // $1,234.56
    /USD?\s*[\d,]+\.?\d*/gi,  // USD 1234.56
    /[\d,]+\.?\d*\s*dollars?/gi  // 1234.56 dollars
  ];
  
  const amounts = [];
  patterns.forEach(pattern => {
    const matches = text.match(pattern);
    if (matches) {
      matches.forEach(match => {
        const cleaned = match.replace(/[^\d.]/g, '');
        const amount = parseFloat(cleaned);
        if (!isNaN(amount)) {
          amounts.push(amount);
        }
      });
    }
  });
  
  return amounts;
}

/**
 * Truncate text to specified length with ellipsis
 */
export function truncateText(text, maxLength = 100, suffix = '...') {
  if (!text || text.length <= maxLength) return text;
  return text.substring(0, maxLength - suffix.length) + suffix;
}

/**
 * Generate slug from text
 */
export function slugify(text) {
  if (!text) return '';
  
  return text
    .toLowerCase()
    .trim()
    .replace(/[^\w\s-]/g, '')
    .replace(/[\s_-]+/g, '-')
    .replace(/^-+|-+$/g, '');
}

// =============================================================================
// NUMBER UTILITIES  
// =============================================================================

/**
 * Parse currency string to number
 */
export function parseCurrency(value) {
  if (typeof value === 'number') return value;
  if (!value) return 0;
  
  const cleaned = value.toString().replace(/[^\d.-]/g, '');
  const parsed = parseFloat(cleaned);
  return isNaN(parsed) ? 0 : parsed;
}

/**
 * Format number as currency
 */
export function formatCurrency(amount, currency = 'USD', locale = 'en-US') {
  if (typeof amount !== 'number' || isNaN(amount)) return '$0.00';
  
  return new Intl.NumberFormat(locale, {
    style: 'currency',
    currency: currency
  }).format(amount);
}

/**
 * Calculate percentage change
 */
export function calculatePercentageChange(oldValue, newValue) {
  if (!oldValue || oldValue === 0) return newValue > 0 ? 100 : 0;
  return ((newValue - oldValue) / Math.abs(oldValue)) * 100;
}

/**
 * Calculate variance percentage
 */
export function calculateVariance(actual, budgeted) {
  if (!budgeted || budgeted === 0) return actual > 0 ? 100 : 0;
  return ((actual - budgeted) / Math.abs(budgeted)) * 100;
}

/**
 * Round to specified decimal places
 */
export function roundToDecimals(value, decimals = 2) {
  if (typeof value !== 'number' || isNaN(value)) return 0;
  return Math.round(value * Math.pow(10, decimals)) / Math.pow(10, decimals);
}

/**
 * Check if number is within range
 */
export function isWithinRange(value, min, max, inclusive = true) {
  if (inclusive) {
    return value >= min && value <= max;
  }
  return value > min && value < max;
}

// =============================================================================
// DATE UTILITIES
// =============================================================================

/**
 * Parse various date formats
 */
export function parseDate(dateValue) {
  if (!dateValue) return null;
  
  // Handle different date formats
  if (dateValue instanceof Date) return dateValue;
  
  // Handle Firebase Timestamp
  if (dateValue && typeof dateValue.toDate === 'function') {
    return dateValue.toDate();
  }
  
  // Handle ISO string or other string formats
  if (typeof dateValue === 'string') {
    const parsed = new Date(dateValue);
    return isNaN(parsed.getTime()) ? null : parsed;
  }
  
  // Handle Unix timestamp (seconds)
  if (typeof dateValue === 'number') {
    const asMs = dateValue < 10000000000 ? dateValue * 1000 : dateValue;
    return new Date(asMs);
  }
  
  return null;
}

/**
 * Format date for display
 */
export function formatDate(date, format = 'YYYY-MM-DD') {
  const parsed = parseDate(date);
  if (!parsed) return '';
  
  const year = parsed.getFullYear();
  const month = String(parsed.getMonth() + 1).padStart(2, '0');
  const day = String(parsed.getDate()).padStart(2, '0');
  
  switch (format) {
    case 'YYYY-MM-DD':
      return `${year}-${month}-${day}`;
    case 'MM/DD/YYYY':
      return `${month}/${day}/${year}`;
    case 'DD/MM/YYYY':
      return `${day}/${month}/${year}`;
    case 'readable':
      return parsed.toLocaleDateString('en-US', { 
        year: 'numeric', 
        month: 'long', 
        day: 'numeric' 
      });
    default:
      return parsed.toISOString().split('T')[0];
  }
}

/**
 * Get relative time description
 */
export function getRelativeTime(date) {
  const parsed = parseDate(date);
  if (!parsed) return 'Unknown';
  
  const now = new Date();
  const diffMs = now - parsed;
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));
  
  if (diffDays === 0) return 'Today';
  if (diffDays === 1) return 'Yesterday';
  if (diffDays < 7) return `${diffDays} days ago`;
  if (diffDays < 30) return `${Math.floor(diffDays / 7)} weeks ago`;
  if (diffDays < 365) return `${Math.floor(diffDays / 30)} months ago`;
  return `${Math.floor(diffDays / 365)} years ago`;
}

// =============================================================================
// OBJECT UTILITIES
// =============================================================================

/**
 * Deep clone object
 */
export function deepClone(obj) {
  if (obj === null || typeof obj !== 'object') return obj;
  if (obj instanceof Date) return new Date(obj.getTime());
  if (obj instanceof Array) return obj.map(item => deepClone(item));
  
  const cloned = {};
  for (const key in obj) {
    if (obj.hasOwnProperty(key)) {
      cloned[key] = deepClone(obj[key]);
    }
  }
  return cloned;
}

/**
 * Deep merge objects
 */
export function deepMerge(target, source) {
  const result = { ...target };
  
  for (const key in source) {
    if (source.hasOwnProperty(key)) {
      if (typeof source[key] === 'object' && source[key] !== null && !Array.isArray(source[key])) {
        result[key] = deepMerge(result[key] || {}, source[key]);
      } else {
        result[key] = source[key];
      }
    }
  }
  
  return result;
}

/**
 * Pick specific properties from object
 */
export function pick(obj, keys) {
  const picked = {};
  keys.forEach(key => {
    if (key in obj) {
      picked[key] = obj[key];
    }
  });
  return picked;
}

/**
 * Omit specific properties from object
 */
export function omit(obj, keys) {
  const omitted = { ...obj };
  keys.forEach(key => {
    delete omitted[key];
  });
  return omitted;
}

/**
 * Check if object has nested property
 */
export function hasNestedProperty(obj, path) {
  const keys = path.split('.');
  let current = obj;
  
  for (const key of keys) {
    if (current === null || current === undefined || !(key in current)) {
      return false;
    }
    current = current[key];
  }
  
  return true;
}

/**
 * Get nested property value
 */
export function getNestedProperty(obj, path, defaultValue = undefined) {
  const keys = path.split('.');
  let current = obj;
  
  for (const key of keys) {
    if (current === null || current === undefined || !(key in current)) {
      return defaultValue;
    }
    current = current[key];
  }
  
  return current;
}

// =============================================================================
// ARRAY UTILITIES
// =============================================================================

/**
 * Group array by property
 */
export function groupBy(array, keyOrFn) {
  if (!Array.isArray(array)) return {};
  
  const keyFn = typeof keyOrFn === 'function' ? keyOrFn : item => item[keyOrFn];
  
  return array.reduce((groups, item) => {
    const key = keyFn(item);
    if (!groups[key]) {
      groups[key] = [];
    }
    groups[key].push(item);
    return groups;
  }, {});
}

/**
 * Remove duplicates from array
 */
export function unique(array, keyOrFn) {
  if (!Array.isArray(array)) return [];
  
  if (!keyOrFn) {
    return [...new Set(array)];
  }
  
  const keyFn = typeof keyOrFn === 'function' ? keyOrFn : item => item[keyOrFn];
  const seen = new Set();
  
  return array.filter(item => {
    const key = keyFn(item);
    if (seen.has(key)) {
      return false;
    }
    seen.add(key);
    return true;
  });
}

/**
 * Sort array by multiple criteria
 */
export function multiSort(array, criteria) {
  if (!Array.isArray(array) || !Array.isArray(criteria)) return array;
  
  return [...array].sort((a, b) => {
    for (const criterion of criteria) {
      const { key, direction = 'asc' } = criterion;
      const aVal = getNestedProperty(a, key);
      const bVal = getNestedProperty(b, key);
      
      let comparison = 0;
      
      if (aVal < bVal) comparison = -1;
      else if (aVal > bVal) comparison = 1;
      
      if (comparison !== 0) {
        return direction === 'desc' ? -comparison : comparison;
      }
    }
    
    return 0;
  });
}

/**
 * Chunk array into smaller arrays
 */
export function chunk(array, size) {
  if (!Array.isArray(array) || size <= 0) return [];
  
  const chunks = [];
  for (let i = 0; i < array.length; i += size) {
    chunks.push(array.slice(i, i + size));
  }
  return chunks;
}

// =============================================================================
// VALIDATION UTILITIES
// =============================================================================

/**
 * Validate email address
 */
export function isValidEmail(email) {
  const pattern = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return pattern.test(email);
}

/**
 * Validate UUID
 */
export function isValidUUID(uuid) {
  const pattern = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
  return pattern.test(uuid);
}

/**
 * Validate cost code format
 */
export function isValidCostCode(costCode) {
  if (!costCode || typeof costCode !== 'string') return false;
  
  const patterns = [
    /^\d{2,3}[A-Z]$/,  // 114L
    /^[A-Z]{2,4}\d{2,4}$/,  // ELEC101
    /^\d{3}\s*-\s*[A-Z\s]+$/i  // 114 - LABOR
  ];
  
  return patterns.some(pattern => pattern.test(costCode.trim()));
}

/**
 * Validate required fields
 */
export function validateRequiredFields(obj, requiredFields) {
  const missing = [];
  
  requiredFields.forEach(field => {
    const value = getNestedProperty(obj, field);
    if (value === undefined || value === null || value === '') {
      missing.push(field);
    }
  });
  
  return {
    valid: missing.length === 0,
    missing
  };
}

// =============================================================================
// PERFORMANCE UTILITIES
// =============================================================================

/**
 * Create performance timer
 */
export function createTimer(label) {
  const startTime = performance.now();
  
  return {
    stop: () => {
      const duration = performance.now() - startTime;
      console.log(`⏱️  ${label}: ${duration.toFixed(2)}ms`);
      return duration;
    },
    lap: (lapLabel) => {
      const lapTime = performance.now() - startTime;
      console.log(`⏱️  ${label} - ${lapLabel}: ${lapTime.toFixed(2)}ms`);
      return lapTime;
    }
  };
}

/**
 * Debounce function execution
 */
export function debounce(func, wait) {
  let timeout;
  
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}

/**
 * Throttle function execution
 */
export function throttle(func, limit) {
  let inThrottle;
  
  return function executedFunction(...args) {
    if (!inThrottle) {
      func.apply(this, args);
      inThrottle = true;
      setTimeout(() => inThrottle = false, limit);
    }
  };
}

// =============================================================================
// CRYPTO UTILITIES
// =============================================================================

/**
 * Generate random ID
 */
export function generateId(length = 8) {
  return crypto.randomBytes(length).toString('hex');
}

/**
 * Generate hash of string
 */
export function generateHash(input, algorithm = 'sha256') {
  return crypto.createHash(algorithm).update(input).digest('hex');
}

/**
 * Generate UUID v4
 */
export function generateUUID() {
  return crypto.randomUUID();
}

// =============================================================================
// CONSTRUCTION-SPECIFIC UTILITIES
// =============================================================================

/**
 * Categorize cost code
 */
export function categorizeCostCode(costCode, description = '') {
  if (!costCode) return 'other';
  
  const code = costCode.toUpperCase();
  const desc = description.toLowerCase();
  
  // Check code patterns first (more reliable)
  if (code.match(/\d+M$/)) return 'material';
  if (code.match(/\d+L$/)) return 'labor';
  if (code.match(/\d+S$/)) return 'subcontractor';
  if (code.match(/\d+O$/)) return 'overhead';
  
  // Check description patterns
  if (desc.includes('material') || desc.includes('supply')) return 'material';
  if (desc.includes('labor') || desc.includes('work')) return 'labor';
  if (desc.includes('subcontractor') || desc.includes('sub ')) return 'subcontractor';
  if (desc.includes('overhead') || desc.includes('management')) return 'overhead';
  
  return 'other';
}

/**
 * Calculate budget health
 */
export function calculateBudgetHealth(actual, budgeted) {
  if (!budgeted || budgeted <= 0) return 'unknown';
  
  const variance = calculateVariance(actual, budgeted);
  
  if (Math.abs(variance) <= 5) return 'on-budget';
  if (variance > 5 && variance <= 15) return 'over-budget';
  if (variance > 15) return 'significantly-over';
  if (variance < -5 && variance >= -15) return 'under-budget';
  if (variance < -15) return 'significantly-under';
  
  return 'unknown';
}

/**
 * Parse construction schedule duration
 */
export function parseDuration(durationStr) {
  if (!durationStr) return 0;
  
  const str = durationStr.toString().toLowerCase();
  
  // Match patterns like "5d", "2w", "1m", "3 days", "2 weeks"
  const patterns = [
    { pattern: /(\d+\.?\d*)\s*d(ays?)?/, multiplier: 1 },
    { pattern: /(\d+\.?\d*)\s*w(eeks?)?/, multiplier: 7 },
    { pattern: /(\d+\.?\d*)\s*m(onths?)?/, multiplier: 30 },
    { pattern: /(\d+\.?\d*)\s*h(ours?)?/, multiplier: 1/24 }
  ];
  
  for (const { pattern, multiplier } of patterns) {
    const match = str.match(pattern);
    if (match) {
      return parseFloat(match[1]) * multiplier;
    }
  }
  
  // Try to parse as plain number (assume days)
  const num = parseFloat(str);
  return isNaN(num) ? 0 : num;
}

// =============================================================================
// EXPORT DEFAULT UTILITIES OBJECT
// =============================================================================

export default {
  // String utilities
  normalizeText,
  extractCostCodes,
  extractMonetaryAmounts,
  truncateText,
  slugify,
  
  // Number utilities
  parseCurrency,
  formatCurrency,
  calculatePercentageChange,
  calculateVariance,
  roundToDecimals,
  isWithinRange,
  
  // Date utilities
  parseDate,
  formatDate,
  getRelativeTime,
  
  // Object utilities
  deepClone,
  deepMerge,
  pick,
  omit,
  hasNestedProperty,
  getNestedProperty,
  
  // Array utilities
  groupBy,
  unique,
  multiSort,
  chunk,
  
  // Validation utilities
  isValidEmail,
  isValidUUID,
  isValidCostCode,
  validateRequiredFields,
  
  // Performance utilities
  createTimer,
  debounce,
  throttle,
  
  // Crypto utilities
  generateId,
  generateHash,
  generateUUID,
  
  // Construction-specific utilities
  categorizeCostCode,
  calculateBudgetHealth,
  parseDuration
};