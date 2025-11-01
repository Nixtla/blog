import fs from "fs";
import path from "path";
import Papa from "papaparse";

const papaparseOptions = {
  header: true,
  dynamicTyping: true,
  skipEmptyLines: true,
  transformHeader: (header) => header.trim(),
  transform: (value) => {
    if (value === "true") return true;
    if (value === "false") return false;
    return value;
  },
}

export default function handler(req, res) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET,OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
  
  if (req.method === "OPTIONS") {
    res.status(200).end();
    return;
  }

  try {
    const { fileName } = req.query;
    
    if (!fileName) {
      return res.status(400).json({ error: "Missing fileName parameter" });
    }

    // SECURITY: Validate and sanitize fileName
    const postSlug = sanitizeFileName(fileName);
    
    // Resolve the absolute path
    const postsDir = path.join(process.cwd(), "posts");
    const mdPath = path.join(postsDir, postSlug, `${postSlug}.md`);
    
    // SECURITY: Ensure the resolved path is within the posts directory
    if (!isPathSafe(mdPath, postsDir)) {
      return res.status(403).json({ 
        error: "Invalid file path" 
      });
    }

    if (!fs.existsSync(mdPath)) {
      return res.status(404).json({ 
        error: "Markdown file not found"
      });
    }

    const raw = fs.readFileSync(mdPath, "utf-8");
    const { frontmatter, content } = parseFrontmatter(raw);
    const { contentWithPlaceholders, charts } = extractCharts(content, postSlug);

    const response = {
      ...frontmatter,
      fileName: postSlug,
      readTimeMinutes: calculateReadTime(content),
      content: contentWithPlaceholders,
      charts,
    };

    res.json(response);
  } catch (error) {
    console.error("API Error:", error);
    res.status(500).json({
      error: "Internal server error",
      message: error.message,
      stack: process.env.NODE_ENV === "development" ? error.stack : undefined,
    });
  }
}

// ============================================
// HELPER FUNCTIONS
// ============================================

/**
 * Sanitize fileName to prevent path traversal attacks
 */
function sanitizeFileName(fileName) {
  if (!fileName || typeof fileName !== 'string') {
    throw new Error('Invalid fileName parameter');
  }

  // Remove .md extension if present
  let sanitized = fileName.replace(/\.md$/, "");
  
  // Remove any path separators and traversal attempts
  sanitized = sanitized
    .replace(/\\/g, '') // Remove backslashes
    .replace(/\//g, '') // Remove forward slashes
    .replace(/\.\./g, '') // Remove parent directory references
    .replace(/^\.+/, ''); // Remove leading dots
  
  // Only allow alphanumeric, hyphens, and underscores
  sanitized = sanitized.replace(/[^a-zA-Z0-9_-]/g, '');
  
  if (!sanitized || sanitized.length === 0) {
    throw new Error('Invalid fileName after sanitization');
  }
  
  return sanitized;
}

/**
 * Verify that the resolved path is within the allowed directory
 */
function isPathSafe(resolvedPath, baseDir) {
  const normalizedPath = path.normalize(resolvedPath);
  const normalizedBase = path.normalize(baseDir);
  
  // Check if the resolved path starts with the base directory
  return normalizedPath.startsWith(normalizedBase + path.sep) || 
         normalizedPath === normalizedBase;
}

/**
 * Sanitize dataSource path for CSV files
 */
function sanitizeDataSource(dataSource) {
  if (!dataSource || typeof dataSource !== 'string') {
    throw new Error('Invalid dataSource parameter');
  }
  
  // Remove path traversal attempts
  let sanitized = dataSource
    .replace(/\\/g, '/') // Normalize to forward slashes
    .replace(/\.\.\/*/g, '') // Remove parent directory references
    .replace(/^\/+/, ''); // Remove leading slashes
  
  // Only allow safe characters for file paths
  if (!/^[a-zA-Z0-9_\-\/\.]+$/.test(sanitized)) {
    throw new Error('Invalid characters in dataSource');
  }
  
  // Ensure it ends with .csv
  if (!sanitized.endsWith('.csv')) {
    throw new Error('dataSource must be a CSV file');
  }
  
  return sanitized;
}

function parseFrontmatter(raw) {
  const match = raw.match(/^---([\s\S]*?)---\s*([\s\S]*)$/);

  if (!match) {
    throw new Error("Invalid markdown frontmatter format");
  }

  const frontmatterRaw = match[1];
  const content = match[2].trim();
  const frontmatter = {};

  frontmatterRaw.split("\n").forEach((line) => {
    const m = line.match(/^([a-zA-Z0-9_\-]+):\s*(.*)$/);
    if (m) {
      let key = m[1].trim();
      let value = m[2].trim();

      // Remove quotes
      if (
        (value.startsWith('"') && value.endsWith('"')) ||
        (value.startsWith("'") && value.endsWith("'"))
      ) {
        value = value.slice(1, -1);
      }

      // Parse arrays
      if (value.startsWith("[") && value.endsWith("]")) {
        try {
          value = JSON.parse(value.replace(/'/g, '"'));
        } catch {}
      }

      frontmatter[key] = value;
    }
  });

  return { frontmatter, content };
}

function extractCharts(content, postSlug) {
  // Early return if no chart blocks exist
  const hasCharts = content.includes('```chart-multiple') || content.includes('```chart');
  
  if (!hasCharts) {
    console.log('No charts found in content');
    return { 
      contentWithPlaceholders: content, 
      charts: {} 
    };
  }

  const charts = {};
  let chartIndex = 0;

  const processChart = (match, chartJson, type) => {
    try {
      const chartData = JSON.parse(chartJson.trim());
      const chartId = chartData.id || `${type}-${chartIndex}`;

      if (chartData.dataSource) {
        try {
          chartData.data = loadChartData(postSlug, chartData.dataSource);
          console.log(`Loaded ${chartData.data.length} rows for ${chartId}`);
        } catch (error) {
          console.error(`Failed to load data for ${chartId}:`, error.message);
          chartData.dataError = error.message;
        }
      }

      chartData.type = type;
      charts[chartId] = chartData;
      chartIndex++;

      return `{{CHART:${chartId}}}`;
    } catch (e) {
      console.error(`Failed to parse ${type} JSON:`, e);
      return match;
    }
  };

  let contentWithPlaceholders = content
    .replace(/```chart-multiple\s*\n([\s\S]*?)\n```/g, (match, json) =>
      processChart(match, json, "chart-multiple")
    )
    .replace(/```chart\s*\n([\s\S]*?)\n```/g, (match, json) =>
      processChart(match, json, "chart")
    );

  console.log(`Extracted ${Object.keys(charts).length} charts`);

  return { contentWithPlaceholders, charts };
}

function loadChartData(postSlug, dataSource) {
  const sanitizedDataSource = sanitizeDataSource(dataSource);
  
  const dataDir = path.join(process.cwd(), "posts", postSlug, "data");
  const csvPath = path.join(dataDir, sanitizedDataSource);
  
  if (!isPathSafe(csvPath, dataDir)) {
    throw new Error('Invalid data source path');
  }

  if (!fs.existsSync(csvPath)) {
    throw new Error(`CSV file not found: posts/${postSlug}/data/${sanitizedDataSource}`);
  }

  const csvContent = fs.readFileSync(csvPath, "utf-8");

  const result = Papa.parse(csvContent, papaparseOptions);

  if (result.errors.length > 0) {
    console.warn(`CSV parsing warnings for ${sanitizedDataSource}:`, result.errors);
  }

  return result.data;
}

function calculateReadTime(content) {
  return Math.round(content.split(" ").length / 200);
}