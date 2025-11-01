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

    const postSlug = sanitizeFileName(fileName);
    
    // Try to find posts directory
    let postsDir = null;
    let mdPath = null;
    
    const possiblePaths = [
      path.join(process.cwd(), "posts"),
      path.join(process.cwd(), "..", "posts"),
      "/var/task/posts",
    ];
    
    for (const testPostsDir of possiblePaths) {
      const testMdPath = path.join(testPostsDir, postSlug, `${postSlug}.md`);
      if (fs.existsSync(testMdPath)) {
        postsDir = testPostsDir;
        mdPath = testMdPath;
        console.log('✓ Found markdown at:', mdPath);
        break;
      }
    }

    if (!mdPath || !postsDir) {
      return res.status(404).json({ 
        error: "Markdown file not found",
        debug: {
          cwd: process.cwd(),
          triedPaths: possiblePaths
        }
      });
    }

    if (!isPathSafe(mdPath, postsDir)) {
      return res.status(403).json({ 
        error: "Invalid file path" 
      });
    }

    const raw = fs.readFileSync(mdPath, "utf-8");
    const { frontmatter, content } = parseFrontmatter(raw);
    const { contentWithPlaceholders, charts } = extractCharts(content, postSlug, postsDir);

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

// HELPER FUNCTIONS

function sanitizeFileName(fileName) {
  if (!fileName || typeof fileName !== 'string') {
    throw new Error('Invalid fileName parameter');
  }

  let sanitized = fileName.replace(/\.md$/, "");

  sanitized = sanitized
    .replace(/\\/g, '')
    .replace(/\//g, '')
    .replace(/\.\./g, '')
    .replace(/^\.+/, '');
  
  sanitized = sanitized.replace(/[^a-zA-Z0-9_-]/g, '');
  
  if (!sanitized || sanitized.length === 0) {
    throw new Error('Invalid fileName after sanitization');
  }
  
  return sanitized;
}

function isPathSafe(resolvedPath, baseDir) {
  const normalizedPath = path.normalize(resolvedPath);
  const normalizedBase = path.normalize(baseDir);
  
  return normalizedPath.startsWith(normalizedBase + path.sep) || 
         normalizedPath === normalizedBase;
}

function sanitizeDataSource(dataSource) {
  if (!dataSource || typeof dataSource !== 'string') {
    throw new Error('Invalid dataSource parameter');
  }
  
  let sanitized = dataSource
    .replace(/\\/g, '/')
    .replace(/\.\.\/*/g, '')
    .replace(/^\/+/, '');
  
  if (!/^[a-zA-Z0-9_\-\/\.]+$/.test(sanitized)) {
    throw new Error('Invalid characters in dataSource');
  }
  
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

      if (
        (value.startsWith('"') && value.endsWith('"')) ||
        (value.startsWith("'") && value.endsWith("'"))
      ) {
        value = value.slice(1, -1);
      }

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

function extractCharts(content, postSlug, postsDir) {
  const hasCharts = content.includes('```chart');
  
  if (!hasCharts) {
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
      const chartId = chartData.id || `${type}-${chartIndex++}`;

      if (chartData.dataSource) {
        try {
          chartData.data = loadChartData(postSlug, chartData.dataSource, postsDir);
          console.log(`✓ Loaded ${chartData.data.length} rows for ${chartId}`);
        } catch (error) {
          console.error(`✗ Failed to load data for ${chartId}:`, error.message);
          chartData.dataError = error.message;
        }
      }

      chartData.type = type;
      charts[chartId] = chartData;

      return `{{CHART:${chartId}}}`;
    } catch (e) {
      console.error(`✗ Failed to parse ${type} JSON:`, e.message);
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

function loadChartData(postSlug, dataSource, postsDir) {
  const sanitizedDataSource = sanitizeDataSource(dataSource);
  
  // Try to find the CSV in multiple locations
  const possibleCSVPaths = [
    path.join(postsDir, postSlug, "data", sanitizedDataSource),
    path.join(postsDir, postSlug, sanitizedDataSource), // Try without data folder
  ];
  
  let csvPath = null;
  let dataDir = path.join(postsDir, postSlug, "data");
  
  for (const testPath of possibleCSVPaths) {
    if (fs.existsSync(testPath)) {
      csvPath = testPath;
      console.log('✓ Found CSV at:', csvPath);
      break;
    }
  }
  
  if (!csvPath) {
    console.error('CSV not found. Tried:');
    possibleCSVPaths.forEach(p => console.error('  -', p));
    
    // Show what actually exists
    const postDir = path.join(postsDir, postSlug);
    if (fs.existsSync(postDir)) {
      console.error('Files in post directory:', fs.readdirSync(postDir));
      
      if (fs.existsSync(dataDir)) {
        console.error('Files in data directory:', fs.readdirSync(dataDir));
      }
    }
    
    throw new Error(`CSV file not found: ${sanitizedDataSource}`);
  }
  
  if (!isPathSafe(csvPath, path.dirname(csvPath))) {
    throw new Error('Invalid data source path');
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