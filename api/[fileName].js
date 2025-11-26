import fs from "fs";
import path from "path";
import Papa from "papaparse";
import zlib from "zlib";

const PAPAPARSE_CONFIG = {
  header: true,
  dynamicTyping: true,
  skipEmptyLines: true,
  transformHeader: (header) => header.trim(),
  transform: (value, header) => {
    // Trim whitespace
    if (typeof value === "string") {
      value = value.trim();
    }

    // List your actual boolean columns here
    const booleanColumns = ["anomaly", "is_anomaly", "threshold"];

    // Only convert to boolean for specific columns
    if (booleanColumns.includes(header)) {
      const lowerValue =
        typeof value === "string" ? value.toLowerCase() : String(value);
      if (lowerValue === "true" || lowerValue === "1") return true;
      if (lowerValue === "false" || lowerValue === "0") return false;
    }

    return value;
  },
};

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET,OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
};

export default function handler(req, res) {
  Object.entries(CORS_HEADERS).forEach(([key, value]) => {
    res.setHeader(key, value);
  });

  if (req.method === "OPTIONS") {
    return res.status(200).end();
  }

  try {
    const { fileName } = req.query;

    if (!fileName) {
      return res.status(400).json({ error: "Missing fileName parameter" });
    }

    const postSlug = sanitizeFileName(fileName);
    const mdPath = findMarkdownFile(postSlug);

    if (!mdPath) {
      return res.status(404).json({ error: "Markdown file not found" });
    }

    const markdownContent = fs.readFileSync(mdPath, "utf-8");
    const { frontmatter, content } = parseFrontmatter(markdownContent);
    const { contentWithPlaceholders, charts, chartMultiples } = extractCharts(
      content,
      postSlug
    );

    const response = {
      title: frontmatter.title || null,
      author_name: frontmatter.author_name || null,
      author_image: frontmatter.author_image || null,
      author_position: frontmatter.author_position || null,
      publication_date: frontmatter.publication_date || null,
      description: frontmatter.description || null,
      image: frontmatter.image || null,
      categories: frontmatter.categories || null,
      tags: frontmatter.tags || null,
      fileName: postSlug,
      readTimeMinutes: calculateReadTime(contentWithPlaceholders),
      content: contentWithPlaceholders,
      charts,
      chartMultiples,
    };

    return res.json(response);
  } catch (error) {
    console.error("API Error:", error);
    return res.status(500).json({
      error: "Internal server error",
      message: error.message,
    });
  }
}

function findMarkdownFile(postSlug) {
  const possiblePaths = [
    path.join(process.cwd(), "posts", `${postSlug}.md`),
    path.join(process.cwd(), "..", "posts", `${postSlug}.md`),
    path.join("/var/task/posts", `${postSlug}.md`),
  ];

  for (const filePath of possiblePaths) {
    if (fs.existsSync(filePath)) {
      return filePath;
    }
  }

  return null;
}

function sanitizeFileName(fileName) {
  if (!fileName || typeof fileName !== "string") {
    throw new Error("Invalid fileName parameter");
  }

  const sanitized = fileName
    .replace(/\.md$/, "")
    .replace(/[\\/]/g, "")
    .replace(/\.\./g, "")
    .replace(/^\.+/, "")
    .replace(/[^a-zA-Z0-9_-]/g, "");

  if (!sanitized) {
    throw new Error("Invalid fileName after sanitization");
  }

  return sanitized;
}

function sanitizeDataSource(dataSource) {
  if (!dataSource || typeof dataSource !== "string") {
    throw new Error("Invalid dataSource parameter");
  }

  const sanitized = dataSource
    .replace(/\\/g, "/")
    .replace(/\.\.\/*/g, "")
    .replace(/^\/+/, "");

  if (!/^[a-zA-Z0-9_\-\/\.]+$/.test(sanitized)) {
    throw new Error("Invalid characters in dataSource");
  }

  // Accept both .csv and .csv.gz files
  if (!sanitized.endsWith(".csv") && !sanitized.endsWith(".csv.gz")) {
    throw new Error("dataSource must be a CSV or CSV.GZ file");
  }

  return sanitized;
}

function parseFrontmatter(raw) {
  const match = raw.match(/^---([\s\S]*?)---\s*([\s\S]*)$/);

  if (!match) {
    throw new Error("Invalid markdown frontmatter format");
  }

  const [, frontmatterRaw, content] = match;
  const frontmatter = {};

  frontmatterRaw.split("\n").forEach((line) => {
    const lineMatch = line.match(/^([a-zA-Z0-9_\-]+):\s*(.*)$/);
    if (!lineMatch) return;

    const key = lineMatch[1].trim();
    let value = lineMatch[2].trim();

    value = removeQuotes(value);
    value = parseArrayValue(value);

    frontmatter[key] = value;
  });

  return { frontmatter, content: content.trim() };
}

function removeQuotes(value) {
  if (
    (value.startsWith('"') && value.endsWith('"')) ||
    (value.startsWith("'") && value.endsWith("'"))
  ) {
    return value.slice(1, -1);
  }
  return value;
}

function parseArrayValue(value) {
  if (value.startsWith("[") && value.endsWith("]")) {
    try {
      return JSON.parse(value.replace(/'/g, '"'));
    } catch {
      return value;
    }
  }
  return value;
}

function extractCharts(content, postSlug) {
  if (!content.includes("```chart")) {
    return { contentWithPlaceholders: content, charts: {}, chartMultiples: {} };
  }

  const charts = {};
  const chartMultiples = {};
  let chartIndex = 0;
  let chartMultipleIndex = 0;

  const processChart = (match, chartJson) => {
    try {
      const chartData = JSON.parse(chartJson.trim());
      const chartId = chartData.id || `chart-${chartIndex++}`;

      if (chartData.dataSource) {
        chartData.data = loadChartData(postSlug, chartData.dataSource);
      }

      charts[chartId] = chartData;

      return `{{CHART:${chartId}}}`;
    } catch (error) {
      console.error(`Failed to process chart:`, error.message);
      return match;
    }
  };

  const processChartMultiple = (match, chartJson) => {
    try {
      const chartData = JSON.parse(chartJson.trim());
      const chartId = chartData.id || `chart-multiple-${chartMultipleIndex++}`;

      if (chartData.dataSource) {
        chartData.data = loadChartData(postSlug, chartData.dataSource);
      }

      chartMultiples[chartId] = chartData;

      return `{{CHART_MULTIPLE:${chartId}}}`;
    } catch (error) {
      console.error(`Failed to process chart-multiple:`, error.message);
      return match;
    }
  };

  let contentWithPlaceholders = content
    .replace(/```chart-multiple\s*\n([\s\S]*?)\n```/g, processChartMultiple)
    .replace(/```chart\s*\n([\s\S]*?)\n```/g, processChart);

  return { contentWithPlaceholders, charts, chartMultiples };
}

function loadChartData(postSlug, dataSource) {
  const sanitizedDataSource = sanitizeDataSource(dataSource);

  // Try .gz version first (prioritize compressed), then fall back to .csv
  const gzPath = path.join(
    process.cwd(),
    "blogCharts",
    postSlug,
    sanitizedDataSource.endsWith(".gz")
      ? sanitizedDataSource
      : sanitizedDataSource + ".gz"
  );

  const csvPath = path.join(
    process.cwd(),
    "blogCharts",
    postSlug,
    sanitizedDataSource.endsWith(".csv.gz")
      ? sanitizedDataSource.replace(".gz", "")
      : sanitizedDataSource
  );

  let csvContent;

  // Check for .gz file first
  if (fs.existsSync(gzPath)) {
    try {
      const compressed = fs.readFileSync(gzPath);
      csvContent = zlib.gunzipSync(compressed).toString("utf-8");
      console.log(`Loaded compressed file: ${gzPath}`);
    } catch (error) {
      console.error(`Error decompressing ${gzPath}:`, error);
      throw new Error(`Failed to decompress CSV file: ${sanitizedDataSource}`);
    }
  }
  // Fall back to regular CSV
  else if (fs.existsSync(csvPath)) {
    csvContent = fs.readFileSync(csvPath, "utf-8");
    console.log(`Loaded regular CSV: ${csvPath}`);
  }
  // File not found
  else {
    throw new Error(
      `CSV file not found: ${sanitizedDataSource} (tried both .csv and .csv.gz)`
    );
  }

  const result = Papa.parse(csvContent, PAPAPARSE_CONFIG);

  if (result.errors.length > 0) {
    console.warn(
      `CSV parsing warnings for ${sanitizedDataSource}:`,
      result.errors
    );
  }

  return result.data;
}

function calculateReadTime(content) {
  const wordCount = content.split(" ").length;
  const wordsPerMinute = 200;
  return Math.round(wordCount / wordsPerMinute);
}
