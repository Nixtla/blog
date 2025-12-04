import fs from "fs";
import path from "path";
import Papa from "papaparse";

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
    const { fileName, maxPoints } = req.query;

    if (!fileName) {
      return res.status(400).json({ error: "Missing fileName parameter" });
    }

    // Use maxPoints from query param or default to 2000
    const parsedMaxPoints = maxPoints ? parseInt(maxPoints, 10) : 2000;

    if (maxPoints && Number.isNaN(parsedMaxPoints)) {
      return res.status(400).json({
        error: "Invalid maxPoints parameter - must be a number",
      });
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
      postSlug,
      parsedMaxPoints
    );

    const response = {
      title: frontmatter.title || null,
      seo_title: frontmatter.seo_title || null,
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

  let sanitized = dataSource.replace(/\\/g, "/");
  // Remove all "../", "..//", etc. recursively to prevent incomplete sanitization
  while (/\.\.\/*/.test(sanitized)) {
    sanitized = sanitized.replace(/\.\.\/*/g, "");
  }
  sanitized = sanitized.replace(/^\/+/, "");

  if (!/^[a-zA-Z0-9_\-\/\.]+$/.test(sanitized)) {
    throw new Error("Invalid characters in dataSource");
  }

  if (!sanitized.endsWith(".csv")) {
    throw new Error("dataSource must be a CSV file");
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

function extractCharts(content, postSlug, maxPoints) {
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
        // Use chart-specific maxPoints if provided, otherwise use global maxPoints
        const chartMaxPoints = chartData.maxPoints || maxPoints;
        chartData.data = loadChartData(
          postSlug,
          chartData.dataSource,
          chartMaxPoints
        );
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
        // Use chart-specific maxPoints if provided, otherwise use global maxPoints
        const chartMaxPoints = chartData.maxPoints || maxPoints;
        chartData.data = loadChartData(
          postSlug,
          chartData.dataSource,
          chartMaxPoints
        );
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

function loadChartData(postSlug, dataSource, maxPoints) {
  const sanitizedDataSource = sanitizeDataSource(dataSource);

  const csvPath = path.join(
    process.cwd(),
    "blogCharts",
    postSlug,
    sanitizedDataSource
  );

  if (!fs.existsSync(csvPath)) {
    throw new Error(`CSV file not found: ${sanitizedDataSource}`);
  }

  const csvContent = fs.readFileSync(csvPath, "utf-8");
  const result = Papa.parse(csvContent, PAPAPARSE_CONFIG);

  if (result.errors.length > 0) {
    console.warn(
      `CSV parsing warnings for ${sanitizedDataSource}:`,
      result.errors
    );
  }

  const data = result.data;

  if (data.length > maxPoints) {
    return downsampleData(data, maxPoints);
  }

  return data;
}

function downsampleData(data, targetPoints) {
  if (data.length <= targetPoints) {
    return data;
  }

  let step = Math.floor(data.length / targetPoints);

  if (step < 2) {
    step = 2;
  }

  const downsampled = [];

  downsampled.push(data[0]);

  for (let i = step; i < data.length - 1; i += step) {
    downsampled.push(data[i]);
  }

  if (data.length > 1) {
    downsampled.push(data[data.length - 1]);
  }

  return downsampled;
}

function calculateReadTime(content) {
  const wordCount = content.split(" ").length;
  const wordsPerMinute = 200;
  return Math.round(wordCount / wordsPerMinute);
}
