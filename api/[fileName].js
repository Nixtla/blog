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
    const { contentWithPlaceholders, charts } = extractCharts(
      content,
      postSlug
    );

    const response = {
      ...frontmatter,
      fileName: postSlug,
      readTimeMinutes: calculateReadTime(content),
      content: contentWithPlaceholders,
      charts,
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

function extractCharts(content, postSlug) {
  if (!content.includes("```chart")) {
    return { contentWithPlaceholders: content, charts: {} };
  }

  const charts = {};
  let chartIndex = 0;

  const processChart = (match, chartJson, type) => {
    try {
      const chartData = JSON.parse(chartJson.trim());
      const chartId = chartData.id || `${type}-${chartIndex++}`;

      if (chartData.dataSource) {
        chartData.data = loadChartData(postSlug, chartData.dataSource);
      }

      chartData.type = type;
      charts[chartId] = chartData;

      return `{{CHART:${chartId}}}`;
    } catch (error) {
      console.error(`Failed to process ${type}:`, error.message);
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

  return { contentWithPlaceholders, charts };
}

function loadChartData(postSlug, dataSource) {
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

  return result.data;
}

function calculateReadTime(content) {
  const wordCount = content.split(" ").length;
  const wordsPerMinute = 200;
  return Math.round(wordCount / wordsPerMinute);
}
