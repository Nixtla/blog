import fs from "fs";
import path from "path";

export default function handler(req, res) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET,OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
  if (req.method === "OPTIONS") {
    res.status(200).end();
    return;
  }
  const { fileName } = req.query;
  if (!fileName) {
    return res.status(400).json({ error: "Missing fileName parameter" });
  }
  const mdPath = path.join(process.cwd(), "posts", `${fileName}.md`);
  if (!fs.existsSync(mdPath)) {
    return res.status(404).json({ error: "Markdown file not found" });
  }
  const raw = fs.readFileSync(mdPath, "utf-8");
  // Parse frontmatter
  const match = raw.match(/^---([\s\S]*?)---\s*([\s\S]*)$/);
  if (!match) {
    return res
      .status(500)
      .json({ error: "Invalid markdown frontmatter format" });
  }
  const frontmatterRaw = match[1];
  const content = match[2].trim();
  // Parse YAML frontmatter manually (simple key: value pairs)
  const frontmatter = {};
  frontmatterRaw.split("\n").forEach((line) => {
    const m = line.match(/^([a-zA-Z0-9_\-]+):\s*(.*)$/);
    if (m) {
      let key = m[1].trim();
      let value = m[2].trim();
      // Remove quotes if present
      if (
        (value.startsWith('"') && value.endsWith('"')) ||
        (value.startsWith("'") && value.endsWith("'"))
      ) {
        value = value.slice(1, -1);
      }
      // Parse arrays (e.g. tags: ["a", "b"])
      if (value.startsWith("[") && value.endsWith("]")) {
        try {
          value = JSON.parse(value.replace(/'/g, '"'));
        } catch {}
      }
      frontmatter[key] = value;
    }
  });
  res.json({
    title: frontmatter.title || null,
    author_name: frontmatter.author_name || null,
    author_image: frontmatter.author_image || null,
    author_position: frontmatter.author_position || null,
    publication_date: frontmatter.publication_date || null,
    description: frontmatter.description || null,
    image: frontmatter.image || null,
    categories: frontmatter.categories || null,
    tags: frontmatter.tags || null,
    fileName: fileName.replace(/\.md$/, ""),
    readTimeMinutes: Math.round(content.split(" ").length / 200),
    content,
  });
}
