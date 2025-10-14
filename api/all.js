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
  const postsDir = path.join(process.cwd(), "posts");
  let files = [];
  try {
    files = fs.readdirSync(postsDir);
  } catch (err) {
    return res.status(500).json({ error: "Unable to read posts directory" });
  }
  const mdFiles = files.filter((file) => file.endsWith(".md"));
  const posts = mdFiles.map((fileName) => {
    const filePath = path.join(postsDir, fileName);
    const contentRaw = fs.readFileSync(filePath, "utf-8");
    // Parse frontmatter
    const match = contentRaw.match(/^---([\s\S]*?)---\s*([\s\S]*)$/);
    let frontmatter = {};
    let content = contentRaw;
    if (match) {
      const frontmatterRaw = match[1];
      content = match[2].trim();
      const lines = frontmatterRaw.split("\n");
      let i = 0;
      while (i < lines.length) {
        const line = lines[i];
        const m = line.match(/^([a-zA-Z0-9_\-]+):\s*(.*)$/);
        if (m) {
          let key = m[1].trim();
          let value = m[2].trim();
          // Multiline array (YAML style)
          if (value === "") {
            // Check for indented lines (array items)
            let arr = [];
            i++;
            while (i < lines.length && lines[i].match(/^\s+-\s*(.*)$/)) {
              const item = lines[i].replace(/^\s+-\s*/, "").trim();
              arr.push(item);
              i++;
            }
            frontmatter[key] = arr;
            continue; // skip i++ at end of loop
          }
          // Multiline JSON-style array
          if (value.startsWith("[") && !value.endsWith("]")) {
            let arrLines = [value];
            i++;
            while (i < lines.length) {
              arrLines.push(lines[i]);
              if (lines[i].includes("]")) break;
              i++;
            }
            // Join all lines, remove newlines and extra spaces
            let arrStr = arrLines
              .join("")
              .replace(/\s+/g, " ")
              .replace(/,\s*\]/, "]")
              .replace(/'/g, '"')
              .trim();
            try {
              frontmatter[key] = JSON.parse(arrStr);
            } catch {
              // Fallback: extract quoted strings as array items
              const matches = arrStr.match(/"(.*?)"/g);
              if (matches) {
                frontmatter[key] = matches.map((s) => s.replace(/"/g, ""));
              } else {
                frontmatter[key] = [];
              }
              // Optionally: console.log('Failed to parse array:', arrStr);
            }
            i++;
            continue;
          }
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
        i++;
      }
    }
    return {
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
    };
  });
  // Sort posts by publication_date (most recent first)
  posts.sort((a, b) => {
    if (!a.publication_date) return 1;
    if (!b.publication_date) return -1;
    return new Date(b.publication_date) - new Date(a.publication_date);
  });
  res.json(posts);
}
