#!/usr/bin/env python3
"""
Clean QMD files for notebook conversion by removing:
- YAML frontmatter
- Specified ## sections (see SECTIONS_TO_REMOVE)
- Adds link to original article after first header
"""

import sys
import re
from pathlib import Path

# List of section headers to remove (case-insensitive matching)
SECTIONS_TO_REMOVE = [
    "Introduction",
    "Conclusion",
    "Final Thoughts",
    "Next Steps",
    "Conclusion and Next Steps",
    "Summary",
    "Related Resources",
]


def find_section_headers(lines):
    """Find all ## section headers and their line numbers."""
    headers = []
    for i, line in enumerate(lines):
        if line.strip().startswith("## ") and not line.strip().startswith("### "):
            headers.append((i, line.strip()))
    return headers


def extract_frontmatter_metadata(lines):
    """
    Extract title from YAML frontmatter.

    Returns:
        str: title or None if not found
    """
    in_frontmatter = False
    title = None

    for i, line in enumerate(lines):
        if i == 0 and line.strip() == "---":
            in_frontmatter = True
            continue
        elif in_frontmatter:
            if line.strip() == "---":
                break
            # Parse title
            if line.strip().startswith("title:"):
                title = line.split("title:", 1)[1].strip().strip('"\'')

    return title


def remove_frontmatter(lines):
    """Remove YAML frontmatter and return set of line numbers to skip."""
    skip_lines = set()
    in_frontmatter = False

    for i, line in enumerate(lines):
        if i == 0 and line.strip() == "---":
            in_frontmatter = True
            skip_lines.add(i)
        elif in_frontmatter:
            skip_lines.add(i)
            if line.strip() == "---":
                in_frontmatter = False
                break

    return skip_lines


def find_sections_to_remove(headers, sections_list, total_lines):
    """
    Find all section ranges that match the sections_list.

    Args:
        headers: List of (line_num, header_text) tuples
        sections_list: List of section names to match (case-insensitive)
        total_lines: Total number of lines in file

    Returns:
        List of (start, end) tuples for sections to remove
    """
    sections_to_remove = []

    for i, (line_num, header) in enumerate(headers):
        # Extract the header text without "## " and anchor tags
        header_text = header.replace("## ", "").split("{")[0].strip()

        # Check if this header starts with any section to remove (case-insensitive)
        for section_name in sections_list:
            # Match if header starts with the section name
            if header_text.lower().startswith(section_name.lower()):
                section_start = line_num
                section_end = headers[i + 1][0] if i + 1 < len(headers) else total_lines
                sections_to_remove.append((section_start, section_end))
                break

    return sections_to_remove


def mark_section_for_removal(skip_lines, start, end):
    """Mark a range of lines for removal."""
    if start is not None and end is not None:
        for line_num in range(start, end):
            skip_lines.add(line_num)


def find_first_header(lines):
    """
    Find the first markdown header in the cleaned lines.

    Returns:
        int or None: Line index of first header, or None if not found
    """
    header_pattern = re.compile(r'^#{1,6}\s+')
    for i, line in enumerate(lines):
        if header_pattern.match(line):
            return i
    return None


def remove_heading_anchors(lines):
    """
    Remove anchor IDs from markdown headings.

    Converts headings like:
        ## Introduction {#introduction}
    to:
        ## Introduction

    Args:
        lines: List of lines to process

    Returns:
        List of lines with anchors removed from headings
    """
    # Pattern to match {#anchor-id} at the end of heading lines
    anchor_pattern = re.compile(r'\s*\{#[a-zA-Z0-9_-]+\}\s*$')
    header_pattern = re.compile(r'^#{1,6}\s+')

    processed_lines = []
    for line in lines:
        # Check if this is a heading line
        if header_pattern.match(line):
            # Remove anchor if present
            cleaned_line = anchor_pattern.sub('', line)
            # Ensure line ends with newline if it was removed
            if not cleaned_line.endswith('\n') and line.endswith('\n'):
                cleaned_line += '\n'
            processed_lines.append(cleaned_line)
        else:
            processed_lines.append(line)

    return processed_lines


def remove_text_blocks(lines):
    """
    Remove ```text blocks from the content.

    These blocks contain sample output that should be removed since
    Jupyter notebooks will display actual code output.

    Args:
        lines: List of lines to process

    Returns:
        List of lines with ```text blocks removed
    """
    processed_lines = []
    in_text_block = False

    for line in lines:
        if line.strip() == '```text':
            in_text_block = True
            continue
        elif in_text_block and line.strip() == '```':
            in_text_block = False
            continue
        elif not in_text_block:
            processed_lines.append(line)

    return processed_lines


def insert_article_link(lines, blog_url, title):
    """
    Insert article link after the first header.

    Args:
        lines: List of cleaned lines
        blog_url: URL to the blog article
        title: Article title (optional, used in link text)

    Returns:
        List of lines with article link inserted
    """
    first_header_idx = find_first_header(lines)

    if first_header_idx is None:
        # No header found, insert at the beginning
        link_text = f"> 📖 Read the full article: [{title or 'Original Article'}]({blog_url})\n\n"
        return [link_text] + lines

    # Insert after the first header
    link_text = f"\n> 📖 Read the full article: [{title or 'Original Article'}]({blog_url})\n\n"

    return (
        lines[:first_header_idx + 1] +
        [link_text] +
        lines[first_header_idx + 1:]
    )


def clean_qmd_file(input_path, output_path, blog_url=None):
    """Clean QMD file by removing specified sections and optionally adding article link."""
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Step 1: Extract metadata from frontmatter
    title = extract_frontmatter_metadata(lines)

    # Step 2: Remove YAML frontmatter
    skip_lines = remove_frontmatter(lines)

    # Step 3: Find all ## section headers
    headers = find_section_headers(lines)

    if not headers:
        # No sections found, get cleaned lines and add article link
        cleaned_lines = [line for i, line in enumerate(lines) if i not in skip_lines]
        # Remove heading anchors and text blocks
        cleaned_lines = remove_heading_anchors(cleaned_lines)
        cleaned_lines = remove_text_blocks(cleaned_lines)
        if blog_url:
            cleaned_lines = insert_article_link(cleaned_lines, blog_url, title)

        # Remove excessive blank lines at the end
        while cleaned_lines and cleaned_lines[-1].strip() == "":
            cleaned_lines.pop()

        # Ensure file ends with single newline
        if cleaned_lines and not cleaned_lines[-1].endswith('\n'):
            cleaned_lines[-1] += '\n'

        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(cleaned_lines)

        print(f"✓ Cleaned QMD file saved to: {output_path}")
        return

    # Step 4: Find and remove all matching sections
    sections_to_remove = find_sections_to_remove(headers, SECTIONS_TO_REMOVE, len(lines))

    for section_start, section_end in sections_to_remove:
        mark_section_for_removal(skip_lines, section_start, section_end)

    # Step 5: Get cleaned lines, remove anchors/text blocks, and optionally insert article link
    cleaned_lines = [line for i, line in enumerate(lines) if i not in skip_lines]
    # Remove heading anchors and text blocks
    cleaned_lines = remove_heading_anchors(cleaned_lines)
    cleaned_lines = remove_text_blocks(cleaned_lines)
    if blog_url:
        cleaned_lines = insert_article_link(cleaned_lines, blog_url, title)

    # Remove excessive blank lines at the end
    while cleaned_lines and cleaned_lines[-1].strip() == "":
        cleaned_lines.pop()

    # Ensure file ends with single newline
    if cleaned_lines and not cleaned_lines[-1].endswith('\n'):
        cleaned_lines[-1] += '\n'

    # Step 6: Write cleaned content
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(cleaned_lines)

    print(f"✓ Cleaned QMD file saved to: {output_path}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python qmd_to_notebook.py <input.qmd> <output.qmd> [blog_url]")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    blog_url = sys.argv[3] if len(sys.argv) > 3 else None

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    clean_qmd_file(input_path, output_path, blog_url)


if __name__ == "__main__":
    main()
