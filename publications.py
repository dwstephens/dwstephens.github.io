#!/usr/bin/env python3

import subprocess
import tempfile
from pathlib import Path

def create_pub_listing(bib_file: str, author: str = "Stephens"):
    """
    Pandoc for BibTeX conversion.

    1.  Splitting the .bib file into individual entries.
    2.  Calling the external `pandoc` command on each entry to create a YAML block.
    3.  Parsing the text output from pandoc to extract data.
    4.  Adding the custom fields (`position`, `first`, `journal-title`, etc.).
    5.  Aggregating the results into the final `publications.yml` and `publications.qmd` files.
    """
    bib_file_path = Path(bib_file)
    if not bib_file_path.exists():
        print(f"Error: Input file not found at '{bib_file_path}'")
        return

    yml_file_path = bib_file_path.with_suffix(".yml")
    qmd_file_path = bib_file_path.with_suffix(".qmd")

    # Read and split the BibTeX file into entries, just like the R script
    with open(bib_file_path, 'r', encoding='utf-8') as f:
        bib_content = f.read()
    
    entries_str = bib_content.strip().split('\n@')
    bib_entries = []
    if entries_str:
        # Handle the first entry, which might not have a leading '@' after the split
        first_entry = entries_str[0].strip()
        if first_entry and not first_entry.startswith('@'):
            first_entry = '@' + first_entry
        if first_entry:
             bib_entries.append(first_entry)
        
        # Prepend '@' to all subsequent entries
        for entry_part in entries_str[1:]:
            if entry_part.strip():
                bib_entries.append('@' + entry_part)

    all_articles_yaml_lines = []
    first_author_count = 0

    # Process each entry using pandoc
    for entry_str in bib_entries:
        # Write the single entry to a temporary file for pandoc to read
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".bib", encoding='utf-8') as temp_bib:
            temp_bib.write(entry_str)
            temp_bib_path = temp_bib.name
        
        try:
            # Execute the same pandoc command as the R script
            result = subprocess.run(
                ["pandoc", temp_bib_path, "--standalone", "--from=bibtex", "--to=markdown"],
                capture_output=True, text=True, check=True, encoding='utf-8'
            )
            pandoc_output_lines = result.stdout.splitlines()

            article_lines = pandoc_output_lines[3:-2]

            # Extract data from the raw pandoc output lines
            authors = []

            container_title, issued_date, doi, url = "", "", "", ""

            for idx, line in enumerate(article_lines):
                if "- family:" in line:
                    family_name = line.split("family:", 1)[1].strip()
                    authors.append(family_name)
                elif line.strip().startswith("publisher:"):
                    publisher = line.split(":", 1)[1].strip()
                elif line.strip().startswith("container-title:"):
                    container_title = line.split(":", 1)[1].strip()
                elif line.strip().startswith("issued:"):
                    issued_date = line.split(":", 1)[1].strip()
                elif line.strip().startswith("doi:"):
                    doi = line.split(":", 1)[1].strip()
                elif line.strip().startswith("url:"):
                    url = line.split(":", 1)[1].strip()
                elif line.strip().startswith("type:"):
                    bib_type = line.split(":", 1)[1].strip()
                    if bib_type == "book":
                        article_lines[idx] = "  type: conference"
                        container_title = publisher
                    elif bib_type == "article-journal":
                        article_lines[idx] = "  type: journal"
            try:
                author_pos = authors.index(author) + 1
                position_line = f"  position: '{author_pos}/{len(authors)}'"

                if author_pos == 1:
                    first_author_count +=1             
            except ValueError:
                position_line = f"  position: 'N/A/{len(authors)}'"
            
            if "type:" in article_lines:
                print("here")

            # Combine original and new lines for this entry
            all_articles_yaml_lines.extend(article_lines)
            if container_title:
                all_articles_yaml_lines.append(f"  type-title: '*{container_title}*'")
            if issued_date:
                all_articles_yaml_lines.append(f"  date: {issued_date}")
            if doi:
                all_articles_yaml_lines.append(f"  path: https://doi.org/{doi}")
            if url:
                all_articles_yaml_lines.append(f"  path: {url}")
                
            all_articles_yaml_lines.append(position_line)
            #all_articles_yaml_lines.append(first_line)
        
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Pandoc failed for an entry. Stderr: {e.stderr}")
        
        finally:
            Path(temp_bib_path).unlink()
    #print(all_articles_yaml_lines)
    # 6. Write the final YAML file
    with open(yml_file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_articles_yaml_lines))
        f.write('\n')

    # 7. Write the QMD file
    total_articles = len(bib_entries)
    title_counts = f"{first_author_count} + {total_articles - first_author_count}"

    qmd_template = f"""---
title: 'Publications ({title_counts})'
title-block-banner: true
date-format: 'MMMM,<br>YYYY'
listing:
  contents:
    - {yml_file_path.name}
  page-size: 10
  sort: 'date desc'
  type: table
  categories: false
  sort-ui: [date, title, type, type-title, position]
  filter-ui: [date, title, type, type-title]
  fields: [date, title, type, type-title, position]
  field-display-names:
    date: Date
    type: Type
    type-title: Type Name
    position: Rank
---
"""
    with open(qmd_file_path, 'w', encoding='utf-8') as f:
        f.write(qmd_template)
    
    print(f"Successfully created '{yml_file_path}' and '{qmd_file_path}'.")


if __name__ == "__main__":
    # This script requires pandoc to be installed and accessible in your system's PATH.
    try:
        subprocess.run(["pandoc", "--version"], check=True, capture_output=True, text=True)
        create_pub_listing("publications.bib", author="Stephens")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: Pandoc is not installed or not found in your PATH.")
        print("Please install pandoc to run this script: https://pandoc.org/installing.html")