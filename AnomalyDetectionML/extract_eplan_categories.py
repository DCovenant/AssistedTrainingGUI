#!/usr/bin/env python3
"""Extract EPLAN component categories from PDF text labels.

EPLAN stores component information as text labels in the PDF.
This script extracts and maps these to categories.
"""

import fitz  # PyMuPDF
import json
from pathlib import Path
from collections import defaultdict

def extract_text_with_position(pdf_path: str, page_num: int = 0) -> list[dict]:
    """Extract all text blocks with their positions and content.

    Returns:
        List of dicts with: {text, x0, y0, x1, y1, bbox}
    """
    doc = fitz.open(pdf_path)
    page = doc[page_num]

    text_items = []
    for text_block in page.get_text("blocks"):
        # text_block format: (x0, y0, x1, y1, text, block_no, block_type)
        if len(text_block) >= 5:
            x0, y0, x1, y1, text, *rest = text_block
            text = text.strip()
            if text and not text.isspace():
                text_items.append({
                    "text": text,
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "width": x1 - x0,
                    "height": y1 - y0,
                })

    doc.close()
    return text_items

def extract_text_words(pdf_path: str, page_num: int = 0) -> list[dict]:
    """Extract individual words with positions (more granular).

    Returns:
        List of dicts with word-level information
    """
    doc = fitz.open(pdf_path)
    page = doc[page_num]

    words = []
    for word_tuple in page.get_text("words"):
        # Format: (x0, y0, x1, y1, word, block_no, line_no, word_no)
        if len(word_tuple) >= 5:
            x0, y0, x1, y1, text, *rest = word_tuple
            words.append({
                "text": text,
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
            })

    doc.close()
    return words

def extract_text_by_font(pdf_path: str, page_num: int = 0) -> dict:
    """Extract text grouped by font properties.

    Returns:
        Dict mapping font names to lists of text occurrences
    """
    doc = fitz.open(pdf_path)
    page = doc[page_num]

    font_dict = defaultdict(list)

    for text_dict in page.get_text("rawdict")["blocks"]:
        if text_dict.get("type") != 1:  # Not text block
            continue

        for line in text_dict.get("lines", []):
            for span in line.get("spans", []):
                font_name = span.get("font", "Unknown")
                text = span.get("text", "")

                if text.strip():
                    font_dict[font_name].append({
                        "text": text,
                        "size": span.get("size"),
                        "flags": span.get("flags"),
                        "color": span.get("color"),
                    })

    doc.close()
    return dict(font_dict)

def identify_categories(text_items: list[dict]) -> dict:
    """Identify potential component categories from text.

    EPLAN typically uses specific text patterns for categories:
    - "CELA" = Cell (compartment)
    - "DISJUNTOR" = Circuit breaker
    - "PROT" = Protection
    - etc.
    """
    categories = defaultdict(list)

    for item in text_items:
        text = item["text"].upper()

        # Look for common EPLAN component keywords
        keywords = {
            "CELA": "Cell/Compartment",
            "DISJUNTOR": "Circuit Breaker",
            "PROT": "Protection Device",
            "TRACÇÃO": "Traction",
            "TENSÃO": "Voltage",
            "CORRENTE": "Current",
            "MEDIDA": "Measurement",
            "TRANSFORMADOR": "Transformer",
            "MOTOR": "Motor",
            "CONTACTOR": "Contactor",
            "TERMINAL": "Terminal",
            "BOBINA": "Coil",
            "RESISTÊNCIA": "Resistor",
            "CAPACITOR": "Capacitor",
        }

        for keyword, category in keywords.items():
            if keyword in text:
                categories[category].append(item)

    return dict(categories)

def main(pdf_path: str = "files/adulterado.pdf", output_file: str = "eplan_categories.json"):
    """Extract EPLAN categories from PDF."""
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        print(f"Error: PDF not found: {pdf_path}")
        return

    print(f"Analyzing PDF: {pdf_path}")

    # Extract text blocks
    print("\n1. Extracting text blocks...")
    text_blocks = extract_text_with_position(str(pdf_path))
    print(f"   Found {len(text_blocks)} text blocks")

    # Extract individual words
    print("\n2. Extracting individual words...")
    words = extract_text_words(str(pdf_path))
    print(f"   Found {len(words)} words")

    # Extract by font
    print("\n3. Extracting text by font...")
    fonts = extract_text_by_font(str(pdf_path))
    print(f"   Found {len(fonts)} different fonts")
    for font_name, items in list(fonts.items())[:5]:
        print(f"   - {font_name}: {len(items)} occurrences")

    # Identify categories
    print("\n4. Identifying EPLAN categories...")
    categories = identify_categories(text_blocks)
    for cat, items in categories.items():
        print(f"   - {cat}: {len(items)} occurrences")
        for item in items[:2]:  # Show first 2 examples
            print(f"     > {item['text']}")

    # Save results
    results = {
        "pdf": str(pdf_path),
        "text_blocks_count": len(text_blocks),
        "words_count": len(words),
        "fonts": {name: len(items) for name, items in fonts.items()},
        "categories_found": {cat: len(items) for cat, items in categories.items()},
        "text_blocks_sample": text_blocks[:10],
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Analysis saved to: {output_file}")

if __name__ == "__main__":
    main()
