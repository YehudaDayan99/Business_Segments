"""
CODE EXCERPT: Description Extraction Functions
File: revseg/react_agents.py (multiple sections)

PURPOSE: Extract product/service descriptions from 10-K filings using
multiple strategies: footnotes, headings, Note 2 prose, LLM fallback.
"""

import re
from typing import Optional, List, Dict, Any

# =============================================================================
# ACCOUNTING/DRIVER SENTENCE FILTER
# =============================================================================

# Pattern to detect accounting/performance driver language
_ACCOUNTING_DENY_RE = re.compile(
    r"(?:increased|decreased|grew|declined)\s+(?:by|due|primarily|mainly)|"
    r"(?:favorable|unfavorable)\s+(?:foreign|currency|impact)|"
    r"period[- ]over[- ]period|"
    r"year[- ]over[- ]year|"
    r"(?:higher|lower)\s+(?:revenue|sales|volume)|"
    r"acquisitions?\s+(?:of|contributed)|"
    r"ASC\s+\d{3}|"
    r"GAAP|"
    r"recognition\s+of\s+revenue|"
    r"revenue\s+recognition|"
    r"performance\s+obligation",
    re.IGNORECASE
)


def strip_accounting_sentences(text: str) -> str:
    """
    Remove sentences containing accounting/regulatory or performance driver language.
    
    This ensures descriptions focus on WHAT the product/service IS,
    not HOW it performed or accounting treatment.
    """
    if not text:
        return ""
    
    # Split into sentences (handling abbreviations like "Inc.")
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    clean_sentences = []
    
    for sentence in sentences:
        if not _ACCOUNTING_DENY_RE.search(sentence):
            clean_sentences.append(sentence)
    
    result = " ".join(clean_sentences).strip()
    return result


# =============================================================================
# TABLE HEADER CONTAMINATION CHECK (Phase 5)
# =============================================================================

TABLE_HEADER_MARKERS = [
    "Year Ended",
    "December 31",
    "June 30", 
    "September 30",
    "(in millions)",
    "(in thousands)",
    "% change",
    "% Change",
    "Total Revenue",
    "Total Net Sales",
    "Year-Over-Year",
    "Fiscal Year",
]


def _is_table_header_contaminated(text: str) -> bool:
    """
    Check if extracted text is contaminated with table header/structure content.
    
    This happens when heading-based extraction captures table structure instead
    of actual product/service definitions (e.g., META Reality Labs before P5 fix).
    
    Returns True if the text appears to be table structure, not a definition.
    """
    if not text:
        return False
    
    # Check for table header markers
    marker_count = sum(1 for marker in TABLE_HEADER_MARKERS if marker.lower() in text.lower())
    
    # If 2+ markers found, likely table header
    if marker_count >= 2:
        return True
    
    # Check for column-like patterns (e.g., "2024 2023 2022")
    year_pattern = re.compile(r'\b20\d{2}\s+20\d{2}\s+20\d{2}\b')
    if year_pattern.search(text):
        return True
    
    # Check for dollar amounts at the start (revenue table data)
    dollar_start = re.compile(r'^\s*\$?\s*\d{1,3}(?:,\d{3})+')
    if dollar_start.match(text):
        return True
    
    # Check for percentage patterns that indicate table data
    pct_pattern = re.compile(r'\b\d+\s*%\s*(change|increase|decrease)\b', re.IGNORECASE)
    if pct_pattern.search(text[:200]):  # Check first 200 chars
        return True
    
    return False


# =============================================================================
# FOOTNOTE EXTRACTION (for AMZN-style tables with (1), (2), etc.)
# =============================================================================

_FOOTNOTE_MARKER_RE = re.compile(r'\((\d+)\)$')


def _extract_footnote_for_label(label: str, html_text: str, table_context_text: str) -> Optional[str]:
    """
    Extract footnote definition for a revenue line label that contains a footnote marker.
    
    Strategy:
    1. Find table separator (_____) then look for (N) Includes pattern
    2. Fallback: Search for (N) Includes in window after label
    3. Fallback: Check pre-extracted table context
    
    Args:
        label: Revenue line label like "Online stores (1)"
        html_text: Full HTML text to search
        table_context_text: Table's nearby text context
        
    Returns:
        Footnote definition text if found, None otherwise.
    """
    # Check if label has a footnote marker
    match = _FOOTNOTE_MARKER_RE.search(label)
    if not match:
        return None
    
    footnote_num = match.group(1)
    label_clean = _FOOTNOTE_MARKER_RE.sub('', label).strip().lower()
    
    # Strategy: Look for table separator (___) followed by footnotes
    low = html_text.lower() if html_text else ""
    
    # Find occurrences of the label
    label_positions = []
    search_pos = 0
    while True:
        idx = low.find(label_clean, search_pos)
        if idx == -1:
            break
        label_positions.append(idx)
        search_pos = idx + len(label_clean)
        if len(label_positions) >= 10:
            break
    
    # For each label position, look for "_____" separator followed by footnotes
    for label_idx in label_positions:
        separator_idx = html_text.find("_____", label_idx, label_idx + 3000)
        if separator_idx != -1:
            footnote_window = html_text[separator_idx:separator_idx + 10000]
            
            # Look for (N) Includes pattern
            pattern = rf'\({footnote_num}\)\s*(Includes\s+[^(]+?)(?=\(\d+\)|_____|$)'
            matches = re.findall(pattern, footnote_window, re.IGNORECASE | re.DOTALL)
            if matches:
                cleaned = _clean(matches[0])
                if len(cleaned) >= 20:
                    return cleaned[:600]
    
    # Fallback: search in window after label
    for label_idx in label_positions:
        window = html_text[label_idx:label_idx + 15000]
        pattern = rf'\({footnote_num}\)\s*(Includes\s+[^(]+?)(?=\(\d+\)|_____|$)'
        matches = re.findall(pattern, window, re.IGNORECASE | re.DOTALL)
        if matches:
            cleaned = _clean(matches[0])
            if len(cleaned) >= 20:
                return cleaned[:600]
    
    return None


# =============================================================================
# HEADING-BASED EXTRACTION (for AAPL-style structured Item 1)
# =============================================================================

def _extract_heading_based_definition(
    html_text: str,
    label: str,
    section_text: Optional[str] = None,
) -> Optional[str]:
    """
    Extract definition for a label that appears as a heading in Item 1.
    
    For Apple-style 10-Ks where products/services have dedicated headings
    (e.g., <b>Services</b>, <strong>iPhone</strong>) followed by descriptive paragraphs.
    
    Phase 3 enhancement: For parent headings like "Services" that have subheadings
    (Advertising, AppleCare, etc.), aggregate ALL child content until a true
    peer heading (another major product category like "iPhone", "Mac").
    
    Strategy:
    1. Find label as a HEADING (bold/strong/h1-h3 tag)
    2. Identify known major section headings (peer headings to stop at)
    3. Continue collecting content past child subheadings
    4. Stop only at a peer heading or max chars
    5. Apply accounting sentence filter
    """
    search_text = section_text if section_text else html_text
    if not search_text:
        return None
    
    label_escaped = re.escape(label.strip())
    
    # Known major product/service headings (peer level) - stop aggregation here
    PEER_HEADINGS = {
        'iphone', 'mac', 'ipad', 'services', 'wearables', 'home', 'accessories',
        'wearables, home and accessories', 'products', 'total net sales',
        'item 2', 'item 3', 'business', 'properties', 'legal proceedings',
    }
    
    # Known child subheadings (not peer level) - continue past these
    CHILD_SUBHEADINGS = {
        'advertising', 'apple care', 'applecare', 'cloud services', 
        'digital content', 'payment services', 'other services',
        'app store', 'apple music', 'apple tv+', 'apple arcade', 'apple news+',
        'apple fitness+', 'icloud', 'apple card', 'apple pay',
    }
    
    heading_tags = ['b', 'strong', 'h1', 'h2', 'h3', 'h4']
    
    for tag in heading_tags:
        pattern = rf'<{tag}[^>]*>\s*{label_escaped}\s*</{tag}>'
        match = re.search(pattern, search_text, re.IGNORECASE)
        
        if match:
            start_pos = match.end()
            remaining_text = search_text[start_pos:]
            
            # Find ALL headings in remaining text
            any_heading_pattern = rf'<(?:{"|".join(heading_tags)})[^>]*>([A-Z][^<]{{1,100}})</(?:{"|".join(heading_tags)})>'
            
            content_end = len(remaining_text)
            max_content = 15000  # Allow up to 15k chars for parent sections with subsections
            
            for heading_match in re.finditer(any_heading_pattern, remaining_text[:max_content], re.IGNORECASE):
                heading_text = heading_match.group(1).strip().lower()
                
                is_peer = heading_text in PEER_HEADINGS
                is_major_section = (
                    len(heading_text.split()) <= 2 and
                    heading_text not in CHILD_SUBHEADINGS and
                    heading_text != label.strip().lower() and
                    heading_match.start() > 200
                )
                
                if is_peer or is_major_section:
                    content_end = heading_match.start()
                    break
            
            content = remaining_text[:min(content_end, max_content)]
            
            # Parse HTML and extract clean text
            from bs4 import BeautifulSoup
            try:
                soup = BeautifulSoup(content, 'lxml')
                text_content = soup.get_text(separator=' ', strip=True)
            except Exception:
                text_content = re.sub(r'<[^>]+>', ' ', content)
            
            # Clean and filter
            cleaned = _clean(text_content)
            filtered = strip_accounting_sentences(cleaned)
            
            if filtered and len(filtered) >= 50:
                # Check for table-header contamination (Phase 5)
                if _is_table_header_contaminated(filtered):
                    continue  # Skip this match
                
                # Limit to first 6 sentences for parent sections
                sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', filtered)
                if len(sentences) > 6:
                    filtered = ' '.join(sentences[:6])
                return filtered[:1500]
    
    return None


# =============================================================================
# NOTE 2 PARAGRAPH EXTRACTION (for META-style prose definitions)
# =============================================================================

NOTE2_SECTION_PATTERN = re.compile(
    r'(?:Note\s*2|NOTE\s*2)[^A-Za-z]*(?:Revenue|REVENUE)',
    re.IGNORECASE
)


def _extract_note2_section(html_text: str, max_chars: int = 50000) -> Optional[str]:
    """Extract Note 2 - Revenue section from the filing."""
    match = NOTE2_SECTION_PATTERN.search(html_text)
    if not match:
        return None
    
    start = match.start()
    end_pattern = re.compile(r'(?:Note\s*3|NOTE\s*3|Note\s*4|NOTE\s*4)', re.IGNORECASE)
    end_match = end_pattern.search(html_text, start + 100)
    
    if end_match:
        end = min(end_match.start(), start + max_chars)
    else:
        end = start + max_chars
    
    return html_text[start:end]


def _extract_note2_paragraph_definition(html_text: str, label: str) -> Optional[str]:
    """
    Extract definition for a label from Note 2 - Revenue section.
    
    This handles META-style filings where definitions are in prose paragraphs
    like "Advertising revenue is generated from marketers advertising on our apps..."
    
    Key insight (Phase 5): For "advertising", search full html_text not just Note 2,
    because the definition may be in Item 1 Business section.
    """
    note2_section = _extract_note2_section(html_text)
    search_text = note2_section if note2_section else html_text
    
    label_clean = label.strip().lower()
    label_escaped = re.escape(label.strip())
    
    # For "advertising", use META-specific pattern on FULL text
    if label_clean in ('advertising', 'advertising revenue'):
        advertising_pattern = re.compile(
            r'(?:substantially all|majority)\s+of\s+(?:our\s+)?revenue\s+'
            r'from\s+(?:selling\s+)?advertising\s+'
            r'([^.]{30,400}\.(?:\s+[A-Z][^.]{20,200}\.)?)',
            re.IGNORECASE | re.DOTALL
        )
        match = advertising_pattern.search(html_text)  # Full text
        if match:
            definition = match.group(1).strip()
            if not _is_table_header_contaminated(definition):
                full_desc = f"Revenue from selling advertising {definition}"
                return strip_accounting_sentences(full_desc)
        
        # Skip generic patterns for advertising - they match wrong context
        return None
    
    # Generic patterns for other labels
    direct_pattern = re.compile(
        rf'{label_escaped}'
        r'(?:\s+(?:revenue|revenues|segment))?\s+'
        r'(?:is generated from|includes|consists of|is comprised of|is derived from|represents|are generated from)'
        r'\s+([^.]{20,500}\.)',
        re.IGNORECASE | re.DOTALL
    )
    
    match = direct_pattern.search(search_text)
    if match:
        definition = match.group(1).strip()
        if not _is_table_header_contaminated(definition):
            return strip_accounting_sentences(definition)
    
    # Special handling for "Other revenue"
    if label_clean in ('other', 'other revenue'):
        other_pattern = re.compile(
            r'Other\s+(?:revenue|revenues)\s+'
            r'(?:consists of|includes|is comprised of|represents)'
            r'\s+([^.]{20,500}\.)',
            re.IGNORECASE | re.DOTALL
        )
        match = other_pattern.search(search_text)
        if match:
            definition = match.group(1).strip()
            if not _is_table_header_contaminated(definition):
                return strip_accounting_sentences(definition)
    
    return None


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _clean(text: str) -> str:
    """Basic text cleaning."""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text
