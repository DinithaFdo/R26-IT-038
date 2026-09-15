import re
import unicodedata


HOMOGLYPH_MAP = {
    # --- Cyrillic lowercase (ORIGINAL, unchanged) ---
    "а": "a", "е": "e", "о": "o", "р": "r",
    "с": "c", "у": "y", "х": "x", "і": "i",
    "ѕ": "s", "ј": "j", "п": "n", "т": "t",
    "н": "n", "м": "m", "в": "v", "б": "b",

    # --- Greek lowercase (ORIGINAL, unchanged) ---
    "ο": "o", "ρ": "p", "ν": "v", "κ": "k",
    "χ": "x", "ε": "e", "α": "a", "τ": "t",

    # --- Fullwidth uppercase + lowercase (ORIGINAL, unchanged) ---
    "Ａ": "A", "Ｂ": "B", "Ｃ": "C", "Ｄ": "D",
    "Ｅ": "E", "Ｆ": "F", "Ｇ": "G", "Ｈ": "H",
    "Ｉ": "I", "Ｊ": "J", "Ｋ": "K", "Ｌ": "L",
    "Ｍ": "M", "Ｎ": "N", "Ｏ": "O", "Ｐ": "P",
    "Ｑ": "Q", "Ｒ": "R", "Ｓ": "S", "Ｔ": "T",
    "Ｕ": "U", "Ｖ": "V", "Ｗ": "W", "Ｘ": "X",
    "Ｙ": "Y", "Ｚ": "Z", "ａ": "a", "ｂ": "b",
    "ｃ": "c", "ｄ": "d", "ｅ": "e", "ｆ": "f",
    "ｇ": "g", "ｈ": "h", "ｉ": "i", "ｊ": "j",
    "ｋ": "k", "ｌ": "l", "ｍ": "m", "ｎ": "n",
    "ｏ": "o", "ｐ": "p", "ｑ": "q", "ｒ": "r",
    "ｓ": "s", "ｔ": "t", "ｕ": "u", "ｖ": "v",
    "ｗ": "w", "ｘ": "x", "ｙ": "y", "ｚ": "z",

    # --- NEW: Cyrillic UPPERCASE (FIX 1 -- previously missing) ---
    "А": "A", "В": "B", "Е": "E", "К": "K",
    "М": "M", "Н": "H", "О": "O", "Р": "P",
    "С": "C", "Т": "T", "Х": "X", "У": "Y",
    "І": "I", "Ѕ": "S", "Ј": "J",

    # --- NEW: Greek UPPERCASE (FIX 1 -- previously missing) ---
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z",
    "Η": "H", "Ι": "I", "Κ": "K", "Μ": "M",
    "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T",
    "Υ": "Y", "Χ": "X",
}

ZERO_WIDTH_CHARS = [
    "\u200B", "\u200C", "\u200D", "\uFEFF",
    "\u00AD", "\u2060", "\u2061", "\u2062",
    "\u2063", "\u2064",
]

HTML_ENTITY_MAP = {
    "&amp;"  : " ",
    "&nbsp;" : " ",
    "&lt;"   : "<",
    "&gt;"   : ">",
    "&quot;" : chr(34),
    "&apos;" : chr(39),
    "&mdash;": "-",
    "&ndash;": "-",
    "&laquo;": "",
    "&raquo;": "",
}


def _remove_null_bytes(text):
    """
    FIX 3: previously only \\x00 was stripped. Now strips the full
    dangerous C0 control-character range (\\x00-\\x1F) EXCEPT tab
    (\\t), newline (\\n), and carriage return (\\r), which are
    legitimate whitespace and must be preserved.
    """
    cleaned = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", text)
    return cleaned, cleaned != text


def _remove_zero_width_chars(text):
    cleaned = text
    for char in ZERO_WIDTH_CHARS:
        cleaned = cleaned.replace(char, "")
    return cleaned, cleaned != text


def _normalize_unicode(text):
    cleaned = unicodedata.normalize("NFKC", text)
    return cleaned, cleaned != text


def _normalize_homoglyphs(text):
    cleaned = "".join(HOMOGLYPH_MAP.get(ch, ch) for ch in text)
    return cleaned, cleaned != text


def _remove_html_entities(text):
    cleaned = text
    for entity, replacement in HTML_ENTITY_MAP.items():
        cleaned = cleaned.replace(entity, replacement)
    cleaned = re.sub(r"&#[0-9]+;", "", cleaned)
    return cleaned, cleaned != text


def _remove_urls(text):
    cleaned = re.sub(r"https?://\S+", "", text)
    cleaned = re.sub(r"www\.\S+",     "", cleaned)
    cleaned = re.sub(r"\S+@\S+\.\S+", "", cleaned)
    return cleaned, cleaned != text


def _remove_structural_patterns(text):
    cleaned = re.sub(r"\[removed\]|\[deleted\]", "", text)
    # FIX (Bug 9, minor): replace u/username and r/subreddit with a
    # placeholder instead of deleting outright. Deleting left broken
    # grammar behind when a trailing possessive followed (e.g.
    # "u/someuser's response" -> "'s response"). A placeholder keeps
    # the sentence grammatically intact: "[user]'s response".
    cleaned = re.sub(r"u/\w+", "[user]", cleaned)
    cleaned = re.sub(r"r/\w+", "[subreddit]", cleaned)
    cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1",       cleaned)
    cleaned = re.sub(r"\*(.*?)\*",       r"\1",       cleaned)
    cleaned = re.sub(r"#{1,6}\s",                    "", cleaned)
    cleaned = re.sub(r"_{2,}(.*?)_{2,}",  r"\1",       cleaned)
    # (cosmetic) remove empty "()" left behind after stripping
    # "(r/subreddit)"-style wrapped mentions -- kept for the rare
    # case a bare "()" still results (e.g. "(r/askreddit)" style
    # with no surrounding words)
    cleaned = re.sub(r"\(\s*\)", "", cleaned)
    return cleaned, cleaned != text


def _normalize_quotes_and_dashes(text):
    """
    FIX 2: previously returned only the cleaned string (no way for
    the caller to know a change happened, so it never reached
    attack_report). Now returns (text, found) like every other
    sanitization function in this file.
    """
    original = text
    replacements = {
        "\u201C": chr(34), "\u201D": chr(34), "\u201E": chr(34),
        "\u00AB": chr(34), "\u00BB": chr(34),
        "\u2018": chr(39), "\u2019": chr(39), "\u201A": chr(39),
        "\u2013": "-",     "\u2014": "-",     "\u2015": "-",
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text, text != original


def _normalize_repeated_punctuation(text):
    """
    Collapses runs of 3+ identical punctuation marks to a single one
    (e.g. "!!!!!" -> "!"). Double punctuation ("??", "!!") is now
    LEFT ALONE -- evidence from full-corpus Model 1 testing showed
    collapsing a single doubled "??" to "?" was enough to flip a
    real human CMV sample's prediction from Human (31.8% AI) to AI
    (55.2% AI). Doubled punctuation is common, mild human emphasis;
    3+ repeats remain the more clearly "attack-like" / unusual
    pattern and are still fully collapsed.
    """
    cleaned = re.sub(r"([!?.,-])\1{2,}", r"\1", text)
    return cleaned, cleaned != text


def _normalize_whitespace(text):
    cleaned = re.sub(r"[ \t]+",  " ",    text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = cleaned.strip()
    return cleaned, cleaned != text


def sanitize(text):
    """
    Sanitize input text by detecting and removing adversarial
    attack characters before AI detection.

    Handles 10 attack categories:
        1.  Null bytes / control characters
        2.  Zero-width characters
        3.  Mathematical Unicode (NFKC normalization)
        4.  Homoglyph substitution (Cyrillic / Greek / Fullwidth,
            upper AND lower case)
        5.  HTML entities
        6.  URLs and email addresses
        7.  Markdown and Reddit structural patterns
        8.  Non-standard quotes and dashes
        9.  Repeated punctuation
        10. Excessive whitespace

    Returns:
        dict:
            clean_text       -- sanitized output text
            attacks_detected -- True if any attack was found
            attack_report    -- list describing each attack found
            original_length  -- character count before sanitization
            clean_length     -- character count after sanitization
    """

    original = text
    report   = []

    text, found = _remove_null_bytes(text)
    if found:
        report.append("Null bytes / control characters removed")

    text, found = _remove_zero_width_chars(text)
    if found:
        report.append("Zero-width characters removed")

    text, found = _normalize_unicode(text)
    if found:
        report.append("Mathematical Unicode characters normalized")

    text, found = _normalize_homoglyphs(text)
    if found:
        report.append("Homoglyph substitution normalized")

    text, found = _remove_html_entities(text)
    if found:
        report.append("HTML entities removed")

    text, found = _remove_urls(text)
    if found:
        report.append("URLs and email addresses removed")

    text, found = _remove_structural_patterns(text)
    if found:
        report.append("Markdown and Reddit patterns removed")

    text, found = _normalize_quotes_and_dashes(text)
    if found:
        report.append("Quotes and dashes normalized")

    text, found = _normalize_repeated_punctuation(text)
    if found:
        report.append("Repeated punctuation normalized")

    text, found = _normalize_whitespace(text)
    if found:
        report.append("Excessive whitespace normalized")

    return {
        "clean_text"       : text,
        "attacks_detected" : len(report) > 0,
        "attack_report"    : report,
        "original_length"  : len(original),
        "clean_length"     : len(text),
    }
