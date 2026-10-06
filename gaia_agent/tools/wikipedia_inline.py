"""Inline text of MediaWiki HTML: which elements are noise and how running text is flattened.

Shared by the block converter (``wikipedia_html``) and the table converter
(``wikipedia_tables``).  Pure functions over BeautifulSoup nodes; no network.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from bs4 import NavigableString, Tag
from bs4.element import PageElement

SEPARATOR = "\x1f"  # stands for a <br> or the end of a list item until the text is cleaned
ICON_MAX_WIDTH = 60  # wider images are photos, not icons
ORDINAL_SUFFIXES = frozenset({"st", "nd", "rd", "th"})
MATH_ALT_PREFIX = "{\\displaystyle"

NOISE_TAGS = frozenset(
    {"style", "script", "link", "meta", "noscript", "figure", "template", "svg", "audio", "video", "iframe"}
    | {"form", "button", "input"}
)
NOISE_CLASSES = frozenset(
    {
        # citations, edit links and other page furniture
        "mw-editsection", "mw-editsection-like", "reference", "references", "reflist", "mw-references-wrap",
        "mw-cite-backlink", "mw-ext-cite-error", "noprint", "mw-empty-elt", "mw-jump-link", "printfooter",
        "catlinks", "mw-collapsible-toggle", "shortdescription", "Inline-Template", "sortkey",
        # navigation, message boxes, sister-project boxes, authority control
        "navbox", "navbox-styles", "navigation-not-searchable", "vertical-navbox", "navbar", "sidebar",
        "hatnote", "dablink", "rellink", "ambox", "ombox", "tmbox", "cmbox", "fmbox", "imbox", "asbox",
        "metadata", "side-box", "sistersitebox", "toc", "authority-control",
        # images, flags, image captions and the Wikidata pen icon
        "thumb", "flagicon", "infobox-caption", "penicon",
    }
)  # fmt: skip
NOISE_IDS = frozenset({"toc", "catlinks"})
BLOCK_TAGS = frozenset(
    {"p", "div", "section", "article", "aside", "blockquote", "center", "dl", "dt", "dd", "ul", "ol", "table"}
    | {"caption", "tr", "td", "th", "pre", "hr", "h1", "h2", "h3", "h4", "h5", "h6", "header", "footer", "nav"}
    | {"main", "details", "summary"}
)

_SPACES = re.compile(r"\s+")
_INVISIBLE = dict.fromkeys((0x200B, 0x2060, 0xFEFF, 0x00AD))  # zero-width space, word joiner, BOM, soft hyphen


def clean(text: str) -> str:
    """Collapse whitespace (non-breaking spaces included) and drop zero-width characters."""
    return _SPACES.sub(" ", text.translate(_INVISIBLE).replace(SEPARATOR, " ")).strip()


def is_noise(tag: Tag) -> bool:
    """True for elements that carry no article content: references, boxes, styles, hidden text."""
    if tag.name in NOISE_TAGS or tag.get("id") in NOISE_IDS:
        return True
    if not NOISE_CLASSES.isdisjoint(tag.get("class") or ()):
        return True
    style = tag.get("style")
    return isinstance(style, str) and "display:none" in style.replace(" ", "").lower()


def inline_text(node: Tag, icons: bool = False) -> str:
    """The running text inside ``node``, still carrying ``SEPARATOR`` marks for line breaks and list items."""
    return inline_children(node.children, icons)


def inline_children(children: Iterable[PageElement], icons: bool = False) -> str:
    """The running text of some nodes (tags and strings; noise, comments and doctypes are left out)."""
    parts: list[str] = []
    for child in children:
        if isinstance(child, Tag):
            if not is_noise(child):
                parts.append(inline_tag(child, icons))
        elif type(child) is NavigableString:  # not comments, doctypes or CDATA
            parts.append(str(child))
    return "".join(parts)


def inline_tag(tag: Tag, icons: bool = False) -> str:
    """The running text of one tag: its content, with line breaks, images, superscripts and list items handled."""
    name = tag.name
    if name == "br":
        return SEPARATOR
    if name == "img":
        return _image_text(tag, icons)
    inner = inline_text(tag, icons)
    if name == "sup":
        return _superscript(inner)
    if name in ("li", "tr"):  # list items and the rows of a nested table end with a separator
        return f" {inner}{SEPARATOR}"
    return f" {inner} " if name in BLOCK_TAGS else inner


def text_of(tag: Tag) -> str:
    """Clean one-line text of ``tag``."""
    return clean(inline_text(tag))


def _image_text(tag: Tag, icons: bool) -> str:
    """Alt text of an image: always for TeX formulas, for small icons only when ``icons`` is set."""
    alt = str(tag.get("alt") or "").strip()
    if alt.startswith(MATH_ALT_PREFIX):
        return alt
    width = str(tag.get("width") or "")
    is_icon = not width.isdecimal() or int(width) <= ICON_MAX_WIDTH
    return f" {alt} " if icons and is_icon and alt else ""


def _superscript(inner: str) -> str:
    text = inner.strip()
    if not text or text in ORDINAL_SUFFIXES:
        return text
    return f"^{text}"
