"""Tests for the MediaWiki HTML to markdown conversion (headings, text, lists, noise removal).

Tables have their own tests in ``test_wikipedia_tables.py``.  The HTML mimics the structure of
MediaWiki parser output but every text is invented.
"""

from __future__ import annotations

import warnings

import pytest

from gaia_agent.tools.wikipedia_html import html_to_markdown, plain_text

EDIT_LINK = (
    '<span class="mw-editsection"><span class="mw-editsection-bracket">[</span>'
    '<a href="/w/index.php?title=Example_Band&amp;action=edit&amp;section=1" title="Edit section: History">'
    '<span>edit</span></a><span class="mw-editsection-bracket">]</span></span>'
)

ARTICLE = (
    '<div class="mw-content-ltr mw-parser-output" lang="en" dir="ltr">'
    '<div class="shortdescription nomobile noexcerpt noprint searchaux" style="display:none">Fictional band</div>'
    '<style data-mw-deduplicate="TemplateStyles:r1">.mw-parser-output .hatnote{font-style:italic}</style>'
    '<div role="note" class="hatnote navigation-not-searchable">For other uses, see <a href="/wiki/E">E</a>.</div>'
    '<table class="infobox vcard"><tbody>'
    '<tr><th colspan="2" class="infobox-above">Example Band</th></tr>'
    '<tr><td colspan="2" class="infobox-image"><span class="mw-default-size"><a href="/wiki/File:X.jpg">'
    '<img alt="Stage photo" src="x.jpg" width="220" height="150"/></a></span></td></tr>'
    '<tr><th scope="row" class="infobox-label">Origin</th><td class="infobox-data">Exampleton</td></tr>'
    '<tr><th scope="row" class="infobox-label">Genres</th><td class="infobox-data">'
    '<div class="plainlist"><ul><li>Rock</li><li>Pop</li></ul></div></td></tr>'
    "</tbody></table>"
    '<p><b>Example Band</b> is a <a href="/wiki/Fiction" title="Fiction">fictional</a> group.'
    '<sup id="cite_ref-1" class="reference"><a href="#cite_note-1">[1]</a></sup> It formed in 1990.</p>\n'
    f'<div class="mw-heading mw-heading2"><h2 id="Discography">Discography</h2>{EDIT_LINK}</div>\n'
    '<table class="wikitable plainrowheaders"><caption>Studio albums</caption><tbody>'
    '<tr><th rowspan="2" scope="col">Title</th><th rowspan="2" scope="col">Year</th>'
    '<th colspan="2" scope="col">Peak position</th></tr>'
    '<tr><th scope="col">US</th><th scope="col">UK</th></tr>'
    '<tr><th scope="row"><i>Blue Morning</i></th><td>1992</td><td>12</td><td>8</td></tr>'
    '<tr><th scope="row"><i>Red Evening</i></th><td>1995</td><td>4</td><td>—</td></tr>'
    "</tbody></table>"
    '<div class="mw-heading mw-heading2"><h2 id="See_also">See also</h2></div>\n'
    '<ul><li><a href="/wiki/Other">Other Band</a></li></ul>'
    '<div class="mw-heading mw-heading2"><h2 id="References">References</h2></div>\n'
    '<div class="reflist"><ol class="references"><li id="cite_note-1"><span class="reference-text">A source.'
    "</span></li></ol></div>"
    '<div class="navbox"><table class="nowraplinks"><tr><td>Navigation</td></tr></table></div>'
    "<!-- NewPP limit report -->"
    "</div>"
)

ARTICLE_MARKDOWN = """\
| Example Band |
| Origin | Exampleton |
| Genres | Rock; Pop |

Example Band is a fictional group. It formed in 1990.

## Discography

Table: Studio albums
| Title | Year | Peak position |
|  |  | US | UK |
| --- | --- | --- | --- |
| Blue Morning | 1992 | 12 | 8 |
| Red Evening | 1995 | 4 | — |

## See also

- Other Band"""


def convert(fragment: str) -> str:
    return html_to_markdown(f'<div class="mw-parser-output">{fragment}</div>')


def test_a_whole_article_keeps_content_and_tables_and_drops_the_noise() -> None:
    assert html_to_markdown(ARTICLE) == ARTICLE_MARKDOWN


def test_empty_input_gives_empty_markdown() -> None:
    assert html_to_markdown("") == ""
    assert html_to_markdown("   \n ") == ""


def test_text_that_looks_like_a_url_is_not_mistaken_for_a_locator() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")

        assert html_to_markdown("https://example.org/page") == "https://example.org/page"


# --- headings, paragraphs and inline text -------------------------------------------------------


def test_headings_use_hashes_and_lose_their_edit_links() -> None:
    html = (
        f'<div class="mw-heading mw-heading2"><h2 id="History">History</h2>{EDIT_LINK}</div>'
        "<p>The band formed in 1990.</p>"
        '<div class="mw-heading mw-heading3"><h3 id="Early">Early years</h3></div><p>Gigs.</p>'
    )

    assert convert(html) == "## History\n\nThe band formed in 1990.\n\n### Early years\n\nGigs."


def test_old_style_headings_are_understood() -> None:
    html = f'<h2><span class="mw-headline" id="Life">Life</span>{EDIT_LINK}</h2><p>Text.</p>'

    assert convert(html) == "## Life\n\nText."


def test_links_and_emphasis_become_plain_text_and_references_disappear() -> None:
    html = (
        '<p>The <b>band</b> played <i>live</i> at <a href="/wiki/Hall" title="Hall">the hall</a>.'
        '<sup id="cite_ref-1" class="reference"><a href="#cite_note-1"><span class="cite-bracket">[</span>1'
        '<span class="cite-bracket">]</span></a></sup>'
        '<sup class="noprint Inline-Template"><a href="/wiki/Help">[<i>citation needed</i>]</a></sup></p>'
    )

    assert convert(html) == "The band played live at the hall."


def test_superscripts_and_subscripts() -> None:
    html = "<p>It cost 10<sup>3</sup> euros on the 5<sup>th</sup> try; water is H<sub>2</sub>O.</p>"

    assert convert(html) == "It cost 10^3 euros on the 5th try; water is H2O."


def test_whitespace_entities_and_invisible_characters_are_normalised() -> None:
    word_joiner = chr(0x2060)
    html = f"<p>  Tom &amp;   Jerry&nbsp;rock&#8203;s \n here{word_joiner} </p>"

    assert convert(html) == "Tom & Jerry rocks here"


def test_math_images_keep_their_tex_alt_text() -> None:
    html = '<p>Then <span class="mwe-math-element"><img alt="{\\displaystyle x^{2}}" src="m.svg"/></span> holds.</p>'

    assert convert(html) == "Then {\\displaystyle x^{2}} holds."


def test_loose_text_and_blocks_keep_their_order() -> None:
    assert convert("loose text<p>para</p>more loose<div>boxed</div>") == "loose text\n\npara\n\nmore loose\n\nboxed"


def test_inline_elements_directly_inside_a_container_join_the_surrounding_text() -> None:
    assert convert("<b>bold</b> text <a href='#x'>link</a><p>para</p><i>tail</i>") == "bold text link\n\npara\n\ntail"


def test_unknown_inline_tags_are_flattened() -> None:
    assert convert("<p>an <custom-tag>odd <b>tag</b></custom-tag> here</p>") == "an odd tag here"


def test_comments_inside_paragraphs_are_ignored() -> None:
    assert convert("<p>Keep<!-- hidden note --> this</p>") == "Keep this"


def test_horizontal_rules_separate_paragraphs_and_vanish() -> None:
    assert convert("<p>One</p><hr/><p>Two</p>") == "One\n\nTwo"


def test_redirect_pages_stay_readable() -> None:
    html = (
        '<div class="redirectMsg"><p>Redirect to:</p><ul class="redirectText"><li>'
        '<a href="/wiki/Example_Band" title="Example Band">Example Band</a></li></ul></div>'
    )

    assert convert(html) == "Redirect to:\n\n- Example Band"


# --- noise ---------------------------------------------------------------------------------------

NOISE = [
    '<div class="hatnote">Hat</div>',
    '<div role="note" class="hatnote navigation-not-searchable">Hat</div>',
    '<table class="box-X plainlinks metadata ambox ambox-content"><tr><td>Needs sources</td></tr></table>',
    '<div class="navbox"><table><tr><td>Nav</td></tr></table></div>',
    '<table class="navbox"><tr><td>Nav</td></tr></table>',
    '<div id="toc" class="toc"><ul><li>Contents</li></ul></div>',
    '<div id="toc"><ul><li>Contents</li></ul></div>',
    '<div class="reflist"><ol class="references"><li>Source</li></ol></div>',
    '<ol class="references"><li>Source</li></ol>',
    "<style>.x{color:red}</style>",
    '<link rel="mw-deduplicated-inline-style" href="mw-data:x"/>',
    "<script>alert(1)</script>",
    '<figure class="mw-default-size"><a><img alt="A" src="a.jpg"/></a><figcaption>Caption text</figcaption></figure>',
    '<div class="thumb tright"><div class="thumbinner"><div class="thumbcaption">Caption</div></div></div>',
    '<span style="display:none">hidden</span>',
    '<div style="display: none;">hidden</div>',
    '<div class="side-box"><p>Sister project</p></div>',
    '<div class="sidebar"><p>Series</p></div>',
    '<div class="noprint">Only on screen</div>',
    '<p class="mw-empty-elt"></p>',
    '<div class="authority-control">Authority</div>',
]


@pytest.mark.parametrize("noise", NOISE)
def test_noise_blocks_are_removed(noise: str) -> None:
    assert convert(f"<p>Keep</p>{noise}<p>Also</p>") == "Keep\n\nAlso"


def test_noise_inside_a_paragraph_is_removed_too() -> None:
    assert convert('<p>Keep<span style="display: none">x</span><span class="noprint">y</span> this</p>') == "Keep this"


def test_cite_errors_are_dropped() -> None:
    html = '<p>Fact.<span class="error mw-ext-cite-error" lang="en" dir="ltr">Cite error: no reflist.</span></p>'

    assert convert(html) == "Fact."


# --- lists, definition lists, preformatted text --------------------------------------------------


def test_nested_lists_are_indented_and_ordered_lists_numbered() -> None:
    html = (
        "<ul><li>One<ul><li>One A</li><li>One B</li></ul></li><li>Two</li></ul>"
        '<ol start="3"><li>Third</li><li>Fourth<ol><li>Inner</li></ol></li></ol>'
    )

    assert convert(html) == "- One\n  - One A\n  - One B\n- Two\n\n3. Third\n4. Fourth\n  1. Inner"


def test_list_items_without_text_are_skipped() -> None:
    assert convert('<ul><li></li><li><span class="reference">[1]</span></li><li>Real</li></ul>') == "- Real"


def test_a_list_item_that_only_holds_a_list_does_not_indent_it() -> None:
    assert convert("<ul><li><ul><li>Nested only</li></ul></li><li>Plain</li></ul>") == "- Nested only\n- Plain"


def test_hidden_list_items_are_skipped_with_their_children() -> None:
    html = (
        '<ul><li>Shown</li><li style="display:none">Hidden<ul><li>Child</li></ul></li>'
        '<li class="noprint">Print</li></ul>'
    )

    assert convert(html) == "- Shown"


def test_hidden_nested_lists_are_dropped_from_a_list_item() -> None:
    assert convert('<ul><li>Item<ul class="noprint"><li>Hidden child</li></ul></li></ul>') == "- Item"


def test_definition_lists() -> None:
    html = "<dl><dt>Term</dt><dd>First meaning</dd><dd>Second meaning</dd></dl>"

    assert convert(html) == "Term\n  First meaning\n  Second meaning"


def test_empty_and_hidden_definition_entries_are_skipped() -> None:
    assert convert('<dl><dt></dt><dd class="noprint">Hidden</dd><dd>Real</dd></dl>') == "  Real"


def test_preformatted_text_keeps_its_lines() -> None:
    assert convert("<pre>line 1\n  line 2</pre>") == "```\nline 1\n  line 2\n```"


def test_blockquotes_are_read_through() -> None:
    assert convert("<blockquote><p>A quote.</p><p>More.</p></blockquote>") == "A quote.\n\nMore."


# --- empty headings ------------------------------------------------------------------------------


def test_headings_without_content_are_dropped() -> None:
    html = (
        "<h2>Kept</h2><p>Body.</p>"
        "<h2>Empty</h2>"
        "<h2>Has child</h2><h3>Child</h3><p>Child body.</p>"
        '<h2>References</h2><div class="reflist"><ol class="references"><li>S</li></ol></div>'
    )

    assert convert(html) == "## Kept\n\nBody.\n\n## Has child\n\n### Child\n\nChild body."


def test_a_chain_of_empty_headings_collapses() -> None:
    assert convert("<p>Intro</p><h2>A</h2><h3>B</h3>") == "Intro"


# --- plain_text ----------------------------------------------------------------------------------


def test_plain_text_strips_tags_and_decodes_entities() -> None:
    fragment = '<span class="searchmatch">Blue</span> &quot;Morning&quot; &amp; <i>more</i>\n  words'

    assert plain_text(fragment) == 'Blue "Morning" & more words'


def test_plain_text_keeps_escaped_angle_brackets() -> None:
    assert plain_text("a &lt;b&gt; c") == "a <b> c"


def test_plain_text_of_nothing_is_empty() -> None:
    assert plain_text(None) == ""
    assert plain_text("") == ""
