"""Tests for turning an HTML page into compact markdown."""

from __future__ import annotations

import time

import pytest
from web_samples import html_page

from gaia_agent.tools import web_docs
from gaia_agent.tools.web_docs import html_to_markdown

PAGE_URL = "https://example.org/dir/page.html"


def test_scripts_styles_and_page_chrome_are_dropped() -> None:
    html = html_page(
        "<header>Site header menu</header><nav><a href='/a'>Home</a></nav>"
        "<style>p {color: red}</style><script>var tracking = 1;</script>"
        "<main><h1>Report</h1><p>The actual content.</p></main>"
        "<footer>Copyright footer</footer>"
    )

    markdown = html_to_markdown(html)

    assert "The actual content." in markdown
    for noise in ("Site header menu", "Home", "color: red", "tracking", "Copyright footer"):
        assert noise not in markdown


def test_hidden_and_embedded_elements_are_dropped() -> None:
    html = html_page(
        "<noscript>Enable scripts</noscript><template><p>Template text</p></template>"
        "<svg><title>Icon title</title></svg><iframe>Frame fallback</iframe><p>Kept.</p>"
    )

    assert html_to_markdown(html) == "# Sample page\n\nKept."


def test_header_and_footer_inside_an_article_are_kept() -> None:
    html = html_page(
        "<article><header><h1>Headline</h1><p>By Jane Doe, 3 May 2021</p></header>"
        "<p>Body text.</p><footer>Last updated 4 May</footer></article>"
    )

    markdown = html_to_markdown(html)

    assert "By Jane Doe, 3 May 2021" in markdown
    assert "Last updated 4 May" in markdown


def test_tables_become_markdown_tables() -> None:
    html = html_page(
        "<table><tr><th>Year</th><th>Album</th></tr>"
        "<tr><td>2001</td><td>Alpha</td></tr><tr><td>2005</td><td>Beta</td></tr></table>"
    )

    markdown = html_to_markdown(html)

    assert "| Year | Album |\n| --- | --- |\n| 2001 | Alpha |\n| 2005 | Beta |" in markdown


def test_the_first_row_of_a_table_without_headers_becomes_its_header() -> None:
    html = html_page("<table><tr><td>Year</td><td>Album</td></tr><tr><td>2001</td><td>Alpha</td></tr></table>")

    markdown = html_to_markdown(html)

    assert "| Year | Album |\n| --- | --- |\n| 2001 | Alpha |" in markdown
    assert "|  |" not in markdown


def test_links_are_kept() -> None:
    html = html_page('<p>See <a href="https://example.org/a">the Alpha page</a>.</p>')

    assert "[the Alpha page](https://example.org/a)" in html_to_markdown(html)


def test_relative_links_are_resolved_against_the_page_url() -> None:
    html = html_page('<a href="/wiki/Beta">Beta</a> <a href="gamma.html">Gamma</a>')

    markdown = html_to_markdown(html, base_url=PAGE_URL)

    assert "[Beta](https://example.org/wiki/Beta)" in markdown
    assert "[Gamma](https://example.org/dir/gamma.html)" in markdown


def test_relative_links_stay_as_they_are_without_a_page_url() -> None:
    assert "[Beta](/wiki/Beta)" in html_to_markdown(html_page('<a href="/wiki/Beta">Beta</a>'))


def test_link_tooltips_that_repeat_the_text_or_the_address_are_dropped() -> None:
    html = html_page(
        '<a href="/wiki/Edmund_Hillary" title="Edmund Hillary">Sir Edmund</a> '
        '<a href="/other" title="Same text">Same text</a>'
    )

    markdown = html_to_markdown(html, base_url=PAGE_URL)

    assert "[Sir Edmund](https://example.org/wiki/Edmund_Hillary)" in markdown
    assert "[Same text](https://example.org/other)" in markdown
    assert '"' not in markdown


def test_link_tooltips_that_add_information_are_kept() -> None:
    html = html_page('<a href="/members" title="Opens the member list">Members</a>')

    assert '[Members](https://example.org/members "Opens the member list")' in html_to_markdown(html, base_url=PAGE_URL)


def test_links_that_lead_nowhere_keep_only_their_text() -> None:
    html = html_page('<a href="#top">Back to top</a> <a href="javascript:void(0)">Open menu</a>')

    markdown = html_to_markdown(html, base_url=PAGE_URL)

    assert "Back to top" in markdown
    assert "Open menu" in markdown
    assert "#top" not in markdown
    assert "javascript" not in markdown


def test_a_link_that_cannot_be_resolved_is_kept_as_written() -> None:
    html = html_page('<a href="http://[::1">Broken</a>')

    assert "[Broken](http://[::1)" in html_to_markdown(html, base_url=PAGE_URL)


def test_images_keep_their_alt_text_but_not_their_url() -> None:
    html = html_page(
        '<p>Chart: <img src="data:image/png;base64,AAAA" alt="Sales by year"> and <img src="/x.png"> end</p>'
    )

    markdown = html_to_markdown(html)

    assert "[image: Sales by year]" in markdown
    assert "base64" not in markdown
    assert "x.png" not in markdown


def test_headings_use_hash_marks() -> None:
    markdown = html_to_markdown(html_page("<h1>Main</h1><h2>Part</h2><p>Text</p>"))

    assert "# Main" in markdown
    assert "## Part" in markdown


def test_underscores_and_asterisks_are_not_escaped() -> None:
    assert "snake_case and 2*3" in html_to_markdown(html_page("<p>snake_case and 2*3</p>"))


def test_lists_have_one_item_per_line() -> None:
    markdown = html_to_markdown(html_page("<ul><li>One</li><li>Two</li></ul>"))

    lines = [line.strip() for line in markdown.splitlines()]
    assert any(line.endswith("One") for line in lines)
    assert any(line.endswith("Two") for line in lines)


def test_the_page_title_is_added_when_the_body_does_not_show_it() -> None:
    markdown = html_to_markdown(html_page("<p>Some text.</p>", title="Annual Report"))

    assert markdown.startswith("# Annual Report\n\n")


def test_the_page_title_is_not_repeated_when_the_body_has_it() -> None:
    markdown = html_to_markdown(html_page("<h1>Annual Report</h1><p>Some text.</p>", title="Annual Report"))

    assert markdown.count("Annual Report") == 1


def test_xml_declarations_and_processing_instructions_are_not_text() -> None:
    html = '<?xml version="1.0" encoding="UTF-8"?><html><body><p>Hello xhtml</p><?php echo 1; ?></body></html>'

    assert html_to_markdown(html) == "Hello xhtml"


def test_non_breaking_and_zero_width_characters_are_normalised() -> None:
    markdown = html_to_markdown(html_page("<p>5&nbsp;km&#8203;/h</p>"))

    assert "5 km/h" in markdown


def test_runs_of_blank_lines_are_collapsed() -> None:
    markdown = html_to_markdown(html_page("<p>a</p><br><br><br><br><p>b</p>"))

    assert "\n\n\n" not in markdown


def test_a_page_without_body_text_gives_nothing() -> None:
    assert html_to_markdown("") == ""
    assert html_to_markdown(html_page("<script>1</script>")) == ""


def test_a_huge_run_of_spaces_does_not_make_the_conversion_crawl() -> None:
    html = html_page("<pre>top" + " " * 60_000 + "bottom</pre>")

    started = time.perf_counter()
    markdown = html_to_markdown(html)

    assert "top" in markdown
    assert "bottom" in markdown
    assert time.perf_counter() - started < 3


def test_unclosed_tags_are_closed_the_way_a_browser_does() -> None:
    items = "".join(f"<li>item {number}" for number in range(1000))
    rows = "<table><tr><td>a1<td>b1<tr><td>a2<td>b2</table>"

    markdown = html_to_markdown(html_page(f"<ul>{items}</ul>{rows}"))

    # One bullet per item: a parser that nests each unclosed <li> would blow the converter's recursion limit.
    assert "* item 0\n* item 1\n" in markdown
    assert "* item 999" in markdown
    assert "| a2 | b2 |" in markdown


def test_extremely_nested_pages_fall_back_to_plain_text() -> None:
    html = html_page("<div>" * 1500 + "buried text" + "</div>" * 1500)

    assert "buried text" in html_to_markdown(html)


def test_html_parser_is_used_when_lxml_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(web_docs, "PREFERRED_PARSER", "no-such-parser")

    markdown = html_to_markdown(html_page("<h2>Part</h2><p>Text</p>"))

    assert "## Part" in markdown
