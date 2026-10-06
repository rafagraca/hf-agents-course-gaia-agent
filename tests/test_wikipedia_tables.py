"""Tests for the conversion of MediaWiki HTML tables to ``| a | b |`` rows (spans, headers, noise in cells).

The HTML mimics the structure of MediaWiki parser output but every text is invented.
"""

from __future__ import annotations

import pytest

from gaia_agent.tools import wikipedia_tables
from gaia_agent.tools.wikipedia_html import html_to_markdown


def convert(fragment: str) -> str:
    return html_to_markdown(f'<div class="mw-parser-output">{fragment}</div>')


def table(rows: str, caption: str = "") -> str:
    return f"<table>{caption}<tbody>{rows}</tbody></table>"


# --- headers and captions ------------------------------------------------------------------------


def test_a_header_row_gets_a_separator() -> None:
    html = table("<tr><th>Title</th><th>Year</th></tr><tr><td>Blue Morning</td><td>1992</td></tr>")

    assert convert(html) == "| Title | Year |\n| --- | --- |\n| Blue Morning | 1992 |"


def test_a_table_with_only_header_cells_has_no_separator() -> None:
    assert convert(table("<tr><th>A</th><th>B</th></tr>")) == "| A | B |"


def test_a_title_row_is_not_a_column_header() -> None:
    html = table("<tr><th colspan='2'>Career</th></tr><tr><th>Born</th><td>1970</td></tr>")

    assert convert(html) == "| Career |\n| Born | 1970 |"


def test_the_caption_becomes_a_line_above_the_table() -> None:
    html = table("<tr><td>x</td></tr>", caption="<caption>Medal table</caption>")

    assert convert(html) == "Table: Medal table\n| x |"


def test_a_caption_alone_does_not_make_a_table() -> None:
    assert convert("<table><caption>Lonely</caption></table><p>Text</p>") == "Text"


# --- spans ---------------------------------------------------------------------------------------


def test_rowspan_repeats_the_value_in_data_rows() -> None:
    html = table(
        "<tr><th>Year</th><th>Title</th></tr>"
        '<tr><td rowspan="2">1992</td><td>Blue Morning</td></tr><tr><td>Red Evening</td></tr>'
    )

    assert convert(html) == "| Year | Title |\n| --- | --- |\n| 1992 | Blue Morning |\n| 1992 | Red Evening |"


def test_a_rowspan_of_three_repeats_through_all_the_rows() -> None:
    html = table(
        "<tr><th>Year</th><th>Title</th></tr>"
        '<tr><td rowspan="3">1992</td><td>One</td></tr><tr><td>Two</td></tr><tr><td>Three</td></tr>'
        "<tr><td>2000</td><td>Four</td></tr>"
    )

    assert convert(html) == (
        "| Year | Title |\n| --- | --- |\n| 1992 | One |\n| 1992 | Two |\n| 1992 | Three |\n| 2000 | Four |"
    )


def test_header_spans_are_not_repeated() -> None:
    html = table(
        '<tr><th rowspan="2">Title</th><th colspan="2">Chart</th></tr><tr><th>US</th><th>UK</th></tr>'
        "<tr><td>Blue Morning</td><td>12</td><td>8</td></tr>"
    )

    assert convert(html) == "| Title | Chart |\n|  | US | UK |\n| --- | --- | --- |\n| Blue Morning | 12 | 8 |"


def test_colspan_in_data_rows_keeps_the_columns_aligned() -> None:
    html = table("<tr><td>Foo</td><td colspan='2'>Did not chart</td><td>x</td></tr>")

    assert convert(html) == "| Foo | Did not chart |  | x |"


def test_rowspan_and_colspan_together_keep_the_columns_aligned() -> None:
    html = table(
        "<tr><th>A</th><th>B</th><th>C</th></tr>"
        '<tr><td rowspan="2" colspan="2">wide</td><td>c1</td></tr><tr><td>c2</td></tr>'
    )

    assert convert(html) == "| A | B | C |\n| --- | --- | --- |\n| wide |  | c1 |\n| wide |  | c2 |"


def test_broken_or_huge_span_values_are_tolerated() -> None:
    html = table('<tr><td colspan="abc">a</td><td rowspan="0">b</td><td colspan="100000">c</td></tr>')

    assert convert(html) == "| a | b | c |"


def test_a_huge_colspan_makes_at_most_the_default_cap_of_columns() -> None:
    row = convert(table('<tr><td colspan="99999">wide</td><td>x</td></tr>'))

    assert len(row.strip("|").split("|")) == 101  # the 100 columns of the cap, then the cell after the span


def test_spans_are_capped_so_a_hostile_table_cannot_explode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wikipedia_tables, "MAX_SPAN", 3)

    assert convert(table('<tr><td colspan="10">a</td><td>b</td></tr>')) == "| a |  |  | b |"


# --- cell contents -------------------------------------------------------------------------------


def test_line_breaks_and_list_items_in_cells_become_semicolons() -> None:
    html = table(
        "<tr><td>Rock<br/>Pop<br/></td><td><ul><li>A</li><li>B</li></ul></td><td>US<br/>"
        '<sup class="reference">[1]</sup></td></tr>'
    )

    assert convert(html) == "| Rock; Pop | A; B | US |"


def test_pipes_in_cells_are_escaped() -> None:
    assert convert(table("<tr><td>a|b</td></tr>")) == "| a\\|b |"


def test_hidden_sort_keys_and_references_in_cells_are_dropped() -> None:
    html = table(
        '<tr><td><span style="display:none">000000001992</span>1 Jan 1992'
        '<sup class="reference"><a href="#n">[3]</a></sup></td></tr>'
    )

    assert convert(html) == "| 1 Jan 1992 |"


def test_comments_inside_cells_are_ignored() -> None:
    assert convert(table("<tr><td>a<!-- note -->b</td></tr>")) == "| ab |"


def test_an_icon_only_cell_keeps_the_alt_text_of_its_icon() -> None:
    html = table(
        '<tr><td><img alt="Yes" src="y.svg" width="20" height="20"/></td>'
        '<td><img alt="Gold medal" src="g.svg"/><img alt="Silver medal" src="s.svg" width="16"/></td></tr>'
    )

    assert convert(html) == "| Yes | Gold medal Silver medal |"


def test_icons_next_to_text_flags_and_photos_add_nothing() -> None:
    html = table(
        '<tr><td><img alt="Wikibooks logo" src="w.svg" width="20" height="20"/> Python at Wikibooks</td>'
        '<td><span class="flagicon"><img alt="Norway" src="n.svg" width="23" height="15"/></span> Norway</td>'
        '<td><img alt="Portrait" src="p.jpg" width="220" height="300"/>Name</td>'
        '<td><img alt="Portrait" src="p.jpg" width="220" height="300"/></td></tr>'
    )

    assert convert(html) == "| Python at Wikibooks | Norway | Name |"


def test_infobox_captions_and_wikidata_pens_are_dropped() -> None:
    html = table(
        '<tr><td colspan="2" class="infobox-image"><div class="infobox-caption">A caption</div></td></tr>'
        '<tr><th>First appeared</th><td>1991 <span class="penicon autoconfirmed-show"><a title="Edit this on '
        'Wikidata"><img alt="Edit this on Wikidata" src="p.svg" width="10" height="10"/></a></span></td></tr>'
    )

    assert convert(html) == "| First appeared | 1991 |"


# --- nested tables, empty rows -------------------------------------------------------------------


def test_nested_table_rows_are_not_listed_twice() -> None:
    inner = "<table><tr><td>in1</td><td>in2</td></tr></table>"
    html = table(f"<tr><td>outer</td><td>{inner}</td></tr>")

    assert convert(html) == "| outer | in1 in2 |"


def test_the_rows_of_a_table_nested_in_a_cell_are_separated() -> None:
    inner = "<table><tr><th>A</th><td>Qatar</td></tr><tr><th>B</th><td>Ecuador</td></tr></table>"

    assert convert(table(f"<tr><td>Group</td><td>{inner}</td></tr>")) == "| Group | A Qatar; B Ecuador |"


def test_empty_rows_hidden_rows_and_empty_tables_vanish() -> None:
    html = (
        table("<tr><td></td></tr>")
        + table('<tr style="display:none"><td>hidden</td></tr><tr><td>shown</td></tr><tr><td> </td></tr>')
        + "<p>After</p>"
    )

    assert convert(html) == "| shown |\n\nAfter"


def test_trailing_empty_cells_are_trimmed_but_inner_ones_are_kept() -> None:
    assert convert(table("<tr><td>a</td><td></td><td>c</td><td></td><td></td></tr>")) == "| a |  | c |"
