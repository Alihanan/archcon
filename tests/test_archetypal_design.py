from archcon.data.archetypal_design import (
    AA_DESIGN_OPTIONS,
    AA_SHARED,
    aa_design_markdown,
    aa_design_svg,
)


def test_aa_design_exposes_multiview_shared_w_option() -> None:
    assert len(AA_DESIGN_OPTIONS) >= 5
    markdown = aa_design_markdown(AA_SHARED)
    assert "shared" in markdown.lower()
    assert "W" in markdown
    svg = aa_design_svg(AA_SHARED)
    assert "svg" in svg.lower()
    assert "clinical" in svg.lower()
