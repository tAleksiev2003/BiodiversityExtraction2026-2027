"""Generate the two invented test PDFs used by tests/test_pdf_to_text.py.

All text is invented. The PDFs are committed, so this script only has to be run again
when the texts change:

    python3 tests/fixtures/make_fixture_pdfs.py

- with_headings.pdf: a short "paper" with numbered section headings (normal chunking);
- without_headings.pdf: plain paragraphs only (fallback chunking).
"""
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

FIXTURE_DIR = Path(__file__).resolve().parent

TITLE = "Invented beetle communities in an imaginary floodplain meadow"

SECTIONS = {
    "1 Introduction": [
        "Floodplain meadows in the invented Kleinau valley are mown twice a year. "
        "We asked whether the mowing date changes the ground beetle community.",
        "Earlier invented surveys reported that late mowing favours large carabid "
        "species, but no study compared both mowing dates on the same site.",
    ],
    "2 Methods": [
        "We placed forty pitfall traps on eight plots of the fictional meadow and "
        "emptied them every two weeks from April to September.",
        "All beetles were identified to species level. Species richness and activity "
        "density were compared between early and late mown plots.",
    ],
    "3 Results": [
        "In total we caught 1234 invented beetles of 56 species. Species richness was "
        "higher on late mown plots than on early mown plots.",
        "Activity density of the imaginary species Carabus fictus decreased after each "
        "mowing event and recovered within three weeks.",
    ],
    "4 Discussion": [
        "Late mowing increased the species richness of ground beetles in this invented "
        "floodplain meadow. Managers could delay the first cut to support beetles.",
    ],
}

PLAIN_PARAGRAPHS = [
    "This invented report describes a walk along an imaginary hedgerow near the "
    "fictional village of Musterdorf on a sunny morning in May.",
    "The hedgerow consisted mainly of hawthorn and blackthorn. Several invented bird "
    "territories were noted along its full length of about two kilometres.",
    "Butterflies were counted on a fixed transect. The most common species was the "
    "imaginary Pieris inventa, followed by a few individuals of other whites.",
    "Along the field margin the vegetation was dense and rich in flowering herbs. "
    "Bumblebees visited mostly clover and knapweed during the whole walk.",
    "The farmer reported that the hedgerow had been cut three years ago. Since then "
    "the shrubs have grown back and now provide shelter for many small animals.",
    "No conclusions are drawn from this single walk. The notes only serve as an "
    "example of a document that contains no headings at all.",
]


def build_with_headings(path: Path) -> None:
    styles = getSampleStyleSheet()
    story = [Paragraph(TITLE, styles["Title"]), Spacer(1, 12)]
    for section_index, (heading, paragraphs) in enumerate(SECTIONS.items()):
        if section_index == 2:
            story.append(PageBreak())  # a second page, so page numbers can be tested
        story.append(Paragraph(heading, styles["Heading1"]))
        story.extend(Paragraph(text, styles["BodyText"]) for text in paragraphs)
    SimpleDocTemplate(str(path), pagesize=A4).build(story)


def build_without_headings(path: Path) -> None:
    styles = getSampleStyleSheet()
    story = []
    for paragraph_index, text in enumerate(PLAIN_PARAGRAPHS):
        if paragraph_index == 3:
            story.append(PageBreak())
        story.append(Paragraph(text, styles["BodyText"]))
        story.append(Spacer(1, 8))
    SimpleDocTemplate(str(path), pagesize=A4).build(story)


if __name__ == "__main__":
    build_with_headings(FIXTURE_DIR / "with_headings.pdf")
    build_without_headings(FIXTURE_DIR / "without_headings.pdf")
    print(f"Written: {FIXTURE_DIR / 'with_headings.pdf'}, {FIXTURE_DIR / 'without_headings.pdf'}")
