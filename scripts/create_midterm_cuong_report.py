from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import html
import struct


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
DOCX_PATH = REPORTS_DIR / "midterm_cuong_report.docx"
DIAGRAM_PATH = FIGURES_DIR / "system_workflow_diagram.png"

FONT_NAME = "Times New Roman"
FONT_SIZE_HALF_POINTS = 26
BLACK = "000000"


def xml_escape(text: str) -> str:
    return html.escape(text, quote=False)


def run_properties(bold: bool = False) -> str:
    bold_xml = "<w:b/>" if bold else ""
    return (
        "<w:rPr>"
        f"<w:rFonts w:ascii=\"{FONT_NAME}\" w:hAnsi=\"{FONT_NAME}\" "
        f"w:eastAsia=\"{FONT_NAME}\" w:cs=\"{FONT_NAME}\"/>"
        f"<w:color w:val=\"{BLACK}\"/>"
        f"<w:sz w:val=\"{FONT_SIZE_HALF_POINTS}\"/>"
        f"<w:szCs w:val=\"{FONT_SIZE_HALF_POINTS}\"/>"
        f"{bold_xml}"
        "</w:rPr>"
    )


def paragraph(text: str = "", *, bold: bool = False, align: str | None = None, spacing_after: int = 120) -> str:
    p_pr_parts = [f"<w:spacing w:after=\"{spacing_after}\"/>"]
    if align:
        p_pr_parts.append(f"<w:jc w:val=\"{align}\"/>")
    p_pr = f"<w:pPr>{''.join(p_pr_parts)}</w:pPr>"
    if not text:
        return f"<w:p>{p_pr}</w:p>"
    return (
        "<w:p>"
        f"{p_pr}"
        "<w:r>"
        f"{run_properties(bold=bold)}"
        f"<w:t xml:space=\"preserve\">{xml_escape(text)}</w:t>"
        "</w:r>"
        "</w:p>"
    )


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as file:
        header = file.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Expected PNG file: {path}")
    return struct.unpack(">II", header[16:24])


def image_paragraph(path: Path, width_inches: float = 6.7) -> str:
    width_px, height_px = png_size(path)
    cx = int(width_inches * 914400)
    cy = int(cx * height_px / width_px)
    return f"""
<w:p>
  <w:pPr><w:jc w:val="center"/><w:spacing w:after="80"/></w:pPr>
  <w:r>
    <w:drawing>
      <wp:inline distT="0" distB="0" distL="0" distR="0">
        <wp:extent cx="{cx}" cy="{cy}"/>
        <wp:effectExtent l="0" t="0" r="0" b="0"/>
        <wp:docPr id="1" name="System Workflow Diagram"/>
        <wp:cNvGraphicFramePr>
          <a:graphicFrameLocks noChangeAspect="1"/>
        </wp:cNvGraphicFramePr>
        <a:graphic>
          <a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
            <pic:pic>
              <pic:nvPicPr>
                <pic:cNvPr id="0" name="{xml_escape(path.name)}"/>
                <pic:cNvPicPr/>
              </pic:nvPicPr>
              <pic:blipFill>
                <a:blip r:embed="rId2"/>
                <a:stretch><a:fillRect/></a:stretch>
              </pic:blipFill>
              <pic:spPr>
                <a:xfrm>
                  <a:off x="0" y="0"/>
                  <a:ext cx="{cx}" cy="{cy}"/>
                </a:xfrm>
                <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
              </pic:spPr>
            </pic:pic>
          </a:graphicData>
        </a:graphic>
      </wp:inline>
    </w:drawing>
  </w:r>
</w:p>
"""


def document_xml() -> str:
    body: list[str] = []
    body.append(paragraph("Speech Command Recognition for Robot Navigation", bold=True, align="center", spacing_after=80))
    body.append(paragraph("Midterm Report - Cuong's Assigned Sections", bold=True, align="center", spacing_after=220))

    body.append(paragraph("1. Problem, Objectives, Input Data, and Expected Outputs", bold=True))
    body.append(paragraph(
        "Problem. The project builds a small speech command recognition system for robot navigation. "
        "The system receives a short voice command and predicts a navigation intent, then maps it to a robot action."
    ))
    body.append(paragraph(
        "Objectives. The main objectives are to prepare command audio data, extract speech features, train a baseline "
        "classifier, and return a predicted command with a confidence score."
    ))
    body.append(paragraph(
        "Input data. The input is a one-second speech clip from Google Speech Commands or a user WAV file. "
        "The implementation uses the labels forward, backward, left, right, stop, and unknown."
    ))
    body.append(paragraph(
        "Expected outputs. The expected outputs are a trained model, predicted command label, confidence score, "
        "and mapped robot action such as MOVE_FORWARD, MOVE_BACKWARD, TURN_LEFT, TURN_RIGHT, STOP, or IGNORE."
    ))

    body.append(paragraph("2. Google Speech Commands Dataset", bold=True))
    body.append(paragraph(
        "Google Speech Commands is a public dataset of short spoken words recorded by many speakers. "
        "It is suitable for this project because the clips are short, already organized by word label, and close to the "
        "type of simple command that a robot navigation system needs."
    ))
    body.append(paragraph(
        "The selected target words are forward, backward, left, right, and stop. Other words are grouped into the "
        "unknown class so the system can ignore commands outside the navigation set."
    ))
    body.append(paragraph(
        "The dataset also provides official training, validation, and testing splits. This makes the evaluation more "
        "consistent and reduces the risk of testing on samples that were used during training."
    ))

    body.append(paragraph("3. Data Preparation Pipeline", bold=True))
    body.append(paragraph(
        "Each audio clip is converted to mono, resampled to 16 kHz, and adjusted to one second. Longer clips are trimmed, "
        "and shorter clips are padded with silence."
    ))
    body.append(paragraph(
        "After length adjustment, the waveform amplitude is normalized. This keeps volume differences from dominating "
        "the model input."
    ))
    body.append(paragraph(
        "Labels are assigned from the folder name in the dataset. The five navigation words keep their original labels, "
        "while all remaining words are mapped to unknown. The unknown class is sampled to keep the dataset more balanced."
    ))

    body.append(paragraph("4. System Workflow", bold=True))
    body.append(paragraph(
        "The system workflow starts with an audio input. The waveform is preprocessed, converted into a Log-Mel "
        "spectrogram, and passed to a CNN classifier. The model returns class probabilities, and the top class is accepted "
        "only if its confidence is high enough."
    ))
    body.append(paragraph(
        "After prediction, the command is mapped to a robot action. If the model confidence is low, the command becomes "
        "unknown and the robot action is IGNORE. This reduces the chance of sending an unsafe or incorrect movement."
    ))
    if DIAGRAM_PATH.exists():
        body.append(image_paragraph(DIAGRAM_PATH))
        body.append(paragraph("Figure 1. End-to-end workflow from audio input to robot action.", align="center", spacing_after=160))

    body.append(
        """
<w:sectPr>
  <w:pgSz w:w="11906" w:h="16838"/>
  <w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134" w:header="720" w:footer="720" w:gutter="0"/>
</w:sectPr>
"""
    )

    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document
  xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
  xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
  xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
  xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
  xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">
  <w:body>
    {''.join(body)}
  </w:body>
</w:document>
"""


def styles_xml() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:docDefaults>
    <w:rPrDefault>
      <w:rPr>
        <w:rFonts w:ascii="{FONT_NAME}" w:hAnsi="{FONT_NAME}" w:eastAsia="{FONT_NAME}" w:cs="{FONT_NAME}"/>
        <w:color w:val="{BLACK}"/>
        <w:sz w:val="{FONT_SIZE_HALF_POINTS}"/>
        <w:szCs w:val="{FONT_SIZE_HALF_POINTS}"/>
      </w:rPr>
    </w:rPrDefault>
  </w:docDefaults>
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:qFormat/>
    <w:rPr>
      <w:rFonts w:ascii="{FONT_NAME}" w:hAnsi="{FONT_NAME}" w:eastAsia="{FONT_NAME}" w:cs="{FONT_NAME}"/>
      <w:color w:val="{BLACK}"/>
      <w:sz w:val="{FONT_SIZE_HALF_POINTS}"/>
      <w:szCs w:val="{FONT_SIZE_HALF_POINTS}"/>
    </w:rPr>
  </w:style>
</w:styles>
"""


def content_types_xml(include_diagram: bool) -> str:
    image_override = '<Override PartName="/word/media/system_workflow_diagram.png" ContentType="image/png"/>' if include_diagram else ""
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="png" ContentType="image/png"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  {image_override}
</Types>
"""


def root_rels_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>
"""


def document_rels_xml(include_diagram: bool) -> str:
    diagram_rel = (
        '<Relationship Id="rId2" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
        'Target="media/system_workflow_diagram.png"/>'
        if include_diagram
        else ""
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
  {diagram_rel}
</Relationships>
"""


def build_docx() -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    include_diagram = DIAGRAM_PATH.exists()

    with ZipFile(DOCX_PATH, "w", ZIP_DEFLATED) as docx:
        docx.writestr("[Content_Types].xml", content_types_xml(include_diagram))
        docx.writestr("_rels/.rels", root_rels_xml())
        docx.writestr("word/document.xml", document_xml())
        docx.writestr("word/styles.xml", styles_xml())
        docx.writestr("word/_rels/document.xml.rels", document_rels_xml(include_diagram))
        if include_diagram:
            docx.write(DIAGRAM_PATH, "word/media/system_workflow_diagram.png")

    return DOCX_PATH


def main() -> None:
    print(build_docx())


if __name__ == "__main__":
    main()
