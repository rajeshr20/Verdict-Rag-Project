"""Script to generate a small sample PDF for Meridian Labs corpus."""
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def generate_meridian_overview_pdf(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(output_path), pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 18)
    c.drawString(72, height - 72, "Meridian Labs: Company Overview & Governance")

    c.setFont("Helvetica", 10)
    c.drawString(72, height - 95, "Document Version: 1.0 | Effective Date: 2024-05-01 | Author: Executive Office")
    c.line(72, height - 105, width - 72, height - 105)

    y = height - 135
    paragraphs = [
        ("Mission and Founding:", True),
        ("Meridian Labs was founded in 2021 by Dr. Aris Sterling with the mission of advancing deterministic, verifiable AI systems for high-stakes enterprise workflows.", False),
        ("", False),
        ("Headquarters and Facilities:", True),
        ("The primary corporate research campus is located in New Eden Science Park, Building 4. The campus houses our primary quantum and neuromorphic computing research clusters.", False),
        ("", False),
        ("Corporate Governance:", True),
        ("All corporate research and development initiatives are subject to internal ethics oversight by the Autonomous Systems Safety Council (ASSC), which convenes bi-monthly.", False),
        ("", False),
        ("Core Intellectual Property:", True),
        ("Meridian Labs holds 14 international patents in verifiable distributed inference, zero-trust enclave routing, and low-latency API proxy gateways.", False),
    ]

    for line, is_heading in paragraphs:
        if is_heading:
            c.setFont("Helvetica-Bold", 12)
            c.drawString(72, y, line)
            y -= 18
        elif line:
            c.setFont("Helvetica", 10)
            c.drawString(72, y, line)
            y -= 16
        else:
            y -= 8

    c.showPage()
    c.save()
    print(f"Generated sample PDF at: {output_path}")


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "data" / "sample_docs" / "meridian_overview.pdf"
    generate_meridian_overview_pdf(out)
