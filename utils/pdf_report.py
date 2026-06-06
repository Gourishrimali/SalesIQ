from io import BytesIO
from textwrap import wrap
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def create_pdf_report(report_text):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)

    width, height = letter
    x = 50
    y = height - 50

    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(x, y, "SalesIQ Gemini AI Business Analyst Report")
    y -= 35

    pdf.setFont("Helvetica", 10)

    for paragraph in report_text.split("\n"):
        lines = wrap(paragraph, 95)

        if not lines:
            y -= 12

        for line in lines:
            if y < 50:
                pdf.showPage()
                pdf.setFont("Helvetica", 10)
                y = height - 50

            pdf.drawString(x, y, line)
            y -= 14

    pdf.save()
    buffer.seek(0)
    return buffer