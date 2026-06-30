from reportlab.lib.pagesizes import LETTER, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image,
    PageBreak, HRFlowable, KeepTogether,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import os

VISUALS_DIR = "final PDF report/visuals for PDF"
OUTPUT_PDF = "final PDF report/NO2_Network_Report.pdf"

SECTIONS = [
    {
        "title": "Section 1 Quantitative EDA",
        "items": [
            ("quant_summary_statistics.png", "Quantitative Summary Statistics"),
            ("quant_boxplot.png", "Quantitative Boxplot"),
        ],
    },
    {
        "title": "Section 2 Training Process",
        "items": [
            ("train_test_counts.png", "Train / Test Split Counts"),
            ("dpw_presplit.png", "DPW Pre-Split"),
            ("dpw_postsplit.png", "DPW Post-Split"),
            ("mjf_presplit.png", "MJF Pre-Split"),
            ("mjf_postsplit.png", "MJF Post-Split"),
            ("pema_presplit.png", "PEMA Pre-Split"),
            ("pema_postsplit.png", "PEMA Post-Split"),
            ("pha_presplit.png", "PHA Pre-Split"),
            ("pha_postsplit.png", "PHA Post-Split"),
            ("Sensor01_training_scatter.png", "Sensor 01 Training Scatter"),
            ("Sensor01_testing_scatter.png", "Sensor 01 Testing Scatter"),
            ("Sensor09_training_scatter.png", "Sensor 09 Training Scatter"),
            ("Sensor09_testing_scatter.png", "Sensor 09 Testing Scatter"),
            ("Sensor13_training_scatter.png", "Sensor 13 Training Scatter"),
            ("Sensor13_testing_scatter.png", "Sensor 13 Testing Scatter"),
            ("Sensor25_training_scatter.png", "Sensor 25 Training Scatter"),
            ("Sensor25_testing_scatter.png", "Sensor 25 Testing Scatter"),
        ],
    },
    {
        "title": "Section 3 — Post-Training Network Analysis",
        "items": [
            ("model_metrics_table.png", "Model Performance Metrics"),
            ("summary_stats_table.png", "Network-Wide Summary Statistics"),
            ("boxplots_all_sites.png", "NO2 Boxplots All Sites"),
            ("boxplots_subset.png", "NO2 Boxplots Excluding MJF & Blackstone"),
            ("median_no2_bar_all.png", "Median NO2 by Site All Sites"),
            ("median_no2_bar_subset.png", "Median NO2 by Site Excluding MJF & Blackstone"),
            ("pct95_exposure_bar.png", "95th Percentile NO2 Exposure by Site"),
            ("site_map.png", "Site Map Median NO2 (Z-Score)"),
            ("correlation_heatmap.png", "Site-to-Site Correlation Heatmap"),
            ("diurnal_heatmap.png", "Diurnal Cycle Heatmap"),
            ("monthly_trend_heatmap.png", "Monthly Trend Heatmap"),
            ("hierarchical_clustering.png", "Hierarchical Clustering Dendrogram"),
        ],
    },
]

#images that look better in landscape
LANDSCAPE_IMAGES = {
    "boxplots_all_sites.png",
    "boxplots_subset.png",
    "correlation_heatmap.png",
    "histograms_5x5_grid.png",
    "median_no2_bar_all.png",
    "median_no2_bar_subset.png",
    "monthly_trend_heatmap.png",
    "diurnal_heatmap.png",
    "hierarchical_clustering.png",
    "pct95_exposure_bar.png",
    "train_test_counts.png",
    "model_metrics_table.png",
    "summary_stats_table.png",
    "quant_summary_statistics.png",
}

def _styles():
    base = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=base["Title"],
        fontSize=22,
        textColor=colors.HexColor("#2c3e50"),
        spaceAfter=6,
        alignment=TA_CENTER,
    )
    section_style = ParagraphStyle(
        "SectionHeader",
        parent=base["Heading1"],
        fontSize=14,
        textColor=colors.white,
        backColor=colors.HexColor("#2c3e50"),
        spaceBefore=18,
        spaceAfter=8,
        leftIndent=-6,
        rightIndent=-6,
        leading=20,
        alignment=TA_LEFT,
    )
    caption_style = ParagraphStyle(
        "Caption",
        parent=base["Normal"],
        fontSize=8,
        textColor=colors.HexColor("#555555"),
        alignment=TA_CENTER,
        spaceAfter=12,
    )
    return title_style, section_style, caption_style


def _fit_image(path, page_w, page_h, margin=0.75, landscape_mode=False):
    from PIL import Image as PILImage
    with PILImage.open(path) as im:
        img_w, img_h = im.size

    if landscape_mode:
        #constrain to fit on portrait page
        max_w = 7.5 * inch
        max_h = 9.0 * inch
    else:
        max_w = page_w - 2 * margin * inch
        max_h = page_h - 2 * margin * inch - 1.2 * inch

    scale = min(max_w / img_w, max_h / img_h)
    return img_w * scale, img_h * scale

def generate_report():
    os.makedirs(os.path.dirname(OUTPUT_PDF), exist_ok=True)

    title_style, section_style, caption_style = _styles()

    story = []

    story.append(Spacer(1, 2 * inch))
    story.append(Paragraph("Breathe Providence NO2 Calibration Report", title_style))
    story.append(Spacer(1, 0.2 * inch))
    story.append(HRFlowable(width="80%", thickness=2, color=colors.HexColor("#2c3e50"), spaceAfter=12))
    story.append(Paragraph("DPP NO2 Calibration Pipeline", title_style))
    story.append(PageBreak())

    for section in SECTIONS:
        story.append(Paragraph(f"  {section['title']}", section_style))
        story.append(Spacer(1, 0.1 * inch))

        for filename, caption in section["items"]:
            lookup = filename
            if filename == "site_map.png":
                for candidate in ("site_map.png", "site_map-png"):
                    if os.path.exists(os.path.join(VISUALS_DIR, candidate)):
                        lookup = candidate
                        break

            filepath = os.path.join(VISUALS_DIR, lookup)
            if not os.path.exists(filepath):
                print(f"  [MISSING] {filename} — skipping")
                continue

            use_landscape = filename in LANDSCAPE_IMAGES
            page_w, page_h = LETTER
            w, h = _fit_image(filepath, page_w, page_h, landscape_mode=use_landscape)

            img = Image(filepath, width=w, height=h)
            img.hAlign = "CENTER"

            story.append(KeepTogether([
                img,
                Paragraph(caption, caption_style),
            ]))
            story.append(PageBreak())

    def on_page(canvas, doc):
        pass

    doc = SimpleDocTemplate(
        OUTPUT_PDF,
        pagesize=LETTER,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        title="Breathe Providence NO2 Calibration Report"
    )
    doc.build(story)
    print(f"PDF report saved to: {OUTPUT_PDF}")


if __name__ == "__main__":
    generate_report()