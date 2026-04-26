"""
Creates a realistic company quarterly report PDF with:
- Text paragraphs (company overview, analysis)
- Bar chart (revenue by region)
- Line chart (monthly growth trend)
- Table (department performance)

Run this ONCE to generate the PDF we'll use for Project 3.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import os


def create_chart_revenue():
    """Create a revenue by region bar chart."""
    regions = ['North America', 'Europe', 'Asia Pacific', 'Latin America']
    q2_revenue = [12.4, 8.1, 9.7, 3.2]
    q3_revenue = [14.2, 8.8, 11.3, 3.9]

    x = range(len(regions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar([i - width/2 for i in x], q2_revenue, width, label='Q2 2024', color='#4A90D9')
    bars2 = ax.bar([i + width/2 for i in x], q3_revenue, width, label='Q3 2024', color='#2ECC71')

    ax.set_ylabel('Revenue ($ Millions)')
    ax.set_title('Quarterly Revenue by Region')
    ax.set_xticks(x)
    ax.set_xticklabels(regions)
    ax.legend()

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2,
                f'${bar.get_height()}M', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2,
                f'${bar.get_height()}M', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('chart_revenue.png', dpi=150)
    plt.close()
    print("  Created chart_revenue.png")


def create_chart_growth():
    """Create a monthly growth trend line chart."""
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']
    revenue = [28.5, 29.1, 30.8, 31.2, 32.5, 33.4, 35.1, 36.8, 38.2]
    costs = [22.1, 22.8, 23.2, 23.5, 24.1, 24.8, 25.2, 25.9, 26.3]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(months, revenue, 'o-', color='#2ECC71', linewidth=2, label='Revenue')
    ax.plot(months, costs, 's--', color='#E74C3C', linewidth=2, label='Operating Costs')
    ax.fill_between(months, costs, revenue, alpha=0.1, color='green')

    ax.set_ylabel('Amount ($ Millions)')
    ax.set_title('2024 Revenue vs Operating Costs (Monthly)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('chart_growth.png', dpi=150)
    plt.close()
    print("  Created chart_growth.png")


def create_report_pdf():
    """
    Build the PDF page by page using PyMuPDF.
    Combines text, images, and table-like content.
    """
    import fitz  # PyMuPDF

    doc = fitz.open()

    # ── PAGE 1: Title + Executive Summary ──────────
    page = doc.new_page()
    # Title
    page.insert_text(fitz.Point(180, 60), "NovaTech Inc.", fontsize=24, fontname="Helvetica-Bold")
    page.insert_text(fitz.Point(150, 90), "Q3 2024 Quarterly Business Report", fontsize=14, fontname="Helvetica")
    page.insert_text(fitz.Point(210, 115), "Confidential — Internal Use Only", fontsize=10, fontname="Helvetica-Oblique", color=(0.5, 0.5, 0.5))

    summary = """Executive Summary

NovaTech Inc. delivered strong results in Q3 2024 with total revenue reaching $38.2 million, representing a 14.7% increase compared to Q3 2023. This growth was primarily driven by expansion in the Asia Pacific region and the successful launch of our CloudSync Enterprise platform.

Key highlights for the quarter include:
- Total revenue of $38.2 million, up from $33.3 million in Q3 2023
- Net profit margin improved to 31.2%, up from 28.4% in Q2 2024
- Customer base grew to 12,400 active enterprise accounts
- Employee headcount increased to 847 across 6 global offices
- CloudSync Enterprise generated $8.4 million in its first full quarter

Operating costs were well controlled at $26.3 million, maintaining a healthy gross margin despite significant investment in R&D for next-generation AI features. The company ended the quarter with $52.8 million in cash reserves.

The board has approved a $15 million investment in expanding our Singapore data center to support growing Asia Pacific demand. Construction is expected to complete by Q2 2025."""

    text_rect = fitz.Rect(50, 150, 550, 780)
    page.insert_textbox(text_rect, summary, fontsize=11, fontname="Helvetica")

    # ── PAGE 2: Revenue Chart + Analysis ──────────
    page = doc.new_page()

    revenue_text = """Regional Revenue Analysis

North America remains our largest market, contributing $14.2 million in Q3 2024, a 14.5% increase from Q2. The growth was fueled by 340 new enterprise contracts, primarily in the financial services and healthcare sectors.

Europe showed steady growth at $8.8 million, up 8.6% from Q2. Our London office expansion and GDPR-compliant cloud offering have been key differentiators. Germany and the UK remain the top-performing markets.

Asia Pacific was the standout performer with $11.3 million in revenue, up 16.5% from Q2. The Singapore office, opened in January 2024, has exceeded all targets. Japan and Australia are the primary growth drivers, with India showing promising early traction.

Latin America contributed $3.9 million, up 21.9% from Q2, making it our fastest-growing region by percentage. Brazil and Mexico account for 78% of regional revenue."""

    text_rect = fitz.Rect(50, 40, 550, 330)
    page.insert_textbox(text_rect, revenue_text, fontsize=11, fontname="Helvetica")

    # Insert revenue chart
    chart_rect = fitz.Rect(70, 340, 530, 710)
    page.insert_image(chart_rect, filename="chart_revenue.png")

    # ── PAGE 3: Growth Chart + Department Performance ──
    page = doc.new_page()

    # Insert growth chart
    page.insert_text(fitz.Point(50, 40), "Financial Growth Trends", fontsize=16, fontname="Helvetica-Bold")
    chart_rect = fitz.Rect(50, 60, 530, 380)
    page.insert_image(chart_rect, filename="chart_growth.png")

    dept_text = """Department Performance Summary

Engineering: Headcount grew from 312 to 358 employees. Shipped 47 features and resolved 892 bugs. Average deployment frequency improved to 12 per week, up from 8 in Q2. System uptime maintained at 99.97%.

Sales: Closed 847 new deals worth $18.4 million in total contract value. Average deal size increased to $21,700, up 15% from Q2. Sales cycle reduced from 34 days to 28 days. Customer acquisition cost decreased by 12% to $3,200 per enterprise account.

Customer Success: Net Promoter Score improved to 72, up from 68 in Q2. Customer churn rate decreased to 2.1%, down from 2.8%. Average support ticket resolution time improved to 4.2 hours from 6.1 hours. Upsell revenue reached $4.2 million.

Product: Launched CloudSync Enterprise, NovaTech Analytics 3.0, and the Mobile SDK. Currently 14 features in development for Q4 release. User engagement metrics show 34% increase in daily active usage across all products.

Human Resources: 127 new hires across all departments. Employee satisfaction score of 4.3 out of 5. Voluntary turnover rate of 8.2%, below industry average of 13.5%. Launched new mentorship program with 89 participants."""

    text_rect = fitz.Rect(50, 395, 550, 790)
    page.insert_textbox(text_rect, dept_text, fontsize=10, fontname="Helvetica")

    # ── PAGE 4: Outlook + Risk Factors ──────────
    page = doc.new_page()

    outlook_text = """Q4 2024 Outlook and Strategic Priorities

Management projects Q4 2024 revenue between $40 million and $42 million, representing 5-10% sequential growth. Key strategic priorities include:

1. CloudSync Enterprise expansion: Target 500 enterprise customers by end of Q4, up from current 340. Planned features include real-time collaboration, advanced analytics dashboard, and SOC2 Type II certification completion.

2. Asia Pacific investment: Complete Singapore data center phase 2 construction. Hire 45 additional staff for the region. Launch Japanese-language product by November 2024.

3. AI Integration: Deploy machine learning-powered anomaly detection across all CloudSync tiers. Beta launch of AI-assisted customer support expected in December 2024.

4. Platform reliability: Target 99.99% uptime SLA for Enterprise customers. Implement multi-region failover for all Tier 1 services.

Risk Factors

- Increasing competition from major cloud providers entering our market segment
- Currency fluctuation exposure, particularly EUR/USD and JPY/USD
- Potential regulatory changes in EU data sovereignty requirements
- Dependency on AWS infrastructure for 73% of our cloud services
- Tight labor market for senior engineering talent in key markets
- Customer concentration risk: top 10 customers represent 22% of revenue

Capital Allocation

The board has approved the following capital allocation for Q4 2024:
- R&D investment: $8.5 million (focus on AI and platform scalability)
- Sales and marketing: $6.2 million (Asia Pacific expansion campaign)
- Infrastructure: $4.8 million (Singapore data center, network upgrades)
- General and administrative: $3.1 million
- Share buyback program: $2.0 million authorized

Total planned expenditure: $24.6 million against projected revenue of $40-42 million, maintaining target profit margin above 30%.

Report prepared by: Sarah Chen, Chief Financial Officer
Approved by: David Park, Chief Executive Officer
Date: October 15, 2024"""

    text_rect = fitz.Rect(50, 40, 550, 790)
    page.insert_textbox(text_rect, outlook_text, fontsize=11, fontname="Helvetica")

    # Save
    output_path = "novatech_q3_report.pdf"
    doc.save(output_path)
    doc.close()
    print(f"  Created {output_path} ({4} pages)")
    return output_path


# ─────────────────────────────────────────────
# BUILD EVERYTHING
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Creating report assets...")
    create_chart_revenue()
    create_chart_growth()

    print("\nBuilding PDF report...")
    pdf_path = create_report_pdf()

    print(f"\n✅ Done! Open {pdf_path} in Preview/browser to see it.")
    print("This is the source document for Project 3 Multimodal RAG.")