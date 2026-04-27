"""
Creates the employee handbook PDF using ReportLab.
Handles long text properly — auto-flows across pages.
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch

def create_handbook():
    doc = SimpleDocTemplate(
        "novatech_handbook.pdf",
        pagesize=letter,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('CustomTitle', parent=styles['Title'], fontSize=20, spaceAfter=20)
    heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading2'], fontSize=14, spaceBefore=20, spaceAfter=10)
    body_style = ParagraphStyle('CustomBody', parent=styles['Normal'], fontSize=10, leading=14, spaceAfter=8)

    content = []

    # Title
    content.append(Paragraph("NovaTech Inc.", title_style))
    content.append(Paragraph("Employee Handbook 2024", styles['Heading3']))
    content.append(Spacer(1, 20))

    sections = {
        "1. Work Schedule and Remote Policy": """
Standard working hours are 9 AM to 5 PM, Monday through Friday. Core hours when all employees must be available are 10 AM to 3 PM regardless of timezone.
<br/><br/>
Remote work is permitted up to 3 days per week for all employees. Fully remote positions are available for engineering and product roles with director-level approval. Remote employees must maintain a dedicated workspace and reliable internet connection with minimum 50 Mbps download speed.
<br/><br/>
Employees working remotely must be responsive on Slack during core hours. Camera should be on during team meetings. All remote workers must visit their assigned office at least once per quarter for team building events.
""",
        "2. Compensation and Stock Options": """
Salaries are benchmarked annually against industry standards using Radford and Levels.fyi data. Annual merit increases range from 2% to 8% based on performance review scores. Promotions typically include a 10% to 20% salary adjustment.
<br/><br/>
All full-time employees receive stock option grants. The standard vesting schedule is 4 years with a 1-year cliff. Initial grants are made on the start date. Refresh grants are awarded annually based on performance and are typically 25% of the initial grant size. Stock options expire 10 years from the grant date or 90 days after termination.
""",
        "3. Time Off and Leave Policies": """
Paid Time Off: 20 days per year for all employees. PTO accrues at 1.67 days per month. Maximum accrual cap is 30 days. Once you hit 30 days, accrual pauses until PTO is used. Unused PTO is not paid out upon termination in states where not required by law.
<br/><br/>
Sick Leave: 10 days per year, non-rollover. A doctors note is required for absences exceeding 3 consecutive days. Mental health days count as sick leave and require no documentation.
<br/><br/>
Parental Leave: 16 weeks paid leave for all new parents regardless of gender. Can be taken anytime within the first year after birth or adoption. Parental leave does not count against PTO.
<br/><br/>
Bereavement: 5 days for immediate family, 3 days for extended family.
<br/><br/>
Company Holidays: 11 paid holidays per year. If a holiday falls on a weekend, it is observed on the nearest weekday.
""",
        "4. Health Insurance and Benefits": """
NovaTech offers three health plan tiers effective from day one of employment:
<br/><br/>
Basic Plan ($50/month employee cost): Medical coverage only. $1500 deductible, 80/20 coinsurance after deductible. In-network only. Covers preventive care at 100%.
<br/><br/>
Standard Plan ($120/month): Medical plus dental and vision. $1000 deductible, 80/20 coinsurance. Dental covers 2 cleanings per year, basic and major restorative work. Vision covers annual exam and $200 frame allowance.
<br/><br/>
Premium Plan ($200/month employee cost): Everything in Standard plus mental health coverage (12 therapy sessions per year), gym membership reimbursement up to $100/month, and telehealth unlimited visits. $500 deductible, 90/10 coinsurance.
<br/><br/>
The company pays 80% of employee premiums and 50% of dependent premiums. Dependents include spouse, domestic partner, and children up to age 26.
""",
        "5. 401(k) Retirement Plan": """
NovaTech matches employee 401(k) contributions dollar-for-dollar up to 5% of salary. Eligibility begins after 90 days of employment. Employer match follows a 3-year graded vesting schedule: 33% after year 1, 66% after year 2, 100% after year 3.
<br/><br/>
Employees may contribute up to the IRS annual limit ($23,000 for 2024, $30,500 for employees over 50). Both traditional pre-tax and Roth after-tax options are available.
""",
        "6. Equipment and Security": """
All employees receive their choice of a MacBook Pro 14-inch or ThinkPad X1 Carbon and a 27-inch 4K monitor on their first day. Additional peripherals (keyboard, mouse, headset) are provided through a $300 equipment allowance.
<br/><br/>
Equipment must be returned within 5 business days of the last day of employment. Damaged or missing equipment will be deducted from the final paycheck at depreciated value.
<br/><br/>
All company devices must have FileVault or BitLocker full-disk encryption enabled. The corporate VPN must be used when accessing internal systems from any network outside the office. Passwords must be at least 12 characters with uppercase, lowercase, number, and special character. Passwords expire every 90 days. Two-factor authentication is mandatory for all company accounts.
<br/><br/>
Report security incidents to security@novatech.com within 1 hour of discovery. The security team responds 24/7 within 30 minutes of report.
""",
        "7. Performance Reviews": """
Reviews are conducted twice per year in June and December. The review process uses a 1-5 rating scale: 1 Does not meet expectations, 2 Partially meets expectations, 3 Meets expectations, 4 Exceeds expectations, 5 Significantly exceeds expectations.
<br/><br/>
Self-reviews are due 2 weeks before the review meeting. Manager reviews include peer feedback from at least 3 colleagues. Performance ratings directly influence merit increases, bonus payouts, and stock refresh grants.
""",
        "8. Professional Development": """
Education reimbursement covers up to $5,000 per year for job-related courses, certifications, and conferences. Requires manager pre-approval. Receipts must be submitted within 30 days.
<br/><br/>
Each employee gets 5 dedicated learning days per year (separate from PTO) for conferences, workshops, or self-directed study. Unused learning days do not roll over.
<br/><br/>
NovaTech maintains licenses for LinkedIn Learning, OReilly Safari, and Coursera for Business for all employees.
""",
    }

    for heading, body in sections.items():
        content.append(Paragraph(heading, heading_style))
        content.append(Paragraph(body.strip(), body_style))

    doc.build(content)
    print("Created novatech_handbook.pdf")


if __name__ == "__main__":
    create_handbook()