"""Generate three FICTIONAL company policy PDFs for testing InfoDeck.

Run:  python scripts/make_sample_docs.py
Output: data/sample_docs/*.pdf

The company "Kestrel Ridge Technologies Ltd." is made up. These files are safe
to commit to a public repository.
"""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

COMPANY = "Kestrel Ridge Technologies Ltd."
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_docs"

styles = getSampleStyleSheet()
TITLE = ParagraphStyle("T", parent=styles["Title"], fontSize=20, spaceAfter=6)
SUB = ParagraphStyle("S", parent=styles["Normal"], fontSize=10, textColor=colors.grey,
                     alignment=1, spaceAfter=12)
H1 = ParagraphStyle("H1", parent=styles["Heading2"], fontSize=13, spaceBefore=12, spaceAfter=6)
H2 = ParagraphStyle("H2", parent=styles["Heading3"], fontSize=11, spaceBefore=8, spaceAfter=4)
BODY = ParagraphStyle("B", parent=styles["Normal"], fontSize=11, leading=16, spaceAfter=7)
CELL = ParagraphStyle("C", parent=BODY, fontSize=9.5, leading=12, spaceAfter=0)
CELL_HEAD = ParagraphStyle("CH", parent=CELL, fontName="Helvetica-Bold")


def table(rows, widths=None):
    data = [[Paragraph(str(c), CELL_HEAD if i == 0 else CELL) for c in row]
            for i, row in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DDE6F0")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#8899AA")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return [t, Spacer(1, 8)]


def bullets(items):
    return [ListFlowable([ListItem(Paragraph(i, BODY), leftIndent=12) for i in items],
                         bulletType="bullet", start="•", leftIndent=14), Spacer(1, 4)]


def build(filename, title, doc_id, meta, sections):
    """sections: list of flowables-producing items: ('h1'|'h2'|'p', text) or flowable lists."""
    path = OUT_DIR / filename

    def on_page(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawString(20 * mm, 287 * mm, f"{COMPANY}  |  {title}  |  {doc_id}")
        canvas.drawString(20 * mm, 10 * mm, "FICTIONAL SAMPLE DOCUMENT - for testing only")
        canvas.drawRightString(190 * mm, 10 * mm, f"Page {doc.page}")
        canvas.restoreState()

    story = [Paragraph(title, TITLE), Paragraph(COMPANY, SUB)]
    story += table([["Document ID", "Version", "Effective date", "Policy owner", "Next review"], meta],
                   widths=[30 * mm, 20 * mm, 32 * mm, 50 * mm, 32 * mm])
    for item in sections:
        if isinstance(item, tuple):
            kind, text = item
            story.append(Paragraph(text, {"h1": H1, "h2": H2, "p": BODY}[kind]))
        else:
            story += item

    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
                            topMargin=18 * mm, bottomMargin=18 * mm, title=title,
                            author=COMPANY, subject="Fictional sample policy")
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    return path


# --------------------------------------------------------------------------- #
# 1. Leave policy
# --------------------------------------------------------------------------- #
LEAVE = [
    ("h1", "1. Purpose and Scope"),
    ("p", "1.1 This policy explains the types of leave available to employees of Kestrel Ridge "
          "Technologies Ltd. (the \"Company\"), how leave is earned, and how to apply for it."),
    ("p", "1.2 The policy applies to all permanent and fixed-term employees in all Company offices. "
          "Interns and contractors are covered only by Section 9 (Public Holidays) unless their "
          "contract states otherwise."),
    ("h1", "2. Definitions"),
    *[bullets([
        "<b>Leave year:</b> 1 January to 31 December.",
        "<b>Working day:</b> Monday to Friday, excluding public holidays.",
        "<b>Probation period:</b> the first 6 months of employment.",
        "<b>Immediate family:</b> spouse or partner, children, parents, siblings and parents-in-law.",
        "<b>HRConnect:</b> the Company's online HR portal used to apply for and approve leave.",
    ])],
    ("h1", "3. Leave Entitlements"),
    ("p", "3.1 Full-time employees are entitled to the following leave in each leave year. "
          "Part-time employees receive leave in proportion to their contracted hours."),
    table([
        ["Leave type", "Entitlement per year", "Carry forward", "Advance notice required"],
        ["Earned (annual) leave", "18 working days", "Up to 10 days", "7 days; 14 days for 4 or more consecutive days"],
        ["Sick leave", "10 working days", "Not allowed", "Inform manager before 10:00 on the first day"],
        ["Casual leave", "6 working days", "Not allowed", "1 working day"],
        ["Maternity leave", "26 weeks (paid)", "Not applicable", "8 weeks before expected date"],
        ["Paternity leave", "10 working days", "Not applicable", "4 weeks before expected date"],
        ["Bereavement leave", "5 working days per event", "Not applicable", "As soon as possible"],
        ["Marriage leave", "3 working days (once during employment)", "Not applicable", "30 days"],
    ], widths=[38 * mm, 40 * mm, 30 * mm, 62 * mm]),
    ("p", "3.2 Paternity leave must be taken within 3 months of the birth or adoption of the child."),
    ("p", "3.3 Casual leave may not be taken for more than 2 consecutive days and may not be "
          "combined with earned leave."),
    ("h1", "4. Accrual and Probation"),
    ("p", "4.1 Earned leave accrues at 1.5 days for each completed month of service. Employees "
          "joining part-way through the year receive a pro-rated entitlement."),
    ("p", "4.2 During the probation period an employee may take a maximum of 3 days of earned "
          "leave. Sick leave and casual leave are available from the first day of employment."),
    ("h2", "4.3 Additional leave for long service"),
    ("p", "Employees receive additional earned leave based on completed years of service on "
          "1 January of each leave year:"),
    table([
        ["Completed years of service", "Additional earned leave per year", "Total earned leave"],
        ["0 - 2 years", "0 days", "18 days"],
        ["3 - 5 years", "2 days", "20 days"],
        ["6 - 10 years", "4 days", "22 days"],
        ["More than 10 years", "6 days", "24 days"],
    ], widths=[55 * mm, 60 * mm, 55 * mm]),
    ("h1", "5. Applying for Leave"),
    ("p", "5.1 All leave must be applied for in HRConnect. The reporting manager must approve or "
          "reject the request within 2 working days. Requests not actioned in 2 working days are "
          "escalated automatically to the department head."),
    ("p", "5.2 Emergency leave: if an employee cannot apply in advance because of an emergency, "
          "they must inform their manager by phone or message on the same day and apply in "
          "HRConnect within 2 working days of returning to work."),
    ("p", "5.3 Leave taken without approval is treated as unauthorised absence and may lead to "
          "loss of pay and disciplinary action."),
    ("h1", "6. Sick Leave"),
    ("p", "6.1 A medical certificate from a registered doctor is required for sick leave of more "
          "than 2 consecutive working days. The certificate must be uploaded in HRConnect within "
          "3 working days of returning to work."),
    ("p", "6.2 If sick leave is exhausted, further absence due to illness is deducted from earned "
          "leave, and then treated as leave without pay."),
    ("p", "6.3 Sick leave cannot be encashed and lapses at the end of the leave year."),
    ("h1", "7. Carry Forward and Encashment"),
    ("p", "7.1 A maximum of 10 unused earned leave days may be carried forward to the next leave "
          "year. Carried-forward days must be used by 31 March of the next year; any unused "
          "carried-forward days lapse after that date."),
    ("p", "7.2 Earned leave cannot be encashed while in employment. On resignation or retirement, "
          "unused earned leave up to a maximum of 30 days is paid out at the employee's basic "
          "salary rate."),
    ("p", "7.3 Employees dismissed for misconduct are not eligible for leave encashment."),
    ("h1", "8. Leave Without Pay and Sabbatical"),
    ("p", "8.1 Leave without pay (LWP) of up to 30 days per year may be granted by the department "
          "head, but only after all earned leave has been used."),
    ("p", "8.2 Employees with at least 5 years of continuous service may apply for an unpaid "
          "sabbatical of up to 3 months. A sabbatical requires approval from the department head "
          "and the HR Director, and must be requested at least 60 days in advance."),
    ("h1", "9. Public Holidays"),
    ("p", "9.1 The Company observes 10 fixed public holidays each year. The list for the next year "
          "is published by HR in December."),
    ("p", "9.2 In addition, each employee may choose 2 optional holidays from the published "
          "optional holiday list."),
    ("p", "9.3 Employees required to work on a public holiday receive a compensatory day off, "
          "which must be taken within 30 days."),
    ("h1", "10. Special Leave"),
    ("p", "10.1 The following paid special leave is available in addition to the entitlements in "
          "Section 3. It does not reduce earned, sick or casual leave balances."),
    table([
        ["Situation", "Paid leave allowed", "Evidence required"],
        ["Jury duty or court witness", "As long as required by the court", "Court summons"],
        ["Voting in a national or state election", "1 day (polling day)", "None"],
        ["Blood or platelet donation", "Half a day per donation, maximum 4 per year", "Donation certificate"],
        ["Serious illness of an immediate family member", "Up to 5 days per year", "Medical letter"],
        ["Relocation at the Company's request", "2 days", "Transfer letter"],
    ], widths=[62 * mm, 60 * mm, 48 * mm]),
    ("h1", "11. Leave During the Notice Period"),
    ("p", "11.1 Earned leave may not be taken during the notice period unless approved in writing by "
          "the department head. Approved leave does not extend the last working day."),
    ("p", "11.2 Sick leave may be taken during the notice period, but a medical certificate is "
          "required for every day of sick leave."),
    ("h1", "12. Return to Work"),
    ("p", "12.1 After sick leave of more than 10 consecutive working days, the employee must submit "
          "a fitness-to-work certificate before returning."),
    ("p", "12.2 Employees returning from long illness, maternity leave or a sabbatical may request "
          "a phased return of up to 4 weeks with reduced hours, agreed with their manager and HR."),
    ("h1", "13. Worked Examples"),
    ("p", "<b>Example 1:</b> Priya joined on 1 April and finished probation on 30 September. By "
          "31 December she has completed 9 months of service and has accrued 13.5 days of earned "
          "leave (9 x 1.5). During probation she could use at most 3 of those days."),
    ("p", "<b>Example 2:</b> Arjun has 14 unused earned leave days on 31 December. He can carry "
          "forward 10 days; the other 4 days lapse. The 10 carried-forward days must be used by "
          "31 March, otherwise they also lapse."),
    ("p", "<b>Example 3:</b> Meera is sick for 3 consecutive working days. Because this is more than "
          "2 days, she must upload a medical certificate in HRConnect within 3 working days of "
          "returning."),
    ("h1", "14. Contact"),
    ("p", "Questions about this policy should be sent to the HR team at hr@kestrelridge.example."),
    ("h2", "Revision history"),
    table([
        ["Version", "Date", "Change"],
        ["2.0", "1 April 2024", "Added paternity leave and HRConnect portal."],
        ["3.0", "1 April 2025", "Increased maternity leave to 26 weeks; added sabbatical."],
        ["3.1", "1 April 2026", "Added long-service leave table and special leave (Section 10)."],
    ], widths=[20 * mm, 35 * mm, 115 * mm]),
]

# --------------------------------------------------------------------------- #
# 2. IT security policy
# --------------------------------------------------------------------------- #
IT = [
    ("h1", "1. Purpose and Scope"),
    ("p", "1.1 This policy protects the Company's information, systems and customers from loss, "
          "misuse and cyber attacks."),
    ("p", "1.2 It applies to all employees, interns, contractors and visitors who use Company "
          "devices, networks, accounts or data, whether in the office or working remotely."),
    ("h1", "2. Roles and Responsibilities"),
    *[bullets([
        "<b>IT Security team:</b> owns this policy, monitors systems and leads incident response.",
        "<b>IT Service Desk:</b> first point of contact for security issues; phone extension 4357 "
          "or servicedesk@kestrelridge.example.",
        "<b>Managers:</b> ensure their teams complete training and follow this policy.",
        "<b>All users:</b> protect their accounts and devices and report incidents immediately.",
    ])],
    ("h1", "3. Acceptable Use"),
    ("p", "3.1 Company systems are provided for business purposes. Limited personal use is "
          "allowed if it does not affect work, does not break any law and does not create a "
          "security risk."),
    ("p", "3.2 Users must not share accounts, bypass security controls, install unapproved "
          "software, or access data they do not need for their role."),
    ("h1", "4. Passwords and Authentication"),
    ("p", "4.1 Passwords must be unique for each system and must never be shared or written down. "
          "Users should use the Company-approved password manager."),
    table([
        ["Account type", "Minimum length", "Multi-factor authentication (MFA)", "Password change"],
        ["Standard user account", "14 characters", "Required for email, VPN and HRConnect", "Only if compromise is suspected"],
        ["Privileged / admin account", "20 characters", "Required for every login", "Every 90 days"],
        ["Service account", "32 characters (random)", "Not applicable", "Every 180 days, managed by IT"],
    ], widths=[40 * mm, 32 * mm, 52 * mm, 46 * mm]),
    ("p", "4.2 An account is locked for 15 minutes after 5 failed login attempts."),
    ("p", "4.3 MFA codes or approval prompts must never be shared with anyone, including people "
          "who claim to be from IT."),
    ("h1", "5. Data Classification and Handling"),
    ("p", "5.1 All Company information must be classified into one of four levels and handled "
          "as shown below."),
    table([
        ["Level", "Examples", "Handling rules"],
        ["Public", "Marketing website content, published press releases", "No restrictions."],
        ["Internal", "Policies, internal announcements, org charts", "Share only with employees and approved contractors."],
        ["Confidential", "Customer contracts, financial reports, source code", "Encrypt when sent outside the Company; share on a need-to-know basis."],
        ["Restricted", "Employee salary and health data, passwords, customer personal data", "Encrypted at all times; never stored on USB drives or personal devices; access logged."],
    ], widths=[26 * mm, 62 * mm, 82 * mm]),
    ("p", "5.2 If the classification of a document is unclear, treat it as Confidential until "
          "the information owner confirms."),
    ("h1", "6. Devices"),
    ("p", "6.1 Company laptops are protected with full-disk encryption (BitLocker) and endpoint "
          "protection software. Users must not disable these controls."),
    ("p", "6.2 Screens must lock automatically after 5 minutes of inactivity. Users must lock "
          "their screen (Windows key + L) whenever they leave their device."),
    ("p", "6.3 Personal devices (BYOD) may be used only for Company email and chat, and only "
          "after enrolment in the Company mobile device management (MDM) system."),
    ("p", "6.4 USB storage devices are blocked by default. An exception requires a Service Desk "
          "ticket approved by the IT Security team and is valid for a maximum of 30 days."),
    ("h1", "7. Software and Updates"),
    ("p", "7.1 Software may only be installed from the Company Software Center. Requests for new "
          "software are raised through the Service Desk and reviewed by IT Security."),
    ("p", "7.2 Security updates must be installed within 14 days of release. Critical security "
          "updates must be installed within 72 hours. Devices that are not updated may be "
          "blocked from the network."),
    ("h1", "8. Email and Phishing"),
    ("p", "8.1 Users must report suspicious emails using the \"Report Phishing\" button in Outlook, "
          "or by forwarding them to security@kestrelridge.example. Do not click links or open "
          "attachments in suspicious emails."),
    ("p", "8.2 Company email must not be auto-forwarded to personal email accounts."),
    ("p", "8.3 The IT team will never ask for your password by email, phone or chat."),
    ("h1", "9. Remote Work and Networks"),
    ("p", "9.1 The Company VPN must be used when working on public or hotel Wi-Fi networks."),
    ("p", "9.2 Users working from home must change the default password on their home router and "
          "use WPA2 or WPA3 encryption."),
    ("p", "9.3 Confidential and Restricted information must not be viewed in public places where "
          "screens can be seen by others."),
    ("h1", "10. Use of AI Tools"),
    ("p", "10.1 Only AI tools approved by IT Security may be used for Company work. The list of "
          "approved tools is published on the intranet."),
    ("p", "10.2 Confidential or Restricted information must never be entered into public AI "
          "tools or chatbots that are not on the approved list."),
    ("p", "10.3 Output from AI tools must be reviewed by a person before it is shared with "
          "customers."),
    ("h1", "11. Security Incident Reporting"),
    ("p", "11.1 Any suspected security incident must be reported to the IT Service Desk "
          "(extension 4357) within 1 hour of discovery."),
    ("p", "11.2 Lost or stolen Company devices must be reported within 2 hours so the device can "
          "be remotely locked and wiped."),
    table([
        ["Severity", "Example", "IT Security response time"],
        ["Critical", "Ransomware, confirmed data breach", "Within 15 minutes, 24x7"],
        ["High", "Compromised account, lost laptop", "Within 1 hour"],
        ["Medium", "Malware detected and blocked, phishing email clicked", "Within 4 business hours"],
        ["Low", "Policy question, suspicious email not clicked", "Within 2 business days"],
    ], widths=[26 * mm, 84 * mm, 60 * mm]),
    ("h1", "12. Physical Security"),
    ("p", "12.1 Employees must wear their ID badge visibly at all times inside Company offices and "
          "must not let anyone \"tailgate\" through a secure door."),
    ("p", "12.2 Visitors must sign in at reception, wear a visitor badge and be escorted by their "
          "host at all times. Visitors may not connect to the internal network; they may use the "
          "Guest Wi-Fi only."),
    ("p", "12.3 Clean desk rule: Confidential and Restricted papers must be locked away at the end "
          "of the day, and printouts must be collected from printers immediately."),
    ("h1", "13. Backup and Data Retention"),
    ("p", "13.1 Work files must be saved in the Company OneDrive or approved shared drives, which "
          "are backed up every day. Files stored only on a laptop's local disk are not backed up."),
    table([
        ["Data type", "Retention period", "Disposal method"],
        ["Email", "3 years", "Automatic deletion"],
        ["Financial and tax records", "8 years", "Secure deletion by Finance"],
        ["Employee personnel files", "7 years after the employee leaves", "Secure deletion by HR"],
        ["System and security logs", "1 year", "Automatic deletion"],
        ["Paper documents (Confidential or Restricted)", "As per the record type above", "Cross-cut shredding or locked shred bins"],
    ], widths=[62 * mm, 52 * mm, 56 * mm]),
    ("p", "13.2 Old laptops, phones and hard drives must be returned to IT for certified data "
          "wiping. They must never be sold, given away or thrown away by employees."),
    ("h1", "14. Third-Party and Vendor Access"),
    ("p", "14.1 Vendors who need access to Company systems must sign a non-disclosure agreement and "
          "complete a security assessment before access is granted."),
    ("p", "14.2 Vendor accounts are created with an expiry date of no more than 90 days and are "
          "reviewed by the sponsoring manager before renewal."),
    ("h1", "15. Training and Compliance"),
    ("p", "15.1 New joiners must complete security awareness training within 30 days of joining. "
          "All users must complete refresher training every year."),
    ("p", "15.2 The IT Security team runs simulated phishing tests. Users who fail a test are "
          "assigned additional training."),
    ("p", "15.3 Breaches of this policy may result in disciplinary action, up to and including "
          "termination of employment."),
    ("h2", "Revision history"),
    table([
        ["Version", "Date", "Change"],
        ["4.0", "1 January 2025", "Introduced MFA for all standard accounts; USB storage blocked."],
        ["4.1", "1 June 2025", "Added data retention schedule (Section 13)."],
        ["5.0", "1 January 2026", "Added Use of AI Tools (Section 10); removed 90-day password change for standard users."],
    ], widths=[20 * mm, 35 * mm, 115 * mm]),
]

# --------------------------------------------------------------------------- #
# 3. Travel and expense policy
# --------------------------------------------------------------------------- #
TRAVEL = [
    ("h1", "1. Purpose and Scope"),
    ("p", "1.1 This policy sets the rules for business travel and for claiming business expenses "
          "at Kestrel Ridge Technologies Ltd. It aims to keep employees safe and comfortable "
          "while controlling costs."),
    ("p", "1.2 It applies to all employees travelling on Company business. Expenses for "
          "contractors are covered by their contracts."),
    ("h1", "2. Travel Approval and Booking"),
    ("p", "2.1 All business travel must be approved before booking, using the travel request form "
          "in the TravelDesk portal."),
    table([
        ["Type of travel", "Approver", "Book at least"],
        ["Domestic, up to 3 nights", "Reporting manager", "14 days before travel"],
        ["Domestic, more than 3 nights", "Department head", "14 days before travel"],
        ["International", "Department head and Chief Financial Officer (CFO)", "21 days before travel"],
    ], widths=[55 * mm, 70 * mm, 45 * mm]),
    ("p", "2.2 Flights and hotels must be booked through the Company travel agent via TravelDesk. "
          "Bookings made outside TravelDesk are reimbursed only if they are cheaper than the "
          "TravelDesk option and the manager approves the exception."),
    ("p", "2.3 Late bookings (less than the minimum notice above) need a written reason from the "
          "approver."),
    ("h1", "3. Class of Travel"),
    table([
        ["Employee grade", "Flights up to 6 hours", "Flights over 6 hours", "Train"],
        ["L1 - L3", "Economy", "Economy", "AC 3-tier"],
        ["L4 - L5", "Economy", "Premium economy", "AC 2-tier"],
        ["L6 and above", "Economy", "Business class", "AC First class"],
    ], widths=[35 * mm, 42 * mm, 48 * mm, 45 * mm]),
    ("p", "3.1 Upgrades paid for by the Company are not allowed. Employees may upgrade at their own "
          "cost or using personal loyalty points."),
    ("h1", "4. Accommodation"),
    ("p", "4.1 The following are the maximum hotel rates per night, including taxes. Breakfast "
          "may be included in the room rate."),
    table([
        ["Location", "Maximum rate per night"],
        ["Tier 1 cities (Mumbai, Delhi NCR, Bengaluru, Chennai, Hyderabad)", "INR 7,000"],
        ["Tier 2 cities (for example Pune, Jaipur, Kochi, Ahmedabad)", "INR 4,500"],
        ["All other domestic locations", "INR 3,000"],
        ["International - London, New York, Tokyo, Singapore", "USD 250"],
        ["International - all other cities", "USD 180"],
    ], widths=[120 * mm, 50 * mm]),
    ("p", "4.2 If an employee stays with friends or family instead of a hotel, they may claim a "
          "fixed amount of INR 1,000 per night (domestic) or USD 40 per night (international)."),
    ("h1", "5. Daily Allowance (Per Diem)"),
    ("p", "5.1 A daily allowance covers meals, tips and small incidental costs. Receipts are not "
          "required for the daily allowance."),
    table([
        ["Employee grade", "Domestic (per day)", "International (per day)"],
        ["L1 - L3", "INR 1,200", "USD 60"],
        ["L4 - L5", "INR 1,800", "USD 75"],
        ["L6 and above", "INR 2,500", "USD 100"],
    ], widths=[50 * mm, 60 * mm, 60 * mm]),
    ("p", "5.2 Half the daily allowance is paid for the day of departure and the day of return if "
          "travel time on that day is less than 8 hours."),
    ("p", "5.3 If meals are provided free (for example by a client or at a conference), the daily "
          "allowance is reduced by 25% for each main meal provided."),
    ("h1", "6. Local Transport"),
    ("p", "6.1 Employees should use app-based cabs, taxis or public transport for local travel."),
    ("p", "6.2 Use of a personal vehicle for business is reimbursed at INR 12 per km for a car and "
          "INR 5 per km for a two-wheeler. Tolls and parking are reimbursed with receipts."),
    ("p", "6.3 Car rentals require prior approval from the department head."),
    ("h1", "7. Meals and Client Entertainment"),
    ("p", "7.1 Client entertainment above INR 5,000 per event requires pre-approval from the "
          "department head. The claim must list the names and companies of all attendees."),
    ("p", "7.2 Alcohol is not reimbursable, except as part of approved client entertainment."),
    ("h1", "8. Non-Reimbursable Expenses"),
    *[bullets([
        "Traffic fines, parking fines and penalties.",
        "Hotel minibar, in-room movies and spa services.",
        "Personal entertainment, gym or sightseeing.",
        "Laundry on trips shorter than 4 nights.",
        "Seat or class upgrades (see Section 3.1).",
        "Lost personal items, or excess baggage for personal items.",
    ])],
    ("h1", "9. Travel Advances"),
    ("p", "9.1 Employees may request a travel advance of up to 75% of the estimated trip cost "
          "(excluding flights and hotels booked through TravelDesk)."),
    ("p", "9.2 Advances must be settled by submitting an expense claim within 15 days of return. "
          "A new advance is not issued while a previous advance is unsettled."),
    ("h1", "10. Submitting Expense Claims"),
    ("p", "10.1 Expense claims must be submitted in TravelDesk within 30 days of returning from "
          "the trip. Claims submitted more than 60 days after return will not be reimbursed."),
    ("p", "10.2 An original receipt or invoice is required for every expense above INR 500, except "
          "for the daily allowance."),
    ("p", "10.3 Approved claims are paid to the employee's salary bank account within 10 working "
          "days of approval by Finance."),
    ("h1", "11. Corporate Credit Card"),
    ("p", "11.1 Employees at grade L5 and above who travel at least 4 times a year may apply for a "
          "corporate credit card. The card must be used only for business expenses."),
    ("p", "11.2 Personal spending on a corporate card must be repaid within 7 days and repeated "
          "misuse will lead to the card being withdrawn."),
    ("h1", "12. Travel Insurance and Safety"),
    ("p", "12.1 All employees on approved business travel are automatically covered by the "
          "Company's group travel insurance, including medical emergencies, trip cancellation and "
          "lost baggage. No separate insurance needs to be purchased."),
    ("p", "12.2 In an emergency while travelling, call the 24x7 travel assistance line at "
          "+91 80 4000 1234 and inform your manager as soon as possible."),
    ("p", "12.3 Travel to countries rated \"high risk\" by the Company security advisor requires "
          "additional approval from the Chief Operating Officer."),
    ("h1", "13. Visas, Passports and Vaccinations"),
    ("p", "13.1 Visa fees, visa agent charges and mandatory vaccinations for business travel are "
          "reimbursed with receipts."),
    ("p", "13.2 Passport application or renewal fees are not reimbursed, because a passport is a "
          "personal document."),
    ("h1", "14. Combining Business and Personal Travel"),
    ("p", "14.1 Employees may add up to 2 personal days to a business trip, with manager approval. "
          "Earned leave must be applied for the personal days."),
    ("p", "14.2 The Company pays only the costs it would have paid for the business trip alone. "
          "Any extra airfare difference, hotel nights or other costs for personal days are paid by "
          "the employee. Family members may travel with the employee at their own cost."),
    ("h1", "15. Policy Violations"),
    ("p", "15.1 Submitting false or inflated claims is treated as fraud and may lead to "
          "disciplinary action, including termination. Questions about this policy can be sent to "
          "finance@kestrelridge.example."),
    ("h2", "Revision history"),
    table([
        ["Version", "Date", "Change"],
        ["2.0", "1 July 2024", "Introduced TravelDesk portal and city tier hotel limits."],
        ["2.1", "1 January 2025", "Increased mileage rate for cars from INR 10 to INR 12 per km."],
        ["2.2", "1 July 2026", "Added Sections 12 to 14 (insurance, visas, combined travel)."],
    ], widths=[20 * mm, 35 * mm, 115 * mm]),
]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    built = [
        build("KRT_Leave_Policy.pdf", "Leave Policy", "HR-POL-004",
              ["HR-POL-004", "3.1", "1 April 2026", "Head of Human Resources", "31 March 2027"], LEAVE),
        build("KRT_IT_Security_Policy.pdf", "IT Security Policy", "IT-POL-001",
              ["IT-POL-001", "5.0", "1 January 2026", "Chief Information Security Officer", "31 December 2026"], IT),
        build("KRT_Travel_and_Expense_Policy.pdf", "Travel and Expense Policy", "FIN-POL-007",
              ["FIN-POL-007", "2.2", "1 July 2026", "Chief Financial Officer", "30 June 2027"], TRAVEL),
    ]
    for p in built:
        print(f"Created {p}")


if __name__ == "__main__":
    main()
