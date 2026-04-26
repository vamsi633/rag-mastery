"""
Generates realistic CSV data files — simulating exports from
real business tools (Salesforce, BambooHR, Zendesk, etc.)

In production, these would come from:
  - Salesforce export → sales.csv
  - BambooHR export → employees.csv
  - Zendesk export → tickets.csv

You can REPLACE these with your own CSVs and the system adapts.
"""

import csv
import random
import os
from datetime import datetime, timedelta

os.makedirs("data", exist_ok=True)


def generate_employees():
    """Simulates a BambooHR / Workday export."""
    departments = {
        "Engineering": ["Engineer", "Senior Engineer", "Staff Engineer", "Engineering Manager", "VP Engineering"],
        "Sales": ["Sales Rep", "Senior Sales Rep", "Account Executive", "Sales Manager", "VP Sales"],
        "Customer Success": ["CS Representative", "CS Manager", "VP Customer Success"],
        "Product": ["Product Manager", "Senior PM", "VP Product"],
        "Marketing": ["Marketing Specialist", "Content Manager", "Marketing Director"],
        "HR": ["HR Coordinator", "HR Manager", "HR Director"],
    }

    salary_ranges = {
        "Junior": (65000, 95000),
        "Rep": (70000, 100000),
        "Specialist": (75000, 100000),
        "Coordinator": (60000, 80000),
        "Engineer": (120000, 150000),
        "Senior": (145000, 180000),
        "Staff": (170000, 200000),
        "Manager": (130000, 170000),
        "Director": (150000, 195000),
        "VP": (180000, 250000),
        "Account": (100000, 140000),
        "Content": (85000, 110000),
    }

    offices = ["San Francisco", "New York", "London", "Tokyo", "Bangalore", "Singapore"]
    first_names = ["Sarah", "Marcus", "Priya", "Tom", "Yuki", "Alex", "Emma", "James", "Nina", "David",
                   "Lisa", "Robert", "Jennifer", "Carlos", "Amy", "Brian", "Diana", "Mike", "Rachel", "Sam",
                   "Olivia", "Kevin", "Hannah", "Chris", "Maria", "John", "Sophie", "Daniel", "Laura", "Ryan",
                   "Ashley", "Derek", "Fatima", "George", "Ingrid", "Jake", "Kara", "Leo", "Mei", "Nathan"]
    last_names = ["Chen", "Johnson", "Patel", "Wilson", "Tanaka", "Rivera", "Liu", "Brown", "Kowalski", "Park",
                  "Wang", "Kim", "Adams", "Mendez", "Zhang", "Foster", "Ross", "O'Brien", "Green", "Martin",
                  "Wright", "Taylor", "Garcia", "Murphy", "Martinez", "Lee", "Singh", "Davis", "Moore", "Clark"]

    rows = []
    emp_id = 1
    used_names = set()

    for dept, roles in departments.items():
        # Decide how many people per department
        count = random.randint(5, 15) if dept == "Engineering" else random.randint(3, 8)

        for _ in range(count):
            # Generate unique name
            while True:
                name = f"{random.choice(first_names)} {random.choice(last_names)}"
                if name not in used_names:
                    used_names.add(name)
                    break

            role = random.choice(roles)

            # Match salary to role level
            salary_key = next((k for k in salary_ranges if k in role), "Engineer")
            low, high = salary_ranges[salary_key]
            salary = round(random.uniform(low, high), -3)

            # Random hire date in last 5 years
            days_ago = random.randint(30, 1800)
            hire_date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")

            status = random.choices(["active", "left"], weights=[85, 15])[0]
            office = random.choice(offices)

            rows.append([emp_id, name, dept, role, salary, hire_date, status, office])
            emp_id += 1

    with open("data/employees.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "name", "department", "role", "salary", "hire_date", "status", "office"])
        writer.writerows(rows)

    print(f"  employees.csv: {len(rows)} employees")
    return rows


def generate_sales(employees):
    """Simulates a Salesforce deal export."""
    sales_reps = [e for e in employees if "Sales" in e[3] or "Account" in e[3]]
    if not sales_reps:
        sales_reps = [e for e in employees if e[2] == "Sales"]

    customers = [
        ("Acme Corp", "North America"), ("BMW Group", "Europe"), ("Sony Corp", "Asia Pacific"),
        ("JPMorgan Chase", "North America"), ("Petrobras", "Latin America"),
        ("Samsung Electronics", "Asia Pacific"), ("Barclays PLC", "Europe"),
        ("Toyota Motor", "Asia Pacific"), ("Walmart Inc", "North America"),
        ("Banco do Brasil", "Latin America"), ("Siemens AG", "Europe"),
        ("Alibaba Group", "Asia Pacific"), ("Netflix Inc", "North America"),
        ("MercadoLibre", "Latin America"), ("HSBC Holdings", "Europe"),
        ("Tesla Inc", "North America"), ("Tata Group", "Asia Pacific"),
        ("Shell PLC", "Europe"), ("Salesforce", "North America"),
        ("Infosys", "Asia Pacific"), ("Nubank", "Latin America"),
        ("Spotify", "Europe"), ("Uber Technologies", "North America"),
        ("Grab Holdings", "Asia Pacific"), ("Magazine Luiza", "Latin America"),
    ]

    products = ["CloudSync Enterprise", "CloudSync Starter", "DevTools Pro", "NovaTech Analytics", "Mobile SDK"]
    statuses = ["closed_won", "closed_won", "closed_won", "negotiating", "proposal", "lost"]

    rows = []
    for i, (customer, region) in enumerate(customers):
        rep = random.choice(sales_reps)
        product = random.choice(products)
        amount = round(random.uniform(25000, 300000), -3)
        status = random.choice(statuses)

        days_ago = random.randint(1, 120)
        close_date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d") if "closed" in status else ""

        rows.append([i + 1, f"{customer} - {product}", customer, amount, region,
                     rep[1], status, close_date, product])

    with open("data/sales.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "deal_name", "customer", "amount", "region",
                         "sales_rep", "status", "close_date", "product"])
        writer.writerows(rows)

    print(f"  sales.csv: {len(rows)} deals")


def generate_tickets():
    """Simulates a Zendesk / Jira Service Desk export."""
    customers = ["Acme Corp", "BMW Group", "Sony Corp", "JPMorgan Chase",
                 "Samsung Electronics", "Barclays PLC", "Toyota Motor",
                 "Walmart Inc", "Netflix Inc", "Tesla Inc"]

    issues = [
        ("Dashboard not loading after update", "high"),
        ("Data sync delay between regions", "critical"),
        ("API rate limiting errors during peak hours", "high"),
        ("SSO integration failing", "critical"),
        ("Language support missing in dashboard", "medium"),
        ("Bulk export timing out for large datasets", "high"),
        ("Compliance report generation failing", "critical"),
        ("Documentation unclear for API integration", "low"),
        ("Custom widgets not saving", "medium"),
        ("Feature request: real-time collaboration", "low"),
        ("Login timeout on mobile app", "high"),
        ("Data not appearing in analytics", "critical"),
        ("Webhook notifications delayed", "medium"),
        ("User permissions not syncing", "high"),
        ("PDF export formatting broken", "medium"),
    ]

    products = ["CloudSync Enterprise", "DevTools Pro", "NovaTech Analytics", "Mobile SDK"]
    agents = ["Rachel Green", "Sam Patel", "Olivia Martin"]

    rows = []
    for i in range(len(issues)):
        customer = random.choice(customers)
        issue, priority = issues[i]
        product = random.choice(products)
        status = random.choices(["resolved", "in_progress", "open"], weights=[50, 30, 20])[0]
        assigned = random.choice(agents) if status != "open" else ""

        days_ago = random.randint(1, 60)
        created = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        resolved = (datetime.now() - timedelta(days=days_ago - random.randint(1, 3))).strftime("%Y-%m-%d") if status == "resolved" else ""

        rows.append([i + 1, customer, issue, product, priority, status, assigned, created, resolved])

    with open("data/tickets.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "customer", "issue", "product", "priority",
                         "status", "assigned_to", "created_date", "resolved_date"])
        writer.writerows(rows)

    print(f"  tickets.csv: {len(rows)} tickets")


if __name__ == "__main__":
    print("Generating business data CSVs...")
    print("(In production, these come from Salesforce, BambooHR, Zendesk exports)\n")
    employees = generate_employees()
    generate_sales(employees)
    generate_tickets()
    print(f"\nFiles saved in data/ folder.")
    print("You can replace these with YOUR OWN CSVs — the system adapts.")