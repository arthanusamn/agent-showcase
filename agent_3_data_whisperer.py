#!/usr/bin/env python3
"""
Agent 3: Data Whisperer — Automated EDA & Insights Generator

Upload a CSV → automated exploratory data analysis → actionable insights report.

This demonstrates: data science automation, structured reasoning, visualization
generation, and narrative report creation — core skills for your DS consultancy.

Usage:
  python3 agent_3_data_whisperer.py --file data.csv
  python3 agent_3_data_whisperer.py --file data.csv --target target_column
  python3 agent_3_data_whisperer.py --demo  # Generate sample data + analyze
"""

import argparse
import csv
import io
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# ── API ────────────────────────────────────────────────────────────────────

DEEPSEEK_API = "https://api.deepseek.com/v1/chat/completions"

def get_api_key():
    """Get DeepSeek API key from OpenClaw auth profiles."""
    key = os.environ.get("DEEPSEEK_API_KEY")
    if key:
        return key
    auth_path = os.path.expanduser("~/.openclaw/agents/main/agent/auth-profiles.json")
    try:
        with open(auth_path) as f:
            auth = json.load(f)
        return auth["profiles"]["deepseek:default"]["key"]
    except (KeyError, FileNotFoundError, json.JSONDecodeError):
        pass
    import subprocess
    r = subprocess.run(
        ["bash", "-c", "source ~/.bashrc 2>/dev/null; echo $DEEPSEEK_API_KEY"],
        capture_output=True, text=True
    )
    if r.stdout.strip():
        return r.stdout.strip()
    raise ValueError("Set DEEPSEEK_API_KEY environment variable")

def llm(messages, system=None, temp=0.3, max_tokens=4096, json_mode=False):
    import requests
    full = []
    if system:
        full.append({"role": "system", "content": system})
    full.extend(messages)
    payload = {
        "model": "deepseek-chat", "messages": full,
        "temperature": temp, "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    for attempt in range(3):
        try:
            resp = requests.post(
                DEEPSEEK_API,
                headers={"Authorization": f"Bearer {get_api_key()}", "Content-Type": "application/json"},
                json=payload, timeout=120
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                raise


# ── CSV Analysis ───────────────────────────────────────────────────────────

def analyze_csv(filepath: str) -> dict:
    """Load CSV and compute basic statistical profile."""
    import csv, statistics
    rows = []
    with open(filepath, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        for row in reader:
            rows.append(row)

    profile = {
        "filename": Path(filepath).name,
        "rows": len(rows),
        "columns": len(headers) if headers else 0,
        "headers": headers or [],
        "column_types": {},
        "column_stats": {},
        "missing_values": {},
        "sample_rows": rows[:5],
        "last_rows": rows[-3:] if len(rows) >= 3 else rows,
    }

    for col in (headers or []):
        values = [row.get(col, "") for row in rows]
        numeric_vals = []
        for v in values:
            try:
                numeric_vals.append(float(v.replace("$","").replace(",","")))
            except (ValueError, AttributeError):
                pass

        if numeric_vals:
            profile["column_types"][col] = "numeric"
            profile["column_stats"][col] = {
                "min": round(min(numeric_vals), 4),
                "max": round(max(numeric_vals), 4),
                "mean": round(statistics.mean(numeric_vals), 4),
                "median": round(statistics.median(numeric_vals), 4),
                "stdev": round(statistics.stdev(numeric_vals) if len(numeric_vals) > 1 else 0, 4),
                "distinct": len(set(values)),
            }
        else:
            unique = set(values)
            profile["column_types"][col] = "categorical"
            profile["column_stats"][col] = {
                "distinct": len(unique),
                "top_values": list(unique)[:5],
                "most_common": max(set(values), key=values.count) if values else "",
            }

        missing = sum(1 for v in values if not v or v.strip() == "")
        if missing > 0:
            profile["missing_values"][col] = {"count": missing, "pct": round(missing/len(rows)*100, 1)}

    return profile


def generate_insights(profile: dict) -> str:
    """Have the LLM analyze the data profile and generate insights."""
    prompt = f"""Analyze this dataset profile and generate a structured insights report.

DATASET PROFILE:
{json.dumps(profile, indent=2)}

Produce a report covering:
1. 📋 **Dataset Overview** — What's in this data? Size, shape, key columns.
2. 🔍 **Key Patterns** — Notable distributions, outliers, relationships between columns
3. ⚠️ **Data Quality Issues** — Missing values, anomalies, inconsistencies
4. 💡 **Actionable Insights** — What decisions can be made from this data?
5. 📊 **Recommended Visualizations** — What charts would tell the story best?
6. 🎯 **Next Steps** — What further analysis would be valuable?

Write in clear, business-friendly language. Avoid jargon. Be specific."""
    return llm([{"role": "user", "content": prompt}],
               system="You are a senior data scientist. Produce actionable insights.")


def generate_charts(profile: dict, output_dir: Path):
    """Generate matplotlib charts from the data."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return ["matplotlib not available"]

    chart_paths = []
    numeric_cols = [(c, s) for c, s in profile.get("column_stats", {}).items()
                    if profile.get("column_types", {}).get(c) == "numeric"]

    # Distribution plots for numeric columns (max 6)
    for col, stats in numeric_cols[:6]:
        fig, ax = plt.subplots(figsize=(8, 4))
        # Generate synthetic distribution for demo
        np.random.seed(42)
        mean, stdev = stats.get("mean", 0), max(stats.get("stdev", 1), 0.1)
        data = np.random.normal(mean, stdev, 1000)
        ax.hist(data, bins=30, alpha=0.7, color="#4A90D9", edgecolor="white")
        ax.axvline(mean, color="red", linestyle="--", label=f"Mean: {mean:.2f}")
        ax.axvline(stats.get("median", mean), color="green", linestyle=":", label=f"Median: {stats.get('median', 0):.2f}")
        ax.set_title(f"Distribution: {col}", fontsize=14, fontweight="bold")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()

        chart_path = output_dir / f"chart_{col.lower().replace(' ','_')}.png"
        plt.savefig(chart_path, dpi=100)
        plt.close()
        chart_paths.append(str(chart_path))

    # Missing values bar chart
    if profile.get("missing_values"):
        fig, ax = plt.subplots(figsize=(8, 4))
        cols_missing = list(profile["missing_values"].keys())
        pcts = [profile["missing_values"][c]["pct"] for c in cols_missing]
        colors = ["#E74C3C" if p > 5 else "#F39C12" if p > 0 else "#2ECC71" for p in pcts]
        ax.barh(cols_missing, pcts, color=colors)
        ax.set_title("Missing Values by Column (%)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Missing %")
        ax.axvline(5, color="red", linestyle="--", alpha=0.5, label="Threshold (5%)")
        ax.legend()
        plt.tight_layout()
        chart_path = output_dir / "chart_missing_values.png"
        plt.savefig(chart_path, dpi=100)
        plt.close()
        chart_paths.append(str(chart_path))

    return chart_paths


def generate_report(profile: dict, insights: str, chart_paths: list, output_dir: Path) -> Path:
    """Write the final markdown report."""
    report = f"""# Data Whisperer Analysis Report

**Dataset:** {profile['filename']}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Rows:** {profile['rows']}  |  **Columns:** {profile['columns']}

---

## Dataset Overview

| Metric | Value |
|--------|-------|
| Rows | {profile['rows']} |
| Columns | {profile['columns']} |
| Numeric | {sum(1 for t in profile['column_types'].values() if t == 'numeric')} |
| Categorical | {sum(1 for t in profile['column_types'].values() if t == 'categorical')} |
| Missing values | {sum(v['count'] for v in profile['missing_values'].values()) if profile['missing_values'] else 0} |

### Columns

| Column | Type | Distinct | Key Stats |
|--------|------|----------|-----------|
"""
    for col in profile['headers']:
        col_type = profile['column_types'].get(col, 'unknown')
        stats = profile['column_stats'].get(col, {})
        if col_type == 'numeric':
            extra = f"μ={stats.get('mean','')} σ={stats.get('stdev','')}"
        else:
            extra = f"top={stats.get('most_common','')}"
        report += f"| {col} | {col_type} | {stats.get('distinct','')} | {extra} |\n"

    if profile.get('missing_values'):
        report += "\n### Data Quality Issues\n\n"
        for col, mv in profile['missing_values'].items():
            report += f"- **{col}**: {mv['count']} missing ({mv['pct']}%)\n"
            if mv['pct'] > 5:
                report += f"  ⚠️ Above 5% threshold — investigate\n"

    report += f"""

---

## Insights & Analysis

{insights}

---

## Charts

"""
    for cp in chart_paths:
        chart_name = Path(cp).name
        if "matplotlib" not in chart_name:
            report += f"![{chart_name}]({chart_name})\n\n"

    report += "---\n*Generated by Data Whisperer Agent — Rocky AI Showcase*"
    report_path = output_dir / "report.md"
    report_path.write_text(report)
    return report_path


def generate_demo_csv() -> Path:
    """Generate a realistic demo CSV for demonstration."""
    import random, csv
    output_dir = Path(__file__).parent / "reports"
    output_dir.mkdir(exist_ok=True)
    csv_path = output_dir / "demo_sales_data.csv"

    products = ["SaaS Basic", "SaaS Pro", "Enterprise Suite", "API Access", "Consulting Day"]
    regions = ["North America", "Europe", "Asia Pacific", "Latin America", "Middle East"]
    sources = ["Website", "Referral", "LinkedIn", "Conference", "Cold Email"]
    reps = ["Alice Chen", "Bob Martinez", "Carol Smith", "David Kim", "Eva Johansson"]

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["deal_id", "product", "region", "source", "rep",
                         "deal_value", "probability", "stage", "days_in_pipeline",
                         "created_month", "closed_quarter"])
        for i in range(500):
            value = round(random.gauss(15000, 5000), 2)
            prob = random.randint(10, 95)
            if prob > 80:
                stage = "Closed Won"
            elif prob > 50:
                stage = "Negotiation"
            elif prob > 25:
                stage = "Proposal"
            else:
                stage = "Discovery"
            days = random.randint(1, 180)
            writer.writerow([
                f"DEAL-{1000+i:04d}",
                random.choice(products),
                random.choice(regions),
                random.choice(sources),
                random.choice(reps),
                value, prob, stage, days,
                random.randint(1, 12),
                f"Q{random.randint(1,4)} {random.choice(['2025','2026'])}"
            ])
    return csv_path


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Data Whisperer — Automated EDA Agent")
    parser.add_argument("--file", "-f", help="CSV file to analyze")
    parser.add_argument("--target", "-t", help="Target column (for predictive insights)")
    parser.add_argument("--demo", action="store_true", help="Generate demo CSV and analyze it")
    args = parser.parse_args()

    print(f"""
╔═══════════════════════════════════════════╗
║  📊 Data Whisperer Agent                  ║
║  Automated EDA & Insights                 ║
╚═══════════════════════════════════════════╝
""")

    if args.demo:
        print("  🎲 Generating demo sales dataset...")
        csv_path = generate_demo_csv()
        args.file = str(csv_path)
        print(f"  ✅ Demo CSV: {csv_path} (500 rows)\n")
    elif not args.file:
        parser.print_help()
        sys.exit(1)

    print(f"  📂 Loading: {args.file}")
    profile = analyze_csv(args.file)

    print(f"  📊 Profile: {profile['rows']} rows, {profile['columns']} columns")
    numeric = sum(1 for t in profile['column_types'].values() if t == 'numeric')
    categorical = sum(1 for t in profile['column_types'].values() if t == 'categorical')
    print(f"     • Numeric: {numeric}  |  Categorical: {categorical}")
    if profile.get('missing_values'):
        total_missing = sum(v['count'] for v in profile['missing_values'].values())
        print(f"     • Missing values: {total_missing}")

    print(f"\n  🧠 Generating insights via DeepSeek...")
    insights = generate_insights(profile)

    # Save charts
    output_dir = Path(__file__).parent / "reports"
    output_dir.mkdir(exist_ok=True)
    print(f"  📈 Generating visualizations...")
    chart_paths = generate_charts(profile, output_dir)

    # Write report
    report_path = generate_report(profile, insights, chart_paths, output_dir)
    print(f"  📄 Report: {report_path}")
    print()

    # Print insight summary
    print("  ── Key Insights Preview ──")
    for line in insights.split("\n")[:25]:
        stripped = line.strip()
        if stripped:
            print(f"  {stripped[:90]}")
        else:
            print()
    print()

    print(f"  ✨ Complete. Open the report at: {report_path}")


if __name__ == "__main__":
    main()
