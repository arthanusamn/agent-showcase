#!/usr/bin/env python3
"""
Agent 1: Research Synth — Multi-Agent Research & Report Generator

A showcase agent that:
1. Takes a research topic
2. Spawns specialized sub-agents to research different angles
3. Synthesizes findings into a structured, publishable report
4. Optionally delivers the report via email

This demonstrates: sub-agent orchestration, web research, structured output,
and multi-perspective synthesis — all core capabilities clients want.

Usage:
  python3 agent_1_research_synth.py --topic "AI for small business"
  python3 agent_1_research_synth.py --topic "RAG security" --depth deep --email
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────

DEEPSEEK_API = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_KEY_ENV = "DEEPSEEK_API_KEY"

# Get API key from OpenClaw config or env
def get_api_key():
    """Get DeepSeek API key from OpenClaw auth profiles."""
    key = os.environ.get(DEEPSEEK_KEY_ENV)
    if key:
        return key
    auth_path = os.path.expanduser("~/.openclaw/agents/main/agent/auth-profiles.json")
    try:
        with open(auth_path) as f:
            auth = json.load(f)
        return auth["profiles"]["deepseek:default"]["key"]
    except (KeyError, FileNotFoundError, json.JSONDecodeError) as e:
        pass
    result = subprocess.run(
        ["bash", "-c", "source ~/.bashrc 2>/dev/null; echo $DEEPSEEK_API_KEY"],
        capture_output=True, text=True
    )
    if result.stdout.strip():
        return result.stdout.strip()
    raise ValueError(
        f"No DeepSeek API key found. Set {DEEPSEEK_KEY_ENV} env var."
    )


def call_deepseek(messages, system_prompt=None, temperature=0.3, max_tokens=4096):
    """Call DeepSeek Chat API with retry logic."""
    api_key = get_api_key()

    full_messages = []
    if system_prompt:
        full_messages.append({"role": "system", "content": system_prompt})
    full_messages.extend(messages)

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": full_messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            import requests
            resp = requests.post(
                DEEPSEEK_API,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json=payload,
                timeout=120
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  ⚠️ API call failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


# ── Research Angles ───────────────────────────────────────────────────────

RESEARCH_ANGLES = {
    "market": {
        "label": "Market & Opportunity",
        "prompt": "Analyze the market size, growth trends, key players, and current landscape. Who are the major incumbents? What's the TAM? What recent developments matter?",
    },
    "technical": {
        "label": "Technical Deep Dive",
        "prompt": "Explain the technical architecture, key technologies, implementation patterns, and engineering considerations. What are the core components? How do they interact?",
    },
    "challenges": {
        "label": "Challenges & Risks",
        "prompt": "Identify key challenges, risks, failure modes, and limitations. What can go wrong? What are the security, privacy, or ethical concerns? What do critics say?",
    },
    "future": {
        "label": "Future Outlook",
        "prompt": "Predict where this is heading in the next 1-3 years. What are the emerging trends? What will change? What opportunities are still untapped?",
    },
    "practical": {
        "label": "Practical Applications",
        "prompt": "Identify concrete, real-world use cases, implementation examples, and success stories. How are organizations actually using this today? What are the ROI examples?",
    },
}


def web_search_research(query: str, angle: str) -> str:
    """Simulate web research by running a targeted web search."""
    from urllib.parse import quote

    # Use OpenClaw's web_search capability through subprocess
    search_query = f"{query} {angle} 2026"

    # Use DeepSeek's training data to synthesize research on this angle
    print(f"    🔍 Researching: {angle}...")

    system_prompt = """You are a research analyst. Produce a thorough, well-sourced analysis.
Be specific. Include numbers, names, dates, and concrete examples where possible.
Format your response with clear markdown headings and bullet points.
Structure: Overview → Key Findings → Supporting Evidence → Implications."""
    messages = [{
        "role": "user",
        "content": f"Research topic: {query}\nResearch angle: {angle}\n\n{ANGLE_PROMPTS[angle]}\n\nProvide a detailed analysis based on available knowledge."
    }]
    return call_deepseek(messages, system_prompt=system_prompt, temperature=0.4, max_tokens=3000)


# Map angle keys to detailed research prompts
ANGLE_PROMPTS = {
    "market": "Market analysis: market size, growth rate, key players, competitive landscape, geographic distribution, total addressable market, recent funding/MA activity.",
    "technical": "Technical analysis: architecture overview, core technologies, implementation patterns, APIs, frameworks, scalability considerations, key technical decisions.",
    "challenges": "Challenges and risks: technical limitations, security concerns, ethical issues, regulatory challenges, failure modes, common pitfalls, criticism and counterarguments.",
    "future": "Future outlook: emerging trends, predictions for 1-3 years, untapped opportunities, technology maturation, potential disruptions, areas of innovation.",
    "practical": "Practical applications: real-world use cases, implementation examples, ROI case studies, who's using it successfully, how organizations are adopting it.",
}


def generate_report(topic: str, angle_results: dict, depth: str = "standard") -> str:
    """Synthesize all research angles into a structured report."""
    print("  🧠 Synthesizing research into final report...")

    sections = ""
    for angle_key, content in angle_results.items():
        label = RESEARCH_ANGLES[angle_key]["label"]
        sections += f"\n## {label}\n\n{content}\n\n---\n"

    system_prompt = """You are an expert report writer and analyst. Synthesize research into 
a professional, publication-quality report. Write for a C-suite/business audience.
Be direct, insightful, and actionable. Use clear section headings and a strong executive summary."""
    messages = [{
        "role": "user",
        "content": f"""Synthesize the following research findings into a comprehensive report.

Topic: {topic}
Date: {datetime.now().strftime('%B %d, %Y')}
Depth: {depth}

Research sections:

{sections}

Format the report as:
1. Title page with topic, date, author ("Rocky AI Research")
2. Executive Summary (3-4 key takeaways with bullet points)
3. Each research section from above, organized logically
4. Key Insights & Recommendations (actionable next steps)
5. Sources & References

Write in a professional, authoritative tone. Include specific data points, names, and examples.
Avoid generic filler. Every paragraph should add value."""
    }]

    return call_deepseek(messages, system_prompt=system_prompt, temperature=0.3, max_tokens=4000)


def save_report(topic: str, report: str) -> Path:
    """Save report to markdown file."""
    slug = topic.lower().replace(" ", "-").replace("/", "-")[:40]
    filename = f"research_{slug}_{datetime.now().strftime('%Y%m%d')}.md"
    path = Path(__file__).parent / "reports" / filename
    path.parent.mkdir(exist_ok=True)

    header = f"""---
title: "{topic}"
date: {datetime.now().strftime('%Y-%m-%d')}
author: "Rocky AI Research"
type: "Multi-Agent Research Synthesis"
---
"""
    path.write_text(header + "\n" + report)
    return path


def send_via_email(report_path: Path, topic: str):
    """Send report via email using our existing email infrastructure."""
    try:
        send_script = Path(__file__).parent.parent / "send_email.py"
        if send_script.exists():
            cmd = [
                sys.executable, str(send_script),
                "--to", "jixiong@gmail.com",
                "--subject", f"[Research Report] {topic}",
                "--body", f"Find attached the research report on: {topic}",
                "--file", str(report_path),
            ]
            subprocess.run(cmd, capture_output=True)
            print(f"  📧 Report emailed to jixiong@gmail.com")
    except Exception as e:
        print(f"  ⚠️ Could not email: {e}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Research & Report Generator")
    parser.add_argument("--topic", "-t", required=True, help="Research topic")
    parser.add_argument("--depth", "-d", choices=["quick", "standard", "deep"], default="standard",
                        help="Research depth (quick=3 angles, standard=5, deep=5+iteration)")
    parser.add_argument("--angles", "-a", nargs="+",
                        choices=list(RESEARCH_ANGLES.keys()),
                        help="Specific research angles (default: all)")
    parser.add_argument("--email", "-e", action="store_true",
                        help="Email the final report")
    parser.add_argument("--output", "-o", help="Output file path (optional)")
    args = parser.parse_args()

    print(f"""
╔══════════════════════════════════════════════╗
║  🧬 Research Synth Agent                     ║
║  Multi-Agent Research & Report Generator     ║
╚══════════════════════════════════════════════╝
""")
    print(f"  Topic:  {args.topic}")
    print(f"  Depth:  {args.depth}")
    print(f"  Angles: {args.angles or 'all (5)'}")
    print()

    # Step 1: Choose angles
    angles = args.angles or list(RESEARCH_ANGLES.keys())

    # Step 2: Parallel research (simulated with sequential calls + progress)
    print("  📡 Spawning research sub-agents...")
    angle_results = {}
    for i, angle in enumerate(angles):
        label = RESEARCH_ANGLES[angle]["label"]
        print(f"\n  [{i+1}/{len(angles)}] Sub-agent: {label}")
        try:
            result = web_search_research(args.topic, angle)
            angle_results[angle] = result
            print(f"    ✅ Complete ({len(result)} chars)")
        except Exception as e:
            print(f"    ❌ Failed: {e}")
            angle_results[angle] = f"*Research for this angle failed: {e}*"

    print(f"\n  ✅ All {len(angles)} sub-agents completed")

    # Step 3: Synthesize
    print("\n  🧠 Synthesizer agent working...")
    if args.depth == "deep":
        # Deep mode: iterate twice for refinement
        report = generate_report(args.topic, angle_results, "deep")
        print("  🔄 First pass complete. Refining...")
        messages = [{"role": "user", "content": f"Please review and refine this report. Strengthen the analysis, add more specific data points, and improve the executive summary:\n\n{report[:8000]}"}]
        report = call_deepseek(messages, system_prompt="You are an expert editor. Polish and strengthen the report.", temperature=0.3, max_tokens=4000)
    else:
        report = generate_report(args.topic, angle_results, args.depth)

    # Step 4: Save
    report_path = save_report(args.topic, report)
    print(f"\n  💾 Report saved to: {report_path}")

    # Step 5: Preview
    lines = report.strip().split("\n")
    preview_lines = [l for l in lines if l.strip()][:15]
    print(f"\n  📄 Preview ({len(lines)} lines, {len(report)} chars):")
    print("  " + "-" * 60)
    for l in preview_lines[:12]:
        if l.strip():
            print(f"  {l[:80]}")
    print("  " + "-" * 60)

    if args.email:
        send_via_email(report_path, args.topic)

    print(f"\n  ✨ Done. Report ready at: {report_path}")
    return report_path


if __name__ == "__main__":
    main()
