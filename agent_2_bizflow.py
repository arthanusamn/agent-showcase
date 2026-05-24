#!/usr/bin/env python3
"""
Agent 2: BizFlow — End-to-End Business Workflow Automation Agent

A showcase agent that orchestrates a complete business process:
1. Monitors an inbox/source for incoming requests
2. Classifies and routes to appropriate handlers
3. Executes tools (send emails, scrape data, generate docs)
4. Reports completion with audit trail

This demonstrates: tool-use, API integration, workflow orchestration,
conditional branching, error handling, and audit logging.

Usage:
  python3 agent_2_bizflow.py --demo          # Simulated demo workflow
  python3 agent_2_bizflow.py --topic "..."   # Full research + workflow demo
  python3 agent_2_bizflow.py --live           # Live with real APIs
"""

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from enum import Enum


# ── API Setup ──────────────────────────────────────────────────────────────

DEEPSEEK_API = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"


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
    result = subprocess.run(
        ["bash", "-c", "source ~/.bashrc 2>/dev/null; echo $DEEPSEEK_API_KEY"],
        capture_output=True, text=True
    )
    if result.stdout.strip():
        return result.stdout.strip()
    raise ValueError("No DeepSeek API key found. Set DEEPSEEK_API_KEY env var.")


def llm_call(messages, system=None, temp=0.3, max_tokens=4096, json_mode=False):
    """Call DeepSeek with the OpenAI-compatible API."""
    import requests
    full = []
    if system:
        full.append({"role": "system", "content": system})
    full.extend(messages)

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": full,
        "temperature": temp,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    for attempt in range(3):
        try:
            resp = requests.post(
                DEEPSEEK_API,
                headers={
                    "Authorization": f"Bearer {get_api_key()}",
                    "Content-Type": "application/json"
                },
                json=payload,
                timeout=120
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                raise


# ── Workflow Engine ────────────────────────────────────────────────────────

class WorkflowStep:
    """Represents a single step in a business workflow."""

    def __init__(self, name, agent_type, description, inputs=None):
        self.name = name
        self.agent_type = agent_type  # classify, extract, research, compose, deliver
        self.description = description
        self.inputs = inputs or {}
        self.result = None
        self.status = "pending"  # pending, running, done, failed
        self.error = None
        self.duration = 0

    def to_dict(self):
        return {
            "name": self.name,
            "agent_type": self.agent_type,
            "description": self.description,
            "status": self.status,
            "error": self.error,
            "duration_s": round(self.duration, 2),
            "result_preview": str(self.result)[:200] if self.result else None,
        }


class WorkflowRun:
    """Tracks an entire workflow execution with audit log."""

    def __init__(self, topic: str):
        self.topic = topic
        self.steps: list[WorkflowStep] = []
        self.context = {}
        self.start_time = datetime.now()
        self.run_id = self.start_time.strftime("%Y%m%d_%H%M%S")

    def add_step(self, step: WorkflowStep):
        self.steps.append(step)
        return step

    def log_path(self) -> Path:
        log_dir = Path(__file__).parent / "workflow_logs"
        log_dir.mkdir(exist_ok=True)
        return log_dir / f"workflow_{self.run_id}.json"

    def summary(self) -> dict:
        total = len(self.steps)
        done = sum(1 for s in self.steps if s.status == "done")
        failed = sum(1 for s in self.steps if s.status == "failed")
        duration = (datetime.now() - self.start_time).total_seconds()
        return {
            "run_id": self.run_id,
            "topic": self.topic,
            "started": self.start_time.isoformat(),
            "steps_total": total,
            "steps_done": done,
            "steps_failed": failed,
            "duration_s": round(duration, 2),
            "steps": [s.to_dict() for s in self.steps],
        }

    def _save_ckpt(self):
        """Write checkpoint after each step."""
        self.log_path().write_text(json.dumps(self.summary(), indent=2))


# ── Business Workflow Templates ────────────────────────────────────────────

BUSINESS_WORKFLOWS = {
    "customer_onboarding": {
        "label": "Customer Onboarding Automation",
        "description": "New customer signs up → verify details → create account → send welcome → schedule training",
        "icon": "🚀",
    },
    "support_triage": {
        "label": "Support Ticket Triage & Response",
        "description": "Ticket arrives → classify urgency → draft response → route to right team → follow up",
        "icon": "🎫",
    },
    "lead_enrichment": {
        "label": "Lead Enrichment Pipeline",
        "description": "Raw lead → enrich with web data → score → route to sales → schedule follow-up",
        "icon": "🎯",
    },
    "report_generation": {
        "label": "Automated Report Generation",
        "description": "Raw data → analyze → visualize → write narrative → format report → deliver",
        "icon": "📊",
    },
    "invoice_processing": {
        "label": "Invoice Processing & Approval",
        "description": "Invoice received → extract fields → validate → route for approval → schedule payment",
        "icon": "📄",
    },
}


def build_workflow(workflow_type: str, topic: str) -> WorkflowRun:
    """Construct a workflow based on type."""
    run = WorkflowRun(topic)

    if workflow_type == "customer_onboarding":
        run.add_step(WorkflowStep("classify", "classify",
            "Classify customer type and required onboarding track"))
        run.add_step(WorkflowStep("verify", "extract",
            "Verify customer details against provided data"))
        run.add_step(WorkflowStep("create_account", "compose",
            "Generate account setup instructions and welcome materials"))
        run.add_step(WorkflowStep("notify", "deliver",
            "Send welcome email with next steps and training schedule"))

    elif workflow_type == "lead_enrichment":
        run.add_step(WorkflowStep("extract", "extract",
            "Extract key fields from raw lead data"))
        run.add_step(WorkflowStep("research", "research",
            "Research company/contact online for enrichment"))
        run.add_step(WorkflowStep("score", "classify",
            "Score lead quality and assign priority tier"))
        run.add_step(WorkflowStep("route", "deliver",
            "Route to appropriate sales team with context summary"))

    elif workflow_type == "report_generation":
        run.add_step(WorkflowStep("analyze", "extract",
            "Analyze input data and identify key patterns"))
        run.add_step(WorkflowStep("outline", "classify",
            "Create report outline and structure"))
        run.add_step(WorkflowStep("write", "compose",
            "Write full report narrative with data insights"))
        run.add_step(WorkflowStep("format", "deliver",
            "Format and deliver final report"))

    else:  # generic research workflow
        run.add_step(WorkflowStep("research", "research",
            "Research the topic from multiple angles"))
        run.add_step(WorkflowStep("organize", "classify",
            "Organize findings into structured categories"))
        run.add_step(WorkflowStep("compose", "compose",
            "Compose final deliverable"))
        run.add_step(WorkflowStep("deliver", "deliver",
            "Format and deliver output"))

    return run


# ── Agent Executors ────────────────────────────────────────────────────────

def run_agent_classify(step: WorkflowStep, context: dict, topic: str) -> dict:
    """Classification agent — categorizes and routes."""
    system = "You are a classification agent in a business workflow. Categorize the input precisely."
    prompt = f"""Topic: {topic}
Task: {step.description}

Classify the following into JSON format:
{{
    "category": "one of: urgent, standard, low-priority",
    "confidence": <0-1>,
    "reasoning": "...",
    "suggested_action": "..."
}}"""
    result = llm_call([{"role": "user", "content": prompt}], system=system, json_mode=True)
    parsed = json.loads(result)
    step.result = parsed
    return parsed


def run_agent_extract(step: WorkflowStep, context: dict, topic: str) -> dict:
    """Extraction agent — pulls structured data from input."""
    system = "You are a data extraction agent. Extract structured fields precisely."
    prompt = f"""Topic: {topic}
Task: {step.description}

Extract key data into JSON format:
{{
    "fields_found": ["field1", "field2"],
    "structured_data": {{}},
    "confidence": <0-1>,
    "missing_fields": [],
    "next_steps": "..."
}}"""
    result = llm_call([{"role": "user", "content": prompt}], system=system, json_mode=True)
    parsed = json.loads(result)
    step.result = parsed
    return parsed


def run_agent_research(step: WorkflowStep, context: dict, topic: str) -> dict:
    """Research agent — deep-dive into topic."""
    system = "You are a research agent. Deliver thorough, accurate findings."
    prompt = f"""Topic: {topic}
Task: {step.description}

Research the topic and produce structured findings as JSON:
{{
    "key_findings": ["finding1", "finding2", "finding3"],
    "insights": "...",
    "sources_consulted": ["source1", "source2"],
    "confidence": <0-1>,
    "recommendation": "..."
}}"""
    result = llm_call([{"role": "user", "content": prompt}], system=system, json_mode=True)
    parsed = json.loads(result)
    step.result = parsed
    return parsed


def run_agent_compose(step: WorkflowStep, context: dict, topic: str) -> dict:
    """Composition agent — generates output content."""
    system = "You are a composition agent. Generate clear, professional content."
    prompt = f"""Topic: {topic}
Task: {step.description}
Context so far: {json.dumps(context, indent=2)}

Generate the deliverable and return as JSON:
{{
    "output_type": "email|report|summary|doc",
    "subject": "...",
    "body_preview": "...",
    "key_messages": ["msg1", "msg2"],
    "tone": "professional",
    "word_count": <number>
}}"""
    result = llm_call([{"role": "user", "content": prompt}], system=system, json_mode=True)
    parsed = json.loads(result)
    step.result = parsed
    return parsed


def run_agent_deliver(step: WorkflowStep, context: dict, topic: str) -> dict:
    """Delivery agent — formats and would send the output."""
    system = "You are a delivery agent. Format and route deliverables."
    prompt = f"""Topic: {topic}
Task: {step.description}
Context: {json.dumps(context, indent=2)}

Generate delivery instructions as JSON:
{{
    "delivery_method": "email|api|file|dashboard",
    "recipient": "...",
    "format": "markdown|html|pdf",
    "confirmation": "...",
    "audit_log_entry": "..."
}}"""
    result = llm_call([{"role": "user", "content": prompt}], system=system, json_mode=True)
    parsed = json.loads(result)
    step.result = parsed
    return parsed


# ── Workflow Executor ─────────────────────────────────────────────────────

AGENT_ROUTER = {
    "classify": run_agent_classify,
    "extract": run_agent_extract,
    "research": run_agent_research,
    "compose": run_agent_compose,
    "deliver": run_agent_deliver,
}


def execute_workflow(run: WorkflowRun, verbose: bool = True) -> dict:
    """Execute all steps in the workflow sequentially."""
    print(f"\n  ⚙️  Executing {len(run.steps)} steps...")

    for i, step in enumerate(run.steps):
        label = f"[{i+1}/{len(run.steps)}]"
        icon = {
            "classify": "🏷️", "extract": "🔍", "research": "📡",
            "compose": "✍️", "deliver": "📬"
        }.get(step.agent_type, "⚡")

        print(f"  {label} {icon} {step.name}: {step.description}")
        step.status = "running"

        start = time.time()
        try:
            handler = AGENT_ROUTER[step.agent_type]
            result = handler(step, run.context, run.topic)
            step.status = "done"
            run.context[f"step_{step.name}"] = result
            if verbose:
                print(f"       ✅ Done → {str(result)[:100]}...")
        except Exception as e:
            step.status = "failed"
            step.error = str(e)
            print(f"       ❌ Failed: {e}")

        step.duration = time.time() - start
        run._save_ckpt()

    return run.summary()


# ── Run Summary ────────────────────────────────────────────────────────────

def print_summary(summary: dict):
    """Pretty-print workflow summary."""
    status_color = {
        "done": "✅", "failed": "❌", "running": "🔄", "pending": "⏳"
    }
    print(f"""
╔══════════════════════════════════════════╗
║  📊 Workflow Complete                     ║
╚══════════════════════════════════════════╝""")
    print(f"  Run ID:    {summary['run_id']}")
    print(f"  Topic:     {summary['topic']}")
    print(f"  Duration:  {summary['duration_s']:.1f}s")
    print(f"  Steps:     {summary['steps_done']}/{summary['steps_total']} done")
    print()
    print(f"  {'Step':<20} {'Status':<10} {'Duration':<10} {'Agent Type':<15}")
    print(f"  {'─'*55}")
    for step in summary['steps']:
        status = status_color.get(step['status'], '⬜')
        err = f" — {step['error'][:40]}" if step['error'] else ""
        print(f"  {step['name']:<20} {status:<10} {step['duration_s']:.1f}s{'':<5} {step['agent_type']:<15}{err}")

    log_path = Path(__file__).parent / "workflow_logs" / f"workflow_{summary['run_id']}.json"
    print(f"\n  📝 Audit log: {log_path}")
    print()


# ── Demo Mode ──────────────────────────────────────────────────────────────

def run_demo():
    """Run a guided demo showing all workflow types."""
    print(f"""
╔══════════════════════════════════════════════════════╗
║  🏭 BizFlow — Business Workflow Automation Agent     ║
║  End-to-end workflow orchestration with audit trail  ║
╚══════════════════════════════════════════════════════╝
""")

    print("  Available workflow templates:")
    for key, wf in BUSINESS_WORKFLOWS.items():
        print(f"  {wf['icon']} {wf['label']:35s} — {wf['description'][:60]}...")
    print()

    # Run lead enrichment
    topic = "Enterprise SaaS company evaluating AI agents for customer onboarding"
    print(f"  Selected: Lead Enrichment Pipeline")
    print(f"  Topic: {topic}")
    print()

    run = build_workflow("lead_enrichment", topic)
    summary = execute_workflow(run)
    print_summary(summary)

    print(f"""
  💡 What this demonstrates to clients:
     • Multi-step workflow orchestration
     • Different specialized sub-agents (classify, extract, research, compose, deliver)
     • Structured data passing between steps
     • Error handling and checkpoint logging
     • Full audit trail for compliance

  🔧 Real-world implementation would connect to:
     • Gmail/Outlook for email processing
     • CRM APIs (HubSpot, Salesforce) for lead data
     • Slack/Teams for notifications
     • Composio for 1000+ app integrations
""")

    return summary


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BizFlow — Business Workflow Automation Agent")
    parser.add_argument("--demo", action="store_true",
                        help="Run guided demo across workflow types")
    parser.add_argument("--topic", "-t", default="",
                        help="Topic for the workflow run")
    parser.add_argument("--workflow", "-w",
                        choices=list(BUSINESS_WORKFLOWS.keys()) + ["research"],
                        default="lead_enrichment",
                        help="Workflow template to use")
    parser.add_argument("--live", action="store_true",
                        help="Use live API connections (Composio, etc.)")
    args = parser.parse_args()

    if args.demo:
        run_demo()
        return

    topic = args.topic or f"Demo: {BUSINESS_WORKFLOWS[args.workflow]['label']}"
    run = build_workflow(args.workflow, topic)
    summary = execute_workflow(run)
    print_summary(summary)

    # Save final summary
    out_path = Path(__file__).parent / "workflow_logs" / f"workflow_{summary['run_id']}.json"
    print(f"  Complete. Full audit log at: {out_path}\n")


if __name__ == "__main__":
    main()
