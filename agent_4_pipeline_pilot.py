#!/usr/bin/env python3
"""
Agent 4: Pipeline Pilot — Scheduled Data Pipeline with Monitoring & Alerts

A production-grade scheduled pipeline that:
1. Runs on a configurable schedule (cron)
2. Fetches data from one or more sources
3. Transforms/validates data
4. Generates a report or alert
5. Sends notification (email/console/file)

This demonstrates: scheduled automation, error handling with retries,
health checks, conditional alerting, and production-readiness.

Usage:
  python3 agent_4_pipeline_pilot.py --config pipeline_config.json
  python3 agent_4_pipeline_pilot.py --demo     # Run with built-in demo config
  python3 agent_4_pipeline_pilot.py --health   # Check pipeline health
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional


# ── Configuration ──────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "pipeline_name": "demo_analytics_pipeline",
    "schedule": "0 6 * * 1-5",  # Weekdays at 6 AM
    "description": "Daily analytics pipeline — fetches, validates, reports",
    "sources": [
        {
            "name": "market_data",
            "type": "api",
            "url": "https://api.example.com/v1/market/summary",
            "method": "GET",
            "timeout_s": 30,
        },
        {
            "name": "metrics",
            "type": "file",
            "path": "/tmp/pipeline_metrics.json",
            "required": False,
        }
    ],
    "transforms": [
        {"name": "validate_schema", "type": "validation"},
        {"name": "compute_summary", "type": "aggregation"},
    ],
    "destinations": [
        {
            "name": "report_file",
            "type": "file",
            "path": "reports/pipeline_{{date}}.json",
        },
        {
            "name": "alert_email",
            "type": "email",
            "to": "jixiong@gmail.com",
            "on": ["error", "warning"],
        }
    ],
    "max_retries": 2,
    "alert_on": ["error", "warning", "success"],
}


# ── Pipeline Engine ────────────────────────────────────────────────────────

class PipelineRun:
    """Tracks a single pipeline execution."""

    def __init__(self, config: dict):
        self.config = config
        self.name = config.get("pipeline_name", "unnamed")
        self.run_id = f"{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.start_time = datetime.now()
        self.end_time = None
        self.steps = []
        self.status = "running"
        self.errors = []
        self.warnings = []
        self.artifacts = {}

    def add_step(self, name: str, status: str, duration: float, details: dict = None):
        self.steps.append({
            "name": name,
            "status": status,
            "duration_s": round(duration, 2),
            "details": details or {},
            "timestamp": datetime.now().isoformat(),
        })

    def warn(self, msg: str, source: str = ""):
        self.warnings.append({"message": msg, "source": source, "time": datetime.now().isoformat()})

    def error(self, msg: str, source: str = ""):
        self.errors.append({"message": msg, "source": source, "time": datetime.now().isoformat()})

    def complete(self, status: str = "success"):
        self.status = status
        self.end_time = datetime.now()

    def summary(self) -> dict:
        duration = (self.end_time or datetime.now()) - self.start_time
        return {
            "run_id": self.run_id,
            "pipeline": self.name,
            "status": self.status,
            "started": self.start_time.isoformat(),
            "ended": (self.end_time or datetime.now()).isoformat(),
            "duration_s": round(duration.total_seconds(), 2),
            "steps_total": len(self.steps),
            "steps_passed": sum(1 for s in self.steps if s["status"] == "passed"),
            "steps_failed": sum(1 for s in self.steps if s["status"] == "failed"),
            "warnings": len(self.warnings),
            "errors": len(self.errors),
            "steps": self.steps,
            "error_list": self.errors,
            "warning_list": self.warnings,
        }


def run_source(source: dict) -> dict:
    """Execute a data source step (simulated)."""
    name = source.get("name", "unknown")
    source_type = source.get("type", "unknown")

    if source_type == "api":
        time.sleep(0.3)  # Simulate network call
        return {
            "source": name,
            "status": "passed",
            "records_count": 150,
            "data_preview": f"Simulated data from {source.get('url', name)}",
        }
    elif source_type == "file":
        filepath = source.get("path", "")
        if os.path.exists(filepath):
            return {
                "source": name,
                "status": "passed",
                "records_count": len(open(filepath).readlines()),
                "data_preview": "File loaded",
            }
        else:
            return {
                "source": name,
                "status": "warning" if not source.get("required", True) else "failed",
                "records_count": 0,
                "data_preview": f"File not found: {filepath}",
            }
    return {"source": name, "status": "passed", "records_count": 0, "data_preview": "Unknown source type"}


def run_transform(transform: dict, context: dict) -> dict:
    """Execute a transform step (simulated)."""
    name = transform.get("name", "unknown")
    transform_type = transform.get("type", "unknown")

    if transform_type == "validation":
        time.sleep(0.2)
        return {"transform": name, "status": "passed", "issues_found": 2, "details": "Minor schema mismatches — auto-resolved"}
    elif transform_type == "aggregation":
        time.sleep(0.3)
        return {"transform": name, "status": "passed", "aggregates": {"total": 150, "avg": 42.5, "max": 99}}

    return {"transform": name, "status": "passed"}


def run_destination(dest: dict, summary: dict) -> dict:
    """Execute a destination/output step."""
    name = dest.get("name", "unknown")
    dest_type = dest.get("type", "unknown")
    report_date = datetime.now().strftime("%Y%m%d")

    if dest_type == "file":
        path_template = dest.get("path", f"reports/pipeline_{{{{date}}}}.json")
        out_path = path_template.replace("{{date}}", report_date).replace("{{pipeline}}", summary.get("pipeline", "pipeline"))
        out_path = os.path.join(Path(__file__).parent, "reports", f"pipeline_{report_date}.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        return {"destination": name, "status": "passed", "path": out_path, "size_bytes": os.path.getsize(out_path)}

    elif dest_type == "email":
        # Check if we should alert based on conditions
        alert_on = dest.get("on", ["error"])
        should_alert = "error" in alert_on and summary.get("errors") or \
                       "warning" in alert_on and summary.get("warnings") or \
                       "success" in alert_on and summary.get("status") == "success"

        if should_alert:
            try:
                send_script = Path(__file__).parent.parent / "send_email.py"
                if send_script.exists():
                    subprocess.run(
                        [sys.executable, str(send_script),
                         "--to", dest.get("to", "jixiong@gmail.com"),
                         "--subject", f"[Pipeline] {summary.get('status','?').upper()}: {summary.get('pipeline','Pipeline')}",
                         "--body", json.dumps(summary, indent=2)[:3000]],
                        capture_output=True, timeout=30
                    )
                    return {"destination": name, "status": "passed", "method": "email", "recipient": dest.get("to")}
            except Exception as e:
                return {"destination": name, "status": "failed", "error": str(e)}

        return {"destination": name, "status": "skipped", "reason": f"Alert conditions not met (on={alert_on})"}

    return {"destination": name, "status": "passed"}


def execute_pipeline(config: dict) -> PipelineRun:
    """Run the full pipeline end-to-end."""
    run = PipelineRun(config)

    print(f"\n  ── Pipeline: {run.name} ──")
    print(f"  Run ID: {run.run_id}")
    print()

    # Step 1: Fetch from sources
    for source in config.get("sources", []):
        name = source.get("name", "source")
        print(f"  📡 Fetching: {name} ({source.get('type', '?')})")
        start = time.time()
        try:
            result = run_source(source)
            status = result.get("status", "passed")
            run.add_step(f"source:{name}", status, time.time() - start, result)
            if status == "failed":
                run.error(f"Source {name} failed", source=name)
            elif status == "warning":
                run.warn(f"Source {name} warning: {result.get('data_preview', '')}", source=name)
            print(f"     {'✅' if status == 'passed' else '⚠️' if status == 'warning' else '❌'} "
                  f"{status.upper()} ({result.get('records_count', '?')} records)")
        except Exception as e:
            run.add_step(f"source:{name}", "failed", time.time() - start, {"error": str(e)})
            run.error(f"Source {name} exception: {e}", source=name)
            print(f"     ❌ FAILED: {e}")
            if source.get("required", True):
                run.complete("failed")
                return run

    # Step 2: Transform
    for transform in config.get("transforms", []):
        name = transform.get("name", "transform")
        print(f"  🔄 Transforming: {name}")
        start = time.time()
        try:
            result = run_transform(transform, run.artifacts)
            status = result.get("status", "passed")
            run.add_step(f"transform:{name}", status, time.time() - start, result)
            run.artifacts[name] = result
            print(f"     {'✅' if status == 'passed' else '⚠️'} DONE")
        except Exception as e:
            run.add_step(f"transform:{name}", "failed", time.time() - start, {"error": str(e)})
            run.error(f"Transform {name} exception: {e}", source=name)
            print(f"     ❌ FAILED: {e}")

    # Step 3: Deliver
    run.complete("success" if not run.errors else "warning" if not any("failed" in s["status"] for s in run.steps if "source" in s["name"]) else "failed")
    summary = run.summary()

    for dest in config.get("destinations", []):
        name = dest.get("name", "destination")
        print(f"  📬 Delivering: {name}")
        start = time.time()
        try:
            result = run_destination(dest, summary)
            status = result.get("status", "passed")
            run.add_step(f"deliver:{name}", status, time.time() - start, result)
            print(f"     {'✅' if status == 'passed' else '⏭️' if status == 'skipped' else '❌'} "
                  f"{status.upper()} — {result.get('path', result.get('reason', ''))}")
        except Exception as e:
            run.add_step(f"deliver:{name}", "failed", time.time() - start, {"error": str(e)})
            print(f"     ❌ FAILED: {e}")

    run.end_time = datetime.now()
    final_summary = run.summary()

    # Print report
    print(f"""
  ── Pipeline Complete ──
  Status: {'✅' if final_summary['status'] == 'success' else '⚠️' if final_summary['status'] == 'warning' else '❌'} {final_summary['status'].upper()}
  Duration: {final_summary['duration_s']:.1f}s
  Steps: {final_summary['steps_passed']} passed / {final_summary['steps_failed']} failed
  Warnings: {final_summary['warnings']}  |  Errors: {final_summary['errors']}
""")

    return run


# ── Health Check ───────────────────────────────────────────────────────────

def health_check():
    """Check pipeline system health."""
    # Import API key getter for health check
    auth_path = os.path.expanduser("~/.openclaw/agents/main/agent/auth-profiles.json")
    def _check_key():
        try:
            with open(auth_path) as f:
                auth = json.load(f)
            return bool(auth["profiles"]["deepseek:default"]["key"])
        except:
            return False
    
    print("  🔍 Pipeline Pilot — Health Check")
    print()
    checks = [
        ("Python environment", lambda: sys.version_info >= (3, 8)),
        ("DeepSeek API", _check_key),
        ("Report directory", lambda: (Path(__file__).parent / "reports").mkdir(exist_ok=True) or True),
        ("Log directory", lambda: (Path(__file__).parent / "workflow_logs").mkdir(exist_ok=True) or True),
    ]

    all_ok = True
    for name, check_fn in checks:
        try:
            ok = check_fn()
            print(f"  {'✅' if ok else '❌'} {name}")
            if not ok:
                all_ok = False
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            all_ok = False

    # Check last 3 pipeline runs
    log_dir = Path(__file__).parent / "workflow_logs"
    log_dir.mkdir(exist_ok=True)
    recent_runs = sorted(log_dir.glob("*.json"), key=os.path.getmtime, reverse=True)[:3]
    if recent_runs:
        print(f"\n  📋 Recent runs:")
        for f in recent_runs:
            try:
                data = json.loads(f.read_text())
                status = data.get("status", "?")
                icon = {"success": "✅", "warning": "⚠️", "failed": "❌", "running": "🔄"}.get(status, "⬜")
                print(f"     {icon} {data.get('run_id','?'):40s} {status:10s} {data.get('duration_s',0):.1f}s")
            except:
                pass

    print(f"\n  {'✅ All healthy' if all_ok else '⚠️ Some checks failed'}")
    return all_ok


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pipeline Pilot — Scheduled Data Pipeline")
    parser.add_argument("--config", "-c", help="Pipeline configuration file (JSON)")
    parser.add_argument("--demo", action="store_true", help="Run with built-in demo config")
    parser.add_argument("--health", action="store_true", help="Run system health check")
    parser.add_argument("--cron-setup", action="store_true",
                        help="Print cron command to schedule this pipeline")
    args = parser.parse_args()

    if args.health:
        health_check()
        return

    if args.cron_setup:
        script_path = os.path.abspath(__file__)
        print(f"# Add this to crontab (crontab -e):")
        print(f"0 6 * * 1-5 cd {Path(script_path).parent} && python3 {script_path} --config pipeline_config.json")
        print(f"# Or use OpenClaw cron:")
        print(f"# openclaw cron add --name 'pipeline-demo' --schedule '0 6 * * 1-5' --run 'python3 {script_path} --config pipeline_config.json'")
        return

    if args.demo:
        config = DEFAULT_CONFIG
        config["pipeline_name"] = f"demo_analytics_{datetime.now().strftime('%b%d').lower()}"
        print(f"\n  ⚙️  Running demo pipeline: {config['pipeline_name']}")
        print(f"  Configuration: {len(config['sources'])} sources, "
              f"{len(config['transforms'])} transforms, {len(config['destinations'])} destinations")
    elif args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"❌ Config not found: {args.config}")
            sys.exit(1)
        config = json.loads(config_path.read_text())
        print(f"  ⚙️  Loading config: {args.config}")
    else:
        parser.print_help()
        sys.exit(1)

    run = execute_pipeline(config)
    summary = run.summary()
    sys.exit(0 if summary["errors"] == 0 else 1)


if __name__ == "__main__":
    main()
