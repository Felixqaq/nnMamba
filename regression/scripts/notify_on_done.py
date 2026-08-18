"""Watch run_all_rq summary and email (or desktop-notify) when all 10 finish.

Email path uses Gmail SMTP. Credentials are read from local files so no secret
is passed on the command line or seen by anyone but you:
  ~/.rq_smtp_user   (optional; defaults to the recipient below)
  ~/.rq_smtp_pass   (your Gmail App Password; chmod 600)
If no password file exists, falls back to a desktop notification via notify-send.
"""
from __future__ import annotations
import json
import smtplib
import subprocess
import time
from email.message import EmailMessage
from pathlib import Path

REG = Path(__file__).resolve().parents[1]
SUMMARY = REG / "train_log" / "run_all_rq" / "summary.json"
RECIPIENT = "felixchang2010@gmail.com"
POLL_SECONDS = 30

CONFIGS = [
    "config.rq1.normal_v_abnormal.image.yaml",
    "config.rq1.normal_v_abnormal.fusion.yaml",
    "config.rq2a.angle_3class.image.yaml",
    "config.rq2a.angle_3class.fusion.yaml",
    "config.rq2b.angle_binary.image.yaml",
    "config.rq2b.angle_binary.fusion.yaml",
    "config.rq2c.angle_reg.image.yaml",
    "config.rq2c.angle_reg.fusion.yaml",
    "config.rq3.oi_emphysema.image.yaml",
    "config.rq3.oi_emphysema.fusion.yaml",
]


def all_done(summary: dict) -> bool:
    return all(
        summary.get(c, {}).get("status") in {"done", "failed"} for c in CONFIGS
    )


def build_report(summary: dict) -> tuple[str, str]:
    lines, ok, fail = [], 0, 0
    for c in CONFIGS:
        info = summary.get(c, {})
        status = info.get("status", "pending")
        ok += status == "done"
        fail += status == "failed"
        lines.append(f"  {status:8} {c}  uuid={info.get('uuid')}")
    subject = f"[nnMamba RQ] all runs finished: {ok} ok, {fail} failed"
    body = (
        f"RQ1/2/3 two-model matrix (10 runs) finished.\n\n"
        f"OK={ok}  FAILED={fail}\n\n" + "\n".join(lines) +
        f"\n\nSummary: {SUMMARY}\n"
    )
    return subject, body


def send_email(subject: str, body: str) -> bool:
    pass_file = Path.home() / ".rq_smtp_pass"
    if not pass_file.exists():
        return False
    user_file = Path.home() / ".rq_smtp_user"
    user = (user_file.read_text().strip() if user_file.exists() else RECIPIENT)
    password = pass_file.read_text().strip()

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = RECIPIENT
    msg.set_content(body)
    with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as smtp:
        smtp.starttls()
        smtp.login(user, password)
        smtp.send_message(msg)
    return True


def desktop_notify(subject: str, body: str) -> None:
    try:
        subprocess.run(["notify-send", subject, body[:400]], check=False)
    except FileNotFoundError:
        pass


def main() -> None:
    while True:
        if SUMMARY.exists():
            try:
                summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                summary = {}
            if all_done(summary):
                subject, body = build_report(summary)
                emailed = False
                try:
                    emailed = send_email(subject, body)
                except Exception as exc:  # noqa: BLE001
                    body += f"\n[email error] {exc}\n"
                desktop_notify(subject, body)
                print(subject)
                print("emailed" if emailed else "email skipped (no ~/.rq_smtp_pass); desktop-notified")
                print(body)
                return
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
