import os
import logging
from typing import Dict, Any

import requests

logger = logging.getLogger(__name__)


class AlertManager:
    def __init__(self, slack_webhook_url: str | None = None):
        self.slack_webhook_url = slack_webhook_url or os.getenv("SLACK_WEBHOOK_URL")

    def send_alert(self, title: str, message: str, severity: str = "INFO"):
        """Send alert via email/slack/etc"""
        alert_msg = f"[{severity}] {title}: {message}"
        logger.warning(alert_msg)

        if self.slack_webhook_url:
            try:
                requests.post(
                    self.slack_webhook_url,
                    json={"text": alert_msg},
                    timeout=5,
                )
            except Exception as e:  # pragma: no cover - log only
                logger.error("Failed to send Slack alert: %s", e)
        if severity == "CRITICAL" and not self.slack_webhook_url:
            print(f"🚨 CRITICAL ALERT: {alert_msg}")

    def check_model_performance(self, metrics: Dict[str, Any]):
        """Alert if model performance degrades"""
        auc = metrics.get("auc", 0)

        if auc < 0.6:
            self.send_alert(
                "Model Performance Degraded",
                f"Model AUC dropped to {auc:.3f}",
                "CRITICAL",
            )
        elif auc < 0.7:
            self.send_alert(
                "Model Performance Warning", f"Model AUC is {auc:.3f}", "WARNING"
            )
