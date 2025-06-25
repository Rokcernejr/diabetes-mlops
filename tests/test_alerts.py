import sys, pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import responses

from monitoring.alerts import AlertManager


@responses.activate
def test_send_alert_slack():
    webhook = "https://hooks.slack.com/services/test"
    responses.add(responses.POST, webhook, json={}, status=200)
    manager = AlertManager(slack_webhook_url=webhook)
    manager.send_alert("Test", "hello", "INFO")
    assert len(responses.calls) == 1
    assert "Test" in responses.calls[0].request.body.decode()
