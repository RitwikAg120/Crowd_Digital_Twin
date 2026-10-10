"""
Stream 2 — gate counters (IoT) into the twin's data fusion.

Real gate counters report people passing in and out. They reach the twin in
one of two ways, both ending in IoTSimulator.push_real(entry, exit), which
switches the fusion from its simulated gates to the real ones:

  HTTP   POST /api/iot  {"entry": 3, "exit": 1, "gate": "north"}
         (header X-IoT-Token when Config.IOT_TOKEN is set)
  MQTT   python main.py ... --mqtt broker.local:1883 --mqtt-topic cdt/gates/#
         payload {"entry": 3, "exit": 1} or "3,1"; needs `pip install paho-mqtt`
"""

import json
from typing import Callable, Optional, Tuple


def parse_counts(payload) -> Tuple[int, int]:
    """Gate counts from a JSON object / JSON text / "entry,exit" text → (entry, exit)."""
    if isinstance(payload, (bytes, bytearray)):
        payload = payload.decode("utf-8", "replace")
    if isinstance(payload, str):
        s = payload.strip()
        if s.startswith("{"):
            payload = json.loads(s)
        else:
            a, b = (int(float(x)) for x in s.replace(";", ",").split(",")[:2])
            payload = {"entry": a, "exit": b}
    entry, exit_ = int(payload.get("entry", 0)), int(payload.get("exit", 0))
    if entry < 0 or exit_ < 0 or entry > 10000 or exit_ > 10000:
        raise ValueError("entry/exit must be counts between 0 and 10000")
    return entry, exit_


class MQTTGates:
    """Subscribes to gate-counter topics and forwards counts to `on_counts(entry, exit)`."""

    def __init__(self, broker: str, topic: str, on_counts: Callable[[int, int], None],
                 username: Optional[str] = None, password: Optional[str] = None):
        try:
            import paho.mqtt.client as mqtt
        except ImportError as e:
            raise RuntimeError("MQTT needs paho-mqtt: pip install paho-mqtt") from e
        host, _, port = broker.partition(":")
        self.host, self.port, self.topic = host, int(port or 1883), topic
        self.on_counts = on_counts
        self.received = self.rejected = 0
        try:                                               # paho-mqtt 2.x
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        except AttributeError:                             # 1.x
            self.client = mqtt.Client()
        if username:
            self.client.username_pw_set(username, password)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        client.subscribe(self.topic)
        print(f"[IoT] MQTT connected to {self.host}:{self.port}, listening on {self.topic}")

    def _on_message(self, client, userdata, msg):
        try:
            self.on_counts(*parse_counts(msg.payload))
            self.received += 1
        except Exception as e:
            self.rejected += 1
            print(f"[IoT] Ignored message on {msg.topic}: {e}")

    def start(self):
        self.client.connect_async(self.host, self.port)
        self.client.loop_start()                           # reconnects by itself

    def stop(self):
        self.client.loop_stop()
        self.client.disconnect()
