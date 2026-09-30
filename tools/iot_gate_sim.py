"""
Play a gate counter against the running twin — to exercise Stream 2 (IoT)
end to end without hardware.

    python tools/iot_gate_sim.py                               # HTTP, follows the crowd
    python tools/iot_gate_sim.py --rate-in 0.8 --rate-out 0.5  # fixed Poisson rates
    python tools/iot_gate_sim.py --mqtt localhost:1883 --topic cdt/gates/north

"follow" (default) reads the twin's own count from /api/snapshot and
reports entries/exits that move a gate occupancy towards it — like real
turnstiles in front of a camera, with miscounts (--miss) and people the
camera can't see (--hidden). Counts go to POST /api/iot, or to MQTT.
"""

import argparse
import json
import random
import time
import urllib.request


def get_json(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="http://localhost:8000")
    ap.add_argument("--token", help="X-IoT-Token, if the server requires one")
    ap.add_argument("--mqtt", help="HOST[:PORT]: publish to MQTT instead of HTTP")
    ap.add_argument("--topic", default="cdt/gates/sim")
    ap.add_argument("--every", type=float, default=1.0, help="seconds between reports")
    ap.add_argument("--rate-in", type=float, help="Poisson entries per second (instead of follow)")
    ap.add_argument("--rate-out", type=float, default=0.0)
    ap.add_argument("--hidden", type=float, default=0.15, help="share of people the camera misses")
    ap.add_argument("--miss", type=float, default=0.03, help="share of passages the gate misses")
    args = ap.parse_args()

    send = None
    if args.mqtt:
        import paho.mqtt.client as mqtt
        host, _, port = args.mqtt.partition(":")
        try:
            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        except AttributeError:
            client = mqtt.Client()
        client.connect(host, int(port or 1883))
        client.loop_start()
        send = lambda e, x: client.publish(args.topic, json.dumps({"entry": e, "exit": x}))
    else:
        def send(e, x):
            req = urllib.request.Request(f"{args.server}/api/iot", method="POST",
                                         data=json.dumps({"entry": e, "exit": x}).encode(),
                                         headers={"Content-Type": "application/json",
                                                  **({"X-IoT-Token": args.token} if args.token else {})})
            with urllib.request.urlopen(req, timeout=5) as r:
                return json.loads(r.read())

    occupancy = None
    poisson = lambda lam: sum(1 for _ in range(1000) if random.random() < lam / 1000)
    while True:
        if args.rate_in is not None:
            e, x = poisson(args.rate_in * args.every), poisson(args.rate_out * args.every)
        else:
            snap = get_json(f"{args.server}/api/snapshot")
            target = round(snap.get("n_agents", 0) * (1 + args.hidden))
            occupancy = target if occupancy is None else occupancy
            through = poisson(0.02 * max(target, 1))
            e = max(0, target - occupancy) + through
            x = max(0, occupancy - target) + through
            occupancy = target
            e = sum(1 for _ in range(e) if random.random() > args.miss)
            x = sum(1 for _ in range(x) if random.random() > args.miss)
        reply = send(e, x)
        print(f"entry {e:3d}  exit {x:3d}" + (f"  → twin gates: {reply}" if reply else ""))
        time.sleep(args.every)


if __name__ == "__main__":
    main()
