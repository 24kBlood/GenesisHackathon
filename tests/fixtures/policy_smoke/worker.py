"""Benign protocol worker used by the dedicated Linux container check."""

import json
import os
import sys
import time


reset = json.loads(sys.stdin.readline())
assert os.getuid() == os.getgid() == 65532
episode_id = reset["observation"]["manifest"]["episode_id"]
print(json.dumps({"type": "submit", "findings": {"episode_id": episode_id, "findings": []}}))
if episode_id == "timeout":
    time.sleep(20)
acknowledgement = json.loads(sys.stdin.readline())
assert acknowledgement["type"] == "step"
assert acknowledgement["reward"] == 0.0
assert acknowledgement["terminated"] is True
