"""
Async end-to-end test: signs a request, POSTs /run_async, polls /status/<job_id>.

  # terminal 1
  PORT=5600 BYOA_SECRET=test-secret ./venv/bin/python val_json_agent.py
  # terminal 2
  BYOA_SECRET=test-secret ./venv/bin/python test_async.py ACTUAL.json GROUND_TRUTH.json [base_url]
"""
import json
import os
import sys
import time
import uuid

import requests

from byoa_adk.security import sign_request

secret = os.environ.get("BYOA_SECRET", "test-secret")
actual_path, truth_path = sys.argv[1], sys.argv[2]
base = (sys.argv[3] if len(sys.argv) > 3 else "http://localhost:5600").rstrip("/")

task_id = f"t-{uuid.uuid4().hex[:8]}"   # unique per run: replay protection
sig, ts = sign_request(secret, task_id, "g1", "data-matching")
body = {
    "task_id": task_id, "goal_id": "g1", "capability": "data-matching",
    "timeout_secs": 600, "hmac_sig": sig, "timestamp": ts,
    "inputs": {"actual": json.load(open(actual_path)),
               "ground_truth": json.load(open(truth_path))},
}
r = requests.post(f"{base}/run_async", json=body)
print("submit:", r.status_code, r.json())
r.raise_for_status()
job_id = r.json()["job_id"]

t0 = time.time()
while True:
    job = requests.get(f"{base}/status/{job_id}").json()
    print(f"  [{time.time() - t0:5.1f}s] status={job['status']}")
    if job["status"] != "running":
        break
    time.sleep(3)

if job["status"] != "done":
    sys.exit(f"FAILED: {job['error']}")
o = job["result"]["outputs"]
print(f"\nmatch={o['match']} score={o['score']} "
      f"{o['matched_count']}/{o['total_ground_truth_fields']} "
      f"llm_used={o['llm_used']} {o['llm_error']}")
for m in o["mismatches"]:
    print("BAD ", m["ground_truth_path"], repr(m["ground_truth_value"])[:40],
          "vs", m["actual_path"], repr(m["actual_value"])[:40])
for m in o["missing"]:
    print("MISS", m["ground_truth_path"], repr(m["ground_truth_value"])[:50])
print("extra:", o["extra_fields"])
