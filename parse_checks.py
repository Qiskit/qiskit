import json
with open("checks2.json") as f:
    d = json.load(f)
failures = [c for c in d.get("check_runs", []) if c.get("conclusion") == "failure"]
print("Total failures:", len(failures))
for c in failures:
    print("-" * 40)
    print("Name:", c["name"])
    out = c.get("output", {})
    if out:
        print("Title:", out.get("title"))
        print("Summary:", out.get("summary"))
