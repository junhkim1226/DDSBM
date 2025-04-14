from collections import defaultdict
from pathlib import Path

path = Path("/home/mseok/work/DL/D3BM/src/final_results")
for result_dir in path.glob("*"):
    dic = defaultdict(list)
    mds = sorted(result_dir.glob("*.md"))
    mds = [md for md in mds if md.stem != "all"]
    for file_idx, md in enumerate(mds):
        with open(md) as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if file_idx > 0:
                    line = line.split(" | ")[1:]
                    line = " | ".join(line)
                dic[idx].append(line)

    for key, val in dic.items():
        val = " ".join(val)
        dic[key] = val

    print(result_dir / "all.md")
    with open(result_dir / "all.md", "w") as f:
        for key, val in dic.items():
            f.write(val + "\n")
