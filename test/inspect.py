from pathlib import Path
from tabulate import tabulate


def inspect_youtube(source_path: Path):
    headers = ["Name", "# file"]
    data = []

    gt_ = 0
    lt_ = 0

    for p in source_path.glob("*"):
        n_file = len(list(p.glob("*")))
        tm = [p.stem, n_file]
        data.append(tm)

        if n_file > 1:
            gt_ += 1
        else:
            lt_ += 1

    print(tabulate(data, headers=headers))
    print("# Number of file more than 1: ", gt_)
    print("# Number of file less than 2: ", lt_)
