"""Ensure docker-image adds deployment files without changing main's business code."""

import argparse
from pathlib import Path
import subprocess


DEPLOYMENT_FILES = {
    ".github/workflows/build-runpod-image.yml",
    "docs/runpod-session-guide.md",
}
DEPLOYMENT_DIRECTORIES = ("runpod-image/", "scripts/runpod/")


def check_branch(repo: Path, main_ref: str) -> tuple[str, list[str]]:
    def git(*args: str) -> bytes:
        return subprocess.check_output(["git", "-C", str(repo), *args])

    base = git("merge-base", main_ref, "HEAD").decode().strip()
    # Disable rename detection so moving a business file into a deployment
    # directory still reports its removal from the business source tree.
    changed = git("diff", "--no-renames", "--name-only", "-z", base, "HEAD").decode().split("\0")
    unexpected = [path for path in changed if path and path not in DEPLOYMENT_FILES
                  and not path.startswith(DEPLOYMENT_DIRECTORIES)]
    return base, unexpected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-ref", default="origin/main")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    base, unexpected = check_branch(repo, args.main_ref)
    print(f"Main code base: {base}")
    if unexpected:
        print("Business changes must land on main first, then be merged into docker-image:")
        for path in unexpected:
            print(f"  {path}")
        return 1
    print("Deployment branch scope: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
