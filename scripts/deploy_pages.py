"""Publish ct2skel outputs to GitHub Pages (gh-pages branch of this repository).

    python scripts/deploy_pages.py out/s1397 [out/other ...] [--no-volume] [--no-err] [--remote origin]

1. `ct2skel publish` copies the static viewer of every output into ./site/<case>/ (plus a landing page)
2. the ./site tree is committed to an orphan `gh-pages` branch (kept in a git worktree at ./.gh-pages) and pushed
3. GitHub serves it at https://<user>.github.io/<repo>/  (enable Pages once: Settings > Pages > branch gh-pages)

Only the static viewer is deployed: saved preset poses work, live joint sliders need `ct2skel serve`.
Never deploy hospital data — only data you are allowed to publish (the bundled examples are CC BY 4.0).
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*cmd, cwd=ROOT, check=True):
    print("$", " ".join(str(c) for c in cmd))
    return subprocess.run([str(c) for c in cmd], cwd=cwd, check=check, text=True, capture_output=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outputs", nargs="+", help="ct2skel output directories to publish")
    ap.add_argument("--remote", default="origin")
    ap.add_argument("--no-volume", action="store_true")
    ap.add_argument("--no-err", action="store_true")
    ap.add_argument("--message", "-m", default=None)
    a = ap.parse_args()

    site = ROOT / "site"
    extra = (["--no-volume"] if a.no_volume else []) + (["--no-err"] if a.no_err else [])
    for out in a.outputs:
        run(sys.executable, "-m", "ct2skel", "publish", "--out", out, "--dest", site, *extra)

    wt = ROOT / ".gh-pages"
    remote_has_branch = subprocess.run(["git", "ls-remote", "--exit-code", "--heads", a.remote, "gh-pages"],
                                       cwd=ROOT, capture_output=True).returncode == 0
    if not wt.exists():
        if remote_has_branch:
            run("git", "fetch", a.remote, "gh-pages")
            run("git", "worktree", "add", wt, "gh-pages")
        else:
            run("git", "worktree", "add", "--detach", wt)
            run("git", "checkout", "--orphan", "gh-pages", cwd=wt)
            run("git", "rm", "-rf", "--quiet", ".", cwd=wt, check=False)
    # mirror ./site into the worktree (delete what is no longer published)
    for child in wt.iterdir():
        if child.name == ".git":
            continue
        shutil.rmtree(child) if child.is_dir() else child.unlink()
    for child in site.iterdir():
        dst = wt / child.name
        shutil.copytree(child, dst) if child.is_dir() else shutil.copyfile(child, dst)
    run("git", "add", "-A", cwd=wt)
    msg = a.message or f"publish {', '.join(Path(o).name for o in a.outputs)}"
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=wt).returncode == 0:
        print("nothing changed in the site")
    else:
        run("git", "commit", "-m", msg, cwd=wt)
    run("git", "push", a.remote, "gh-pages", cwd=wt)
    url = subprocess.run(["git", "remote", "get-url", a.remote], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    if "github.com" in url:
        owner_repo = url.split("github.com")[-1].strip(":/").removesuffix(".git")
        owner, repo = owner_repo.split("/")[:2]
        print(f"site: https://{owner}.github.io/{repo}/   (first time: enable Pages for branch gh-pages, or run\n"
              f"      gh api -X POST repos/{owner}/{repo}/pages -f build_type=legacy -f source[branch]=gh-pages -f source[path]=/ )")
    return 0


if __name__ == "__main__":
    sys.exit(main())
