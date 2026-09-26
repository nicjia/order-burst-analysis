#!/usr/bin/env python3
"""Split the code base into parts small enough for a diff-based review, and build one branch per part.

A diff review (for example an automated multi-agent review) sees only changed lines and caps them, currently at
8,000 lines and 500 files. The code here is about 43,000 lines. For each part this script makes two commits on top
of HEAD, without touching the working tree or the index:

    review-base-N    HEAD with the part's files removed                     (tag)
    review/part-N    review-base-N with those files restored, i.e. HEAD     (branch)

so `review/part-N` compared with `review-base-N` shows exactly that part as added code, while the checkout still
holds the whole repository for context. Parts are the study groups of src_py/INDEX.md plus fixed folders (PARTS).
Documents, figures, config/*.json, archive/ and tools/ are not reviewed. Review rules are in REVIEW.md.

    python3 tools/review_parts.py            # show the plan
    python3 tools/review_parts.py --create   # (re)build the refs and a clean review worktree next to the repo
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

MAX_LINES, MAX_FILES = 7900, 500           # keep a little under the reviewer's 8,000-line cap
CODE_EXT = (".py", ".sh", ".cpp", ".hpp", ".h", ".c", ".yml", ".yaml")
SKIP = ("archive/", "tools/")
EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"

# "@Name" = every script listed under the src_py/INDEX.md section whose heading starts with Name; else a path prefix.
PARTS = [
    ("burst forecasting and the later studies (ledger 1.32-1.36)",
     ["studies/burst_forecasting/", "@Multi-day fingerprint", "@Forced flow", "@Daily retail", "@Earnings flow"]),
    ("hidden liquidity, shared modules, reversal checks (ledger 1.1-1.17, 1.21-1.23)",
     ["@Shared modules", "@Hidden liquidity", "@Figures", "@Overnight"]),
    ("referee audits, two avenues, fingerprint validation (ledger 1.18-1.20, 1.27)",
     ["@Referee audits", "@Two avenues", "@Fingerprint validation"]),
    ("program evidence, metaorders, burst information (ledger 1.24-1.26, 1.28-1.30)",
     ["@Program evidence", "@Metaorders", "@Burst information"]),
    ("P4 revisit and the legacy pipeline it audits (ledger 1.31)",
     ["@P4 revisit", "@Legacy pipeline"]),
    ("tests, C++ detector, build, examples and Hoffman2 job scripts",
     ["tests/", "src_cpp/", "Makefile", "examples/", ".github/", "hoffman2/"]),
]


def git(*args, env=None, stdin=None):
    r = subprocess.run(["git", *args], capture_output=True, text=True, env=env, input=stdin)
    if r.returncode:
        sys.exit("git %s failed:\n%s" % (" ".join(args), r.stderr))
    return r.stdout


def index_sections():
    """INDEX.md section heading -> list of src_py paths, read from HEAD."""
    out, cur = {}, None
    for line in git("show", "HEAD:src_py/INDEX.md").splitlines():
        if line.startswith("## "):
            cur = line[3:].strip()
            out[cur] = []
        elif cur and line.startswith("| `"):
            out[cur].append("src_py/" + line.split("`")[1])
    return out


def plan():
    lines = {}
    for row in git("diff", "--numstat", EMPTY_TREE, "HEAD").splitlines():
        added, _, path = row.split("\t", 2)
        if path.startswith(SKIP) or not (path.endswith(CODE_EXT) or os.path.basename(path) == "Makefile"):
            continue
        lines[path] = 0 if added == "-" else int(added)
    sections = index_sections()
    parts, taken = [], set()
    for name, rules in PARTS:
        files = []
        for rule in rules:
            if rule.startswith("@"):
                heads = [h for h in sections if h.startswith(rule[1:])]
                if len(heads) != 1:
                    sys.exit("PARTS rule %r matches %d INDEX.md sections" % (rule, len(heads)))
                files += [f for f in sections[heads[0]] if f in lines]
            else:
                files += [f for f in lines if f == rule or f.startswith(rule)]
        files = [f for f in dict.fromkeys(files) if f not in taken]
        taken.update(files)
        parts.append((name, files))
    rest = [f for f in lines if f not in taken]
    if rest:
        parts.append(("unassigned files (add them to PARTS or src_py/INDEX.md)", rest))
    return parts, lines


def show(parts, lines):
    ok = True
    print("part  files  lines  scope")
    for i, (name, files) in enumerate(parts, 1):
        n = sum(lines[f] for f in files)
        flag = "" if n <= MAX_LINES and len(files) <= MAX_FILES else "  <-- over the limit, split this part"
        ok &= not flag
        print("%4d  %5d  %5d  %s%s" % (i, len(files), n, name, flag))
    print("total %5d  %5d" % (sum(len(f) for _, f in parts), sum(lines.values())))
    return ok


def create(parts, worktree):
    if git("status", "--porcelain", "--untracked-files=no").strip():
        print("note: uncommitted changes are not included; the parts are built from HEAD")
    head = git("rev-parse", "HEAD").strip()
    tree = git("rev-parse", "HEAD^{tree}").strip()
    m = len(parts)
    refs = {}
    with tempfile.TemporaryDirectory() as tmp:
        env = dict(os.environ, GIT_INDEX_FILE=os.path.join(tmp, "index"))
        for i, (name, files) in enumerate(parts, 1):
            git("read-tree", head, env=env)
            git("update-index", "--force-remove", "-z", "--stdin", env=env, stdin="\0".join(files) + "\0")
            base_tree = git("write-tree", env=env).strip()
            base = git("commit-tree", base_tree, "-p", head, "-m",
                       "Review base %d/%d: HEAD without part %d\n\nPart %d: %s" % (i, m, i, i, name)).strip()
            part = git("commit-tree", tree, "-p", base, "-m",
                       "Review part %d/%d: %s\n\nRestores %d files. All of them are existing code under review; "
                       "see REVIEW.md." % (i, m, name, len(files))).strip()
            refs[i] = (base, part)
    # a branch checked out in a worktree cannot be moved; detach any review worktree first (it must be clean)
    wt_path = None
    for block in git("worktree", "list", "--porcelain").split("\n\n"):
        fields = dict(l.split(" ", 1) for l in block.splitlines() if " " in l)
        if fields.get("branch", "").startswith("refs/heads/review/part-"):
            if git("-C", fields["worktree"], "status", "--porcelain").strip():
                sys.exit("review worktree %s has local changes; commit or discard them first" % fields["worktree"])
            git("-C", fields["worktree"], "checkout", "-q", "--detach")
            wt_path = fields["worktree"]
    for ref in git("for-each-ref", "--format=%(refname)", "refs/heads/review/", "refs/tags/").split():
        if ref.startswith(("refs/heads/review/part-", "refs/tags/review-base-")):
            git("update-ref", "-d", ref)
    for i, (base, part) in refs.items():
        git("update-ref", "refs/tags/review-base-%d" % i, base)
        git("update-ref", "refs/heads/review/part-%d" % i, part)
    wt = Path(wt_path or worktree).resolve()
    if wt.exists():
        git("-C", str(wt), "checkout", "-q", "review/part-1")
    else:
        git("worktree", "add", "-q", str(wt), "review/part-1")
    print("\nbuilt review/part-1..%d and review-base-1..%d from %s" % (m, m, head[:12]))
    print("review worktree: %s (on review/part-1)\n" % wt)
    print("Run one review per part from that folder, for example:")
    for i in refs:
        print("  git switch review/part-%d   then review against   review-base-%d" % (i, i))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--create", action="store_true", help="build the review refs and worktree")
    ap.add_argument("--worktree", help="review worktree path (default: ../<repo>-review)")
    a = ap.parse_args()
    top = Path(git("rev-parse", "--show-toplevel").strip())
    os.chdir(top)
    parts, lines = plan()
    ok = show(parts, lines)
    if a.create:
        if not ok:
            sys.exit("not creating refs: a part is over the limit")
        create(parts, a.worktree or str(top.parent / (top.name + "-review")))


if __name__ == "__main__":
    main()
