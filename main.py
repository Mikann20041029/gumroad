import os
import json
import time
import csv
import zipfile
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from dateutil.parser import isoparse
import requests

GITHUB_API = "https://api.github.com"

OUT_DIR = "out"
DATA_DIR = "data"
LAST_WATCHLIST_PATH = os.path.join(DATA_DIR, "last_watchlist.csv")
LAST_RUN_PATH = os.path.join(DATA_DIR, "last_run.json")

# ---- Tunables (safe defaults) ----
LANGUAGES = ["python", "javascript", "typescript", "go", "rust"]
PER_LANG = 200                  # 5 * 200 = ~1000 seeds
MIN_STARS = 500                 # avoid tiny repos
MAX_REPOS_TOTAL = 1000          # hard cap processed per run
REQUEST_SLEEP_SEC = 0.12        # be polite

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASE = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)


def gh_headers() -> dict:
    token = os.getenv("GITHUB_TOKEN", "").strip()
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "watchlist-bot",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def gh_get(url: str, params: dict | None = None) -> dict:
    r = requests.get(url, headers=gh_headers(), params=params, timeout=45)
    if r.status_code >= 400:
        raise RuntimeError(f"GitHub API error {r.status_code}: {r.text[:500]}")
    return r.json()


def iso_to_dt(s: str) -> datetime:
    return isoparse(s).astimezone(timezone.utc)


def days_since(dt: datetime) -> int:
    delta = utc_now() - dt
    return max(0, int(delta.total_seconds() // 86400))


def safe_int(x) -> int:
    try:
        return int(x)
    except Exception:
        return 0


def safe_str(x) -> str:
    return "" if x is None else str(x)


def clamp(n: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, n))


@dataclass
class RepoRow:
    # Identity
    repo_full_name: str
    html_url: str
    description: str
    homepage: str
    topics: str

    # Status / policy-like signals
    archived: int
    disabled: int
    is_fork: int
    license_spdx: str

    # Basic stats
    language: str
    stars: int
    forks: int
    watchers: int
    open_issues: int
    size_kb: int

    # Timestamps
    created_at: str
    updated_at: str
    pushed_at: str
    days_since_last_commit: int
    days_since_last_update: int

    # Decision outputs
    risk_score: int
    action: str  # AVOID / CAUTION / OK
    risk_flags: str
    why: str

    # Meta
    updated_at_run: str


def search_seeds() -> list[dict]:
    """
    Build seeds from GitHub Search (stars-desc per language).
    Return the *search items themselves* to avoid extra /repos calls (rate limit safe).
    """
    seeds: list[dict] = []
    seen = set()

    for lang in LANGUAGES:
        q = f"language:{lang} stars:>={MIN_STARS} archived:false fork:false"
        page = 1
        per_page = 100
        collected = 0

        while collected < PER_LANG:
            data = gh_get(f"{GITHUB_API}/search/repositories", params={
                "q": q,
                "sort": "stars",
                "order": "desc",
                "per_page": per_page,
                "page": page,
            })
            items = data.get("items", [])
            if not items:
                break

            for it in items:
                full = it.get("full_name", "")
                if not full or full in seen:
                    continue
                seen.add(full)
                seeds.append(it)  # store item dict (not full_name string)
                collected += 1
                if collected >= PER_LANG:
                    break

            page += 1
            time.sleep(REQUEST_SLEEP_SEC)

    return seeds[:MAX_REPOS_TOTAL]



def compute_risk(repo: dict) -> tuple[int, list[str], str]:
    """
    Deterministic scoring (0-100) + flags + action.
    The score is explainable and stable; AI only adds a compact why (optional).
    """
    flags: list[str] = []
    score = 0

    stars = safe_int(repo.get("stargazers_count"))
    open_issues = safe_int(repo.get("open_issues_count"))
    archived = 1 if repo.get("archived") else 0
    disabled = 1 if repo.get("disabled") else 0
    is_fork = 1 if repo.get("fork") else 0

    license_obj = repo.get("license") or {}
    license_spdx = safe_str(license_obj.get("spdx_id", "")).strip()

    pushed_at = repo.get("pushed_at") or repo.get("updated_at") or utc_now().isoformat()
    updated_at = repo.get("updated_at") or pushed_at

    dslc = days_since(iso_to_dt(pushed_at))
    dslu = days_since(iso_to_dt(updated_at))

    # Hard red flags
    if archived:
        score += 80
        flags.append("ARCHIVED")
    if disabled:
        score += 90
        flags.append("DISABLED")
    if is_fork:
        score += 10
        flags.append("FORK")

    # License clarity matters for risk/compliance
    if not license_spdx or license_spdx.upper() in {"NOASSERTION", "OTHER"}:
        score += 10
        flags.append("LICENSE_UNCLEAR")

    # Staleness (commit activity)
    if dslc >= 365:
        score += 45
        flags.append("NO_COMMIT_365D")
    elif dslc >= 180:
        score += 30
        flags.append("NO_COMMIT_180D")
    elif dslc >= 90:
        score += 18
        flags.append("NO_COMMIT_90D")

    # Update staleness (repo metadata changes)
    if dslu >= 365:
        score += 10
        flags.append("NO_UPDATE_365D")
    elif dslu >= 180:
        score += 6
        flags.append("NO_UPDATE_180D")

    # Issue pressure normalized by adoption
    denom = max(1, stars)
    issues_per_1k = (open_issues * 1000.0) / denom
    if issues_per_1k >= 80:
        score += 25
        flags.append("ISSUE_PRESSURE_HIGH")
    elif issues_per_1k >= 40:
        score += 15
        flags.append("ISSUE_PRESSURE_MED")

    # Low adoption can raise risk for production usage
    if stars < 1000:
        score += 5
        flags.append("LOW_ADOPTION")

    score = clamp(score, 0, 100)

    # Action mapping for buyers
    if score >= 80:
        action = "AVOID"
    elif score >= 50:
        action = "CAUTION"
    else:
        action = "OK"

    return score, flags, action


def deepseek_enrich(rows: list[RepoRow]) -> dict[str, dict]:
    """
    For higher-risk entries only, ask DeepSeek for:
    - why (100-160 chars, must cite at least one numeric fact)
    - action override (AVOID/CAUTION/OK) if strongly justified
    Strict JSON only.

    If no API key, return empty dict and we use rule-based why.
    """
    if not DEEPSEEK_API_KEY:
        return {}

    target = [r for r in rows if r.risk_score >= 50]
    if not target:
        return {}

    def call(batch: list[RepoRow]) -> dict[str, dict]:
        items = []
        for r in batch:
            items.append({
                "repo_full_name": r.repo_full_name,
                "stars": r.stars,
                "open_issues": r.open_issues,
                "days_since_last_commit": r.days_since_last_commit,
                "archived": r.archived,
                "disabled": r.disabled,
                "license_spdx": r.license_spdx,
                "risk_flags": r.risk_flags,
                "action": r.action,
            })

        prompt = (
            "Return STRICT JSON only: an array of objects with keys: "
            "repo_full_name, why, action.\n"
            "Rules:\n"
            "- why: 100-160 characters in English.\n"
            "- Must cite at least one numeric fact from input (e.g., stars, open_issues, days_since_last_commit).\n"
            "- No marketing, no links, no recommendations beyond action.\n"
            "- action must be one of: AVOID, CAUTION, OK.\n"
            "- If you keep the provided action, repeat it.\n"
            f"Input:\n{json.dumps(items, ensure_ascii=False)}"
        )

        payload = {
            "model": DEEPSEEK_MODEL,
            "messages": [
                {"role": "system", "content": "Return strict JSON only. No prose."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }

        r = requests.post(
            f"{DEEPSEEK_BASE}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=90,
        )
        if r.status_code >= 400:
            raise RuntimeError(f"DeepSeek API error {r.status_code}: {r.text[:500]}")

        content = r.json()["choices"][0]["message"]["content"].strip()
        start = content.find("[")
        end = content.rfind("]")
        if start == -1 or end == -1 or end <= start:
            raise RuntimeError(f"DeepSeek returned non-JSON: {content[:200]}")
        arr = json.loads(content[start:end+1])

        out: dict[str, dict] = {}
        for obj in arr:
            k = obj.get("repo_full_name")
            why = obj.get("why")
            act = obj.get("action")
            if isinstance(k, str) and k:
                out[k] = {
                    "why": why if isinstance(why, str) else "",
                    "action": act if isinstance(act, str) else "",
                }
        return out

    out: dict[str, dict] = {}
    batch_size = 40
    for i in range(0, len(target), batch_size):
        part = call(target[i:i+batch_size])
        out.update(part)
        time.sleep(0.4)

    return out


def rule_based_why(r: RepoRow) -> str:
    parts = []
    if r.archived:
        parts.append("Archived repository")
    if r.disabled:
        parts.append("Disabled repository")
    if r.license_spdx == "" or r.license_spdx.upper() in {"NOASSERTION", "OTHER"}:
        parts.append("License unclear")

    if r.days_since_last_commit >= 365:
        parts.append(f"No commits for {r.days_since_last_commit} days")
    elif r.days_since_last_commit >= 180:
        parts.append(f"Inactive for {r.days_since_last_commit} days")
    elif r.days_since_last_commit >= 90:
        parts.append(f"Low activity ({r.days_since_last_commit} days since last commit)")
    else:
        parts.append(f"Active (last commit {r.days_since_last_commit}d ago)")

    # Normalize issue pressure hint
    parts.append(f"{r.open_issues} open issues, {r.stars}★")

    s = "; ".join(parts)
    if len(s) > 160:
        s = s[:160]
    return s


def load_last_watchlist() -> dict[str, dict]:
    if not os.path.exists(LAST_WATCHLIST_PATH):
        return {}
    out = {}
    with open(LAST_WATCHLIST_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out[row.get("repo_full_name", "")] = row
    return out


def write_schema(fields: list[str]):
    path = os.path.join(OUT_DIR, "SCHEMA.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Schema\n\n")
        for col in fields:
            f.write(f"- `{col}`\n")
    return path


def write_quality(rows: list[RepoRow]):
    path = os.path.join(OUT_DIR, "QUALITY.txt")
    missing_why = sum(1 for r in rows if not r.why.strip())
    missing_license = sum(1 for r in rows if not r.license_spdx.strip())
    archived = sum(1 for r in rows if r.archived)
    disabled = sum(1 for r in rows if r.disabled)

    scores = [r.risk_score for r in rows]
    if scores:
        p50 = sorted(scores)[len(scores)//2]
        p90 = sorted(scores)[int(len(scores)*0.9)]
        p99 = sorted(scores)[int(len(scores)*0.99)]
    else:
        p50 = p90 = p99 = 0

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"rows={len(rows)}\n")
        f.write(f"missing_why={missing_why}\n")
        f.write(f"missing_license={missing_license}\n")
        f.write(f"archived={archived}\n")
        f.write(f"disabled={disabled}\n")
        f.write(f"risk_score_p50={p50}\n")
        f.write(f"risk_score_p90={p90}\n")
        f.write(f"risk_score_p99={p99}\n")
        f.write(f"generated_at={utc_now().isoformat()}\n")
    return path


def write_sources():
    path = os.path.join(OUT_DIR, "SOURCES.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Sources\n\n")
        f.write("- GitHub REST API (public repository metadata)\n")
        f.write("- Data is derived from publicly available repo fields; no private data.\n")
    return path


def write_top_risky(rows: list[RepoRow], top_n: int = 200):
    path = os.path.join(OUT_DIR, "TOP_RISKY.md")
    top = sorted(rows, key=lambda r: (r.risk_score, r.stars), reverse=True)[:top_n]
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Top Risky Repositories (Top {top_n})\n\n")
        for r in top:
            f.write(f"- **{r.repo_full_name}** | score={r.risk_score} | action={r.action} | {r.risk_flags}\n")
            f.write(f"  - {r.html_url}\n")
            f.write(f"  - why: {r.why}\n")
    return path


def write_changelog(rows: list[RepoRow], last_map: dict[str, dict]):
    path = os.path.join(OUT_DIR, "CHANGELOG.md")

    new_high = []
    worsened = []
    improved = []

    for r in rows:
        prev = last_map.get(r.repo_full_name)
        if not prev:
            if r.risk_score >= 50:
                new_high.append(r)
            continue
        prev_score = safe_int(prev.get("risk_score", 0))
        if r.risk_score - prev_score >= 15:
            worsened.append((prev_score, r))
        elif prev_score - r.risk_score >= 15:
            improved.append((prev_score, r))

    new_high = sorted(new_high, key=lambda x: x.risk_score, reverse=True)[:50]
    worsened = sorted(worsened, key=lambda t: (t[1].risk_score - t[0]), reverse=True)[:50]
    improved = sorted(improved, key=lambda t: (t[0] - t[1].risk_score), reverse=True)[:50]

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Changelog ({utc_now().date().isoformat()})\n\n")

        f.write("## New entries (score>=50)\n")
        if not new_high:
            f.write("- (none)\n")
        else:
            for r in new_high:
                f.write(f"- {r.repo_full_name} score={r.risk_score} action={r.action} flags={r.risk_flags}\n")

        f.write("\n## Worsened (>= +15)\n")
        if not worsened:
            f.write("- (none)\n")
        else:
            for prev_score, r in worsened:
                f.write(f"- {r.repo_full_name} {prev_score} -> {r.risk_score} action={r.action} flags={r.risk_flags}\n")

        f.write("\n## Improved (>= -15)\n")
        if not improved:
            f.write("- (none)\n")
        else:
            for prev_score, r in improved:
                f.write(f"- {r.repo_full_name} {prev_score} -> {r.risk_score} action={r.action} flags={r.risk_flags}\n")

    return path


def write_outputs(rows: list[RepoRow]):
    # CSV / JSONL fields
    fields = list(asdict(rows[0]).keys()) if rows else []
    csv_path = os.path.join(OUT_DIR, "watchlist.csv")
    jsonl_path = os.path.join(OUT_DIR, "watchlist.jsonl")

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    schema_path = write_schema(fields)
    quality_path = write_quality(rows)
    sources_path = write_sources()
    top_path = write_top_risky(rows)
    last_map = load_last_watchlist()
    changelog_path = write_changelog(rows, last_map)

    # ZIP pack
    zip_path = os.path.join(OUT_DIR, "watchlist_pack.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in [csv_path, jsonl_path, schema_path, quality_path, sources_path, top_path, changelog_path]:
            z.write(p, arcname=os.path.basename(p))

    # Persist state for next diff
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(csv_path, "r", encoding="utf-8") as src, open(LAST_WATCHLIST_PATH, "w", encoding="utf-8") as dst:
        dst.write(src.read())
    with open(LAST_RUN_PATH, "w", encoding="utf-8") as f:
        json.dump({"ran_at": utc_now().isoformat(), "rows": len(rows)}, f, ensure_ascii=False, indent=2)

    return zip_path


def run():
    ensure_dirs()

    seeds = search_seeds()
    rows: list[RepoRow] = []

    for i, it in enumerate(seeds[:MAX_REPOS_TOTAL], start=1):
        data = it  # use Search API item directly (rate limit safe)

        full = data.get("full_name", "")
        if not full:
            continue

    # Fields available in search item
        stars = safe_int(data.get("stargazers_count"))
        forks = safe_int(data.get("forks_count"))
        open_issues = safe_int(data.get("open_issues_count"))
        default_branch = data.get("default_branch") or "main"
        pushed_at = data.get("pushed_at") or data.get("updated_at") or utc_now().isoformat()

        pushed_dt = iso_to_dt(pushed_at)
        dslc = days_since(pushed_dt)

    # Deterministic risk (search-item-safe)
        risk_score, flags = compute_risk(stars, open_issues, dslc)

    # NOTE: search items do not include subscribers_count reliably
        watchers = stars


       


        # Compute deterministic risk
        score, flags, action = compute_risk(data)

        license_obj = data.get("license") or {}
        license_spdx = safe_str(license_obj.get("spdx_id", "")).strip()
        topics_list = data.get("topics") or []
        if isinstance(topics_list, list):
            topics = ",".join([safe_str(t).strip() for t in topics_list if safe_str(t).strip()])[:800]
        else:
            topics = ""

        pushed_at = data.get("pushed_at") or data.get("updated_at") or utc_now().isoformat()
        updated_at = data.get("updated_at") or pushed_at

        row = RepoRow(
            repo_full_name=full,
            html_url=safe_str(data.get("html_url", "")),
            description=safe_str(data.get("description", ""))[:300],
            homepage=safe_str(data.get("homepage", ""))[:300],
            topics=topics,

            archived=1 if data.get("archived") else 0,
            disabled=1 if data.get("disabled") else 0,
            is_fork=1 if data.get("fork") else 0,
            license_spdx=license_spdx if license_spdx else "",

            language=safe_str(data.get("language", "")),
            stars=safe_int(data.get("stargazers_count")),
            forks=safe_int(data.get("forks_count")),
            watchers=watchers,
            open_issues=safe_int(data.get("open_issues_count")),
            size_kb=safe_int(data.get("size")),

            created_at=safe_str(data.get("created_at", "")),
            updated_at=safe_str(updated_at),
            pushed_at=safe_str(pushed_at),
            days_since_last_commit=days_since(iso_to_dt(pushed_at)),
            days_since_last_update=days_since(iso_to_dt(updated_at)),

            risk_score=score,
            action=action,
            risk_flags="|".join(flags),
            why="",
            updated_at_run=utc_now().isoformat(),
        )
        rows.append(row)

        if i % 50 == 0:
            time.sleep(REQUEST_SLEEP_SEC)

    # Stable ordering: risk desc, then stars desc
    rows.sort(key=lambda r: (r.risk_score, r.stars), reverse=True)

    # AI enrich (optional)
    enrich = deepseek_enrich(rows)
    for r in rows:
        e = enrich.get(r.repo_full_name)
        if e:
            why = (e.get("why") or "").strip()
            act = (e.get("action") or "").strip().upper()
            if why:
                r.why = why[:180]
            else:
                r.why = rule_based_why(r)
            if act in {"AVOID", "CAUTION", "OK"}:
                r.action = act
        else:
            r.why = rule_based_why(r)

    zip_path = write_outputs(rows)
    print(f"OK: rows={len(rows)} zip={zip_path}")


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1:] and sys.argv[1] or "run"
    if cmd == "run":
        run()
    else:
        raise SystemExit(f"Unknown command: {cmd}")
