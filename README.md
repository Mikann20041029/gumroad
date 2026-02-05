# Dependency Risk Watchlist (auto)

Weekly OSS repository risk watchlist dataset generator.
Outputs a ZIP to GitHub Releases.

## Output (ZIP)
- watchlist.csv
- watchlist.jsonl
- TOP_RISKY.md
- CHANGELOG.md
- SCHEMA.md
- QUALITY.txt
- SOURCES.md

## Secrets (optional)
- DEEPSEEK_API_KEY: If set, generates concise "why" + "action" with evidence.
(GITHUB_TOKEN is provided automatically by GitHub Actions)

## Run
Actions → "Build Watchlist Dataset" → Run workflow
