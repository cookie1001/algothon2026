# Algothon 2026 Starter Code

Starter code for the Susquehanna x UNSW FinTech Society Algothon 2026 - the seventh year of Australia's first student-led algorithmic trading hackathon.

Full rules, scoring, schedule, and submission details live on the **[Algothon 2026 Wiki](https://wiki.algothon.au/)** - this README only covers what's in this repo and how to run it. If anything here ever seems to disagree with the wiki, the wiki is correct.

## What's in this repo

| File | What it's for |
| :--- | :--- |
| `teamName.py` | Boilerplate for your algorithm - implement `getMyPosition(prcSoFar)` here. Only renamed to `<YourTeamName>.py` at submission time (see below). |
| `eval.py` | The official evaluation script - scores the last 250 days of whatever `prices.txt` you give it. |
| `prices.txt` | Current stage's price data. |
| `requirements-dev.txt` | The exact package set available at grading time, for setting up a local environment that matches the sandbox. **Do not include this in your submission.** |

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # macOS/Linux
pip install -r requirements-dev.txt
```

1. Implement `getMyPosition(prcSoFar)` in `teamName.py`. No need to rename anything or touch `eval.py` while developing - `eval.py` imports from `teamName` by default.
2. Run `python eval.py` to backtest locally.

## Submitting

Only when you're ready to submit: copy `teamName.py` to `<YourTeamName>.py` (matching your registered team name) and zip it up - `eval.py` and `prices.txt` are not part of the submission, only your algorithm file (and `requirements.txt`, if you used extra packages). See the [Submission Guide](https://wiki.algothon.au/submission/) for exact packaging requirements. Submit through the [live leaderboard](https://www.algothon.au/leaderboard).

## Questions

Post in the questions forum on our Discord - moderators are there to help.
