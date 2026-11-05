# Your next work — 5 November 2026

The fixed experiment is complete and archived at 4a64ed4. This is a writing,
understanding and contribution-review stage. No training command is needed now.

## 1. Read the contribution assessment — about 15 minutes

Open `NOVELTY_ASSESSMENT.md`. Our possible contribution is a careful example
showing how comparison choices change conclusions. It is not the invention of
joint heat inversion or a new PINN. Journal publication remains uncertain.

## 2. Read the new manuscript sections — about 30–45 minutes

Read `INTRODUCTION_DRAFT.md`, followed by `DISCUSSION_DRAFT.md`. The Introduction
explains why we asked the question and how prior work relates. The Discussion
explains what the results support and where their reach stops.

Mark unclear wording and any statement you cannot yet explain. The existing
plain-language guides `methods.md` and `result.md` are available for reference.
Do not add claims merely because they sound impressive.

## 3. Write a short explanation in your own words — about 20 minutes

Create your own `AUTHOR_UNDERSTANDING.md` in this docs folder. Answer:

1. What specific question does our study investigate?
2. What do earlier papers already establish?
3. Give one result favoring a PINN and one favoring CV12, stating the condition
   and outcome for each.
4. Name the two most important limits of our comparison.
5. What can our completed study contribute without claiming a universal winner?

Five short paragraphs are sufficient. This is practice for explaining the work
and finding concepts we should discuss together. It is not a condition for
continuing the manuscript work.

## 4. Save the writing checkpoint in Git

From the same repository CMD window, after reading the drafts:

```bat
git add docs/METHODS_DRAFT.md docs/RESULTS_DRAFT.md docs/methods.md docs/result.md
git add docs/INTRODUCTION_DRAFT.md docs/DISCUSSION_DRAFT.md docs/NOVELTY_ASSESSMENT.md docs/NEXT_STEPS.md
git add docs/LOG.md audit/manuscript_table_sources.json audit/literature_update_2026-11-05.md
git diff --cached --stat
git commit -m "Draft research sections and document contribution limits"
git push
```

If you created your own understanding note and want it versioned, stage that
specific file too before the commit. The drafts remain local until you run the
commit and push commands. These commands preserve the frozen experiment archive.

## What comes after this checkpoint

Review the assembled argument, check references and figures, then choose a
suitable journal using its current official scope and fee policy. Prefer a route
with zero mandatory author fees. Acceptance and review times cannot be promised.
Prepare the abstract after the contribution and limitations are settled. A new
experiment, if scientifically justified, needs a separate plan and fresh data.
