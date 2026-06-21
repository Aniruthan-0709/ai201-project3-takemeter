# TakeMeter — Planning Document
**Project:** Fine-tuned discourse quality classifier for IPL cricket community  
**Model:** DistilBERT fine-tuned on r/Cricket IPL posts and comments

---

## Community

**Chosen community:** r/Cricket, filtered to IPL (Indian Premier League) content.

The IPL subreddit discourse is an ideal classification target for several reasons. First, it is extraordinarily active during the IPL season — thousands of posts and comments per match day — meaning the discourse spans the full quality spectrum from deep tactical analysis to pure emotional noise. Second, IPL fans are unusually vocal about the distinction between good and bad takes. Phrases like "classic hot take," "actually decent analysis," and "just a reaction post" are part of the community's own vocabulary, which means the label distinctions we are drawing are ones the community itself already makes informally. Third, the subject matter — franchise cricket — naturally generates all four discourse types we are classifying: statistics-heavy analysis of player performance, bold opinions about teams and captains, real-time emotional reactions to match events, and open questions seeking community opinion. No single type dominates, which means the classification problem is genuinely interesting rather than trivially imbalanced.

---

## Labels

We use four labels. Each post or comment receives exactly one label.

### `analysis`
**Definition:** The post makes a structured argument supported by specific statistics, historical comparisons, or tactical observations. The evidence cited is specific and verifiable — real player names, realistic numbers, match contexts — rather than vague or impressionistic.

**Example 1:**  
> "Bumrah's death-over economy of 8.4 this IPL is actually his worst since 2019, which is concerning given MI's dependency on him in the final 4 overs."

**Example 2:**  
> "The data is clear: teams batting second have won 62% of matches at Wankhede this season, so MI's decision to bowl first after winning the toss is consistently baffling."

---

### `hot_take`
**Definition:** A bold, confident opinion stated without real supporting evidence. The post asserts rather than argues — the claim may be true or false, but the post provides no data or reasoning to back it up, and often uses charged or provocative language.

**Example 1:**  
> "Rohit Sharma is completely finished as a T20 batter and MI should have released him two seasons ago."

**Example 2:**  
> "RCB will never win the IPL with Kohli as the centerpiece. His ego ruins team balance every single season."

---

### `reaction`
**Definition:** An immediate emotional response to a specific match moment or event. The post expresses a feeling in the moment — excitement, frustration, shock, joy — with little to no argument or claim being made. Typically written during or immediately after a match.

**Example 1:**  
> "WHAT A SIX BY HARDIK OH MY GOD THIS MAN IS UNREAL"

**Example 2:**  
> "We were so close. So so close. I'm not okay right now."

---

### `discussion`
**Definition:** A question or conversation prompt that invites the community to share views. The post makes no strong claim of its own — it is opening a debate, asking for predictions, or seeking opinions rather than asserting anything.

**Example 1:**  
> "If you could pick one player from any franchise for a do-or-die final, who would it be and why?"

**Example 2:**  
> "Do you think the impact player rule has improved IPL cricket or made it worse? Genuine question."

---

## Hard Edge Cases

### Edge case 1: `analysis` vs `hot_take`
The hardest boundary in this taxonomy is between a post that looks like analysis but is actually a hot take dressed up with one statistic.

**Example ambiguous post:**  
> "Virat Kohli is overrated in T20s — his strike rate of 130 is below what you'd expect from a top-order batter at this level."

This post cites a number (strike rate of 130) but the framing is accusatory and the stat is cherry-picked to support a conclusion rather than part of a genuine argument.

**Decision rule:** If the evidence cited would genuinely support the claim even if you removed the opinion framing, label it `analysis`. If the stat is decorative — present to sound credible but not doing real argumentative work — label it `hot_take`. A single cherry-picked stat with no context or comparison is `hot_take`. A stat placed in historical or comparative context (e.g. "his worst since 2019" or "below the league average of 138") is `analysis`.

### Edge case 2: `reaction` vs `hot_take`
Short, emotionally charged posts that also make a claim are hard to place.

**Example ambiguous post:**  
> "Dhoni is washed, that stumping proved it."

This is written in the heat of the moment (reaction energy) but also makes a claim without evidence (hot take framing).

**Decision rule:** If the post is clearly triggered by a specific live event and the "claim" is just emotional venting rather than a considered position, label it `reaction`. If the claim is stated as a general truth about a player or team that goes beyond the moment, label it `hot_take`. The test: would this post make sense posted 3 days after the match? If yes, it's `hot_take`. If it only makes sense in-the-moment, it's `reaction`.

---

## Data Collection Plan

**Source:** Posts and comments scraped from r/Cricket, filtered to IPL-related content using team names, player names, and tournament keywords (IPL, T20 league, auction, etc.). Both top-level posts and top comments from high-engagement threads were collected to ensure variety in length and style.

**Target distribution:** 50 examples per label, 200 total. Equal distribution was chosen deliberately to avoid the model learning a majority-class shortcut. The project requirement states at least 20% per label — our 25% per label exceeds this.

**If a label is underrepresented:** If after collection a label had fewer than 40 examples, the plan was to supplement by specifically searching for that post type. For example, if `analysis` was underrepresented, we would search IPL match threads for long-form comments with statistics. In practice, `reaction` posts were the most abundant and `analysis` the least, which was anticipated.

**Split:**
- Train: 140 examples (70%)
- Validation: 30 examples (15%)
- Test: 30 examples (15%)

Stratified splitting will be used to maintain the 25% per label ratio across all three sets.

---

## Evaluation Metrics

**Primary metric: Macro F1**  
We use macro F1 (the unweighted average of per-class F1 scores) as our primary metric because our dataset is balanced and we care equally about performance on all four labels. A model that performs well on `reaction` and `hot_take` but fails on `analysis` and `discussion` would be useless for the actual task, even if its overall accuracy looks acceptable. Macro F1 penalises this unevenness.

**Secondary metric: Per-class precision and recall**  
We report precision and recall separately for each label. This matters because the failure modes are asymmetric:
- Low recall on `analysis` means we're missing real analysis posts, which is bad if the classifier is used to surface quality content.
- Low precision on `hot_take` means we're mislabelling other posts as hot takes, which is bad if the classifier is used to flag or moderate content.

Understanding which direction errors go in tells us more than accuracy alone.

**Tertiary: Confusion matrix**  
A full 4x4 confusion matrix lets us see which label pairs are most confusable, which directly informs what the model actually learned vs what we intended.

**Baseline comparison:** We compare our fine-tuned DistilBERT model against a zero-shot Groq (llama-3.3-70b-versatile) baseline on the same held-out test set. The baseline tells us whether fine-tuning actually helped.

---

## Definition of Success

**Minimum threshold (acceptable for submission):**
- Macro F1 ≥ 0.65 on the test set
- No individual class F1 below 0.50
- Fine-tuned model matches or outperforms the zero-shot baseline on macro F1

**Target threshold (genuinely useful classifier):**
- Macro F1 ≥ 0.75 on the test set
- All class F1 scores above 0.65
- The confusion matrix shows that errors are mostly between adjacent labels (e.g. `analysis` vs `hot_take`) rather than distant ones (e.g. `reaction` vs `analysis`)

**What would make this deployable in a real community tool:**  
A macro F1 of 0.75+ with no class below 0.65 would make this classifier useful as a soft tagging tool — for example, automatically surfacing `analysis` posts in a weekly digest or flagging `reaction` posts during live match threads. We would not deploy it as a moderation tool (i.e. removing content) at any threshold given the subjectivity of the task and the small training set.

---

## AI Tool Plan

### Label stress-testing
Before annotating, we gave Claude our four label definitions and the two edge case descriptions and asked it to generate 10 posts that sit at the boundary between `analysis` and `hot_take`, and 10 between `reaction` and `hot_take`. Several of the generated boundary posts were genuinely hard to classify, which led us to sharpen the decision rules now documented in the Hard Edge Cases section above — specifically the "decorative stat" rule for analysis vs hot_take and the "3-day test" for reaction vs hot_take.

### Annotation assistance
We used Claude (claude-sonnet-4-6) to pre-label batches of scraped posts using the taxonomy definitions as the prompt. Each batch of 10 posts was sent with the full label definitions and decision rules, and Claude returned a JSON array with label and confidence score per post. Every pre-labeled example was reviewed manually against the taxonomy before being accepted. Posts where the pre-label conflicted with our own reading were re-labeled by hand. This is disclosed in the README.

### Failure analysis
After generating test set predictions from both models, we will feed the list of wrong predictions to Claude and ask it to identify systematic error patterns — for example, "the model consistently confuses short analysis posts with hot takes" or "all reaction misclassifications involve posts that don't include explicit emotional language." We will verify each identified pattern manually by reading the examples ourselves before including it in the evaluation report. Patterns suggested by the AI that we cannot confirm by reading the data will not be reported.

---

## Stretch Features

*(To be updated if stretch features are attempted)*

None planned at this stage. This document will be updated before starting any stretch work.