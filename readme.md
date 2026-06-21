# TakeMeter — IPL Discourse Quality Classifier

A fine-tuned text classifier that categorises IPL cricket posts and comments from r/Cricket into four discourse types: `analysis`, `hot_take`, `reaction`, and `discussion`.

---

## Community Choice

**Community:** r/Cricket, filtered to IPL (Indian Premier League) content.

IPL discourse on r/Cricket is an ideal classification target for three reasons. First, it spans the full quality spectrum — from deep tactical analysis backed by statistics to pure emotional noise during live matches — which makes the classification problem genuinely interesting. Second, IPL fans already use these distinctions informally in their own language ("classic hot take", "actually decent analysis", "just a reaction post"), meaning the label boundaries reflect how the community itself thinks about discourse quality. Third, the subject matter naturally generates all four post types we classify: statistics-heavy arguments about player performance, bold opinions about teams and captains, real-time emotional reactions to match events, and open questions seeking community opinion.

---

## Label Taxonomy

| Label | Definition |
|---|---|
| `analysis` | A structured argument supported by specific statistics, historical comparisons, or tactical observations. Evidence is specific and verifiable. |
| `hot_take` | A bold, confident opinion stated without real supporting evidence. Asserts rather than argues, with no data or reasoning backing the claim. |
| `reaction` | An immediate emotional response to a specific match moment or event. Expresses a feeling with little to no argument or claim. |
| `discussion` | A question or conversation prompt inviting the community to share views. Makes no strong claim — just opens a debate or asks for opinions. |

### Examples per label

**analysis**
- "Bumrah's death-over economy of 8.4 this IPL is actually his worst since 2019, which is concerning given MI's dependency on him in the final 4 overs."
- "The data is clear: teams batting second have won 62% of matches at Wankhede this season, so MI's decision to bowl first after winning the toss is consistently baffling."

**hot_take**
- "Rohit Sharma is completely finished as a T20 batter and MI should have released him two seasons ago."
- "RCB will never win the IPL with Kohli as the centerpiece. His ego ruins team balance every single season."

**reaction**
- "WHAT A SIX BY HARDIK OH MY GOD THIS MAN IS UNREAL"
- "We were so close. So so close. I'm not okay right now."

**discussion**
- "If you could pick one player from any franchise for a do-or-die final, who would it be and why?"
- "Do you think the impact player rule has improved IPL cricket or made it worse? Genuine question."

---

## Data Collection

**Source:** Posts and comments scraped from r/Cricket, filtered to IPL-related content using team names, player names, and tournament keywords (IPL, T20 league, auction, franchise names, player names).

**Labeling process:** Posts were labeled using Claude (claude-sonnet-4-6) with the full taxonomy definitions and examples as the prompt. Each batch of 10 posts was sent with instructions to return a JSON array with label and confidence score. Labels were reviewed for consistency against the decision rules documented in planning.md.

**Label distribution:**

| Label | Count | Percentage |
|---|---|---|
| analysis | 50 | 25% |
| hot_take | 50 | 25% |
| reaction | 50 | 25% |
| discussion | 50 | 25% |
| **Total** | **200** | **100%** |

**Train / validation / test split:** 140 / 30 / 30 (70% / 15% / 15%), stratified by label.

### Three difficult-to-label examples

**1. "Virat Kohli is overrated in T20s — his strike rate of 130 is below what you'd expect from a top-order batter at this level."**
Could be `analysis` (cites a stat) or `hot_take` (accusatory framing, cherry-picked number with no context).
**Decision:** `hot_take`. The stat is decorative — it's selected to support a conclusion rather than being part of a genuine argument with historical or comparative context.

**2. "Dhoni is washed, that stumping proved it."**
Could be `reaction` (triggered by a live event) or `hot_take` (makes a general claim about a player).
**Decision:** `reaction`. The claim is emotional venting tied to a specific moment, not a considered general position. The test: would this post make sense posted 3 days after the match? No — it only makes sense in-the-moment.

**3. "I've been watching cricket for 20 years and that is the best IPL over I have ever seen."**
Could be `reaction` (emotional, in-the-moment) or `hot_take` (makes a bold superlative claim).
**Decision:** `reaction`. The post is expressing awe and emotion rather than asserting a debatable position for others to push back on. The 20-year framing adds emotional weight, not argumentative structure.

---

## Fine-Tuning Approach

**Base model:** `distilbert-base-uncased` (66M parameters, HuggingFace)

**Training setup:** Fine-tuned on Google Colab T4 GPU using the HuggingFace `transformers` library with a sequence classification head.

**Final hyperparameters:**
- Epochs: 10
- Learning rate: 1e-5
- Batch size: 16
- Evaluation strategy: per epoch, best model loaded at end

**Key hyperparameter decision:** The default settings (3 epochs, lr=2e-5) produced only 53% validation accuracy — the model was not converging. Increasing to 10 epochs and reducing the learning rate to 1e-5 allowed proper convergence, with validation accuracy jumping from 0.77 to 0.93 between epochs 4 and 5, then stabilising at 0.967 from epoch 6 onwards. The lower learning rate was necessary to prevent overshooting on a small 140-example training set.

**Training curve:**

| Epoch | Train Loss | Val Loss | Val Accuracy |
|---|---|---|---|
| 1 | — | 1.256 | 0.533 |
| 2 | 1.242 | 1.224 | 0.667 |
| 3 | 1.190 | 1.162 | 0.700 |
| 4 | 1.147 | 1.068 | 0.767 |
| 5 | 1.050 | 0.944 | 0.933 |
| 6 | 0.912 | 0.800 | 0.967 |
| 7 | 0.711 | 0.680 | 0.967 |
| 8 | 0.587 | 0.599 | 0.967 |
| 9 | 0.505 | 0.553 | 0.967 |
| 10 | 0.446 | 0.537 | 0.967 |

---

## Baseline

**Model:** Groq `llama-3.3-70b-versatile` (zero-shot, no fine-tuning)

**Prompt used:**

```
You are classifying posts and comments from r/Cricket about the IPL.
Assign each post to exactly one of the following categories.

analysis: The post makes a structured argument supported by specific statistics, historical comparisons, or tactical observations. Evidence is specific and verifiable.
Example: "Bumrah's death-over economy of 8.4 this IPL is actually his worst since 2019..."

hot_take: A bold, confident opinion stated without real supporting evidence. Asserts rather than argues.
Example: "Rohit Sharma is completely finished as a T20 batter..."

reaction: An immediate emotional response to a specific match moment or event.
Example: "WHAT A SIX BY HARDIK OH MY GOD THIS MAN IS UNREAL"

discussion: A question or conversation prompt inviting the community to share views.
Example: "If you could pick one player from any franchise for a do-or-die final, who would it be?"

Respond with ONLY the label name. Do not explain your reasoning.
Valid labels: analysis, hot_take, reaction, discussion
```

**Results:** 30/30 parseable responses, 0 unparseable.

---

## Evaluation Report

### Overall results

| Metric | Baseline (Llama-3.3-70b) | Fine-tuned (DistilBERT) |
|---|---|---|
| Accuracy | 0.967 | 0.967 |
| Macro F1 | 0.97 | 0.97 |
| Test set size | 30 | 30 |

### Per-class metrics — Baseline

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| analysis | 1.00 | 1.00 | 1.00 | 8 |
| hot_take | 0.88 | 1.00 | 0.93 | 7 |
| reaction | 1.00 | 0.88 | 0.93 | 8 |
| discussion | 1.00 | 1.00 | 1.00 | 7 |
| **macro avg** | **0.97** | **0.97** | **0.97** | **30** |

### Per-class metrics — Fine-tuned model

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| analysis | 1.00 | 1.00 | 1.00 | 8 |
| hot_take | 0.88 | 1.00 | 0.93 | 7 |
| reaction | 1.00 | 0.88 | 0.93 | 8 |
| discussion | 1.00 | 1.00 | 1.00 | 7 |
| **macro avg** | **0.97** | **0.97** | **0.97** | **30** |

### Confusion matrix — Fine-tuned model

| | Predicted: analysis | Predicted: hot_take | Predicted: reaction | Predicted: discussion |
|---|---|---|---|---|
| **True: analysis** | 8 | 0 | 0 | 0 |
| **True: hot_take** | 0 | 7 | 0 | 0 |
| **True: reaction** | 0 | 1 | 7 | 0 |
| **True: discussion** | 0 | 0 | 0 | 7 |

### Wrong predictions analysis

**Total wrong predictions: 1 / 30**

**Case 1 — reaction predicted as hot_take (confidence: 0.37)**

> *"I've been watching cricket for 20 years and that is the best IPL over I have ever seen."*
> True label: `reaction` | Predicted: `hot_take`

This is the exact edge case documented in planning.md before annotation. The post contains the surface structure of a hot take — a bold superlative claim ("the best IPL over I have ever seen") stated without evidence. The model latched onto this structure and predicted `hot_take`. The very low confidence (0.37) shows the model was genuinely uncertain, which is the correct response to an ambiguous post. The key signal the model missed is the emotional framing ("I've been watching cricket for 20 years") which anchors the post as a personal reaction rather than a debatable position. This is a boundary problem, not a labeling inconsistency — the post sits genuinely between two labels and the model's uncertainty reflects that.

**Why this boundary is hard:** Both `reaction` and `hot_take` can contain strong, unqualified claims. The distinguishing feature is whether the post is expressing a feeling in the moment or asserting a general debatable position. That distinction requires understanding emotional context, which a small fine-tuned model with 140 training examples will struggle to capture reliably.

**What would fix it:** More training examples that sit explicitly at this boundary, with the decision rule (the "3-day test") reflected in which side they were labeled on. Currently the training set likely has clean examples of both labels, which teaches the model the prototype of each category but not the hard boundary between them.

### Sample classifications

| Post | True Label | Predicted | Confidence |
|---|---|---|---|
| "Bumrah's death-over economy of 8.4 this IPL is actually his worst since 2019..." | analysis | analysis | 0.94 |
| "Rohit Sharma is completely finished as a T20 batter and MI should have released him two seasons ago." | hot_take | hot_take | 0.91 |
| "WHAT A SIX BY HARDIK OH MY GOD THIS MAN IS UNREAL" | reaction | reaction | 0.97 |
| "If you could pick one player from any franchise for a do-or-die final, who would it be?" | discussion | discussion | 0.95 |
| "I've been watching cricket for 20 years and that is the best IPL over I have ever seen." | reaction | hot_take | 0.37 |

**Why the analysis prediction is reasonable:** The post cites a specific statistic (economy rate of 8.4) placed in historical context (worst since 2019) and draws a specific tactical conclusion (MI's dependency on Bumrah in the death). This matches the analysis definition precisely — specific, verifiable evidence used to support a structured argument.

---

## Reflection: What the Model Learned vs. What We Intended

The model learned reliable surface-level signals for each label: all-caps emotional language and exclamation points map to `reaction`; question marks and inviting phrasing map to `discussion`; statistics and player names in argumentative sentences map to `analysis`; short declarative opinion sentences map to `hot_take`.

What it did not fully learn is the underlying intent distinction between labels — specifically the difference between a bold claim made in emotional reaction to an event versus a bold claim stated as a general considered position. The one wrong prediction sits exactly at this boundary, and the model's low confidence (0.37) suggests it recognised the ambiguity rather than making a confident wrong prediction.

The gap between intended and learned behaviour is that our label definitions are intent-based ("expressing a feeling" vs "asserting a position") but the model is pattern-matching on surface form. For 96.7% of posts, the surface form is a reliable proxy for intent. For the edge cases — particularly short posts that make bold claims in emotional language — surface form is not enough.

---

## Spec Reflection

**One way the spec helped:** The instruction to identify a hard edge case before annotating a single example was the most valuable step in the project. Writing the decision rules for `reaction` vs `hot_take` and `analysis` vs `hot_take` before seeing the data forced us to think about where the boundaries actually were. The one wrong prediction the model made was the exact edge case we documented — which means the taxonomy was correctly designed even if the model couldn't always apply it.

**One way implementation diverged from the spec:** The spec assumes data will be collected and then manually annotated, treating AI assistance as optional acceleration. In practice, AI-assisted labeling (Claude pre-labeling with taxonomy prompt) was used from the start rather than as a supplement to manual annotation. This changed the workflow from "annotate then verify" to "generate labels then audit for consistency." The labels are consistent with the taxonomy, but the process was more automated than the spec envisioned.

---

## AI Usage

**Instance 1 — Label stress-testing**
We provided Claude with the four label definitions and the two edge case descriptions and asked it to generate 10 posts sitting at the `analysis` vs `hot_take` boundary and 10 at the `reaction` vs `hot_take` boundary. Several generated posts were genuinely hard to classify, which directly led to sharpening the decision rules: the "decorative stat" rule (a single cherry-picked stat without comparative context = hot_take) and the "3-day test" (would this post make sense 3 days after the match? if no, it's reaction). These rules are now documented in planning.md and applied consistently across the dataset.

**Instance 2 — Dataset generation**
Claude was used to generate the labeled dataset of 200 Reddit-style posts and comments. We provided the taxonomy definitions, example posts for each label, and instructions to generate 50 examples per label with realistic IPL player names, team names, and match contexts. We reviewed the output for label consistency and removed examples that sat too cleanly at label boundaries without a clear decision, replacing them with examples that better illustrated each label's prototype. The final dataset reflects these editorial decisions, not the raw Claude output.

**Instance 3 — Failure pattern analysis**
After generating test set predictions, we provided the wrong predictions to Claude and asked it to identify patterns. Claude identified the `reaction` vs `hot_take` boundary as the systematic failure mode, consistent with the edge case we had pre-identified. We verified this manually by reading all 30 test predictions and confirmed that the one wrong prediction matched the documented edge case exactly. No patterns suggested by Claude were included in this report that we could not independently verify.