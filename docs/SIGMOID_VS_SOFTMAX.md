# 🧠 Why Sigmoid Instead of Softmax? - A Complete Guide

## Understanding Multi-Label vs Multi-Class Classification

Our stutter detection system uses **Sigmoid** activation instead of **Softmax**. This document explains why and how the entire inference pipeline works.

---

## 📊 The Core Difference

### Multi-Class Classification (Softmax)
- Classes are **MUTUALLY EXCLUSIVE**
- Example: "Is this image a Cat OR a Dog?" (can't be both)
- All probabilities **MUST sum to 1.0**

### Multi-Label Classification (Sigmoid) ✅ **What We Use**
- Classes are **INDEPENDENT**
- Example: "Does this audio have Block AND/OR SoundRep AND/OR other stutters?"
- Each class gets its **own independent probability** (0 to 1)
- Probabilities **DON'T need to sum to 1**

---

## 🎯 Why Sigmoid is Correct for Stutter Detection

Our problem: A single 3-second audio chunk can have **MULTIPLE stutters simultaneously**.

Example scenario:
- Someone says "I-I-I w-w-want..." 
- This contains BOTH `WordRep` ("I-I-I") AND `SoundRep` ("w-w-want")
- We need to detect BOTH, not choose one!

### If We Used Softmax (❌ Wrong):

```
Logits:     [2.5,  -1.0,  1.8,  2.2,  -0.5]
Softmax:    [0.45, 0.01, 0.22, 0.31, 0.02]  (sums to 1.0)
            Prol  Block SndRp WrdRp Intrj

Problem: Model says "45% Prolongation is most likely"
         But we MISS that WordRep (0.31) and SoundRep (0.22) are also present!
```

### With Sigmoid (✅ Correct):

```
Logits:     [2.5,  -1.0,  1.8,  2.2,  -0.5]
Sigmoid:    [0.92, 0.27, 0.86, 0.90, 0.38]  (independent, don't sum to 1)
            Prol  Block SndRp WrdRp Intrj

With threshold 0.5:
- Prolongation: 0.92 > 0.5 ✅ DETECTED
- Block: 0.27 < 0.5 ❌ Not detected
- SoundRep: 0.86 > 0.5 ✅ DETECTED
- WordRep: 0.90 > 0.5 ✅ DETECTED
- Interjection: 0.38 < 0.5 ❌ Not detected

Result: ["Prolongation", "SoundRep", "WordRep"] - All three detected!
```

---

## 📐 The Math

### Softmax Formula
$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}$$

- Divides by sum of all exponentials
- Forces all outputs to sum to 1
- Increasing one probability DECREASES others

### Sigmoid Formula
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

- Applied to each logit **independently**
- Maps any real number to (0, 1)
- Each output is independent of others

| Logit Value | Sigmoid Output | Meaning |
|-------------|----------------|---------|
| -5 | 0.007 | Very unlikely |
| -2 | 0.12 | Unlikely |
| 0 | 0.50 | Neutral (50-50) |
| +2 | 0.88 | Likely |
| +5 | 0.993 | Very likely |

---

## 🔄 Training vs Inference Flow

### During Training

```
┌─────────────────────────────────────────────────────────────────┐
│  Input: Audio waveform (3 seconds, 16kHz)                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  WavLM Model → Linear Layer                                     │
│  Output: LOGITS (raw scores, -∞ to +∞)                         │
│  Example: [2.5, -3.0, 0.1, -1.5, -2.0]                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Loss Function: BCEWithLogitsLoss                               │
│  - "WithLogits" = applies Sigmoid INTERNALLY                    │
│  - More numerically stable than separate Sigmoid + BCELoss      │
│  - Compares each prediction to ground truth independently       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Backpropagation: Update weights to minimize loss               │
└─────────────────────────────────────────────────────────────────┘
```

### During Inference

```
┌─────────────────────────────────────────────────────────────────┐
│  Input: Audio waveform (3 seconds, 16kHz)                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  WavLM Model → Linear Layer                                     │
│  Output: LOGITS (same as training)                              │
│  Example: [2.5, -3.0, 0.1, -1.5, -2.0]                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  MANUALLY apply Sigmoid (since no loss function to do it)       │
│  probs = torch.sigmoid(logits)                                  │
│  Result: [0.92, 0.05, 0.52, 0.18, 0.12]                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Apply THRESHOLD (e.g., 0.5) to each probability                │
│  - 0.92 > 0.5 → Prolongation ✅                                 │
│  - 0.05 < 0.5 → Block ❌                                        │
│  - 0.52 > 0.5 → SoundRep ✅                                     │
│  - 0.18 < 0.5 → WordRep ❌                                      │
│  - 0.12 < 0.5 → Interjection ❌                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Final Label:                                                   │
│  - If ANY detected → "Prolongation, SoundRep"                   │
│  - If NONE detected → "Fluent"                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🎚️ Understanding the Threshold

### What is the Threshold?
The threshold is the probability cutoff for deciding if a stutter is "detected" or not.

### Why 0.5 as Default?
- `sigmoid(0) = 0.5` (the neutral point)
- If the model outputs a positive logit → probability > 50% → likely a stutter
- If the model outputs a negative logit → probability < 50% → likely not a stutter

### Adjusting the Threshold

| Threshold | Sensitivity | Use Case |
|-----------|-------------|----------|
| **0.3 - 0.4** | HIGH (more sensitive) | Catch subtle stutters, clinical screening |
| **0.5** | BALANCED | General use |
| **0.6 - 0.7** | LOW (more strict) | Reduce false positives, high confidence only |

### Same Threshold for All Classes
In our implementation, we use the **same threshold for all 5 stutter types**. This is the simplest approach. Advanced implementations could use different thresholds per class based on:
- Class imbalance in training data
- Clinical importance of each stutter type
- Precision/Recall tradeoffs

---

## 🏷️ The "Fluent" Label

Since we use Sigmoid (independent probabilities), we don't have a "Fluent" output neuron. Instead:

```python
# If NO stutter probabilities exceed the threshold
if not result['detected']:  # Empty list = []
    result['label'] = "Fluent"
else:
    result['label'] = ", ".join(result['detected'])
```

**"Fluent" = Absence of all stuttering types**

This is more elegant than adding a 6th "Fluent" class because:
1. No contradictory outputs (can't have "Block" AND "Fluent" both high)
2. Cleaner model architecture
3. Matches the real-world definition: fluent speech has no stutters

---

## 📝 Code Examples

### In Training (`train_waveLM.py`):
```python
# Loss function handles sigmoid internally
loss_fn = nn.BCEWithLogitsLoss()

# Forward pass - model outputs raw logits
logits = model(waveforms, masks)
loss = loss_fn(logits, labels)  # Sigmoid applied inside
```

### In Inference (`WaveLM_inference.py`):
```python
# Get raw logits from model
logits = model(waveform)

# Manually apply sigmoid to get probabilities
probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

# Check each against threshold
for i, label in enumerate(STUTTER_LABELS):
    if probs[i] >= threshold:
        result['detected'].append(label)

# Label as Fluent if nothing detected
if not result['detected']:
    result['label'] = "Fluent"
```

---

## 🎯 Summary Table

| Aspect | Softmax | Sigmoid (Our Choice) |
|--------|---------|----------------------|
| Problem Type | Multi-class | Multi-label |
| Classes | Mutually exclusive | Independent |
| Probabilities Sum | Always = 1 | Can be anything |
| Multiple Detections | ❌ No | ✅ Yes |
| "Fluent" Handling | Needs 6th class | Implicit (no detections) |
| Loss Function | CrossEntropyLoss | BCEWithLogitsLoss |

---

## 📚 Further Reading

- [PyTorch BCEWithLogitsLoss Documentation](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)
- [Multi-Label Classification Explained](https://en.wikipedia.org/wiki/Multi-label_classification)
- [Sigmoid vs Softmax: When to Use Which](https://stats.stackexchange.com/questions/233658/softmax-vs-sigmoid-function-in-logistic-classifier)
