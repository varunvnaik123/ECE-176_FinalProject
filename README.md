# ECE 176 Final Project — Chart Pattern Recognition with CNNs

**Course:** ECE 176: Introduction to Deep Learning  
**Team:** Varun Naik and Sujal Gour

---

#Demo video: https://drive.google.com/file/d/1FSg0V32TzEsSwo7rnUd_z5Tt2OzAMKAR/view?usp=sharing


## What We Built

We trained CNNs to recognize stock chart patterns from candlestick images, things
like Head & Shoulders, Double Tops, and Ascending Triangles. The big question we
wanted to answer:

> *If you train a model on synthetic chart images, does it actually learn anything
> useful when you throw real stock data at it?*

To test this, we generated our own labeled dataset, trained two models (a small
custom CNN and a fine-tuned ResNet-18), used Grad-CAM to see what the models were
actually looking at, and then ran a simple trading backtest to see if the
detections meant anything in practice.

---

## Patterns We Classify

| Pattern | Direction |
|---------|-----------|
| Head & Shoulders | Bearish |
| Double Top | Bearish |
| Descending Triangle | Bearish |
| Inverse Head & Shoulders | Bullish |
| Double Bottom | Bullish |
| Ascending Triangle | Bullish |
| No Pattern (random walk) | Neutral |

---

## How It Works

1. **Generate data** — we simulate 10,500 synthetic candlestick charts (1,500 per
   class) with added noise so the model doesn't just memorize clean shapes.

2. **Train models** — a 3-layer CNN from scratch and a ResNet-18 fine-tuned from
   ImageNet weights. We compare both on a held-out validation set.

3. **Grad-CAM** — we visualize which parts of the chart each model focuses on.
   This is how we sanity-check that it's learning the actual pattern shape and not
   something random.

4. **Real data test** — we run the trained model on real S&P 500 stocks using a
   sliding window and record every detection above 55% confidence.

5. **Backtest** — we simulate entering a trade on every detection and check if the
   price moved in the predicted direction 5 days later.

---

## What We Expect to Find

Honestly, we're not expecting the model to be a money printer. Our hypothesis is
that the model *can* learn the pattern shapes (Grad-CAM should confirm this), but
confidence will drop noticeably on real noisy data vs. clean synthetic charts.
If directional accuracy on real detections beats 50% with p < 0.05, we'd call
that a win. If not, the domain gap analysis is still the interesting result.

---

