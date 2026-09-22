# TikTok Video Engagement Prediction — In-Class Competition

**Author:** Elaf Alyoubi
**Competition:** In-Class Competition: Predictive Modelling (WeCloudData DS Bootcamp)
**Goal:** Predict the cumulative view count a video will reach at Day 30, using only video metadata and the first 5 days (Day 0–Day 5) of engagement and creator statistics.

---

## 0. Approach & Philosophy

This project followed four principles throughout:

1. **Reframe before modeling.** The literal target (`target_day30_views`) is not the most learnable quantity, most of it is already explained by Day-5 views alone. Reframing the target as a *growth factor* (see Section 1) turns an extreme, heavy-tailed regression problem into a much more tractable one.
2. **Never invent data.** Wherever a day of engagement or a creator statistic was missing, it was left as a true missing value (`NaN`) rather than filled with an assumed or interpolated number. The model (LightGBM) handles missingness natively, so there was no need to guess.
3. **Every idea is a hypothesis, not a fact.** Three feature ideas, creator popularity/reach, engagement-quality ratios, and "uncomfortable" emotional content, were treated as hypotheses to be tested, not assumptions to be trusted. Each was measured by removing it from the model and checking whether local validation error got worse (see Section 3). Two of the three hypotheses were confirmed; one was not, and that negative result is reported honestly rather than hidden.
4. **Trust a local validation harness over a single leaderboard number.** All modeling decisions (target transform, sample weighting, feature groups) were made by comparing average error and its variance across a 5-fold local validation split, a leaderboard score is a single, noisy sample and was not used to drive iteration.

---

## 1. Problem Framing

Rather than predicting `target_day30_views` directly, I reframed the problem as predicting a **growth factor**:

```
growth_factor = target_day30_views / last_known_views
```

where `last_known_views` is the most recent Day 0–5 view count available for a video (Day 5 for ~98% of videos; an earlier day when Day 5 was missing).

**Why:** An exploratory check showed that Day-5 views alone already have a Spearman correlation of ~0.99 with the Day-30 target — most of the "answer" is already visible by Day 5. The real prediction problem is *how much further a video grows after that point*, and the growth factor is heavily right-skewed (median ≈ 1.16x, but up to 767x for a few viral videos), which matches the "few videos explode, most stay quiet" pattern described in the competition brief.

The final prediction is: `predicted_views = last_known_views × predicted_growth_factor`.

## 2. Data & Key Decisions

| Decision | Choice | Reason |
|---|---|---|
| Missing days in `engagement_daily` | Left as `NaN`, no imputation | Avoids inventing engagement numbers that were never observed |
| Missing creator stats | Left as `NaN` | Same principle, LightGBM handles missing values natively |
| Local validation | Random 5-fold `KFold` (not grouped by creator) | 97.5% of test-set creators already appear in train, so a random split mirrors the real test distribution better than a creator-grouped split |
| Training target | `log(growth_factor)`, clipped at a minimum of 1.0 | Raw growth factor has a long right tail; training on it directly caused the model to occasionally predict wildly (and even negative) growth. Log-transforming stabilized training |
| Sample weighting | None (tested weighting by views and views²) | Unweighted training gave the best and most stable score on both raw and log RMSE across folds |
| Data leakage control | Only Day 0–5 engagement used; creator stats looked up at the video's post date and post date + 5 days only | Matches the competition's "no leakage" rule explicitly |

## 3. Feature Engineering

Three hypotheses were tested going into this project, plus one feature group I added based on early data exploration (momentum). Each was evaluated by **removing it from the model and measuring the increase in local validation error** (5-fold log RMSE), and by inspecting LightGBM's feature importances.

| Feature group | Hypothesis | Result (log RMSE without it, vs. 0.283 with all features) | Verdict |
|---|---|---|---|
| **Momentum** (`last_day_views_share`, `last_two_days_views_share`, `views_growth_day1_to_day5`) | Videos still accelerating late in the observed window keep growing after Day 5 | **0.299** (largest increase) | **Strongest signal in the whole model** also ranked #1 and #2 in feature importance |
| **Creator features** (follower count, gain in followers/videos/favorites over Days 0–5, views-per-follower) | A popular creator's videos take off faster; a video that outperforms its creator's usual reach ("breaks out" of the creator's normal audience) grows more | **0.289** | **Confirmed** measurable, above-noise improvement, and several creator features rank in the top 10 by importance |
| **Engagement rate quality** (likes/comments/shares/saves per view at Day 5) | Higher-quality engagement (not just raw views) signals a video that will keep growing | 0.283 (no measurable difference) | **Not confirmed** in this formulation, the ratios appear in the model's top features individually but removing the whole group didn't hurt accuracy |
| **Emotion / "uncomfortable content" score** (`fear + disgust + anger + sadness`) vs. `joy` | Emotionally uncomfortable content spreads more than content that is merely funny | 0.284 (negligible difference) | **Not confirmed** weak correlation with growth in both raw and post-Day-5 checks |

**Other features included but not individually ablated:** raw daily engagement counts (Day 0–5, 7 metrics × 6 days), video metadata (duration, language, AI-generated flag, description word/emoji/hashtag counts, speaking rate, topic, music source, resolution, post hour/weekday).

**Columns explicitly excluded from the model:** `video_id`, `author_id` (identifiers, no predictive meaning as raw numbers), `create_date` (duplicate of `create_time`), `is_ads` (constant, single value across all rows), `desc_language` (redundant with `is_english`), `music_id` / `music_author` / `music_owner_id` (high-cardinality, largely unseen in the test set, 20–72% overlap only), and of course `target_day30_views` itself.

## 4. Model

- **Algorithm:** LightGBM Regressor (gradient-boosted trees), chosen for native handling of missing values and categorical features, and its suitability for tabular data with mixed numeric/categorical columns.
- **Target:** `log(growth_factor)`, clipped at a minimum growth of 1.0 (views cannot decrease, since they are cumulative).
- **Validation:** 5-fold random K-Fold (fixed random seed for reproducibility).
- **Baselines used for comparison:**
  - Predict `last_known_views` as-is (no growth): log RMSE 0.423
  - Predict `last_known_views × median(growth_factor)`: log RMSE 0.356, raw RMSE 74,430
  - **Final model:** log RMSE 0.283 (≈20% improvement over the best baseline), raw RMSE 74,025 (roughly tied with the baseline, raw RMSE is dominated by a handful of very large outlier videos, which is why the log-scale comparison is more informative for judging incremental improvements)

The competition's evaluation metric (RMSE on raw view counts) means that raw-scale performance is what ultimately matters; the log-scale metric was used internally as a more stable tool for comparing modeling choices, since it is far less dominated by a handful of extreme videos.

## 5. Repository Contents

- `tiktok_engagement_prediction.ipynb` : full, self-contained notebook: data loading → feature engineering → local validation → model training → prediction → submission file generation. Runs top to bottom with **Restart & Run All** to reproduce the exact submission.
- `baseline_submission.csv` : final Kaggle submission file.
- `README.md` : this file.

## 6. Honest Limitations

- One test-set video had its last known engagement day at Day 0 (no comparable case existed in training), so its prediction is an extrapolation and is less reliable than the rest.
- The engagement-rate and emotion feature groups did not show a measurable benefit in this validation setup; they were kept in the final model for completeness and because individual features from these groups appeared among the top-15 most-used features, but their group-level contribution is not statistically distinguishable from noise here.
- Raw RMSE is highly sensitive to a small number of very large videos, making fold-to-fold raw RMSE noisy (std ≈ 13,000–45,000 depending on the experiment); the log-scale metric was used as the primary tool for comparing feature groups and modeling choices.
