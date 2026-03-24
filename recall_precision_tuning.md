# Recall vs. Boundary Precision Tuning

This note collects every adjustment we can make—data, preprocessing, architecture/flow, training, inference, and evaluation—to push the structure recall (aiming for >100 detected objects per image/dataset) while keeping the boundary-level precision of each mask intact. Each idea includes a short rationale, a **priority** assessment, and a **success likelihood** score.

| Stage | Idea | Priority | Success likelihood | Notes |
| --- | --- | --- | --- | --- |
| Preprocessing | Apply high-frequency-preserving denoising (e.g., wavelet shrinkage, anisotropic diffusion) that keeps contours sharp. | Medium | High | Reduces noise-triggered false negatives while leaving boundaries crisp. |
| Preprocessing | Normalize intensity per image or per batch with percentile clipping to handle brightness/contrast variability. | Medium | Medium | Prevents intensity-driven missed detections without influencing boundary precision. |
| Inference / Post-processing | Reduce size/score thresholds slowly to admit more segments, then filter for boundary quality (e.g., ratio of flow divergence). | High | Medium | Allows more structures while pruning at the boundary-confidence level. |
| Inference / Post-processing | Use ensemble/multi-threshold decoding and merge masks only when both preserve high boundary IoU. | Medium | Medium | Adds structures from different settings but the merge step ensures boundaries stay tight. |
| Inference / Post-processing | Apply boundary refinement (CRF, fast contour smoothing) only on confident masks to avoid ruining precision. | Medium | High | Keeps boundaries crisp even if more masks are produced. |
| Evaluation & Monitoring | Track boundary IoU/contour F1 in addition to recall; use Pareto charts to expose where recall bumps start hurting contours. | High | High | Helps decide which ideas actually maintain precision. |

Be mindful that improving recall above the existing counts usually means adding more structures that are borderline cases. Whenever the constraint is "no retraining", focus on preprocessing tweaks and inference/post-processing, and use boundary-aware evaluation to ensure precision stays high. If you later lift that constraint, revisit the removed ideas.
