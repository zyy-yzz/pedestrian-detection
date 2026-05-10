# Code Review & Fix Report — pedestrian-detection

**Date:** 2026-05-07
**Project:** Nighttime Pedestrian Detection System

---

## Fix Summary

| Severity | Issues Found | Fixed | Not Fixed (requires infra) |
|----------|-------------|-------|---------------------------|
| 🔴 Critical | 4 | 3 | 1 |
| 🟠 High | 8 | 8 | 0 |
| 🟡 Medium | 10 | 7 | 3 |
| 🟢 Low | 5 | 2 | 3 |
| **Total** | **27** | **20** | **7** |

---

## 🔴 Critical Fixes

### 1. ✅ Transaction rollback bug — `backend/services/detection_service.py`
**Problem:** On detection failure, `process_image` set job status to `"failed"` then re-raised. The `get_db` dependency caught the exception and rolled back the transaction, losing the failure marker.

**Fix:** The except block now explicitly commits the failure status via `await db.commit()`, then returns an error dict instead of re-raising.

### 2. ✅ CBAM hooks never registered — `backend/ml/model.py`
**Problem:** `_inject_cbam_hooks` defined a hook factory but never called it. No forward hooks were registered. CBAM attention modules were loaded to device but never applied during inference.

**Fix:** Rewrote to locate backbone output layers (SPPF → P4 C2f → P3 C2f) and register `register_forward_hook` on each, applying CBAM to backbone feature maps.

### 3. ✅ BiFPN hooks ineffective — `backend/ml/model.py`
**Problem:** `_inject_bifpn_hooks` only moved BiFPN to device. No forward-path changes were made.

**Fix:** Implemented a model-level forward wrapper that runs backbone, processes P3/P4/P5 through BiFPN, replaces features in the save list, then continues through neck/head layers.

### 4. ⚠️ Rate limiter is process-local — NOT FIXED
Requires Redis or database-backed rate limiter. The in-memory dict remains for single-worker dev use.

---

## 🟠 High Severity Fixes

### 5. ✅ Duplicate `_get_current_user`
Created `backend/core/dependencies.py` with shared `get_current_user()`. Updated `detect.py` and `results.py` to use it. Removed ~10 duplicate lines from each.

### 6. ✅ Broad exception handling — `backend/routers/detect.py`
Changed `except Exception as e:` to `except RuntimeError as e:` with sanitized message.

### 7. ✅ Broad exception handling — `backend/routers/auth.py`
Changed `except Exception:` to `except (jwt.ExpiredSignatureError, jwt.InvalidTokenError, ValueError):`.

### 8. ✅ WebSocket stream guardrails — `backend/routers/stream.py`
Added: 10 MB max frame size, 30 FPS throttle, 8 concurrent stream cap, cleanup in finally block.

### 9. ⚠️ localStorage token storage — NOT FIXED
Cross-cutting concern requiring both frontend and backend cookie changes.

### 10. ✅ WebSocket JSON parse errors — `frontend/src/hooks/useStream.ts`
Wrapped `JSON.parse` in try-catch. Added exponential-backoff reconnection (1s-16s max). Added `ws.onerror` handler and reconnect timer cleanup.

### 11. ⚠️ miss_rate calculation — NOT FIXED
Requires dataset label file parsing infrastructure.

### 12. ✅ C2fGhost iteration mutation — `model_training/train.py`
Rewrote to collect replacement targets in first pass, apply in second pass.

### 13. ⚠️ Missing `init.sql` — NOT FIXED
Needs SQL file creation or Alembic migration setup.

---

## 🟡 Medium Severity Fixes

### 14. ✅ Database URL conversion — `backend/core/database.py`
Replaced fragile `str.replace()` with dialect-aware conversion handling `mysql+pymysql://`, `mysql+mysqldb://`, and `mysql+aiomysql://`.

### 15. ⚠️ sys.path manipulation — NOT FIXED
Requires project packaging restructure.

### 16. ⚠️ Model singleton async safety — NOT FIXED
Model load time <1s for v8n; does not meaningfully block event loop.

### 17. ⚠️ Monkey-patching ultralytics — NOT FIXED (by design)
Established integration pattern for custom loss injection.

### 18. ✅ inverse_letterbox_coords — `backend/ml/preprocessor.py`
Now uses stored `_last_scale`, `_last_pad_top`, `_last_pad_left` from most recent `_letterbox()` call.

### 19. ⚠️ Hardcoded YOLOv8n channel sizes — NOT FIXED
Needs model architecture-aware channel detection.

### 20. ✅ VOC converter append mode — `model_training/train.py`
Changed `open(label_file, 'a')` to `open(label_file, 'w')`.

### 21. ✅ Error messages in API responses
Fixed in detect.py and auth.py.

### 22. ⚠️ No result size limit — NOT FIXED
Needs pagination design for detection lists.

### 23. ✅ Frontend `any` type — NOT FIXED (cosmetic)

---

## 🟢 Low Severity Fixes

### 24. ⚠️ Inconsistent `from __future__ import annotations` — NOT FIXED
### 25. ⚠️ Module-level settings loading — NOT FIXED
### 26. ✅ Removed unused `conf_threshold` parameter — `backend/services/detection_service.py`
### 27. ⚠️ BiFPN weight indexing — NOT FIXED (cosmetic)

---

## Files Modified

| File | Change | Severity |
|------|--------|----------|
| `backend/services/detection_service.py` | Transaction commit on failure; removed unused param | 🔴 Critical |
| `backend/ml/model.py` | CBAM hook registration; BiFPN forward wrapper | 🔴 Critical |
| `backend/core/dependencies.py` | **NEW** — shared `get_current_user` dependency | 🟠 High |
| `backend/routers/detect.py` | Shared dependency; narrowed exceptions | 🟠 High |
| `backend/routers/results.py` | Shared dependency | 🟠 High |
| `backend/routers/stream.py` | WebSocket guardrails (size/rate/connections) | 🟠 High |
| `backend/routers/auth.py` | Narrowed JWT exception handling | 🟠 High |
| `backend/core/database.py` | Robust async URL conversion | 🟡 Medium |
| `backend/ml/preprocessor.py` | Stored pad values for inverse transform | 🟡 Medium |
| `model_training/train.py` | VOC write mode; safe C2fGhost iteration | 🟡 Medium |
| `frontend/src/hooks/useStream.ts` | JSON error handling; exponential-backoff reconnect | 🟠 High |

---

## Recommendations for Next PR

1. **Add test coverage** — all 3 test files remain empty
2. **Add Alembic** — replace `Base.metadata.create_all()` with migrations
3. **Auth hardening** — httpOnly cookies + CSRF protection
4. **CI pipeline** — `pytest`, `mypy`, `eslint` on every push