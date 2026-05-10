# Code Review Report — pedestrian-detection

**Date:** 2026-05-07
**Project:** Nighttime Pedestrian Detection System
**Tech Stack:** Python/FastAPI + React/TypeScript + PyTorch/YOLOv8
**Files Reviewed:** 55+ source files across backend, frontend, model training, and infra

---

## Severity Legend

| Icon | Level    | Meaning                                                               |
| ---- | -------- | --------------------------------------------------------------------- |
| 🔴   | Critical | Bug / data loss / security vulnerability — must fix before production |
| 🟠   | High     | Design flaw / correctness issue — should fix before merge             |
| 🟡   | Medium   | Code quality / maintainability — fix soon                             |
| 🟢   | Low      | Style / minor improvement — non-blocking                              |

---

## 🔴 Critical Issues

### 1. DB transaction: status update lost on exception

**`backend/services/detection_service.py:100-104`**

```python
except Exception as exc:
    await JobService.update_status(db, job_id, "failed", error_message=str(exc))
    raise  # <-- exception propagates → db.rollback() called in get_db()
```

The `get_db` dependency (database.py:22-31) calls `session.rollback()` in its `except` block. When `detection_service.process_image` re-raises, the rollback path runs and the `"failed"` status update is discarded. The job remains stuck at `"processing"` forever.

**Fix:** Return an error response instead of re-raising, or use a separate transaction for the failure marker.

### 2. CBAM hooks never registered — inference silently degraded

**`backend/ml/model.py:258-275`**

```python
def _inject_cbam_hooks(model) -> None:
    ...
    def _make_cbam_hook(cbam, name):
        def _hook(_m, _inp, output):
            saved[name] = cbam(output)
        return _hook
    # _make_cbam_hook is never called! No hooks are registered.
```

The function defines the hook factory but never calls it. CBAM attention modules will **never be applied** during inference, silently degrading detection quality on low-light images. Same issue for `_inject_bifpn_hooks` — it moves BiFPN to the device but doesn't hook it into the forward pass.

**Fix:** Complete the hook registration by calling `model.register_forward_hook(...)` with the created hooks.

### 3. Empty test files — zero test coverage

**`backend/tests/test_auth.py`, `test_detect.py`, `test_preprocess.py`**

All three test files are empty (1 line each). The project has zero test coverage for:

- Authentication (JWT creation, password hashing, rate limiting)
- Detection pipeline (image processing, model inference, result persistence)
- Preprocessing (CLAHE, letterbox, normalization)

**Fix:** Implement tests for at minimum: auth token lifecycle, detection service happy path, and preprocessing pipeline correctness.

### 4. Rate limiter is process-local — broken with multi-worker

**`backend/routers/auth.py:20-21`**

```python
_login_attempts: dict[str, list[float]] = {}
```

The in-memory rate limiter dictionary is per-process. With multiple Uvicorn workers (or horizontal scaling), each worker maintains an independent counter — an attacker can distribute login attempts across workers to bypass the limit entirely.

**Fix:** Use Redis (redis-py + sliding window) or a database-backed rate limiter.

---

## 🟠 High Severity

### 5. Duplicate auth dependency

**`backend/routers/detect.py:16-24`** and **`backend/routers/results.py:14-22`**

`_get_current_user` is copy-pasted verbatim between two router modules. If the token extraction logic changes, both copies must be updated.

**Fix:** Extract to `backend/core/dependencies.py`:

```python
async def get_current_user(authorization: str = Header(...)) -> dict:
    ...
```

### 6. Broad exception handling exposes internals

**`backend/routers/detect.py:53-54`**

```python
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Detection failed: {e}")
```

Catches `Exception` too broadly and leaks raw exception messages to clients. This can expose internal paths, model structure, or database state.

### 7. WebSocket stream has no guardrails

**`backend/routers/stream.py:13-42`**

The `/stream` WebSocket endpoint lacks:

- Frame size limit (attacker can send 100MB frames)
- Per-connection rate limit (unlimited frames/second)
- Max connections cap (can exhaust GPU memory)
- Origin validation (cross-site WebSocket hijacking)

### 8. JWT tokens stored in localStorage — XSS risk

**`frontend/src/api/client.ts:10`**

```typescript
const token = localStorage.getItem('access_token');
```

Tokens in `localStorage` are readable by any JavaScript running on the page. An XSS vulnerability in any frontend dependency immediately compromises all user tokens.

**Fix:** Use httpOnly cookies with the backend setting them via `Set-Cookie`.

### 9. WebSocket onmessage has no error handling

**`frontend/src/hooks/useStream.ts:28-29`**

```typescript
ws.onmessage = (event) => {
  const frame: StreamFrame = JSON.parse(event.data);  // throws on bad JSON
```

Malformed JSON silently breaks the connection. No reconnection logic exists.

### 10. miss_rate calculation is semantically wrong

**`model_training/evaluate.py:131-140`**

```python
num_det = len(results[0].boxes.cls)
num_gt = len(results[0].boxes)  # WRONG: boxes is predictions, not GT
```

`results[0].boxes` contains model predictions, not ground truth. The miss rate compares predictions against predictions — producing meaningless numbers.

**Fix:** Parse ground truth annotations from the dataset label files.

### 11. C2fGhost replacement mutates during module iteration

**`model_training/train.py:113-131`**

`_replace_c2f_with_ghost` iterates `model.named_modules()` while replacing modules in-place. This can cause the iterator to skip modules or crash.

**Fix:** Collect module names first, then replace in a second pass.

### 12. Missing `backend/sql/init.sql` referenced in Docker

**`docker-compose.yml:46`**

```yaml
volumes:
  - ./backend/sql/init.sql:/docker-entrypoint-initdb.d/init.sql:ro
```

This file does not exist in the repository. MySQL starts without schema initialization — tables are only created by SQLAlchemy's `create_all`, which has no migration history.

**Fix:** Create the init SQL script, or use Alembic for database migrations.

---

## 🟡 Medium Severity

### 13. Fragile async MySQL URL conversion

**`backend/core/database.py:10-11`**

```python
_async_url = settings.database_url.replace("mysql+pymysql://", "mysql+aiomysql://")
```

Hard-coded string replacement fails if URL uses a different dialect, different casing, or includes query parameters.

**Fix:** Use `sqlalchemy.engine.url.make_url()` to parse and reconstruct with the async driver.

### 14. Module-level sys.path manipulation

**`backend/ml/model.py:17-20`** and **`model_training/train.py:22-23`**

```python
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
```

Modifying `sys.path` at import time is fragile. Install the project as a proper package (`pip install -e .`) or use relative imports within packages.

### 15. Model singleton is not async-safe

**`backend/services/model_loader.py:16-31`**

Double-checked locking uses `threading.Lock()`, but `load()` is called from async contexts in FastAPI. Loading a large model (~several seconds) will block the event loop.

**Fix:** Wrap `get_model()` calls in `asyncio.to_thread()`.

### 16. Monkey-patching ultralytics internals

**`backend/ml/model.py:227-242`** and **`model_training/train.py:91-108`**

Both files monkey-patch `ultralytics.utils.metrics.bbox_iou`. Ultralytics updates can break the patch silently.

### 17. inverse_letterbox_coords recalculates from scratch

**`backend/ml/preprocessor.py:124-155`**

Recomputes `scale`, `pad_h`, `pad_w` instead of using stored `_last_scale` and pad values. If any parameter differs between forward and inverse calls, coordinate mapping will be wrong.

### 18. Hardcoded YOLOv8n channel sizes

**`model_training/train.py:55-59`**

```python
backbone_channels = {
    'P3': 128,  # YOLOv8n only
    'P4': 256,
    'P5': 512,
}
```

These are specific to YOLOv8n. Using v8s/m/l/x will cause tensor shape mismatches.

### 19. VOC-to-YOLO converter appends to files

**`model_training/train.py:213`**

```python
with open(label_file, 'a') as f:  # 'a' = append mode
```

If the script is accidentally run twice, labels will be duplicated.

### 20. Error messages leak to production responses

**Multiple locations in routers**

Several endpoints pass raw `str(e)` into response detail fields, exposing internal error details to API consumers.

### 21. No result size limit on detection endpoint

**`backend/routers/results.py:29-40`**

The `/results/{job_id}` endpoint returns all detections without pagination. A video job could return an enormous JSON payload.

### 22. Frontend: `any` type on error catch

**`frontend/src/hooks/useDetection.ts:21`**

```typescript
} catch (err: any) {
```

Using `any` defeats TypeScript's type safety.

---

## 🟢 Low Severity / Style

### 23. Inconsistent import style

Some files use `from __future__ import annotations`, others don't. Standardize across the project.

### 24. settings loaded at module level

**`backend/core/security.py:11`**, **`backend/core/database.py:8`**

`settings = get_settings()` called at module level makes testing harder (can't easily override per test).

### 25. README.md is empty

The project README is a 1-line placeholder. No setup instructions, API docs, or architecture overview.

### 26. process_stream_frame ignores conf_threshold parameter

**`backend/services/detection_service.py:137-139`**

The `conf_threshold` parameter is accepted but ignored — threshold comes from model config instead.

### 27. BiFPNFusion weight indexing is unintuitive

**`model_training/modules/bifpn.py:70-76`**

`td_weights[1]` for P4 and `td_weights[0]` for P3 — the reverse mapping is confusing.

---

## Summary

| Severity    | Count  |
| ----------- | ------ |
| 🔴 Critical | 4      |
| 🟠 High     | 8      |
| 🟡 Medium   | 10     |
| 🟢 Low      | 5      |
| **Total**   | **27** |

### Top 3 fixes to make immediately:

1. **CBAM/BiFPN hooks** (`backend/ml/model.py`) — the enhanced architecture is disabled at inference time
2. **Transaction rollback bug** (`backend/services/detection_service.py`) — failed jobs stuck in "processing" forever
3. **Write tests** (`backend/tests/`) — zero test coverage for a system that processes user uploads and has authentication

---

## What's Done Well

- **Architecture**: Clean separation between routers → services → models with clear responsibilities per layer
- **Type annotations**: Consistent use of Python type hints and TypeScript interfaces throughout
- **Async throughout**: Database access uses SQLAlchemy async properly with generator-based session management
- **Model design**: CBAM, C2fGhost, BiFPN, and WIoU implementations are well-structured PyTorch modules with clear docstrings
- **Preprocessing pipeline**: CLAHE + bilateral denoising + letterbox is well-chosen for low-light conditions
- **Docker deployment**: Multi-service setup with health checks and resource limits in production override
- **Token refresh queue**: The axios interceptor's queue-based refresh pattern avoids redundant refresh calls
- **Error UI states**: Frontend properly shows loading spinners, error messages, empty states, and graceful transitions
