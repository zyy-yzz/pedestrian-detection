# Enhancement Toolbar + Compare + Export — 实现报告

**日期:** 2026-05-07
**项目:** Nighttime Pedestrian Detection System

---

## 新增功能

### 1. 6种增强方式选择工具栏
| 增强方式 | 方法 | 作用 |
|---------|------|------|
| CLAHE 对比度增强 | LAB空间L通道自适应直方图均衡 | 提升暗区对比度 |
| 双边滤波去噪 | Bilateral Filter | 抑制红外噪声保留边缘 |
| 直方图均衡化 | 全局直方图均衡 | 整体对比度提升 |
| 锐化增强 | Unsharp Mask | 增强边缘清晰度 |
| Gamma 校正 | LUT曲线调整 | 亮度曲线矫正 |
| 自适应阈值 | CLAHE局部对比度归一化 | 局部对比度自适应 |

### 2. 增强型 vs 原生 YOLOv8 并排对比
- 左右并排显示增强型和原生YOLOv8n的检测结果
- 独立统计面板（检测数/延迟/置信度）
- Delta对比指示器（检测数差异、延迟差异）

### 3. 图片导出
- PNG 标注图下载（canvas渲染含检测框）
- JSON 检测数据导出（bbox + confidence + class_name）

### 4. 参数调节
- 置信度阈值滑块 (0.05–0.95)
- IoU 阈值滑块 (0.1–0.9)

---

## 修改文件清单

### 后端 (7 files)
| 文件 | 改动 |
|------|------|
| `backend/ml/preprocessor.py` | 重构为可切换管线，新增6种增强方法 + VanillaPreprocessor |
| `backend/ml/model.py` | EnhancedYOLOv8.detect()支持enhancements参数，新增VanillaYOLOv8类 |
| `backend/services/model_loader.py` | 新增get_vanilla_model()单例，修复load()异常处理 |
| `backend/services/detection_service.py` | 支持enhancements列表+compare模式，阈值正确传递 |
| `backend/routers/detect.py` | 接收enhancements(逗号分隔)和compare表单参数 |
| `backend/models/schemas.py` | 新增CompareDetectionResponse schema |
| `backend/ml/__init__.py` | 导出VanillaYOLOv8和VanillaPreprocessor |

### 前端 (8 files)
| 文件 | 改动 |
|------|------|
| `frontend/src/types/index.ts` | 新增EnhancementOption, CompareResult, DetectionConfig等类型 |
| `frontend/src/store/detectionStore.ts` | 扩展compareData, config, toggleEnhancement, setConfig |
| `frontend/src/hooks/useDetection.ts` | 基于config提交，支持compare模式 |
| `frontend/src/components/detection/EnhancementToolbar.tsx` | **NEW** 6个增强开关+2个滑块+对比切换 |
| `frontend/src/components/detection/CompareView.tsx` | **NEW** 并排canvas渲染 |
| `frontend/src/components/detection/StatsPanel.tsx` | 扩展CompareStats双栏对比组件 |
| `frontend/src/components/ui/ExportButton.tsx` | **NEW** PNG+JSON下载 |
| `frontend/src/components/detection/ImageDetector.tsx` | 集成所有新组件，修复错误显示条件 |

---

## 代码审查结果

| 严重度 | 发现 | 状态 |
|--------|------|------|
| 🔴 Critical | conf/iou阈值写入preprocessor导致被忽略 | ✅ 已修复 — 现在直接写入model对象 |
| 🟠 High | 首次检测失败错误被条件渲染隐藏 | ✅ 已修复 — 错误移到条件外 |
| 🟠 High | load()异常时单例返回损坏实例 | ✅ 已修复 — try/except包围+置None |
| 🟡 Medium | 预处理器_letterbox代码重复 | ⚠️ 已知，两个类独立维护 |
| 🟡 Medium | 函数命名不一致 | ⚠️ 已知 |
| 🟢 Low | detect_batch未实现真正批处理 | ⚠️ 后续优化 |
| 🟢 Low | gamma LUT缓存失效风险 | ⚠️ 当前不可变，风险低 |
| 🟢 Low | store reset不恢复config | ✅ 设计如此 — 保留用户偏好 |

---

## API 变更

### POST /detect/image — 新增表单参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enhancements` | string | `"clahe,denoise"` | 逗号分隔的增强方式列表 |
| `compare` | bool | `false` | 是否启用原生YOLOv8对比 |

### 响应格式 (compare=true)

```json
{
  "job_id": "uuid",
  "status": "completed",
  "enhancements_used": ["clahe", "denoise", "sharpen"],
  "enhanced": {
    "pedestrian_count": 5,
    "processing_time_ms": 45,
    "detections": [...],
    "annotated_image": "base64..."
  },
  "vanilla": {
    "pedestrian_count": 3,
    "processing_time_ms": 38,
    "detections": [...],
    "annotated_image": "base64..."
  },
  "comparison": {
    "delta_count": 2,
    "delta_time_ms": 7,
    "avg_conf_enhanced": 0.82,
    "avg_conf_vanilla": 0.71
  }
}
```