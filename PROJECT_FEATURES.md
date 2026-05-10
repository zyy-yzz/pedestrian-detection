# 夜间行人检测系统 — 功能概述

**项目名称：** Nighttime Pedestrian Detection System (yoloPro)
**技术栈：** Python/FastAPI + React/TypeScript + PyTorch/YOLOv8 + Docker
**版本：** 1.0.0

---

## 一、项目定位

基于增强型 YOLOv8 的夜间行人检测全栈系统。针对低光照、红外图像场景进行了专门的模型优化和图像预处理，提供从模型训练、后端推理到前端交互界面的完整解决方案。

---

## 二、核心功能

### 1. AI 模型推理

#### 1.1 增强型 YOLOv8 模型 (EnhancedYOLOv8)

- **CBAM 注意力机制** — 在骨干网络 P3/P4/P5 层后插入通道+空间双注意力模块，增强对暗光下行人的特征提取
- **C2f-Ghost 模块** — 用 Ghost 卷积替换标准 C2f 模块，减少约 30% 参数量同时保持感受野
- **BiFPN 特征金字塔** — 带学习权重的双向跨尺度特征融合，替代标准 PANet 颈部结构，提升小目标检测
- **WIoU Loss** — Wise-IoU 损失函数，动态聚焦难样本（低 IoU 的预测框获得更高损失权重）
- **自动设备适配** — 支持 CUDA GPU 和 CPU 推理，CUDA 不可用时自动回退

#### 1.2 夜间图像预处理管线

对每张输入图像执行三步预处理：

| 步骤           | 算法                                             | 作用                |
| ------------ | ---------------------------------------------- | ----------------- |
| CLAHE 增强     | 限制对比度自适应直方图均衡化 (clip=3.0, tile=8×8)            | 提升暗区对比度，增强行人轮廓可见性 |
| 双边滤波去噪       | Bilateral Filter (d=9, σ_color=75, σ_space=75) | 抑制红外传感器噪声，保留边缘    |
| Letterbox 缩放 | 保持长宽比的 640×640 缩放 + 灰边填充                       | 统一输入尺寸，不产生几何畸变    |

#### 1.3 后处理

- 置信度色彩编码：≥85% 绿色 / ≥65% 琥珀色 / <65% 红色
- 边界框 + 标签渲染到原图
- 支持 Base64 编码的 JPEG 标注图直接返回

---

### 2. 后端 API 服务

#### 2.1 检测接口 (`/detect`)

| 端点                        | 方法   | 功能                                               |
| ------------------------- | ---- | ------------------------------------------------ |
| `/detect/image`           | POST | 上传单张图片进行行人检测，返回标注图和检测列表                          |
| `/detect/video`           | POST | 上传视频进行检测（框架已就绪，实现标记为 501）                        |
| `/detect/status/{job_id}` | GET  | 查询检测任务状态（queued → processing → completed/failed） |

**支持的图片格式：** JPEG, PNG, BMP
**支持的视频格式：** MP4, AVI, MOV (待实现)
**文件大小限制：** 20 MB (可配置)
**可调参数：** 置信度阈值 (conf_threshold) 和 IoU 阈值 (iou_threshold)

#### 2.2 实时流式检测 (`/stream`)

- **WebSocket 协议** — 低延迟双向通信

- **逐帧检测** — 客户端发送原始帧字节，服务端返回检测结果 JSON

- **返回数据结构：**
  
  ```json
  {
    "frame_index": 0,
    "pedestrian_count": 3,
    "detections": [
      {"bbox": [x1,y1,x2,y2], "confidence": 0.92, "class_id": 0}
    ],
    "inference_ms": 45
  }
  ```

- **安全限制：** 单帧最大 10MB、帧率上限 30 FPS、最大并发 8 路

#### 2.3 用户认证系统 (`/auth`)

| 端点               | 功能                                   |
| ---------------- | ------------------------------------ |
| `/auth/register` | 用户注册 (用户名 + 邮箱 + 密码)                 |
| `/auth/login`    | 用户登录，返回 access_token + refresh_token |
| `/auth/refresh`  | 使用 refresh_token 刷新 access_token     |

- **密码安全：** bcrypt 哈希存储
- **JWT 认证：** HS256 签名，access_token 30分钟过期，refresh_token 7天
- **速率限制：** 登录接口每 IP 每分钟 5 次尝试上限
- **角色系统：** admin / user 双角色

#### 2.4 历史记录管理 (`/results`, `/history`)

| 端点                        | 功能                       |
| ------------------------- | ------------------------ |
| `/results/{job_id}`       | 查询已完成检测的详细结果（含所有检测框）     |
| `/results/{job_id}/image` | 下载标注后图片（待实现）             |
| `/history`                | 分页查询当前用户的历史检测任务列表        |
| `/history/{job_id}`       | DELETE 删除历史检测记录（所有者或管理员） |

#### 2.5 系统状态

| 端点            | 功能                       |
| ------------- | ------------------------ |
| `/health`     | 健康检查                     |
| `/model/info` | 返回模型版本、输入尺寸、阈值、设备、类别等元信息 |

---

### 3. 前端检测工作室

#### 3.1 检测功能页 (DetectionStudio)

- **图片检测** — 拖拽上传或点击选择图片，实时显示检测结果叠加图
- **视频检测** — 界面框架就绪
- **实时流检测** — WebSocket 连接实时摄像头画面进行逐帧检测

#### 3.2 检测结果显示

- 标注图叠加 — 在原图上绘制彩色边界框和置信度标签
- 统计面板 — 行人数量、处理耗时、平均置信度
- 检测列表 — 每个检测框的详细信息和置信度徽章
- 置信度颜色编码 — 绿/琥珀/红直观显示检测质量

#### 3.3 交互体验

- 拖拽上传 (DropZone)
- Tab 切换三种检测模式
- 加载动画和错误状态提示
- 深色主题 UI，低光环境下使用舒适

#### 3.4 状态管理 (Zustand)

- `detectionStore` — 当前检测任务状态和结果
- `streamStore` — WebSocket 连接状态、实时检测人数和延迟
- `historyStore` — 检测历史记录列表和分页

#### 3.5 网络层

- Axios 拦截器自动附加 Bearer Token
- Token 自动刷新 — 401 响应时使用 refresh_token 静默续期
- 并发请求的 Token 刷新队列 — 多个同时过期的请求只触发一次刷新

---

### 4. 模型训练管线

#### 4.1 LLVIP 数据集支持

- 自动将 VOC XML 标注转换为 YOLO txt 格式
- 自动划分训练集/验证集 (85%/15%)
- 支持成对可见光+红外图像训练

#### 4.2 训练增强 (NightAugmentation)

为夜间场景定制的数据增强管线：

| 增强方法                 | 参数                   | 作用       |
| -------------------- | -------------------- | -------- |
| 随机暗化 (RandomDarken)  | Δ亮度 ∈ [-40, 0]       | 模拟更暗场景训练 |
| 高斯噪声 (GaussianNoise) | σ ~ U(5, 25)         | 模拟传感器噪声  |
| HSV 增强               | H±1.5%, S±70%, V±40% | 光照和颜色变化  |
| 水平翻转                 | p=0.5                | 空间数据增强   |
| 随机仿射变换               | 平移±10%, 缩放 0.5-1.5×  | 几何鲁棒性    |
| Mosaic               | p=1.0                | 4图拼接训练   |
| MixUp                | p=0.15               | 混合样本训练   |

#### 4.3 训练配置

- **优化器：** AdamW (lr=0.01, weight_decay=0.0005)
- **学习率策略：** 余弦退火 (cosine schedule)
- **预热：** 3 epochs 线性预热
- **混合精度训练 (AMP)：** 支持，加速训练
- **Mosaic 关闭：** 最后 10 epochs 关闭 Mosaic 增强以提升精度

#### 4.4 评估指标

- mAP@0.5 和 mAP@0.5:0.95
- Precision / Recall
- Miss Rate (行人检测专用指标)
- FPS 推理速度基准测试

---

### 5. 数据持久化

#### 5.1 数据库模型 (MySQL + SQLAlchemy ORM)

```
┌──────────────┐       ┌──────────────────┐       ┌───────────────────┐
│    User      │       │  DetectionJob    │       │  DetectionResult  │
├──────────────┤       ├──────────────────┤       ├───────────────────┤
│ id (PK)      │──┐    │ id (PK, UUID)    │──┐    │ id (PK)           │
│ username     │  │    │ user_id (FK)     │  │    │ job_id (FK)       │
│ email        │  │    │ input_type       │  │    │ frame_index       │
│ password_hash│  │    │ original_filename│  │    │ bbox_x1,y1,x2,y2  │
│ role         │  │    │ file_path        │  │    │ confidence        │
│ created_at   │  │    │ status (enum)    │  │    │ class_id          │
│ last_login   │  │    │ model_version    │  │    │ track_id          │
└──────────────┘  │    │ pedestrian_count │  │    └───────────────────┘
                  ├───>│ processing_time  │  │
                  │    │ error_message    │  │    ┌───────────────────┐
                  │    │ created_at       │  ├───>│  JobAuditLog      │
                  │    │ completed_at     │  │    ├───────────────────┤
                  │    └──────────────────┘  │    │ id (PK)           │
                  │                          │    │ job_id (FK)       │
                  │                          │    │ event             │
                  │                          │    │ detail (JSON)     │
                  │                          │    │ created_at        │
                  │                          │    └───────────────────┘
                  │
                  └── 一对多关系 (One-to-Many)
```

- **审计日志 (JobAuditLog)：** 记录每次状态变更，JSON 格式存储详细信息
- **索引优化：** user_id / status / created_at 列建立索引

#### 5.2 异步数据库访问

- SQLAlchemy 2.0 异步引擎 (mysql+aiomysql)
- 连接池：10 常驻 + 20 溢出
- 生成器模式会话管理 (自动 commit/rollback/close)

---

### 6. 部署与运维

#### 6.1 Docker Compose 三服务架构

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Frontend   │────>│   Backend    │────>│   MySQL:8.0  │
│  (Nginx:80)  │     │ (FastAPI:    │     │  (内部通信)   │
│              │     │  8000)       │     │              │
└──────────────┘     └──────────────┘     └──────────────┘
       │                     │                     │
       │  /api → 代理到后端   │  /app/models (只读)   │  mysql_data 卷
       │  /ws  → WebSocket   │  /app/media (读写)    │
       │                     │                     │
```

#### 6.2 生产环境增强 (docker-compose.prod.yml)

- 前端：CPU 1核 / 内存 256MB 限制，自动重启
- 后端：CPU 4核 / 内存 8GB 限制，健康检查 (HTTP /health 端点)
- 数据库：CPU 2核 / 内存 2GB 限制，MySQL ping 健康检查
- 持久化卷使用本地驱动

#### 6.3 环境配置

- 12-Factor App 风格，通过环境变量配置
- `.env` 文件开发环境配置模板 (`.env.example`)
- 敏感信息（密钥、密码）通过 Docker secrets 或环境变量注入

---

### 7. 安全特性

| 特性       | 实现                           |
| -------- | ---------------------------- |
| 密码存储     | bcrypt 哈希 (passlib)          |
| API 认证   | JWT Bearer Token (HS256)     |
| Token 刷新 | access + refresh token 双令牌机制 |
| 登录保护     | 基于 IP 的速率限制                  |
| 文件类型校验   | 服务端白名单验证 (MIME type)         |
| 文件大小限制   | 上传大小上限 (20MB)                |
| CORS 配置  | 可配置的跨域允许列表                   |
| 用户隔离     | 检测历史按用户隔离，管理员可管理所有记录         |
| 生产模式     | Debug 关闭，错误信息不泄露内部细节         |

---

## 三、技术架构图

```
┌─────────────────────────────────────────────────────────────┐
│                      Frontend (React 18)                    │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────────┐ │
│  │ Detection│ │  Video   │ │  Live    │ │  Job History  │ │
│  │  Studio  │ │ Detector │ │ Detector │ │   (Paginated) │ │
│  └──────────┘ └──────────┘ └──────────┘ └───────────────┘ │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Zustand Stores (detection / stream / history)       │  │
│  │  Axios Client (auto token refresh + retry queue)     │  │
│  └──────────────────────────────────────────────────────┘  │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP REST + WebSocket
┌──────────────────────┴──────────────────────────────────────┐
│                    Backend (FastAPI)                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────────┐ │
│  │  /auth   │ │ /detect  │ │ /stream  │ │ /results      │ │
│  │ Router   │ │ Router   │ │ Router   │ │ Router        │ │
│  └──────────┘ └──────────┘ └──────────┘ └───────────────┘ │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  DetectionService (orchestration)                    │  │
│  │  JobService (CRUD + audit log)                       │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  EnhancedYOLOv8 Model Wrapper                        │  │
│  │  NightPreprocessor (CLAHE + Denoise + Letterbox)     │  │
│  │  Postprocessor (NMS + Box Decode + Draw + Encode)    │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  SQLAlchemy 2.0 Async ORM + JWT Auth + Rate Limit    │  │
│  └──────────────────────────────────────────────────────┘  │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────────┐
│                   MySQL 8.0 Database                         │
│  users │ detection_jobs │ detection_results │ job_audit_logs │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│              Model Training (PyTorch + Ultralytics)          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────────┐ │
│  │  CBAM    │ │ C2fGhost │ │  BiFPN   │ │   WIoU Loss   │ │
│  │ (Attn)   │ │ (Ghost)  │ │ (Neck)   │ │   (Loss Fn)   │ │
│  └──────────┘ └──────────┘ └──────────┘ └───────────────┘ │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  NightAugmentation (Mosaic/MixUp/Darken/Noise/HSV)   │  │
│  │  LLVIP Dataset Parser (VOC XML → YOLO TXT)           │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 四、关键性能目标

| 指标                  | 目标                        |
| ------------------- | ------------------------- |
| 推理延迟 (单张图片)         | < 100ms (GPU)             |
| 实时流帧率               | ≥ 20 FPS (GPU)            |
| mAP@0.5 (LLVIP 测试集) | > 85%                     |
| mAP@0.5:0.95        | > 55%                     |
| 模型体积                | < 15MB (YOLOv8n 变体)       |
| 并发检测任务              | 8 路 WebSocket 流 + REST 请求 |

---

## 五、项目结构

```
pedestrian-detection/
├── backend/                    # FastAPI 后端服务
│   ├── main.py                # 应用入口 + lifespan 管理
│   ├── core/                  # 核心模块
│   │   ├── config.py          # Pydantic Settings 配置
│   │   ├── database.py        # SQLAlchemy 异步引擎和会话
│   │   ├── security.py        # JWT + bcrypt 认证
│   │   └── dependencies.py    # 共享 FastAPI 依赖项
│   ├── routers/               # API 路由
│   │   ├── auth.py            # 认证路由
│   │   ├── detect.py          # 检测路由
│   │   ├── results.py         # 结果和历史路由
│   │   └── stream.py          # WebSocket 实时流路由
│   ├── services/              # 业务逻辑层
│   │   ├── detection_service.py  # 检测编排
│   │   ├── job_service.py        # 任务 CRUD
│   │   └── model_loader.py       # 模型单例加载
│   ├── ml/                    # 机器学习模块
│   │   ├── model.py           # EnhancedYOLOv8 模型封装
│   │   ├── preprocessor.py    # 夜间图像预处理
│   │   └── postprocessor.py   # 后处理 (NMS + 标注)
│   ├── models/                # 数据模型
│   │   ├── schemas.py         # Pydantic API Schema
│   │   └── db_models.py       # SQLAlchemy ORM 模型
│   └── tests/                 # 测试 (待实现)
├── frontend/                   # React 前端
│   └── src/
│       ├── api/client.ts      # Axios 客户端 (含 Token 刷新)
│       ├── types/index.ts     # TypeScript 类型定义
│       ├── store/             # Zustand 状态管理
│       ├── hooks/             # 自定义 React Hooks
│       ├── components/        # UI 组件
│       │   ├── detection/     # 检测相关组件
│       │   ├── layout/        # 布局组件
│       │   └── ui/            # 通用 UI 组件
│       └── pages/             # 页面组件
├── model_training/             # 模型训练
│   ├── train.py               # 训练脚本
│   ├── evaluate.py            # 评估脚本
│   ├── configs/               # 训练配置 YAML
│   └── modules/               # 自定义 PyTorch 模块
│       ├── cbam.py            # CBAM 注意力模块
│       ├── c2f_ghost.py       # Ghost 卷积模块
│       ├── bifpn.py           # BiFPN 特征金字塔
│       └── wiou_loss.py       # WIoU 损失函数
├── docker-compose.yml          # Docker 编排
├── docker-compose.prod.yml     # 生产环境覆盖配置
└── datasets/LLVIP/             # LLVIP 数据集 (可见光+红外)
```