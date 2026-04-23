# AutoAlpha v2 — Claude Code 工作规范

## 项目概况

当前分支：`v2`  
目标：在 Scientech Alpha Quantitative Challenge 中通过 LLM 自动挖掘高质量 alpha 因子。

## 强制要求：每次修改后必须执行

1. **更新 DEV 时间线**：在 `frontend/src/data/devTimeline.ts` 追加一条新记录，说明本次修改的时间、标题、摘要、标签（2-4个）和关键改动（2-4条 bullet）。
2. **git push**：执行 `git add <修改的文件>` → `git commit -m "..."` → `git push origin v2`。

只有"结构变化、链路打通、评估口径变化、前端可见变化、模块重构"才需要记录；零碎样式调整不用单独开条目。

## 代码风格

- 不写无意义注释，变量命名自解释。
- 不引入未被当前任务需要的抽象或功能。
- 不加只为"以后可能用到"的错误处理。

## 关键路径

| 模块 | 路径 |
|------|------|
| AutoAlpha v2 研究包 | `autoalpha_v2/` |
| 前端（Vite + React） | `frontend/src/` |
| 后端 API | `frontend/backend/app.py` |
| 评估器 | `core/genalpha.py` |
| 算子库 | `factors/operators.py` |
| 提交导出 | `outputs/export_submission.py` |
| DEV 时间线数据 | `frontend/src/data/devTimeline.ts` |

## Git 规范

- 分支：始终在 `v2` 上工作。
- remote：`origin` → `https://github.com/yinshuo-thu/autoalpha.git`
- 每次 push 到 `origin v2`。
- commit message 用英文，简洁描述"做了什么"。
