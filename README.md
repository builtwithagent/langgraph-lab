# LangGraph Lab

跟着 LangChain 官方课程 + 自定实操手册做的学习记录。每个目录一个任务,一次 commit。

## 进度台账

- [x] **T0** 环境 + 第一个 `create_agent`(`01-first-agent/`)
- [ ] **T1** 不用框架,手搓 agent loop
- [ ] **T2** 第一个 StateGraph
- [ ] **T3** 状态与记忆
- [ ] **T4** 持久化与断点续跑
- [ ] **T5** 人在环 interrupt
- [ ] **T6** 流式输出 + 前端界面

## 环境

```bash
# 激活虚拟环境(Git Bash)
source .venv/Scripts/activate
# 或 PowerShell
.venv\Scripts\Activate.ps1
```

## 配置

复制 `.env.example` 为 `.env` 并填入你的 API Key:

```bash
cp .env.example .env
```

`.env` 已在 `.gitignore` 里,**不会被提交**。

## 运行示例

```bash
.venv/Scripts/python 01-first-agent/agent.py
```

## 说明

- 目标不是"学完某框架",而是**把每个任务做成可展示、可复现的小实验**
- 每个任务目录下有独立 README,记录:做了什么 / 卡在哪 / 学到什么

---

*LangChain 1.x + LangGraph 1.x(2025-10-22 GA)*
