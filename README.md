# LangGraph Lab

跟着 LangChain 官方课程 + 自定实操手册做的学习记录。每个目录一个任务,一次 commit。

## ⚠️ 先读这个:[`SOURCES.md`](SOURCES.md)

本仓库所有内容(代码 / 概念 / 数字 / 建议)**都要能追到出处**。
`SOURCES.md` 里定了五级标注(`官方` / `源码` / `实测` / `推论` / `建议`)、核验方法,
以及**一份"本项目自造说法"的登记表**(哪些词不是官方术语,别拿去找资料)。

> 原因很直接:这是拿来学的东西,**写错就是学错**。
> 所以宁可标"这是我的推测",也不把推测写成好像有依据。

## 进度台账

- [x] **T0** 环境 + 第一个 `create_agent`(`01-first-agent/`)
- [x] **T1** 不用框架,手搓 agent loop(`02-raw-agent-loop/`)
- [x] **T2** 第一个 StateGraph(`03-first-graph/`)
- [x] **T3** 状态与记忆:State Schema / 裁剪 / 摘要(`04-state-memory/`)
- [ ] **T4** 持久化与断点续跑(`05-persistence/`)
- [ ] **T4b** 长期记忆 store —— 跨会话记住用户偏好(`05b-long-term-memory/`)
- [ ] **T5** 人在环 interrupt
- [ ] **T6** 流式输出 + 前端界面
- [ ] **T7** RAG Agent(主力作品)
- [ ] **T8** 评测
- [ ] **T9** 多步 + MCP(`Send` 在这里)
- [ ] **T10** 包装

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
