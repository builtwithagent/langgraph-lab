# 来源与核验规范

> 这个项目的所有内容(代码、概念、数字、建议)**都要能追到出处**。
> 原因很直接:你以这里的内容为准来学,写错就是学错。
> 所以宁可标"这是我的推测",也不把推测写成好像有依据。

---

## 一、五级标注

每一条实质性的论断,都要能归到下面某一级。**标注写在论断旁边,不能只在文末统一说明。**

| 级别 | 标注 | 含义 | 必须附带 |
|---|---|---|---|
| **A** | `【官方原文】` | 官方文档/官方博客的原话 | 链接 + **可直接复核的原文片段** |
| **B** | `【源码】` | 从本机已安装的包里读出来的实际代码 | 文件路径 + 行号 |
| **C** | `【实测】` | 本机跑出来的数据 | 可复现的脚本 + 运行环境 |
| **D** | `【推论】` | 基于 A/B/C 的推理,**官方没说** | 显式标注 + 说明推理链 |
| **E** | `【建议】` | 经验判断,**没有权威出处** | 显式标注"这是我的建议" |

### 三条铁律

1. **D 和 E 绝不能写成 A/B 的样子。** 一旦发现自己在写"业界通常…""最佳实践是…"这种模糊兜底的话,先停下来问:这是 A 还是我在编?
2. **A 级必须给出原文片段**,不能只给链接 —— 链接会变,原文片段不会。读者要能当场对照。
3. **数字必须有出处。** 要么是 C(实测,带脚本),要么是 A(官方公布)。**不许"大约""左右"地估一个数然后当结论用。**

---

## 二、核验方法(照做就行,别凭记忆)

### 链接类

```bash
curl -s -o /dev/null -w '%{http_code}' -L --max-time 25 "<URL>"
```

返回 `200` 才算通过。返回 `000` 要区分是**域名被墙/抖动**还是**链接失效**:

```bash
# 对照测试:同域名根路径 + 其他已知可用域名
for u in https://github.com https://api.github.com https://docs.langchain.com; do
  echo "$(curl -s -o /dev/null -w '%{http_code}' -L --max-time 20 "$u")  $u"
done
```

若只有某个子域名不通、其他都通 → 环境问题,链接本身有效,**在文档里注明**。

### 源码类

游大街(走 GitHub)不如就近读本地 —— 包已经装在 venv 里,读它最快且**版本一定对得上**:

```bash
.venv/Scripts/python.exe -c "import inspect, langgraph.graph.state as m; print(inspect.getsource(m._is_field_binop))"
```

或者直接看文件:

```
.venv/Lib/site-packages/langgraph/graph/state.py
.venv/Lib/site-packages/langgraph/graph/message.py
.venv/Lib/site-packages/langchain_core/messages/utils.py
```

引用源码时**必须写清是哪个文件、哪一行**,因为版本一升行号就变 —— 所以要连版本号一起记。

### 概念/术语类

按优先级查,**第一个查到的就是主依据**:

1. 官方文档(spec / API reference)
2. 官方博客(比文档更适合"为什么"和"最佳实践")
3. 原始论文(arXiv / 会议论文)
4. 知名开源项目的实际代码

⚠️ **别用搜索引擎的二手解读当依据** —— 那些本身就是转述,可能已经错了。

### 数值类

必须本机实测,并且**把测量脚本写进仓库**,让别人(和未来的自己)能复现。

---

## 三、已核验的权威来源清单

### 官方文档

| 来源 | 链接 | 用于 |
|---|---|---|
| LangGraph · Graph API | https://docs.langchain.com/oss/python/langgraph/graph-api | state / reducer / super-step |
| LangGraph · Add memory | https://docs.langchain.com/oss/python/langgraph/add-memory | 裁剪 / 删除 / 摘要 / checkpoint |
| `trim_messages` API 参考 | https://reference.langchain.com/python/langchain-core/messages/utils/trim_messages | 参数语义 |
| 智谱 GLM-4.7-Flash 模型页 | https://docs.bigmodel.cn/cn/guide/models/free/glm-4.7-flash | 上下文窗口 200K / 最大输出 128K |

### 官方博客 / 工程文章

| 来源 | 链接 | 用于 |
|---|---|---|
| **LangChain · Context Engineering** | https://blog.langchain.com/context-engineering-for-agents/ | **Write / Select / Compress / Isolate 四分类法**、trimming vs summarization 的区别、Claude Code 的 95% 阈值 |
| **Anthropic · Effective context engineering for AI agents** | https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents | **context rot**、**attention budget**、**compaction** 的定义 |

### 论文

| 来源 | 链接 | 用于 |
|---|---|---|
| Liu et al., *Lost in the Middle: How Language Models Use Long Contexts*, TACL 2023 | https://arxiv.org/abs/2307.03172 | 长上下文里"中间的信息容易被忽略"这个结论 |

### 官方/半官方工具(仅登记,本项目未安装)

| 来源 | 链接 | 说明 |
|---|---|---|
| `langmem`(LangChain 官方记忆库) | https://pypi.org/project/langmem/ | 官方 add-memory 文档的摘要范例用了它的 `SummarizationNode`。`【实测】` 2026-10-15 元数据:版本 `0.0.30`、依赖含 langchain / langchain-anthropic / langchain-openai / langgraph / langsmith / trustcall;本项目**未安装**,理由见 T3 README 3.9 |

### 本项目内的实测脚本

| 脚本 | 产出哪些 C 级数据 |
|---|---|
| `04-state-memory/02_trim_messages.py` | token 计数公式、裁剪开销、工具 schema 开销、边界行为、稳态模拟、额度算例 |
| `04-state-memory/03_summarize_messages.py` | 摘要节点的 state 变更行为(可用 `--offline` 免联网) |

---

## 四、⚠️ 明确标注为"非权威"的说法

下面这些词/说法是**本项目为了方便理解自己起的**,或者在业界是口语,**不是官方术语**。
看到它们时要知道:概念本身有 A 级依据,但**这个词没有**。

| 本项目用的说法 | 官方/权威叫法 | 说明 |
|---|---|---|
| 「滑动窗口」 | LangChain 归在 **Compress → Context Trimming**;Anthropic 叫 **compaction** 的一部分 | "sliding window" 在 LangChain 官方博客和 Anthropic 文章里**出现 0 次**(2026-10-15 grep 实测),是社区口语说法 |
| 「孤儿 ToolMessage」 | 无官方名词;官方表述是 *an assistant message with `tool_calls` must be followed by tool messages responding to each `tool_call_id`* | "孤儿"是本项目为了让 `tool_call_id` 找不到配对这件事好记而起的 |
| 「回溯砍」 | 无官方名词 | 对 `_first_max_tokens` 里 `end_on` 过滤逻辑的通俗描述,依据是**源码 B 级** |
| 「额度」「预算」 | 官方用 *max_tokens*;Anthropic 用 *attention budget* | "预算"是本项目的比喻 |
| 「失忆」 | Anthropic 用 **context rot** | 描述同一现象,但有个更准确的名字 |
| 摘要的「6 条阈值」 | **Claude Code 用"超过上下文窗口的 95%"** | ⚠️ 之前项目里随手写的"6 条"是**举例用的假数据**,不是推荐值,见下 |

---

## 五、已修正的记录

诚实记录一下发现问题并改掉的地方(留着提醒自己别重犯):

| 日期 | 问题 | 修正 |
|---|---|---|
| 2026-10-11 | 写"设计受 Pregel 和 **Apache Beam** 启发" | 抓官方 `graph-api` 原文核对,全文无 "Beam" → 删除,改为直接贴官方原文 |
| 2026-10-13 | `count_tokens_approximately` 公式凭印象写成 `内容长度/4 + 3` | 读源码后发现漏了 `len(role)` 等;第三条测试数据与公式对不上才暴露 → 改为逐项对照源码 |
| 2026-10-15 | 说裁剪"额度不够装不下所以返回空" | 读 `_first_max_tokens` 后发现是 `end_on` **回溯砍**到空 → 重写 |
| 2026-10-15 | 用「滑动窗口」「lost in the middle」但**没给出处** | grep 官方文档发现两词均 0 次 → 补 `arXiv:2307.03172`;「滑动窗口」降级为"非权威说法"并登记在第四节 |
| 2026-10-15 | 摘要触发阈值举例写"6 条",容易被当成推荐值 | 补权威依据:**Claude Code 在超过上下文窗口 95% 时触发 auto-compact**(LangChain 官方博客原文),并明确"6 条"是示意 |
| 2026-10-15 | 摘要节点写成像是本项目自己的设计 | 抓官方 `summarize_conversation` 原文逐条对照,确认**结构完全一致**(`[:-2]`、增量摘要、先算再删),并把对照片段放进 T3 README 3.7 与脚本 `part_6` |
| 2026-10-15 | 摘要"该留什么、丢什么"只有我的建议 | 补 Anthropic 官方原文:Claude Code 保留 *architectural decisions, unresolved bugs, implementation details*,丢弃 *redundant tool outputs or messages*;并补官方警告 *overly aggressive compaction can result in the loss of subtle but critical context* |

---

## 六、遇到"查不到依据"怎么办

**不要写。** 三条处理方式,按顺序:

1. **标 `【待核实】`** 并说明卡在哪(比如"官方文档未提及此参数")—— 留个位置,以后查到再补
2. **标注为 `【推论】`/`【建议】`**,并写清推理链,让读者自己判断
3. **直接删掉** —— 如果它既不是必需的,又找不到依据,那就是在凑字数

> 宁可少写,不可写错。**这份文档的价值在于"每一句都可信",不在于"覆盖得多"。**
