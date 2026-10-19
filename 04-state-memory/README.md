# T3 · 状态与记忆(State Schema / 裁剪 / 摘要)

> 对应实操手册 **T3**:
> 定义 State Schema,用 `add_messages` 之类 reducer 处理状态更新,做消息裁剪/摘要。
> **完成标准**:长对话不爆上下文;能解释 reducer 为什么必要。

---

## 📌 本文的来源标注

本文每一条实质结论都标注了出处级别,规则见仓库根目录 **[`SOURCES.md`](../SOURCES.md)**。

| 标注 | 含义 |
|---|---|
| `【官方】` | 官方文档 / 官方博客,**附原文片段** |
| `【源码】` | 从本机 venv 里读出的实际代码,附文件路径 |
| `【实测】` | 本机跑出的数据,脚本在 `04-state-memory/` 下可复现 |
| `【推论】` | 基于前者的推理,**官方没说** |
| `【建议】` | 经验判断,**无权威出处** |

> ⚠️ **本项目自己起的说法**(如"滑动窗口""孤儿 ToolMessage""回溯砍")在 `SOURCES.md` 第四节
> 有登记 —— 概念本身有依据,但**这些词不是官方术语**,别拿它们去找资料。

---

## 🧭 先定位:裁剪和摘要在官方框架里的位置

**`【官方】`** LangChain 官方博客把上下文管理分成四类:

> "we break down some common strategies — **write, select, compress, and isolate** — for context engineering"
> —— https://blog.langchain.com/context-engineering-for-agents/

| 类别 | 做什么 | 本项目覆盖 |
|---|---|---|
| Write | 把信息**写出去**(存文件、存记忆) | T3 的 `summary` 字段 |
| Select | 把信息**拉进来**(检索、挑要用的) | 部分(裁剪属于"挑留哪些") |
| **Compress** | **只保留完成任务必需的 token** | ✅ **本文件的主体** |
| Isolate | 把上下文**拆开**(多 agent、子上下文) | 后续 T9 |

**Compress 下面又分两种,官方明确区分了它们的差别:**

> **Context Summarization**:"Agent interactions can span hundreds of turns and use token-heavy tool calls.
> Summarization is one common way to manage these challenges."
>
> **Context Trimming**:"Whereas summarization typically **uses an LLM to distill** the most relevant pieces of
> context, trimming can often filter or, as Drew Breunig points out, '**prune**' context.
> This can use **hard-coded heuristics like removing older messages from a list**."
> —— https://blog.langchain.com/context-engineering-for-agents/

**`【实测】`** 这两句正好对应本文件的两半:

| | 用 LLM 吗 | 手段 | 本项目 |
|---|---|---|---|
| Context Trimming | ❌ 不用 | 硬编码规则(从列表里移除更老的消息) | `02_trim_messages.py` |
| Context Summarization | ✅ 用 | 让模型蒸馏出要点 | `03_summarize_messages.py` |

**为什么"只保留必需的 token"这件事重要** —— Anthropic 的表述:

> "Context, therefore, must be treated as a **finite resource** with **diminishing marginal returns**.
> … LLMs have an '**attention budget**' that they draw on when parsing large volumes of context.
> **Every new token introduced depletes this budget by some amount**."
> —— https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents

**这个现象有个正式名字,叫 context rot:**

> "as the number of tokens in the context window increases, the model's ability to accurately recall
> information from that context **decreases**. While some models exhibit more gentle degradation than others,
> **this characteristic emerges across all models**."
> —— 同上

---

## 🎯 本节的范围:T3 **不等于**"记忆系统的设计与实现"

先把边界划清楚,免得学完以为全了。

`【官方】` LangGraph 的 Memory 文档把记忆分成**两大类**:

> **短期记忆**:"**Short-term** memory (**thread-level persistence**) enables agents to track
> multi-turn conversations." —— 同一轮会话内的多轮对话
>
> **长期记忆**:"Use long-term memory to store user-specific or application-specific data
> **across conversations**." —— 跨会话
> —— https://docs.langchain.com/oss/python/langgraph/add-memory

`【官方】` 官方目录结构是这样的(章节名一字未改):

```
## Add short-term memory        ← 怎么"存下来"(checkpointer + thread_id)
## Add long-term memory         ← 跨会话的 store
## Manage short-term memory     ← 怎么"管理"已经存下来的消息
    ### Trim messages
    ### Delete messages
    ### Summarize messages
    ### Manage checkpoints
```

**本项目 T3 只做了其中一格:**

| 层次 | 要解决的问题 | 怎么做 | 在哪学 |
|---|---|---|---|
| 短期记忆 | **存下来** —— 让 state 跨轮次活着 | `checkpointer` + `thread_id` | **T4**(尚未学) |
| 短期记忆 | **管理它** —— 太长怎么办 | `trim` / `delete` / `summarize` | ✅ **T3(本文件)** |
| 长期记忆 | **跨会话记住** —— 记住用户偏好等 | `store` | ⚠️ **T0–T10 计划里没有专门覆盖** |

### 所以准确的表述是

> **T3 = "短期记忆的上下文管理"的设计与实现。**
> 它不是完整的记忆系统 —— 缺了"怎么存"(T4)和"跨会话的长期记忆"(计划缺口)。

### 为什么可以"先学管、后学存"

`【推论】` 因为裁剪和摘要**不依赖 checkpointer 也能演示** —— 你给一个手写的 messages 列表就能跑。

但要注意一个后果:**摘要的价值要等 T4 才真正兑现**。
现在摘要在一次 `invoke` 结束就没了(没有 checkpointer 就没地方存),
只有上了 checkpointer,"跨轮次记住"这件事才成立。

---

## 做了什么

| 文件 | 干什么 | 要调模型吗 |
|---|---|---|
| `01_state_schema.py` | State Schema 与 reducer 的机制、签名规则、`add_messages` vs `operator.add`、**为什么清空不能写 `= []`** | ❌ 秒出 |
| `02_trim_messages.py` | 消息裁剪:`trim_messages` + `count_tokens_approximately`,12 段:参数对比 / 两个坑 / **SystemMessage 去留** / **`start_on` 真实语义** / **触发时机** / **滑动窗口稳态** / **额度怎么算** / **额度怎么估** | ❌ 秒出 |
| `03_summarize_messages.py` | 消息摘要:6 段 —— `RemoveMessage` 两种删法 / 摘要节点 / 条件边触发 / **对照官方范例** / **该留什么丢什么** / **现成方案 langmem** | ✅ 要(可用 `--offline`) |

跑法:

```bash
cd D:/VibeCoding/langgraph-lab

.venv/Scripts/python.exe 04-state-memory/01_state_schema.py
.venv/Scripts/python.exe 04-state-memory/02_trim_messages.py
.venv/Scripts/python.exe 04-state-memory/03_summarize_messages.py            # 真调模型
.venv/Scripts/python.exe 04-state-memory/03_summarize_messages.py --offline  # 不联网
```

---

## 📎 知识点补齐(卡住就点开)

| 概念 | 入口 |
|---|---|
| State Schema / reducer 官方文档 | https://docs.langchain.com/oss/python/langgraph/graph-api#schema |
| **记忆系统全篇**(裁剪 / 删除 / 摘要 / checkpoint 管理) | https://docs.langchain.com/oss/python/langgraph/add-memory |
| `trim_messages` API 参考 | https://reference.langchain.com/python/langchain-core/messages/utils/trim_messages |
| `count_tokens_approximately` API 参考 | https://reference.langchain.com/python/langchain-core/messages/utils/count_tokens_approximately |
| **`trim_messages` 与 `_last_max_tokens` 源码**(`include_system` / `start_on` 的真相都在这) | https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/messages/utils.py |
| **GLM-4.7-Flash 官方参数**(上下文窗口 200K / 最大输出 128K) | https://docs.bigmodel.cn/cn/guide/models/free/glm-4.7-flash |
| `add_messages` reducer 源码 | https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/graph/message.py |
| Python `typing.Annotated`(不懂这个就看不懂 reducer 声明) | https://docs.python.org/3/library/typing.html#typing.Annotated |
| Python `operator` 模块(`operator.add` 是什么) | https://docs.python.org/3/library/operator.html |
| Python `inspect.signature`(看 reducer 签名规则用) | https://docs.python.org/3/library/inspect.html#inspect.signature |
| 官方免费课 Module 2(状态与记忆) | https://academy.langchain.com/courses/intro-to-langgraph |
| ⭐ **LangChain 官方博客 · Context Engineering**(Write/Select/Compress/Isolate 四分类、trimming vs summarization、Claude Code 的 95% 阈值) | https://blog.langchain.com/context-engineering-for-agents/ |
| ⭐ **Anthropic · Effective context engineering for AI agents**(context rot、attention budget、compaction) | https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents |
| ⭐ **论文** · Lost in the Middle: How Language Models Use Long Contexts(TACL 2023) | https://arxiv.org/abs/2307.03172 |
| 本仓库的来源标注规范 | [`../SOURCES.md`](../SOURCES.md) |

> 💡 **看源码别只走 GitHub** —— `github.com` 在国内经常连不上(2026-10-15 实测过:
> `github.com` 返回 000,但 `api.github.com` 正常)。既然包已经装在本机,
> 直接读 venv 里的源码最快、最稳,而且**版本一定对得上你装的那份**:
>
> ```
> .venv/Lib/site-packages/langchain_core/messages/utils.py     ← trim_messages / _last_max_tokens
> .venv/Lib/site-packages/langgraph/graph/message.py           ← add_messages / RemoveMessage
> .venv/Lib/site-packages/langgraph/graph/state.py             ← reducer 判定逻辑(:1825 / :1904)
> ```
>
> 命令行看某个函数的源码:
>
> ```bash
> .venv/Scripts/python.exe -c "import inspect, langchain_core.messages.utils as u; print(inspect.getsource(u._last_max_tokens))"
> ```

---

## 一、State Schema:字段类型后面那个 `Annotated` 是干什么的

### 1.1 先接住直觉:`TypedDict` 就是"给字典的键标上类型"

```python
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    steps:    Annotated[list[str], operator.add]
    summary:  str
```

- `AgentState` 本质还是个 dict,冒号后面是**类型标注**,运行时不做校验。
- 框架要拿它做的事:**在每个 super-step 边界,决定每个字段的"合并方式"**。

### 1.2 三种字段,三种表现(实测)

| 字段写法 | 合并行为 | 谁在管 |
|---|---|---|
| `messages: Annotated[list, add_messages]` | 按 id **upsert**(重复 id 就地替换)+ 支持删除 | `add_messages` |
| `steps: Annotated[list, operator.add]` | 单纯**追加**(list + list) | `operator.add` |
| `summary: str`(没有 `Annotated`) | **后写覆盖先写** | `LastValue` |

**实测输出**:

```
messages = ['HumanMessage', 'AIMessage']
steps    = ['读了输入', '干了点活']
summary  = '第二版摘要(覆盖掉了上一版)'   ← 没写 Annotated 就是这个下场
```

👉 **一句话**:没写 `Annotated` 的字段 = 每个节点都能覆盖它,谁最后跑完谁说了算。

### 1.3 机制挖到源码:框架怎么知道要用 `operator.add`?

这是 T3 里最值钱的一段。结论:**框架根本不认 `operator.add` 这个名字。**

`langgraph/graph/state.py` 里三处关键代码(行号按 `langgraph 1.2.14` 实测核对):

```python
# :1825  —— 读类型标注时必须加 include_extras,否则 Annotated 里的元数据会被剥掉
type_hints = get_type_hints(schema, include_extras=True)

# :1867  —— 逐个字段判断:能不能当 binop 用
elif channel := _is_field_binop(annotation):

# :1904  —— _is_field_binop():判断"这个东西能不能当 reducer"
meta[-1]                       # 只看最后一个元数据
callable(...)                  # 必须是可调用的
len(positional_params) == 2    # 位置参数必须正好 2 个

# :1920  —— 不满足就在这里抛
f"Invalid reducer signature. Expected (a, b) -> c. Got {sig}"
```

三处必须全部通过 → 用 `BinaryOperatorAggregate(typ, meta[-1])` 建这个字段;
任何一个不过 → **静默退化成 `LastValue`**(覆盖;或者签名不合法时直接抛错)。

所以:

- 源码里**从头到尾没有出现过 `operator.add` 这个标识符**。
- `operator.add`、`add_messages`、你自己写的任何函数,在框架眼里**是同一种东西**。
- 判断标准只有一条:**签名是不是 `(a, b) -> c`。**

### 1.4 reducer 签名规则(实测,别背错)

**只数 `POSITIONAL_ONLY` / `POSITIONAL_OR_KEYWORD` 位置参数的个数,必须 == 2。**
**不看参数名、不看类型注解、不看函数体。**

| 签名 | 结果 |
|---|---|
| `def f(a, b)` | ✅ 通过 |
| `def f(existing, new)` | ✅ 通过(名字随便起) |
| `def f(a, b=1)`(带默认值) | ✅ 通过(还是 2 个位置参数) |
| `def f(x)` | ❌ 只有 1 个 |
| `def f(*args)` | ❌ `*args` 算 0 个 |
| `def f(*, new)` | ❌ 关键字专用算 1 个 |

签名写错,**在建图/编译时就抛**,不会等到运行时:

```
ValueError: Invalid reducer signature. Expected (a, b) -> c. Got (x)
```

### 1.5 reducer 的契约:`(旧值, 新值) -> 新值`

- 调用形式永远是 `reducer(existing, new)`。
- **开局还会被调一次** `reducer([], [])` —— 初始 state 也会走一遍 reducer,所以你的 reducer 必须能接受两个空值。

### 1.6 `add_messages` 的四条合并规则(实测)

1. 新 id → **追加**
2. **重复 id → 就地替换那一格**(不是追加!)
3. `RemoveMessage(id=...)` → **删除**
4. 一次给多条 → 逐条处理

👉 关键:**它不是 `list += `,它是"按 id 的 upsert + 支持删除"**。

### 1.7 `RemoveMessage` 的三种姿势

```python
# ① 删单条
RemoveMessage(id="some-uuid")

# ② 传一列删多条
[RemoveMessage(id=m.id) for m in state["messages"][:-2]]

# ③ 一键清空
from langgraph.graph.message import REMOVE_ALL_MESSAGES
RemoveMessage(id=REMOVE_ALL_MESSAGES)
```

⚠️ 用 `REMOVE_ALL_MESSAGES` 的那一次更新,**不能混非删除消息**。

### 1.8 `operator.add` vs `add_messages`:契约相同,语义不同

这是 `01_state_schema.py` 第 9 部分做的对照实验:

| 场景 | `Annotated[list, operator.add]` | `Annotated[list, add_messages]` |
|---|---|---|
| 同一条消息(id 相同)再交一次 | 变成 **3 条**(重复) | 变成 **2 条**(就地替换) |
| 交一个 `RemoveMessage` | **当成普通消息堆进去,不报错** | **真删掉** |

💥 **实战坑**:把 `messages` 写成 `Annotated[list, operator.add]` 的症状是
"**消息删不掉、改不掉、越滚越长**",而且**完全不报错**。

### 1.9 ⚠️ 想清空 `messages`:`= []` 和 `return []` **全都不行**

这是 T3 里最反直觉的一个点:**三种"直觉写法"全部静默失效**。
`01_state_schema.py` 第 10 部分做了完整实测。

#### 先看 reducer 本身(不涉及图)

`【实测】` `add_messages(旧列表, 新列表)` 的语义是"把新列表**合并**进旧列表":

```
旧列表              : [HUM:m1, AI:m2]   (2 条)
add_messages(旧, []) -> [HUM:m1, AI:m2]   (2 条)   ← 一条没少
```

**合并一个空列表 = 什么都没发生。** 这就是根因。

#### 图里四种写法实测

起始 state 有 2 条消息:

| 写法 | 结果 |
|---|---|
| **A.** `return {"messages": []}` | ❌ 没清掉,还是 2 条 |
| **B.** `state["messages"] = []` 然后 `return {}` | ❌ 没清掉,还是 2 条 |
| **C.** `state["messages"] = []` 然后 `return dict(state)` | ❌ **没清掉,还是 2 条** |
| **D.** `return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}` | ✅ 清空了 |

#### 三种失败各是什么原因

**A:`return {"messages": []}`**

> 你的意思是"**往 messages 里加 0 条**",不是"**把 messages 设成空**"。
> 框架把它交给 reducer:`add_messages(旧, [])` → 旧列表原样返回。
> **不报错,也不生效。**

**B:节点里 `state["messages"] = []`,再 `return {}`**

> 节点拿到的 state 是**当次读出来的快照**。
> 你在里面改,改的是"你手里这份";框架在节点返回后**不会把它写回去**——
> 它只看你 `return` 了什么。而你返回的是空更新,所以图里一点没动。

**C:节点里改完,再 `return dict(state)`** ← **最反直觉的一条**

> 看起来是"我把改好的整个 state 交回去了",但**框架仍然把它当更新指令**,
> 还是走 `add_messages(旧, [])` —— 依然是"加 0 条",依然什么都没发生。

#### 一句话记住

> **节点的返回值永远是"我要改哪些字段、增量是什么",永远不是"新的 state 长这样"。**

在 LangGraph 里**你不是在"操作 state",你是在"提交变更"**。
清空 = 提交一条"全删"变更;不是提交一个空 state。

#### ✅ 正确姿势

```python
from langgraph.graph.message import REMOVE_ALL_MESSAGES

# 清空
return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}

# 删指定的几条(点名 id)
return {"messages": [RemoveMessage(id=m.id) for m in state["messages"][:-2]]}
```

> 💡 这也解释了 1.8 那条实战坑的另一面:
> `operator.add` 和 `add_messages` **都不认"空列表=清空"**。
> 想删,只能在 reducer 能理解的语义里表达 —— 也就是 `RemoveMessage`。

---

## 二、消息裁剪:`trim_messages`

### 2.1 最重要的认知:裁剪**不需要**调模型

`trim_messages` 要一个"算 token"的东西(`token_counter` 参数)。
最常用的是 `count_tokens_approximately` —— **纯字符估算,零成本、毫秒级、不联网**。

#### 真实公式(读源码得到,我自己第一版写错了)

```
每条消息:
    chars  = len(content)
    chars += len(repr(tool_calls))     # 仅 AI 且带工具调用
    chars += len(tool_call_id)         # 仅 ToolMessage
    chars += len(role)                 # ★ 关键:system=6 / user=4 / assistant=9 / tool=4
    chars += len(name)                 # 仅当有 name 且 count_name=True
    token  = ceil(chars / chars_per_token)      # 默认 4
    token += extra_tokens_per_message           # 默认 3
```

> ⚠️ 我第一版想当然写成 `len(content)/4 + 3`。实测三条内容长度 10 / 11 / 5,
> 结果**全是 7** —— 内容和长度差一倍、结果一样,说明公式漏了东西。
> 去读源码才发现漏的是 **role 字符串长度**。

手算 vs 实测(可复现):

| 类型 | content 字符 | role 长 | chars | 手算 | 实测 |
|---|---|---|---|---|---|
| SystemMessage | 10 | 6 | 16 | 7 | 7 ✅ |
| HumanMessage | 11 | 4 | 15 | 7 | 7 ✅ |
| AIMessage | 5 | 9 | 14 | 7 | 7 ✅ |

**两处额外开销也实测了**:

- `AIMessage` 内容都为空串:不带 `tool_calls` → 6 token;**带 → 28 token**
- `ToolMessage` 内容都是 5 字符:`tool_call_id='c1'` → 6;**`tool_call_id` 长 18 → 9**

### 2.2 四种参数搭配的实测对比

用 17 条带工具调用的历史(约 120 token):

| 参数 | 保留 | 首条类型 | 结果 |
|---|---|---|---|
| 不设 `start_on`, `max_tokens=40` | 2 条 | **ToolMessage** | 💥 孤儿 tool_call_id |
| `start_on="human"`, `max_tokens=40` | **0 条** | — | 💥 空列表 |
| `start_on="human"`, `max_tokens=80` | 4 条 | HumanMessage | ✅ |
| C + `include_system=True`, `max_tokens=80` | 5 条 | SystemMessage | ✅ |

### 2.3 💥 坑 1:不设 `start_on` → 裁出"孤儿 ToolMessage"

`ToolMessage` 靠 `tool_call_id` 跟在发起它的 `AIMessage` 后面。
如果 AI 那一半被裁掉、只剩 Tool 这一半,`tool_call_id` 就成了孤儿 ——
发给模型**直接 400**:

```
An assistant message with 'tool_calls' must be followed by tool messages
responding to each 'tool_call_id'
```

✅ **修法**:永远传 `start_on="human"`(或 `"system"`)——
告诉它"裁剪后的开头只能是 Human / System",它就绝不会在工具调用链中间下刀。

### 2.4 💥 坑 2:`max_tokens` 抠到临界 → **返回空列表**

固定 `start_on="human"`,只调额度:

| `max_tokens` | 20 | 30 | 40 | **50** | **55** | 60 | 80 | 100 | 200 |
|---|---|---|---|---|---|---|---|---|---|
| 返回条数 | 0 | 0 | 0 | **0** | **0** | 4 | 4 | 4 | 8 |

**为什么空**,见下面 **2.7**(`start_on` 是"回溯砍",不是"找到就停")。

**空的后果比报错还阴**:`model.invoke([])` 有的模型 400、有的返回一句莫名其妙的话;
`state["messages"][0]` 直接 `IndexError`。**错误信息指向的地方,跟真正的病因完全不搭边。**

### 2.5 官方推荐写法 + `end_on`

```python
def call_model(state: MessagesState):
    messages = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=128,                      # ⚠️ 官方教学用的极小值,别照抄
        start_on="human",
        end_on=("human", "tool"),
    )
    return {"messages": [model.invoke(messages)]}
```

- `start_on` 管**头**,`end_on` 管**尾**。
- `end_on=("human", "tool")`:裁剪后最后一条只能是 Human 或 Tool
  —— 也就是**不能以一条"只说话没动作"的 AIMessage 收尾**,
  否则模型会以为自己上一句还没说完。

### 2.6 SystemMessage 会不会被裁掉?`include_system` 的真相 ⚠️

**先说结论:默认会丢,而且丢得悄无声息。**

裁剪是"从后往前尽量收,再从收进来的那批**头上往回砍**"。
而排在**列表最开头**的 SystemMessage(通常写着"你是谁、要守什么规矩"),
天然就是最容易被砍掉的那个。

实测(9 条历史 / 约 120 token,首条是 9 token 的系统提示):

| 场景 | 参数 | 结果 | 首条 |
|---|---|---|---|
| 1 | 默认(`include_system=False`) | `HUM \| AI \| TOOL \| AI` | HumanMessage ❌ **系统提示没了** |
| 2 | `include_system=True` | `SYS \| HUM \| AI \| TOOL \| AI` | SystemMessage ✅ 保住了 |

#### 💥 但开启它是要付代价的:系统提示要"抢额度"

| `max_tokens` | 30 | 45 | 50 | 55 | 60 | 65 | **70** | 80 | 120 |
|---|---|---|---|---|---|---|---|---|---|
| 总条数 | 1 | 1 | 1 | 1 | 1 | 1 | **5** | 5 | 9 |
| 其中对话 | 0 | 0 | 0 | 0 | 0 | 0 | **4** | 4 | 8 |

`max_tokens ≤ 65` 时,**只剩一条光杆系统提示,对话一条不剩** ——
而且同样**不报错**。

#### 源码就三行(`langchain_core.messages.utils._last_max_tokens`)

```python
# ① 只有「列表第一条」是 SystemMessage 时才特殊照顾
if include_system and len(messages) > 0 and isinstance(messages[0], SystemMessage):
    system_message = messages[0]     # 摘出来单独保管
    messages = messages[1:]          # 从待裁列表里拿掉

# ② 它要先占走一份额度
remaining_tokens = max(0, max_tokens - system_tokens)

# ③ 裁完塞回最前面
if system_message:
    result = [system_message, *result]
```

#### 三个结论

1. **默认(False)确实会丢**,而且是"悄无声息地丢"——不报错、不警告,
   你只会发现模型突然不守规矩了。
2. **只认第一条**。判断条件是 `messages[0]`,不是"任意一条"。
   实测:把 SystemMessage 挪到中间,`include_system=True` **完全失效**。
3. **它抢额度**。极端情况下把对话挤到一条不剩。

#### 实战三条

- 系统提示**永远放在 `messages[0]`**(几乎所有框架都这么假设);
- 用 `include_system=True`,并**留出系统提示那份额度**;
- 系统提示尽量短 —— 它是**每轮都要付的固定成本**,2000 token 的系统提示等于每轮先烧 2000。

### 2.7 `start_on` 的真实语义是「回溯砍」,不是「找到就停」⚠️

直觉上会以为 `start_on="human"` 是:
"从后往前找,碰到第一条 HumanMessage 就停,从它开始往后全保留。"

**不对。** 真实过程是四步:

```
第一步  额度内尽量从后往前收,能收多少收多少
第二步  检查收进来的这批,头上那条是不是 HumanMessage
第三步  不是 → 砍掉头上那条,再看新的头上那条;是 → 停
第四步  一路砍到空 → 返回空列表
```

**先收一批,再往回砍** —— 这就是"空列表"的真正成因。

额度从大到小的实测:

| `max_tokens` | 200 | 120 | 90 | 80 | 70 | 60 | **55** | 50 | 45 |
|---|---|---|---|---|---|---|---|---|---|
| 条数 | 8 | 8 | 4 | 4 | 4 | 4 | **0** | 0 | 0 |

**这张表有两件事最该看懂:**

1. **额度大的那几档(200、120)结果不是全保留** —— 头上那条 SystemMessage
   照样被砍。因为 `start_on="human"` 要求"头一条必须是 HumanMessage",
   系统提示不满足,**哪怕额度用不完也要砍**。
   ⚠️ 这就是最阴的地方:**你以为"额度够就不会有事",其实照样出事。**
2. **额度缩小到某个点,结果不是"保留得更少",而是突然变成空。**

走一遍具体例子(假设额度只够收进 `[AI, TOOL, AI]`):

```
收进来: [AI, TOOL, AI]    头上是 AI     ✗ 砍掉
砍一刀: [TOOL, AI]        头上是 TOOL   ✗ 砍掉
砍一刀: [AI]              头上是 AI     ✗ 砍掉
砍一刀: []                → 返回空列表
```

三个"看起来还能用"的消息,**最后一条都没剩下**。

#### ✅ 生产写法:两道兜底

```python
def safe_trim(messages, max_tokens=2000):
    if not messages:
        return messages
    trimmed = trim_messages(
        messages,
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=max_tokens,
        start_on="human",        # 避开坑 1(孤儿 ToolMessage)
        include_system=True,     # 避开 2.6(系统提示被砍)
    )
    # 兜底 1:裁到一条不剩 —— 额度太小
    if not trimmed:
        return messages
    # 兜底 2:只活下来系统提示,对话全被挤没了 —— 同样不能用
    #         ⚠️ 这种情况"判空"拦不住,因为结果不是空的
    if all(isinstance(m, SystemMessage) for m in trimmed):
        return messages
    return trimmed
```

### 2.8 ⏰ 触发时机:到底什么时候该裁?

#### 先纠正一个常见误解

很多人以为裁剪是"**防爆机制**":等上下文快超过模型窗口了,才裁一刀救火。

**不对。`max_tokens` 不是模型的上下文窗口,是你自己设的预算。**

| | 是什么 | 典型的量级 | 超了会怎样 |
|---|---|---|---|
| 模型上下文窗口 | **红线** | 128K ~ 200K | API 直接报错 |
| 你设的 `max_tokens` | **预算** | 2K ~ 8K | —— |

两者差了一两个数量级。裁剪的界线画在**预算**上,离红线远得很。
所以"每轮调模型前都裁"不是"不管三七二十一先砍一刀",而是
"**始终把上下文压在预算以内**"——是**常态控制**,不是救火。

> 📌 官方文档的文字部分说的是"count tokens and truncate whenever it **approaches that limit**"
> (接近上限时才裁),但它给的**代码例子**是无条件裁。两者不一致,见下面的实测。

#### 8.1 关键实测:历史本来就装得下,`trim` 会不会照样动手?

历史 `SYS | HUM | AI` 一共 **20 token**,故意给一个用不完的额度 `max_tokens=9999`:

| 传了哪些边界 | 结果 | 判定 |
|---|---|---|
| **都不传** | `SYS \| HUM \| AI` | ✅ **原样返回,什么都没做** |
| 只传 `start_on="human"` | `HUM \| AI` | ❌ 被砍了 3 → 2 条 |
| 只传 `end_on=("human","tool")` | `SYS \| HUM` | ❌ 被砍了 3 → 2 条 |
| **两个都传**(官方例子) | `HUM` | ❌ 被砍了 3 → 1 条 |

**💥 这张表是本节最值钱的东西:**

- **不传 `start_on` / `end_on`** → 装得下就**原样返回**,一条不动。
  说明 `trim_messages` 内部有"够了就收手"的短路逻辑。
- **只要传了任意一个边界** → 短路逻辑之后**仍然会做边界过滤**,
  哪怕额度根本用不完,它照样按规则砍。
- 官方例子**两个边界都传了**,所以那个写法实际上是"**每一轮都会动刀**"。

> ⚠️ 但这不是 bug。`end_on=("human","tool")` 存在的意义是
> "保证发给模型的最后一条不是 AI 自己刚说的那句话"——它是**故意**要动刀的。

#### 8.2 "每轮都裁"会不会很慢?实测

100 条消息(含工具调用)的规模:

```
裁剪一次(含二分查找): 约 0.25 ms
光算一次 token:        约 0.07 ms
→ 裁剪内部要调用约 log2(100) ≈ 6 次计数,倍数对得上
```

**每轮给裁剪留 1 毫秒都嫌多。** 所以"每轮都裁"在性能上完全不是问题。
真正的成本不是 CPU,是**它裁掉的东西**。

#### 8.3 能不能用"模型自带的精确计数"代替估算?

| 方案 | 开销 | 说明 |
|---|---|---|
| `count_tokens_approximately` | ~0.07 ms | 纯字符串、本地、零依赖 |
| tiktoken(模型 `gpt-4o`) | ~0.6 ms | 本地 tiktoken,**慢 8 倍** |
| 模型 `glm-4.7-flash` | **不能用** | `NotImplementedError`:该模型未实现计数接口 |

**最后一行是关键:你想"精确计数"也做不到**——很多模型(包括智谱)根本没实现这个接口。

所以 `count_tokens_approximately` 不是"退而求其次的将就方案",
而是**唯一可行的方案**。它是估算(可能有 10~20% 误差),
但裁剪本来就要"留余量",留余量这件事刚好把误差吃掉了。

#### 8.4 完整答案:其实是"两个独立的触发点"

```
┌──────────────────────────────────────────────────────────────────┐
│ 触发点 1 · 每一轮调模型之前                                        │
│   动作:裁剪 trim                                                 │
│   目的:保证这一次请求不超预算 / 不超窗口                           │
│   特点:高频 · 零成本 · 掉了就真没了                                │
├──────────────────────────────────────────────────────────────────┤
│ 触发点 2 · 消息数或 token 超过阈值                                 │
│   动作:摘要 summarize                                            │
│   目的:在"要丢东西"之前,先把结论压出来保住                         │
│   特点:低频 · 要调模型 · 有损但保留语义                            │
└──────────────────────────────────────────────────────────────────┘
```

**这正是最容易被搞混的地方**:
- 裁剪是"**每轮都做**"的日常动作
- 摘要是"**攒够了才做**"的定期动作

#### "阈值触发裁剪"这种写法呢?

也有人这么写(官方文档文字部分描述的就是这个):

```python
if 当前 token > 阈值:
    才裁
```

它跟"每轮都裁"的差别:

- ✅ 省下每次的 0.25 ms(**真的可以忽略**)
- ❌ **风险**:如果某一条消息特别大(用户粘贴一整篇文档),
  上一次检查时还不到阈值,这一次直接冲爆 ——
  你得在**调用失败之后**才知道爆了。

所以现在的做法一般是:

> **每轮都裁(便宜的常态控制) + 超阈值时摘要(贵的效果补救)**

**裁剪保证"不炸",摘要保证"不忘"——两件事,别指望一个机制干完。**

### 2.9 📉 固定额度 + 多轮对话 = 滑动窗口稳态

`max_tokens` 是个**固定值**。那对话变长之后会怎样?直接模拟 20 轮
(`max_tokens=150`,故意设小方便观察):

| 轮次 | state 条数 | state token | 请求条数 | 请求 token | 本次动作 |
|---:|---:|---:|---:|---:|---|
| 1 | 3 | 33 | 3 | 33 | 没裁,全留着 |
| 5 | 11 | 125 | 11 | 125 | 没裁,全留着 |
| 6 | 13 | **148** | 13 | 148 | 没裁(刚好卡在 150 之下) |
| **7** | 15 | 171 | **13** | **148** | **砍 2 条,窗口从第 2 轮开始** |
| 8 | 17 | 194 | 13 | 148 | 砍 4 条,窗口从第 3 轮开始 |
| 11 | 23 | 265 | 13 | 150 | 砍 10 条,窗口从第 6 轮开始 |
| 15 | 31 | 361 | 11 | 130 | 砍 20 条,窗口从第 11 轮开始 |
| 20 | 41 | **481** | 11 | **130** | 砍 30 条,窗口从第 16 轮开始 |

#### 三个阶段

1. **前 6 轮:完全没裁** —— state 和请求一模一样。这就是"消息少,永远用不到额度"。
2. **第 7 轮起:开始裁** —— 累计 token 越过了 150。
3. **之后每一轮:都在裁,而且裁掉的条数恒定。** 你说的"不停地截",**完全正确**。

#### 为什么会"恒定"

裁剪的目标是"裁到不超过 `max_tokens`"。裁到位之后,下一轮新消息进来 → 又超了
→ 再裁回 `max_tokens` 附近。**进多少、出多少,总量卡住。**

```
  老消息 ← 被挤出去 ┊ 新消息 → 进来
  ───────────────────┊────────────────
                     ┊
              窗口大小恒定(≈ max_tokens)
```

> ⚠️ **「滑动窗口」这个叫法是本项目的通俗说法,不是官方术语。**
> `【实测】` 2026-10-15 grep 官方 add-memory 文档与 LangChain / Anthropic 两篇上下文工程文章,
> "sliding window" 均出现 **0 次**。
> **`【官方】`** 官方对这个动作的表述是
> "**hard-coded heuristics like removing older messages from a list**"(LangChain 官方博客
> *Compressing Context → Context Trimming*)。
> 概念本身有依据,但**这个词没有** —— 拿它去找资料会找不到。登记在 `SOURCES.md` 第四节。

#### ⚠️ 但有两列数字的走向完全不同

| | 稳态之后 |
|---|---|
| **请求 token** | 卡住不动(就在 150 附近) |
| **state token** | **一直在涨,永远不回落** |

因为官方那个 `call_model` **只裁发给模型的那份,根本没动 state 里的 messages**。

**两个推论:**

1. `state.messages` 会**无限增长** —— 上了 checkpointer 之后,这条 thread 的记录一直变大。
2. 每轮裁剪要处理的列表越来越长(实测很快,可忽略)。

#### 完整心智模型

```
state.messages  ← 一直在长,像一本完整的会议纪要
      │
      ├── 裁剪 ──→ 只发最近的 N 条给模型(滑动窗口)
      │
      └── 摘要 ──→ 把"即将被挤出去"的那批压成一段话
                   (这一步会真的删 state,见 03 那份脚本)
```

> **裁剪管的是"这一次发什么",摘要管的是"忘了什么还能捡回来"。**

#### 🛡️ 安全落点:进入稳态 = 模型开始失忆

稳态之后,最早的消息一轮一轮被挤出去 ——
用户第 1 轮说的"我叫 Bob"、第 2 轮给的订单号,**迟早会被挤没**。
这就是为什么必须配一个**摘要触发点**。

而且**稳态来得比想象的早**:上面 150 token 只装了 6 轮就满了 ——
因为每条消息都带 role、每条都算"信封开销"(见 2.1),
**你设的预算能装的实际轮数,是打过折的。**

### 2.10 📐 `max_tokens` 该设多大?跟模型窗口是什么关系?

#### 核心认知:窗口是"硬上限",不是"你的预算"

模型窗口是**输入 + 输出共享**的一份空间。所以不能贴着窗口设:

```
窗口 200K
├──────────────────────────────────────────────┤
│  你的输入(max_tokens)  │  留给输出  │ 余量   │
└──────────────────────────────────────────────┘
                          ↑
                这块不给够,模型一开口就被截断
```

#### 消耗方不止"对话消息"

一次请求至少占四块:

| # | 消耗方 | 说明 |
|---|---|---|
| ① | 系统提示 | 每轮都付的固定成本 |
| ② | 对话消息 | **裁剪管的就是这块** |
| ③ | **工具定义的 schema** | ← 最容易忘掉 |
| ④ | 模型这次的输出 | 要预留 |

③ 是真占额度 —— `count_tokens_approximately` 源码里就有:

```python
for tool in tools:
    tools_chars += len(json.dumps(tool_dict))
token_count += math.ceil(tools_chars / chars_per_token)
```

**实测**:

```
普通对话(2 条消息):            17 token
同样对话 + 挂上 3 个工具定义:  263 token
→ 光是工具定义就吃掉 246 token
```

⚠️ 挂 20 个工具的 agent,光 schema 就可能上千 token,**而且每一轮都要付**。

#### 怎么算:两条约束取小

```python
def compute_max_tokens(
    context_window: int,
    want: int = 8000,             # 业务上你希望留多少历史
    reserve_output: int = 4000,   # 给模型生成回复留的空间
    reserve_tools: int = 1500,    # 工具 schema 占的空间
    safety: float = 0.9,          # 给估算误差留的垫子
) -> int:
    hard_limit = int((context_window - reserve_output - reserve_tools) * safety)
    return max(1000, min(want, hard_limit))
```

| 模型窗口 | 业务想要 | 硬上限 | 最终 `max_tokens` | 谁在起作用 |
|---:|---:|---:|---:|---|
| 200,000(GLM-4.7-Flash) | 8,000 | 175,050 | **8,000** | 业务需求 |
| 128,000(中档) | 8,000 | 110,250 | **8,000** | 业务需求 |
| 32,000(小窗口) | 8,000 | 23,850 | **8,000** | 业务需求 |
| **8,000(老旧小模型)** | 8,000 | 2,250 | **2,250** | **硬上限(被压下来了)** |

> 📌 GLM-4.7-Flash 官方参数:**上下文窗口 200K / 最大输出 128K**
> (来源:https://docs.bigmodel.cn/cn/guide/models/free/glm-4.7-flash)

#### 三个结论

1. **窗口给了 200K,不代表你该设 200K。** 设太大会每轮烧钱、延迟高,
   还会触发 **context rot**(见 2.11)。200K 窗口的模型,业务上设 8K 往往就够。
2. **窗口小了就被迫往下压。** 8K 窗口的模型想留 8K 历史根本不可能 ——
   扣掉输出和工具预留只剩 2K 左右,对话聊两轮就滑一遍。
3. **换模型必须重新估一次。** 从 200K 换到 8K,如果 `max_tokens` 还是 8000,
   请求会**直接超窗口报错**。而这个错误开发时不一定测出来,上线换配置才炸。

#### 🛡️ 安全落点

- **工具定义也是成本**,每轮都付 → 精简工具数量和 docstring,
  别把整个 API 文档塞进工具说明
- **输出预留不能省** —— 省了的表现是"回复被莫名截断",不是报错,很难查
- **换模型是配置变更,要走一遍测试** —— 窗口差异是这类事故的高发区

### 2.11 「业务上想留多少」到底怎么估?

#### ⚠️ 先躲开一个混淆:工程里有两个 `max_tokens`

| 位置 | 管什么 |
|---|---|
| `trim_messages(max_tokens=X)` | **输入侧** —— 裁完之后,最多带多少进模型 |
| `model.bind(max_completion_tokens=Y)` | **输出侧** —— 模型最多生成多少 token |

**一个管"喂进去多少",一个管"吐出来多少"。名字像,作用正好相反。**

实测(2026-10-15,langchain-openai 1.7.0):

```
ChatOpenAI 构造参数里跟 token 有关的是:
  ['custom_get_token_ids', 'max_completion_tokens', 'tiktoken_model_name']

trim_messages 的输入侧参数:['max_tokens', 'token_counter', ...]
```

> 💥 **`ChatOpenAI` 已经改名成 `max_completion_tokens` 了,老的 `max_tokens` 不再生效。**
> 但 `bind()` **不做任何校验** —— 你写什么它都收下:
>
> ```python
> m.bind(max_tokens=128)            # -> 收下了, kwargs={'max_tokens': 128}      ← 过时参数名
> m.bind(max_completion_tokens=128) # -> 收下了, kwargs={'max_completion_tokens': 128}  ← 正确
> ```
>
> 官方 add-memory 文档里那行 `summarization_model = model.bind(max_tokens=128)`
> 按现在这个版本属于**"能跑但不一定生效"**,别照抄。

#### 怎么反推"需要几轮"?—— 这才是核心

不要问"我这个业务有多少 token",要问:

> **"回答用户当前这句话,模型至少需要回看多少历史?"**

这个答案取决于 **agent 的类型**,不取决于业务领域:

| Agent 类型 | 需要回看 | 为什么 |
|---|---|---|
| 单轮问答 / 分类 / 抽取 | 0 ~ 1 轮 | 每次独立,历史没用 |
| 客服对话(查单、改地址) | 3 ~ 10 轮 | 要记住用户前面给的订单号 / 诉求 |
| 多步任务(帮我把这活干完) | 几乎全部 | 目标在开头,每一步都依赖它 |
| 代码助手(改文件) | 全部 + 文件内容 | 上下文本身就是工作区 |

所以"业务想要多少"翻译成可操作的话是:

> **"这个 agent 在第 N 轮回答时,还看不看得见第 1 轮说的话?"**

#### ⚠️ 但它不是"越大越好"

1. **成本** —— 每轮都按输入 token 付费
2. **延迟** —— 输入越长,首个 token 出得越慢
3. **质量** —— 上下文越长,模型越难准确回忆里面的信息

**第 3 条有正式依据,不是感觉:**

`【官方】` **Anthropic** 把这个现象叫 **context rot**:

> "as the number of tokens in the context window increases, the model's ability to accurately **recall**
> information from that context **decreases**. While some models exhibit more gentle degradation than others,
> **this characteristic emerges across all models**."
> —— https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents

`【官方】` 同一篇文章还给了原因解释:

> "LLMs have an '**attention budget**' that they draw on when parsing large volumes of context.
> **Every new token introduced depletes this budget by some amount**, increasing the need to carefully
> curate the tokens available to the LLM."
> —— 同上

`【官方】` 更细的机制有论文专门研究 —— 相关信息落在上下文的**不同位置**,效果差别很大:

> "performance is often **highest when relevant information occurs at the beginning or end** of the input
> context, and **significantly degrades when models must access relevant information in the middle** of
> long contexts"
> —— Liu et al., *Lost in the Middle: How Language Models Use Long Contexts*, TACL 2023
> https://arxiv.org/abs/2307.03172

> 📌 顺带纠一条:本项目之前把"越长的上下文效果越差"笼统叫 **"lost in the middle"**。
> 更准确的说法是 —— **"变长导致召回下降"叫 context rot**;
> **"中间位置特别差"才是 lost in the middle(论文里的 U 型曲线)**。两者相关但不是同一个东西。

**目标不是"尽量多留",而是"刚好够用"。**

#### 算例:一个客服 agent 该设多少?(可复现)

```
① 系统提示                             |      17
② 工具 schema(5 个工具)                |     399
③ 一轮典型对话(用户+AI)                |      33
④ 需要保留的轮数(客服类,估计)          |       8
----------------------------------------------
→ 对话部分 = ③ × ④                     |     264
→ trim_messages(max_tokens) = ①+对话    |     281

输出预留(model 侧)                      |    1500
安全余量(15%)                           |     327
→ 总共要从窗口里扣掉                    |    2507
```

**三个值得看的点:**

- **工具 schema(399)比系统提示(17)大 20 倍。** 5 个工具的描述文字,
  比精心写的系统提示贵得多 —— 因为每个工具的 schema 是一整段 JSON。
  **精简工具比精简提示划算。**
- 一轮对话才 33 token,所以"留 8 轮"很便宜(264)。
  真正贵的是**固定成本**(系统提示 + 工具),它们**每轮都要付**。
- 数字是这例子专用的 —— 你的提示写多长、挂几个工具,要拿**自己的真实内容**去量。

#### 工程上怎么定?

`【官方】` 先看权威参照 —— **Claude Code 的做法**:

> "Claude Code runs 'auto-compact' **after you exceed 95% of the context window**"
> —— https://blog.langchain.com/context-engineering-for-agents/

`【建议】` **以下四步是本项目的建议,不是官方最佳实践**(在 `SOURCES.md` 登记为 E 级):

1. 先按上面的算例定一个**保守值**(比如 4K)
2. 上线跑真实流量,**统计每轮请求的实际 token 分布**
3. 看中位数和 **P95** —— 你会发现**绝大多数请求远用不到那么多**,
   是少数长对话把额度撑起来的
4. 把 `max_tokens` 定在 **P95 附近**,剩下的长尾交给裁剪兜底

> ⚠️ 上面的 `compute_max_tokens()` 函数和"取小"的公式,也都是 `【建议】` ——
> **它俩不是官方 API,是本文为了把这几个约束讲清楚而写的推演**。
> 官方只提供 `trim_messages(max_tokens=...)` 这个参数,至于填多少,官方没有规定。

> 🎯 **一句话**:`max_tokens` 的正确目标不是"最大能装多少",
> 而是"**覆盖 95% 的请求刚好够用,极端长对话让它裁**"。

---

## 三、消息摘要:`RemoveMessage` + 摘要节点

### 3.1 为什么光裁剪不够

`【官方】` 官方原文直接承认了裁剪的这个问题:

> "The problem with trimming or removing messages, as shown above, is that
> **you may lose information from culling of the message queue**. Because of this,
> some applications benefit from a more sophisticated approach of
> **summarizing the message history using a chat model**."
> —— https://docs.langchain.com/oss/python/langgraph/add-memory

`【官方】` 摘要这件事有个更正式的名字 —— Anthropic 叫它 **compaction**:

> "**Compaction** is the practice of taking a conversation **nearing the context window limit**,
> **summarizing its contents**, and **reinitiating a new context window with the summary**."
>
> "Compaction typically serves as **the first lever in context engineering** to drive better
> long-term coherence."
> —— https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents

裁剪丢掉的信息**没有任何地方记着它**。所以这些一旦被裁掉就出事:

- 用户最早说的偏好:"我只买蓝色那款,预售的不考虑"
- 已确认的事实:"我叫小李,会员号是 M88231"
- 已经执行过的动作:"刚才已经帮你查过订单 88345 了"

模型下一轮不是在敷衍你,**它是真的忘了**。

摘要的思路:**不是丢掉,而是压成一段话。**

| | 信息损失 | 成本 | 可逆 | 权威叫法 |
|---|---|---|---|---|
| 裁剪 | 高(永久) | 0 | 不可逆 | `【官方】` Trimming |
| 摘要 | 中(有损保留语义) | 要调模型 | 不可逆 | `【官方】` Summarization / Compaction |

`【官方】` 官方博客明确区分这两者的手段差异:

> "Whereas summarization typically **uses an LLM to distill** the most relevant pieces of context,
> trimming can often filter or, as Drew Breunig points out, '**prune**' context.
> This can use **hard-coded heuristics like removing older messages from a list**."
> —— https://blog.langchain.com/context-engineering-for-agents/

### 3.2 摘要状态放在自定义字段上

```python
class SummaryState(MessagesState):   # MessagesState 是 TypedDict,可以直接继承
    summary: str                     # 没写 Annotated → LastValue → 后写覆盖先写
```

继承 `MessagesState` 后,`messages` 字段连带它的 `add_messages` reducer 一起拿来,
只需补一个 `summary`。摘要正好需要"覆盖"语义,所以**不需要 reducer**。

### 3.3 💥 坑 3:消息没有 `id`,`RemoveMessage` **静默删不掉**

实测:手工 `HumanMessage(content=...)` 造的消息,`id` 全是 `None`;
`[RemoveMessage(id=m.id) for m in msgs[:-2] if m.id]` → 得到空列表
→ 合并后**还是 17 条,一条没删,而且没有任何报错**。

```
💥 尝试删掉前 15 条 → 实际剩 17 条   (无报错)
```

**原因**:`RemoveMessage` 完全靠 `id` 定位,`id=None` 它找不到目标,但**不抛异常**。

**为什么真实图里不会遇到**:消息进 state 时经过 `add_messages` reducer,
reducer 会给没 id 的消息**自动补 uuid**。
所以手工造数据时,先过一遍 `add_messages([], msgs)` 把 id 补上。

### 3.4 两种删消息的姿势(实测对照)

```python
# 姿势 A · 精确删除(官方范例用的)—— 结果:17 条 → 2 条
"messages": [RemoveMessage(id=m.id) for m in to_summarize if m.id]

# 姿势 B · 一键清空 —— 结果:2 条 → 0 条
"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]
```

**为什么删消息不能直接赋值**:节点的返回值**不能覆盖** state,
只能"作为一次更新"交给该字段的 reducer 处理。
所以要用 `RemoveMessage` 这种"**删除指令**"来表达意图。

**实战推荐 A**:姿势 B 全清之后,下一轮模型手里只剩一段摘要,
**它会不知道用户最后一句说了什么**,容易答非所问。
A 保留的那 2 条就是"上下文引子"。

### 3.5 增量摘要优于每次重压

摘要本身也会越来越长。如果每次都"全新压一遍",等于把已经压过的内容再压一次
——**信息衰减很快**。正确做法是把旧摘要带上:

```python
if previous_summary:
    prompt = (
        f"这是截至目前对话的摘要:\n{previous_summary}\n\n"
        "请结合下面的新增对话,把摘要**更新**一下(合并新旧内容,不要丢已有关键信息…)"
    )
```

### 3.6 用条件边自动触发

```python
def should_summarize(state: SummaryState) -> str:
    if len(state["messages"]) > 6:      # ⚠️ 6 是示意值,不是推荐值,见下
        return "summarize"
    return END

b.add_conditional_edges("chat", should_summarize, {"summarize": "summarize", END: END})
```

阈值判断放在**条件边**里,而不是写在节点内部
—— 这就是 T2 学的"**条件边是图里唯一的分支来源**"的实战用法。

实测图结构与运行结果:

```
START → chat ─┬─(消息数 ≤ 6)→ END
              └─(消息数 >  6)→ summarize → END
```

```
塞进去 17 条 → 跑完 messages = 2 条,summary 里保留了关键约束 ✅
```

#### ⚠️ "阈值该定多少"—— 有权威依据,别自己拍

`【官方】` LangChain 官方博客记载了一个**生产环境**的做法:

> "**Claude Code runs 'auto-compact' after you exceed 95% of the context window**
> and it will summarize the full trajectory of user-agent interactions."
> —— https://blog.langchain.com/context-engineering-for-agents/

**所以权威参照是"上下文窗口的 95%",不是拍一个"6 条"。**

| | 本文件里的 6 条 | Claude Code 的 95% |
|---|---|---|
| 性质 | `【建议】` **为了让演示在几行输出里看出效果而设的示意值** | `【官方】` 真实产品在生产里用的阈值 |
| 判断依据 | 消息**条数** | 占**上下文窗口的比例** |

**两者的差距是有原因的** —— 按条数判断不可靠:
一条消息可能 10 token,也可能 1000 token(用户粘一篇长文)。
**按"占窗口的比例"判断才稳。**

`【推论】` 用比例判断还有个好处:换模型(窗口变了)时阈值自动跟着变,不用改代码。
建议写成:当 `当前 token 数 / 模型窗口 > 0.9~0.95` 时触发摘要。

### 3.7 ✅ 对照官方范例:我们写的和官方给的是一回事

`【官方】` 官方 `summarize_conversation` 原文(一字不改):

```python
def summarize_conversation(state: State):
    # First, we get any existing summary
    summary = state.get("summary", "")

    if summary:
        # A summary already exists
        summary_message = (
            f"This is a summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"

    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = model.invoke(messages)

    # Delete all but the 2 most recent messages
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}
```
—— https://docs.langchain.com/oss/python/langgraph/add-memory

`【实测】` 逐条对照本项目的 `03_summarize_messages.py`:

| 官方范例 | 本项目 | 一致? |
|---|---|---|
| `state.get("summary", "")` | `state.get("summary", "")` | ✅ |
| 有旧摘要就 `"Extend the summary"` | `previous_summary` 分支 | ✅ |
| `model.invoke(messages)` | `_invoke_with_retry(...)` | ✅ |
| **先 invoke,再构造删除** | 先 `summarize`,再 `deletes` | ✅ 顺序相同 |
| `state["messages"][:-2]` | `state["messages"][:-2]` | ✅ |
| 返回 `{summary, messages}` | 返回 `{summary, messages}` | ✅ |

> **本项目的摘要节点没有自创写法,就是官方范例的结构。**
> 唯一的差别是**工程增强**(不改变语义):429 重试、`--offline` 模式、
> 打印被删掉的 id 以便看清机制。

`【官方】` 官方在"删除消息"那节还给了一条**必须遵守的有效性警告**:

> "When deleting messages, **make sure** that the resulting message history is valid.
> Check the limitations of the LLM provider you're using. For example:
> * Some providers expect message history to start with a `user` message
> * Most providers require `assistant` messages with tool calls to be followed by
>   corresponding `tool` result messages."
> —— 同上

💡 **这就是 2.3 那个"孤儿 ToolMessage"问题的官方依据** ——
官方在"删除"这一节又警告了一遍。**裁剪和删除是同一个坑。**

### 3.8 `【官方】` 摘要该留什么、丢什么

Anthropic 公开了 Claude Code 的 compaction 具体保留/丢弃什么:

> "The model **preserves architectural decisions, unresolved bugs, and implementation details**
> while **discarding redundant tool outputs or messages**. The agent can then continue with this
> compressed context plus the **five most recently accessed files**."
> —— https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents

翻成通用说法:

| | 保留 | 丢弃 |
|---|---|---|
| **决策类** | 定过的方案、做过的选择 | 讨论过程中的反复 |
| **未完成的事** | 还没解决的问题、待办 | 已完成且无后续的 |
| **具体细节** | 参数、路径、订单号等标识 | 冗余的工具原始输出 |
| **格式类** | 用户明确要求的偏好 | 寒暄、重复确认 |

#### ⚠️ 官方明确警告"压太狠"的后果

> "The art of compaction lies in **the selection of what to keep versus what to discard**,
> as **overly aggressive compaction can result in the loss of subtle but critical context
> whose importance only becomes apparent later**."
> —— 同上

**这是"摘要不能当事实记录"的权威依据** —— 它丢的可能正是
"当时看不出重要、后来才发现重要"的东西。所以钱、订单、是否已执行必须**独立于模型**记录。

#### 💡 一个很实用的技巧:tool result clearing

`【官方】` Anthropic 点名了**最安全的轻量压缩方式**:

> "An example of low-hanging superfluous content is **clearing tool calls and results** –
> once a tool has been called deep in the message history, why would the agent need to see
> the raw result again? One of the **safest lightest touch forms of compaction** is
> **tool result clearing**."
> —— 同上

这跟 2.3 讲的"别裁出孤儿 ToolMessage"是**互补的两面**:

| 做法 | 安全? | 为什么 |
|---|---|---|
| 裁掉**单条** ToolMessage | ❌ 危险 | 配对断了,`tool_call_id` 成孤儿 |
| **成对清除**老的 tool_calls + 它的 tool results | ✅ 安全 | 配对完整地一起消失 |

而且工具返回往往是整个对话里**最占地方**的部分,所以这一招性价比最高。

#### 📐 官方推荐的提示词调优顺序:先保召回,再提精度

> "For engineers implementing compaction systems, we recommend **carefully tuning your prompt
> on complex agent traces**. **Start by maximizing recall** to ensure your compaction prompt
> captures every relevant piece of information from the trace, **then iterate to improve precision**
> by eliminating superfluous content."
> —— 同上

#### 先把两个词搞清楚

这两个是**信息检索 / 机器学习里的标准指标**(不是 Anthropic 自创):

| 指标 | 问的问题 | 用摘要的场景说 |
|---|---|---|
| **召回(recall)** | **该找到的,我找到了吗?** | 原来对话里 5 条关键信息,摘要保住了几条? |
| **精度(precision)** | **我找到的,都是该要的吗?** | 摘要里写的内容,有几条是真正重要的? |

#### 2×2 看就懂了

把"一段对话里所有信息"分成两类:**关键**和**非关键**;
再看摘要**收没收**:

| | **摘要收进来了** | **摘要没收到** |
|---|---|---|
| **关键信息** | ✅ **命中**(好) | ❌ **漏掉** → **召回低 → 模型失忆** |
| **非关键信息** | ⚠️ **废话** → **精度低 → 浪费 token** | ✅ **正确丢弃**(好) |

**"先保召回"= 先把左下角那个"❌ 漏掉"清空;"再提精度"= 再去清右上角那个"⚠️ 废话"。**

#### 为什么必须先召后精?因为两种错误的代价**极不对称**

| 错误 | 代价 |
|---|---|
| **漏一条关键信息**(召回低) | 模型答错 / 重复执行 / 用户投诉 —— **要上线才能发现,排查成本极高** |
| **多留一条废话**(精度低) | 多花几十个 token —— **几乎无感** |

所以顺序不能反。**你先追求"压得短",就是在拿"会不会漏关键信息"去换"省几个 token",完全不划算。**

#### 具体怎么操作

**第一步:保召回(第一版提示词应该"贪心")**

```text
请保留以下所有内容:
  · 用户提过的任何数字、编号、地址、名称
  · 用户表达过的任何偏好或禁忌
  · 所有已经执行过的动作(做了什么、结果如何)
  · 所有还没解决的问题 / 待办
  · 用户明确拒绝过的选项
```

然后跑一批**真实对话**,逐条检查:关键信息有漏的吗?

**第二步:提精度(看着摘要里反复出现的废话,在提示词里明确排除)**

```text
以下内容不要写进摘要:
  · 寒暄与礼貌用语("谢谢""好的""不客气")
  · 重复的确认("您确定吗?""是的我确定")
  · 已经被后续消息推翻的中间方案
```

再跑一遍,确认**没把该留的也删掉**。

#### ⚠️ 一个容易忽略的细节:官方说要"在复杂轨迹上调"

> "carefully tuning your prompt **on complex agent traces**"

`【推论】` 这句话的意思是:**别用三五轮的简单例子调提示词**。
简单例子里信息本来就少,你的提示词怎么写都能"不漏";一上真实流量(几十轮 + 多工具调用),
才会暴露出漏信息的问题。**测试集要够脏、够长。**

### 3.9 有没有现成的轮子?有,但本项目不用

`【官方】` 官方范例里用了 `langmem` 包里的 `SummarizationNode`:

```python
from langmem.short_term import SummarizationNode, RunningSummary

summarization_node = SummarizationNode(
    token_counter=count_tokens_approximately,
    model=summarization_model,
    max_tokens=256,
    max_tokens_before_summary=256,   # ← 阈值按 token 算,不是按消息条数
    max_summary_tokens=128,
)
```
—— https://docs.langchain.com/oss/python/langgraph/add-memory

`【实测】` 本项目**没有装它**,理由(2026-10-15 查 PyPI 元数据):

| 发现 | 说明 |
|---|---|
| 版本还是 **0.0.30** | 早期版本,API 可能变 |
| 依赖很重 | `langchain` / `langchain-anthropic` / `langchain-openai` / `langgraph` / `langsmith` / `trustcall` 一整套 |
| 定位冲突 | 本项目目标是**看懂摘要机制**,手搓版已与官方范例逐条一致;套一层封装反而分不清"哪些是框架给的、哪些要自己写" |

`【建议】` 真到生产、又不想自己维护摘要逻辑时,再考虑 `pip install langmem`。

> 📌 注意那个参数名 —— `max_tokens_before_summary` 是**按 token 设阈值**,
> 跟 3.6 说的"按比例而不是按条数"是同一个思路。

---

## 四、三种记忆策略:不是三选一,是叠着用

| 手段 | 信息损失 | 成本 | 可逆 | 什么时候用 |
|---|---|---|---|---|
| 不处理 | 无 | 0 | — | 短对话、demo |
| 裁剪 trim | 高(永久) | 0(估算) | 不可逆 | 上下文刚满、纯聊天记录 |
| 摘要 summarize | 中(有损) | 要调模型 | 不可逆 | 有必须记住的偏好/约束 |
| 外部存储 | 低 | 读写一次 | **可逆** | **已发生的动作**必须用它 |

实战组合:

```
每次调模型前    trim(摘要 + 最近 N 条)          ← 控住体积
消息数超阈值    生成/更新摘要                    ← 保语义
写类动作发生时  写进外部存储(DB / checkpointer) ← 保事实
```

---

## 五、踩坑记录

### 坑 1:`count_tokens_approximately` 的公式不能想当然 ⚠️

我第一版写 `len(content)/4 + 3`,实测三条长度 10/11/5 结果全是 7。
**必须去读源码** —— 漏的是 `len(role)`(system=6 / user=4 / assistant=9 / tool=4),
以及 AI 的 `tool_calls` repr、Tool 的 `tool_call_id`。

### 坑 2:不设 `start_on` 会裁出孤儿 `ToolMessage`

症状是发给模型直接 400,报错里提到 `tool_calls` / `tool_call_id`。
**修法**:`start_on="human"`。

### 坑 3:`max_tokens` 太小返回空列表,不报错

实测悬崖在 50/55 之间。空的后果比报错更阴。
**修法**:`safe_trim()` 判空兜底 + 别用 `result[0]` 硬索引。

### 坑 4:默认会把 SystemMessage 裁掉,而且不报警 ⚠️

`include_system` 默认 `False`,而 SystemMessage 恰好在列表最开头,
正好是第一个被砍的。加 `include_system=True` 能保住,
**但它要抢额度** —— 额度紧时会把对话挤到一条不剩(只剩一条光杆系统提示)。
而且它**只认 `messages[0]` 那一条**,系统提示挪到中间就完全失效。

### 坑 5:`start_on` 是"回溯砍",额度充足也会砍

不是"找到一个锚点就停",而是"先收一批,再从头往回砍,砍光了返回空"。
**实测:额度给到 9999 时,`start_on="human"` 照样把开头的 SYS 砍掉。**
以为"额度够就没事"是最容易踩的误判。

### 坑 6:传了边界 → 装得下也照样动刀 ⚠️

实测:`SYS | HUM | AI`(20 token),额度给 9999(远远用不完):

| 传了哪些边界 | 结果 |
|---|---|
| 都不传 | 原样返回 ✅ |
| 只传 `start_on` / 只传 `end_on` | 各砍 1 条 ❌ |
| **两个都传**(官方例子) | 3 条 → **1 条** ❌ |

**不传边界时 `trim_messages` 有"够了就收手"的短路逻辑;
只要传了任一边界,短路之后仍然会做边界过滤。**
所以官方那个例子是"每轮都会动刀"的写法 —— 这是设计意图,不是 bug。

### 坑 7:消息没有 `id`,`RemoveMessage` 静默失效 ⚠️⚠️

**最阴的一个** —— 删 0 条、无异常、无警告。
**修法**:手工造数据先过 `add_messages([], msgs)` 补 id。

### 坑 8:节点返回值是 partial update,不能"覆盖" state

所以删消息只能发 `RemoveMessage` 指令,不能 `state["messages"] = []`。
这是 T2 那条"**有 reducer 的字段,给得越多 ≠ 越保险**"的延伸。

### 坑 9:想清空 state,`return {"messages": []}` **也不生效** ⚠️⚠️

这条是把坑 8 推到极限之后发现的,**三种"直觉写法"全部静默失效**(实测见 1.9):

| 写法 | 结果 |
|---|---|
| `return {"messages": []}` | ❌ 还是原样 |
| `state["messages"] = []; return {}` | ❌ 还是原样 |
| `state["messages"] = []; return dict(state)` | ❌ **还是原样** |
| `return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}` | ✅ 清空 |

根因:`add_messages(旧, [])` = "合并 0 条" = 什么都不变。
**节点的返回值永远被当"更新指令",不是"最终状态"。**

### 坑 10:`model.bind(max_tokens=...)` 已经不生效了 ⚠️

`ChatOpenAI` 现在叫 **`max_completion_tokens`**,老的 `max_tokens` 是过时参数名。
但 **`bind()` 不做任何校验** —— 传错了它照样收下,只是在真正调用时被忽略或报错。

```python
m.bind(max_tokens=128)            # kwargs={'max_tokens': 128}          ← 静默失效
m.bind(max_completion_tokens=128) # kwargs={'max_completion_tokens': 128} ← 正确
```

> ⚠️ 官方 add-memory 文档里那行 `summarization_model = model.bind(max_tokens=128)`
> 按现在这个版本属于"能跑但不一定生效",别照抄。

**顺带记住:两个 `max_tokens` 不是一回事**
—— `trim_messages(max_tokens=X)` 管**输入**,`max_completion_tokens=Y` 管**输出**。

### 坑 11:智谱免费档限流是真的狠

本轮实测连续 12 次调用**全部 429**,两种错误码轮着来:

```
429 - {'error': {'code': '1302', 'message': '您的账户已达到速率限制…'}}   ← 你调太快
429 - {'error': {'code': '1305', 'message': '该模型当前访问量过大…'}}     ← 服务端挤
```

**两者都归到"瞬时错误"**(对应 T1 笔记里的第 ① 类),处理方式是**等待 + 重试**,
不要去改业务逻辑。

**本项目的应对**:`03_summarize_messages.py` 内置 `--offline` 模式,
用"本地假摘要"顶替模型,让 `RemoveMessage` / state 合并 / 条件边这套**管道**能独立验证。
真摘要必须联网跑。

> 📌 这是环境限制,不是代码问题。等额度宽松时去掉 `--offline` 重跑即可。

---

## 六、🎯 范式落点 / 🛡️ 安全落点

### 🎯 范式落点:这套就是 Reflexion 的地基

| 手段 | 对应范式里的哪一步 |
|---|---|
| 裁剪(trim) | 上下文管理的**降级兜底** —— 最便宜,信息永久损失 |
| 摘要(summary) | **Reflexion / 长期记忆的存储机制** —— 把老信息压成可携带的表示 |
| 外部存储 | **Reflexion 的"经验"落地处** —— 让教训跨轮次存活 |

**Reflexion 和"摘要"的区别只在于写什么进去**:
- 摘要写的是"对话里发生过什么"
- Reflexion 写的是**结构化的教训**:哪一步错了、为什么错、下次怎么改

所以 T3 做完,你已经有了 Reflexion 的**存储机制**(`summary` 字段 + 裁剪 + 条件边触发),
缺的只是"写什么进去"这一层。

### 🛡️ 安全落点:摘要**不能当"事实记录"**

#### 这句话翻译成人话

> **摘要 = 模型对过去对话的一次"改写"。它是二手信息。**
> **凡是需要精确、需要核对、需要追溯的东西,不能只靠它记着。**

`【官方】` 依据 —— Anthropic 原话:

> "overly aggressive compaction can result in **the loss of subtle but critical context
> whose importance only becomes apparent later**."
> —— https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents

注意它说的是"**丢了,而且你当时还发现不了**"——这比"记错了"更麻烦。

#### 摘要为什么会失真:三个具体机制

| 机制 | 具体怎么发生 |
|---|---|
| **压缩必然丢细节** | 20 轮对话压成 100 字,细节被概括掉是数学上必然的 |
| **模型会"合理化"改写** | 原文"用户说要退订单 88345" → 摘要写成"用户想退款"(订单号没了) |
| **模型分不清"说过"和"做过"** | 用户问"能退吗" → 摘要写成"用户已申请退款" |

#### 走一遍具体例子(客服场景)

**原始对话:**

```
用户:  我的订单 88345 能不能退?蓝色 XL 款,299 块
助手:  可以退的,需要我帮您提交退款申请吗?
用户:  好,提交吧
```

**摘要大概率写成:**

```
用户想退一个订单,助手同意协助处理
```

**丢了什么:** 订单号 `88345`、金额 `299 元`、规格 `蓝色 XL`,
以及最要命的 —— **"到底提交了没有"这个状态**。

**下一轮的两种事故:**

| 事故 | 怎么发生的 |
|---|---|
| **重复执行** | 摘要里没写"已提交" → agent 再提交一次 → **退两次款** |
| **该做没做** | 摘要里写了"已提交"但实际没执行 → 用户等半天没到账 → 投诉 |

两种都是真实生产事故的形态,而且**都不会报错**。

#### 正确做法:事实和摘要**分开存**

```python
class State(MessagesState):
    summary: str        # ← 模型写的,用来"理解上下文",允许它不精确
    order_id: str       # ← 系统写的,精确,可核对
    refund_status: str  # ← 系统写的:"未申请" / "已提交" / "已完成"
```

| 谁看什么 | 用途 |
|---|---|
| 模型看 `summary` | 知道"我们正在聊退款这件事" |
| 系统看 `refund_status` | 决定"要不要真的去执行"——**不信任摘要** |

**摘要只用来"理解上下文",不能用来"做事实判断"。**

#### 一条可操作的判断标准

> 当你准备用某个信息**做决策或执行动作**时,先问自己:
> **"这条信息是从摘要里读来的,还是从结构化字段 / 数据库里读来的?"**
> **如果是前者 —— 停下来,去查真实状态。**

`【建议】` 这条是本项目给的检查方法,不是官方规定,但能挡住绝大多数这类事故。

另一条:**裁剪是有损且不可逆的**。
裁掉的动作记录会让模型**忘记自己做过什么 → 可能重复执行**。
写入类场景,这类动作必须有 checkpointer 或库表兜底。

### 🛡️ 安全落点(裁剪部分):三个"不报错的失效"

裁剪这块最危险的地方在于:**出问题的时候往往没有任何报错**。
下面三种失效,全都静默:

| 失效 | 症状 | 怎么防 |
|---|---|---|
| 裁出孤儿 `ToolMessage` | 发给模型 400,报错提到 `tool_call_id` | `start_on="human"` |
| 裁成空列表 | `model.invoke([])` 行为未定义;`messages[0]` 报 `IndexError` | 判空兜底 |
| 只剩一条系统提示 | **完全看不出异常** —— 结果不是空的,但一条对话都没有 | 判 `all(isinstance(m, SystemMessage))` |

还有两条成本意识:

- **系统提示是"每轮都要付的固定成本"**。一份 2000 token 的系统提示,
  等于每一轮都先烧掉 2000 —— 它比任何一次裁剪都贵。所以系统提示要**写短、写好**,
  不要把所有规则都往里堆(该拆去 RAG 检索的拆走)。
- **裁剪是在"省钱"和"断片"之间做取舍**。省下的每一个 token,
  都是从模型记忆里割掉的。所以裁之前先想:
  **这段内容丢了,会导致什么后果?** 会导致"重复执行动作"的,一律不能只存在 message 里。

---

## 七、这一步攒下的伏笔(后面会兑现)

| 伏笔 | 在哪兑现 |
|---|---|
| `RemoveMessage` 是"删除指令",靠 `id` 定位 | T4 checkpointer —— 状态持久化后,跨轮次删改才有意义 |
| 摘要/裁剪做完了,但 state 只活在**一次** `invoke` 里 | **T4 checkpointer** —— 官方原文:只在 **super-step 边界**存快照 |
| `summary` 字段用 `LastValue` 语义 | T5 人在环:同样用"覆盖型字段"存 interrupt 决策 |
| 条件边是唯一分支来源 | **T9 `Send`** —— 动态扇出,那时回看 super-step 会更清楚 |
| 摘要文件里那句"摘要不能当事实记录" | T7 RAG Agent / T10 包装 —— 写入类动作的外部状态校验 |

---

## 产出

- `04-state-memory/01_state_schema.py` —— 10 段,不调模型,秒出
- `04-state-memory/02_trim_messages.py` —— 12 段,不调模型,秒出
- `04-state-memory/03_summarize_messages.py` —— 6 段,可 `--offline`
- `04-state-memory/README.md` —— 本文件
