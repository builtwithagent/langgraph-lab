# T2 · 第一个 StateGraph

## 做了什么

把 T1 手搓的 `while` 循环,改成 LangGraph 的状态图。**同一个 agent,两种写法。**

| 文件 | 内容 |
|---|---|
| `01_basic_graph.py` | **不碰 LLM**,纯本地把图的四个零件摸一遍(秒跑通) |
| `02_agent_as_graph.py` | 把 T1 的 agent 装进图,带 `ToolNode` + 错误处理 + `RetryPolicy` |
| `03_superstep_and_update.py` | **不碰 LLM**,实测 superstep 边界 + 节点返回值语义(秒跑通) |

跑法:
```bash
cd /d/VibeCoding/langgraph-lab
./.venv/Scripts/python 03-first-graph/01_basic_graph.py          # 秒出,不调模型
./.venv/Scripts/python -u 03-first-graph/02_agent_as_graph.py    # 要调模型
```

---

## 📎 知识点补齐(卡住就点开,全部实测可访问)

`01_basic_graph.py` 只用到 5 个新概念,一个个给入口:

| # | 概念 | 去哪看 |
|---|---|---|
| 1 | **State / Node / Edge / 图的心智模型** | [Graph API 文档](https://docs.langchain.com/oss/python/langgraph/graph-api) · [「用图的方式思考」官方概念篇](https://docs.langchain.com/oss/python/langgraph/thinking-in-langgraph) ← 先看这个,比 API 文档好懂 |
| 2 | **reducer**(`Annotated` + `operator.add`) | [Graph API 的 State/Reducers 小节](https://docs.langchain.com/oss/python/langgraph/graph-api) · [`Annotated`](https://docs.python.org/3/library/typing.html#typing.Annotated) · [`operator.add`](https://docs.python.org/3/library/operator.html#operator.add) |
| 3 | **TypedDict**(State 为什么写成它) | [Python 官方文档](https://docs.python.org/3/library/typing.html#typing.TypedDict) |
| 4 | **条件边 + `path_map`** | [Graph API 的 Conditional edges 小节](https://docs.langchain.com/oss/python/langgraph/graph-api) |
| 5 | **Mermaid**(文件末尾会打印图的源码) | [语法速查](https://mermaid.js.org/syntax/flowchart.html) · [在线预览 mermaid.live](https://mermaid.live/)(把打印的源码粘进去就能看图) |
| 6 | **super-step / 执行模型(Pregel)** | [Graph API 的 Graphs 小节](https://docs.langchain.com/oss/python/langgraph/graph-api#graphs) —— 官方对 super-step、Pregel 的原始定义 |

**想跟着视频过一遍**(中文,B站;下面的 BV 号 2026-10-11 用 B 站 API 校验过标题,不是瞎写的):

| 视频 | UP | 为什么看它 |
|---|---|---|
| [挑战19分钟搞定,LangGraph快速入门与原理剖析](https://www.bilibili.com/video/BV1HVFLzfEeS) | 敲代码的码农 | **最短**,19 分钟把概念过一遍,适合现在 |
| [尚硅谷 LangGraph 教程](https://www.bilibili.com/video/BV1z3NY66EY1) | 尚硅谷 | 体系化、分 P 讲,当参考书用 |
| [B站讲的最细的 LangGraph 快速入门教程](https://www.bilibili.com/video/BV16gV8zNE9M) | 大模型碎碎念 | 有「核心组件:Graphs/State/Nodes/Edges」专门分 P |
| [官方免费课程 intro-to-langgraph](https://academy.langchain.com/courses/intro-to-langgraph) | LangChain 官方 | 英文,讲得比大多数中文教程细;想深入就这个 |

---

## 一、图一共就四个零件

| 零件 | 是什么 | 注意 |
|---|---|---|
| **State** | 一块共享内存,所有节点都能读写 | 是个 `TypedDict`(运行时就是 dict) |
| **Node** | 一个普通函数:收 state,返回「我要更新哪些字段」 | 返回值是 **partial update**,不是整个 state |
| **Edge** | 固定的下一步 | `START` / `END` 是两个特殊标记 |
| **条件边** | 由函数决定下一步走哪 | **图里唯一能做分支的地方** |

**从 T1 到 T2 换掉的,本质是一句话:把「怎么流转」从命令式代码,变成声明式结构。**

### 图解:两个示例各长什么样

**示例一(最小图)—— 一条直线**

```
START --> [ a ] --> [ b ] --> END
```

三个节点共用同一块 State,它是这样被一格一格填满的
(Q = 输入的问题,A = b 算出来的回答):

| 时刻 | `question` | `answer` | `trail` |
|---|---|---|---|
| `invoke` 时 | Q | — | `[]` |
| `a` 跑完 | Q | — | `["node_a 跑过了"]` |
| `b` 跑完 | Q | A | `["node_a 跑过了", "node_b 跑过了"]` |

两个关键点:

1. **节点返回的都是 partial update** —— `a` 只返回 `trail`,`b` 只返回 `answer` + `trail`。谁都没交回整个 state。
2. **`trail` 被两个节点写,却没互相覆盖** —— 这不是默认行为,是 `trail` 声明了 reducer 才有的(见下一节)。

**示例二(条件边)—— 一条岔路**

```
START --> [ read_change ] --> [ judge ] --+--(涨停)--> [ limit_up ] --> END
                                          |
                                          +--(其他)--> [ normal   ] --> END
```

| 时刻 | `symbol` | `change_pct` | `verdict` | `trail` |
|---|---|---|---|---|
| `invoke` 时 | S | — | — | `[]` |
| `read_change` 跑完 | S | 10.01 | — | `["读取涨跌幅 10.01%"]` |
| `judge` 跑完 | S | 10.01 | 涨停 | `[…, "判定:涨停"]` |
| 走涨停分支后 | S | 10.01 | 涨停 | `[…, "→ 记入涨停复盘池"]` |

**这张图最该盯住的是 `judge`:** 它不做分支,只把结论写进 `verdict` 字段。
真正决定走哪的是 `route()` —— 这个函数就是条件边。

> **对照 T1**:`route()` 就是你手写 `while` 循环里那个 `if`。
> 区别在于 `if` 藏在代码里、只有读源码才知道有分支;条件边被**声明**进了图结构,
> 所以 `get_graph().draw_mermaid()` 能把岔路画出来 —— 图本身成了文档。

---

## 二、reducer:为什么两个节点写同一个字段不会互相覆盖

`01_basic_graph.py` 第一段里,`trail` 字段是这样声明的:

```python
trail: Annotated[list[str], operator.add]   # ← 这就是 reducer
```

- **不加 reducer**:默认行为是**后者覆盖前者**。node_a 写 `["a"]`,node_b 写 `["b"]`,最后只剩 `["b"]`
- **加了 reducer**:变成**累加**。结果是 `["a", "b"]`

实测输出:
```
最终 state: trail = ['node_a 跑过了', 'node_b 跑过了']
```

> 💡 **这个机制是理解 `MessagesState` 的钥匙。** LangGraph 内置的 `MessagesState` 就是
> 「一个 messages 列表 + `add_messages` 这个 reducer」——
> 所以你 `return {"messages": [新消息]}` 时,框架是**追加**,不是覆盖。
> T1 里你要手写 `messages.append(msg)`,这里不用。

**reducer 还有个隐藏作用**:它决定了并行分支的结果怎么合并。没有 reducer 的多路写入会直接报错 —— 这也是后面阶段 4 玩 `Send` 并行时必须先搞清的东西。

---

## 三、条件边:图里唯一的分支来源

```python
def route(state) -> str:
    """返回「下一个节点的名字」"""
    return "limit_up" if state["verdict"] == "涨停" else "normal"

builder.add_conditional_edges("judge", route, {"limit_up": "limit_up", "normal": "normal"})
```

对比 T1:这相当于你手写循环里的那个 `if`——只不过它被**声明**出来了,而不是埋在代码里。

### ⚠️ 第三个参数 `path_map` 要写

不传也能跑(靠返回值直接找同名节点),**但有个副作用**:`get_graph().draw_mermaid()` 画不出分支,只会画一条假的 `judge → END`。

实测对比:

```
# 不传 path_map
judge --> __end__;              ← 假的,两条分支都没了

# 传了 path_map
judge -.-> limit_up;            ← 虚线表示条件边,正确
judge -.-> normal;
```

**习惯:条件边一律把 `path_map` 写上。** 反正架构图早晚要出。

---

## 四、把 agent 装进图

`02_agent_as_graph.py` 的全部结构就这几行:

```python
builder = StateGraph(MessagesState)

builder.add_node("agent", call_model, retry_policy=RetryPolicy(max_attempts=3, initial_interval=1.0))
builder.add_node("tools", ToolNode(TOOLS, handle_tool_errors=True))

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", "__end__": END})
builder.add_edge("tools", "agent")

graph = builder.compile()
```

自动生成的图结构:
```
graph TD;
	__start__ --> agent;
	agent -.-> __end__;
	agent -.-> tools;
	tools --> agent;
```

**这就是 T1 那个 while 循环的样子,只不过是画出来的。**

### 对照 T1:框架到底收走了什么

| T1 你要自己写 | T2 谁来做 |
|---|---|
| 手写 JSON Schema | `@tool` 装饰器从**函数签名 + docstring** 自动生成 |
| 手写 `TOOL_MAP` 映射表 | `ToolNode(TOOLS)` 自动维护 |
| `messages.append(...)` | `MessagesState` 的 `add_messages` reducer |
| `tool_call_id` 手动配对 | `ToolNode` 自动配对 |
| `while` + `if not msg.tool_calls` | `tools_condition` —— **现成的路由函数** |
| 并行执行多个工具 | `ToolNode` 默认并发 |
| `try/except` 工具异常 | `handle_tool_errors=True` |
| 自己写重试逻辑 | `RetryPolicy` 挂在节点上 |
| 自己加 `max_turns` 防死循环 | `recursion_limit`(默认 25 步) |

**结论跟 T1 一样:框架不是更聪明,是把你手写过的那些工程细节收进了声明式结构里。**
而你手写过一遍,所以现在能一眼看出每一行对应 T1 的哪一段 —— **这就是 T1 的价值。**

### 代码行 ↔ 边的对照:T1 的 7 件事分别落在图的哪里

| T1 手搓(`02-raw-agent-loop/agent_loop.py`) | T2 图里对应什么 |
|---|---|
| `messages = [HumanMessage(q)]` | `add_edge(START, "agent")` |
| `client.chat.completions.create(...)` | `agent` 节点(`call_model`) |
| `messages.append(resp)` | `MessagesState` 的 `add_messages` reducer |
| `if resp.tool_calls:` → 执行工具 | 条件边的 `"tools"` 那一支 |
| 执行工具 + 把结果 append 回去 | `tools` 节点 + `add_edge("tools", "agent")` |
| `if not resp.tool_calls: break` | 条件边的 `"__end__"` 那一支 |
| **`while True:`** | **没有对应物 —— while 被「图的形状」替代了** |

⚠️ 两个最容易看漏的点:

1. **图是环,不是直线。** `tools --> agent` 是一条**回环边**。整个 `02_agent_as_graph.py` 里**一个 `while` 都没有**,但编译出来的图是个环 —— 控制流被翻译成了「结构」,不是「代码」。
2. **「停」和「继续」是平级的两条路。** T1 里"什么时候停"是 `while` 的判断条件;T2 里它变成了条件边的一个**分支**(`__end__`)。不是判断语句,是一条边。

> 补一句:"一步步走"只在**单节点串行**时成立。LangGraph 实际按 **superstep** 调度,同一层里若有多节点/多分支是并发的(阶段 4 讲 `Send` 才展开)。

### ToolNode 的 partial update 在哪?(源码级证据)

**问**:`01_basic_graph.py` 里每个节点都 `return {"trail": [...]}`,但 `ToolNode` 从没见过它 return 什么 —— 工具结果到底怎么塞回 state 的?

**答**:`ToolNode` 是**框架预置节点**,那个 `return` 不由我们写,所以在自己的文件里当然看不到。实现在:

```
.venv/Lib/site-packages/langgraph/prebuilt/tool_node.py
```

第 **886** 行就是那个 return:

```python
if not any(isinstance(output, Command) for output in flat_outputs):
    # TypedDict, pydantic, dataclass, etc. should all be able to load from dict
    return (
        flat_outputs
        if input_type == "list"
        else {self._messages_key: flat_outputs}      # ← 就是它
    )
```

`self._messages_key` 在 **776** 行被赋成 `"messages"`。所以它返回的正是:

```python
{"messages": [ToolMessage(...), ToolMessage(...)]}
```

**标准的 partial update,跟你手写的节点一模一样。**

**实测佐证(不调模型)**:搭一个只有 `ToolNode` 的最小图,喂一条带 `tool_calls` 的 `AIMessage`:

```
  [AIMessage] ''
  [ToolMessage] '3'
```

`ToolMessage` 是**图自己 append 进去的**,靠 `MessagesState` 的 `add_messages` reducer。也就是说 `ToolNode` 只负责"返回",合并这件事由 state 的 reducer 干。

> **这就是「节点契约」**:不管是你写的函数,还是框架预置的 `ToolNode`,**入参都是完整 state,返回值都必须是 partial update**。
>
> **一个推论**:`ToolNode` 硬编码写 `messages` 这个 key。所以如果你的 State 里没有 `messages` 字段,直接丢个 `ToolNode` 进去会报错 —— 想用别的 key 得传 `messages_key=` 参数。

### 怎么保证不无限打转:`recursion_limit`

T1 里你自己写 `for i in range(10)`,图里对应的是 **`recursion_limit`** —— 在**调用时**通过 config 传:

```python
graph.invoke({"messages": [...]}, config={"recursion_limit": 25})
```

**默认 25。超了抛 `GraphRecursionError`,不是静默截断:**

```
Recursion limit of 25 reached without hitting a stop condition.
You can increase the limit by setting the `recursion_limit` config key.
```

⚠️ **它数的不是「轮」,是 superstep。** 实测(纯本地图,不调模型):

| 图的形状 | `recursion_limit` | 实际发生 |
|---|---|---|
| 单节点自环 `loop → loop` | 3 / 5 / 25 | 节点执行 **3 / 5 / 25** 次 |
| `agent → tools → agent` 环 | 25 | `agent` **13** 次 / `tools` **12** 次 = 25 个 superstep |

**也就是说:默认 25 大约只够 12 轮工具调用。** 长任务 agent 很容易撞上限 —— 撞到了先调大 `recursion_limit`,而不是去怀疑模型。

对照 T1:

| | T1 手搓 | T2 图 |
|---|---|---|
| 上限写在哪 | `for i in range(10)`,写死在循环里 | `config={"recursion_limit": N}`,**调用时**传 |
| 数的是什么 | 迭代轮数 | **superstep 数** |
| 超了会怎样 | 自己 `break`,静默停 | 抛 `GraphRecursionError`,**不静默** |

> **"不静默"这点比 T1 好**:T1 里 `for i in range(10)` 跑完就安静地退出,你可能压根不知道它是因为超限才停的 —— 拿到一个"看起来正常"的答案,其实是被截断的。

### superstep 到底是什么(实测)

**先记住一句话:一个 super-step 不是「跑了几个节点」,而是「state 被更新了几次」。**

为什么这么定义?因为**并行分支会同时读到同一份旧 state** —— a 在算、b 也在算,谁先写谁后写就乱了。所以框架必须划一条界:**这一批节点全干完,再把它们的返回值一起合并进 state。** 这条界就是一个 super-step。

于是有个更好用的说法:

> **一个 superstep = 一次 state 更新所对应的那一批节点。**

- 图是**链条** → 每次更新恰好来自 **1 个**节点 → 看起来就是「一个节点 = 一步」
- 图有**分叉** → 一次更新可以来自 **好几个**节点 → 「3 个节点 = 1 步」

所以"一个节点执行一次 = 一个 superstep"**在串行图里是对的**,只是在分叉图里不成立。

官方定义(原话,2026-10-12 核对):

> "Inspired by Google's **Pregel** system, the program proceeds in discrete **'super-steps'**.
> A super-step can be considered a single iteration over the graph nodes. Nodes that run in parallel are part of the same super-step, while nodes that run sequentially belong to separate super-steps. …
> At the end of each super-step, nodes with no incoming messages vote to halt by marking themselves as inactive."

拆开就三句话:

1. 一个 super-step = **图上的一次迭代**
2. **并行跑的节点属于同一个 super-step;串行跑的节点属于不同的 super-step**
3. 每一步结束时,没有新消息的节点投票 `halt`;所有节点都 `halt` 且没有消息在途,图就结束

**注意官方拼写是带连字符的 `super-step`**(我前面图省事写成 superstep,指同一个东西)。

> 这套模型来自 Google 的 **Pregel** 论文(Malewicz et al., SIGMOD 2010)—— 典型的 **BSP**(整体同步并行):一批节点跑完 → 屏障同步、应用更新 → 再进下一轮。
>
> 这也解释了一件事:**同一个 super-step 里的多个节点,如果都写同一个字段,就必须靠 reducer 合并** —— 因为它们互相看不见对方的写入。
>
> **顺手一个 T4 伏笔**:官方文档还有一句 —— checkpointer **只在 super-step 边界存快照,不会存节点函数执行到一半的中间态**。所以"断点续跑"的粒度就是 super-step:T4 做持久化时,恢复点永远落在两步之间,不会落在一个节点内部。

实测(3 个节点,每个节点睡 0.6s,用耗时"称重"):

| 图的形状 | 耗时 | superstep 数 |
|---|---|---|
| `START` 一次性扇出 a/b/c | **0.61s** | **1** —— 3 个节点挤在同一步并发跑 |
| `START → fanout → a/b/c` | **1.20s** | **2** —— fanout 独占 1 步,a/b/c 在下一步并发 |
| `START → a → b → c`(纯串行) | **1.81s** | **3** —— 只能排队 |

> 补一层:所以图的执行模型不是"节点依次调用",而是**一轮一轮的调度** —— 这直接决定了并行分支怎么写、reducer 为什么必需。

**「界」具体指什么?**

通俗说法 —— **它就是「全班停笔、一起收卷」的那个时刻。**

- 并行分支一起开跑,但**谁也不许一边算一边往 state 里写**
- 必须等这一批节点**全部停笔**,才允许把它们的返回值**一起**合并进 state
- 这个"一起写"的时刻,就是界(官方叫 **super-step 边界**,BSP 里叫 **barrier**)

**它解决什么问题?** 保证结果**不依赖谁跑得快**。假如没有这条界:

- `a` 算完先写 `x=1`;`b` 后算完时读到的就不是最初那份 state,而是"已经含 x=1"的版本了
- 那结果就取决于线程调度 —— **同一份输入跑两次,结果可能不一样**

**实测:让节点自己报告「我读到的 state」**(脚本里的实验 5):

| 图 | `a` 读到 | `b` 读到 | 说明 |
|---|---|---|---|
| 并行 `START → a \| b` | `[]` | `[]` | **互相看不见** —— 读的是同一份旧 state |
| 串行 `START → a → b` | `[]` | `['a 写的']` | `a` 交卷后界已发生过一次,`b` 才看得见 |

**所以串行图里界照样存在,只是每步只有 1 个节点,你察觉不到。**

> 用程序员的词:**界就是并行任务的那个 `join` / 屏障**。
> 用生活的话:**"人都到齐了再开饭"—— 不是谁做好谁先吃。**

**`recursion_limit` 的精确换算**(接着上面那节):

| 图的形状 | 节点执行的 superstep 数 | 需要的 `recursion_limit` |
|---|---|---|
| 单节点 | 1 | 2 |
| `START` 扇出 3 个 | 1 | 2 |
| 串行 3 个 | 3 | 4 |

→ 公式:**`recursion_limit ≥ 节点执行的 superstep 数 + 1`**。多出来的那 1 格是「走到 END」那一步,它同样计数。

(死循环永远到不了 END,所以 `limit=N` 就是白跑 N 步 —— 前面 agent 图 "25 步 ≈ 12 轮" 的算法就来自这里。)

### 节点的返回值能不能给完整 state?(能,但别)

**能** —— 语法上完全合法。但**语义上危险**,因为返回值**不是「覆盖整个 state」**,而是**逐字段交给该字段自己的 reducer**:

- **无 reducer 的字段**:你给什么就是什么(所以给全量 = 给部分,效果一样)
- **有 reducer 的字段**:你给的值会被**合并**到现有值上 —— **给全量就重复了**

实测(前置:`n1` 先跑过,`trail` 已经是 `['n1']`):

| probe 节点返回什么 | 最终 `trail` |
|---|---|
| `{"trail": ["n2"]}` —— partial | `['n1', 'n2']` ✅ |
| `{"trail": s["trail"] + ["n2"]}` —— 带上历史 | `['n1', 'n1', 'n2']` ⚠️ **n1 被算了两次** |

**所以「给得越多 ≠ 越保险,而是重复越多」。**

返回值还有两个边界情况(实测):

| 返回什么 | 结果 |
|---|---|
| 未声明的 key,如 `{"trail": [...], "bogus": 123}` | **静默丢弃**,不报错,也写不进 state |
| `{}` 空 dict | 完全合法,表示「这一步不改任何字段」 |

> **一句话总结:入参永远是完整 state(框架保证),返回值永远只写你要改的字段。**

复现命令(不调模型,秒出):
```bash
./.venv/Scripts/python 03-first-graph/03_superstep_and_update.py
```

---

## 五、实测输出(2026-10-11)

### 第一次提问:`现在几点?顺便算一下 1234 + 5678。`

```
[HumanMessage] 现在几点?顺便算一下 1234 + 5678。
[AIMessage]   要求调用工具:["add({'a': 1234, 'b': 5678})", 'get_current_time({})']
[ToolMessage] 6912.0
[ToolMessage] 2026-10-11 19:32:42
[AIMessage]   现在的时间是 2026年10月11日 19:32:42。计算结果:1234 + 5678 = 6912
```

**跟 T1 手搓版一模一样的流转**(模型并发调了 2 个工具、结果回灌、模型汇总),
但代码从 40 行循环体缩成了 7 行声明。

### 第二次提问(故意触发错误):`帮我算 10 除以 0 等于几?`

```
[HumanMessage] 帮我算 10 除以 0 等于几?
[AIMessage]   要求调用工具:["divide({'a': 10, 'b': 0})"]
[ToolMessage] Error: ZeroDivisionError('除数不能为 0')  Please fix your mistakes.
[AIMessage]   10 除以 0 是没有定义的,因为除数不能为 0。...
```

**这一行是重点:**

`divide` 抛了 `ZeroDivisionError`,但**流程没有崩** —— `ToolNode(handle_tool_errors=True)` 自动把它包成了 `ToolMessage`,内容就是**异常类型 + 消息**,还附了一句 `Please fix your mistakes.` 引导模型修正。

对照 T1 总结的四类错误,这就是**第 ② 类「工具执行失败 → 模型修」的框架实现**:

| 四类错误 | 手搓版(T1) | 框架版(T2) |
|---|---|---|
| ① 瞬时错误 → 系统修 | 自己写 `call_with_retry` | `retry_policy=RetryPolicy(...)` |
| ② 工具失败 → 模型修 | 自己 `except` 后塞回 messages | `ToolNode(handle_tool_errors=True)` |
| ③ 需授权 → 人修 | 阻塞式问人(假 interrupt) | `interrupt()` + checkpointer(**T5**,还没到) |
| ④ bug → 开发者修 | 不写 `except` | 不写 `except`(一样的) |

**注意 ③ 还是缺的** —— 因为它必须等 checkpointer(T4)。这就是「**T4 必须在 T5 前面**」那条因果的又一次体现。

---

## 六、踩坑记录

### 坑 1:`add_messages` 的 import 路径变了 ⚠️

网上 1.0 之前的教程几乎都写:
```python
from langchain_core.messages import add_messages   # ❌ ImportError
```
在 langgraph 1.2.14 / langchain-core 1.6.9 上直接报错。正确的位置是:
```python
from langgraph.graph import add_messages           # ✅
```

**这就是学习计划里说的「版本陷阱」的活标本** —— 看到 `from langchain_core.messages import add_messages` 就知道是老教程。

(用了 `MessagesState` 的话根本不用自己 import 它,但手写 state schema 时会用到。)

### 坑 2:条件边不传 `path_map`,图结构会画错

见上面第三节。不报错,但可视化会骗你 —— 而且**你可能会照着错的图去理解流程**。

### 坑 3:`HumanMessage` 没有 `tool_calls` 属性

```python
for m in result["messages"]:
    if m.tool_calls:        # ❌ HumanMessage / ToolMessage 上会 AttributeError
```

`tool_calls` 只有 `AIMessage` 有。得用 `getattr(m, "tool_calls", None)`。

> 顺带一个观察:**这其实是好事**。T1 里 messages 是裸 dict,你随便 `.get()` 什么都不会报错;这里是**带类型的消息对象**,写错字段会立刻炸 —— 问题在开发时暴露,而不是在线上。

### 坑 4:免费模型限流

`call_model` 里加了 `time.sleep(4)`,不然两轮之间容易被 429 打断。
节点上挂了 `RetryPolicy`,理论上限流也能靠它自动重试回来 —— 但别指望它,主动降速更省事。

---

## 七、这一步攒下的伏笔(后面会兑现)

| 已经出现的东西 | 什么时候真正用上 |
|---|---|
| **条件边** | 它就是 **Plan-and-Execute 的骨架** —— 加一条回环边(`executor → planner`)就成了 |
| **reducer** | 阶段 4 用 `Send` 扇出并行分支时,**结果怎么合并全靠它** |
| **`ToolNode` 默认并发** | 呼应 T1 的「工具级并发 vs 图级并行」 |
| **`RetryPolicy`** | 呼应 T1 四类错误的第 ① 类 |
| **`handle_tool_errors`** | 呼应第 ② 类 |
| **`MessagesState`** | T3(状态与记忆)的起点 |
| **缺 checkpointer** | **T4 一整块就是补它**,补完才能做 T5 的真 interrupt |
| **`recursion_limit`** | 做长任务 agent(T7 RAG)必然要调 —— 默认 25 只够约 12 轮工具调用 |

---

## 产出

- [x] 最小图跑通(State + Node + Edge)
- [x] 条件边 + reducer 跑通
- [x] agent 装进状态图,跟 T1 对照
- [x] 工具异常被 `ToolNode` 转成 `ToolMessage`(四类错误第 ② 类的框架实现)
- [x] 图结构导出 Mermaid
- [x] 弄清 `ToolNode` 的 partial update 从哪来(源码 `tool_node.py:886`)
- [x] 实测 `recursion_limit`(数的是 **superstep**,默认 25 ≈ 12 轮工具调用)
- [x] 实测 superstep 的真实边界(扇出 3 个 = 1 步;串行 3 个 = 3 步)
- [x] 实测节点返回值语义(给完整 state 会让 reducer 重复累加)
- [ ] commit 推送
