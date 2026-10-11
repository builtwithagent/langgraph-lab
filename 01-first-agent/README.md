# T0 · 环境 + 第一个 agent

## 做了什么

用 LangChain 1.x 的 `create_agent` 跑通「模型 + 工具」的最小闭环 —— 让它自己决定调不调工具。

## 怎么跑

```bash
cd /d/VibeCoding/langgraph-lab
cp .env.example .env      # 然后填 API Key
./.venv/Scripts/python 01-first-agent/agent.py
```

## 这个任务真正要理解的三件事

1. **函数的 docstring 就是工具说明书**
   模型靠 docstring 判断"这个工具干什么、什么时候用、参数怎么传"。docstring 写得含糊,工具就调不对 —— 这不是代码注释,是给模型的 prompt。

2. **`create_agent` 底层就是 LangGraph**
   1.0 之后,LLM agent 的标准入口是 `create_agent`(它内部跑在 LangGraph 运行时上)。这一步先当黑盒用,后面 T2 开始拆开看。

3. **完整消息链里能看到 agent 的"思考过程"**
   输出里有:用户提问 → AI 决定调工具 → 工具返回结果 → AI 汇总回答。这就是 agent loop 的最小形态。T1 会让你不用框架手写一遍。

## 运行结果(2026-10-11 实测)

提问:`现在几点?顺便算一下 1234 + 5678 等于多少`

消息链是这样的:

```
Human Message    → 用户提问
Ai Message       → 模型决定调工具(一次并列调了 2 个!)
  Tool Calls: add(a=1234, b=5678)
              get_current_time()
Tool Message     → 6912.0
Tool Message     → 2026-10-11 09:12:42
Ai Message       → 现在时间是 2026-10-11 09:12:42;1234 + 5678 = 6912
```

**这就是 agent loop 的完整形态**:提问 → 决策 → 调工具 → 拿结果 → 汇总。

## 踩坑记录

1. **LangSmith 报错**:`LANGSMITH_TRACING=true` 但没填 `LANGSMITH_API_KEY` 会报错。没 key 时把 `LANGSMITH_TRACING` 设成 `false`;拿到 key 再改回 `true`。

2. **模型一次并行调用了两个工具**(重点观察)
   它没有"先算加法,等结果,再查时间"这么串行地跑两轮,而是在**同一次决策里并列发出两个 Tool Call**(看输出的两个 call_id 是并列的)。
   → 这说明:模型可以把互不依赖的工具调用**并发**发起,这是 agent 效率的关键来源,也是后面 LangGraph 做并行节点(T6)的基础。

3. **`add` 返回 `6912.0` 而不是 `6912`**
   因为函数签名写的是 `a: float, b: float`。想让工具返回整数,把类型标注改成 `int` 即可 —— **顺便体会到一件事:工具的返回类型也是模型能感知的**。

## 附:LangSmith 为什么不用在代码里写一行?

**因为开关是环境变量,而"上报的钩子"是框架默认埋好的。**

完整链路(源码位置基于本机 langchain-core 1.6.9,可自行核验):

1. `.env` → `load_dotenv()` → 写进**进程环境变量**(这是唯一的"开关")
2. 任何 Runnable 执行时都要建一个 `CallbackManager`,而**建的时候自动就调了** `_get_trace_callbacks()`
   → `langchain_core/callbacks/manager.py:103`
3. 它先问一句"追踪开着吗":`_tracing_v2_is_enabled()`
   → `langchain_core/tracers/context.py:132` → 再往下走到 `langsmith/utils.py:tracing_is_enabled()`
4. 开着 → 当场实例化一个 `LangChainTracer`,挂进 callbacks 里
   → `langchain_core/tracers/context.py:112`
5. 之后 LLM、工具的每个生命周期事件(`on_llm_start` / `on_tool_start` / `on_tool_end` …)都被这个 tracer 接住,**批量上报**到 LangSmith

**关键点**:所有组件执行都必须经过 callback manager —— tracer 只是**插在这条必经之路上的一个监听器**。所以业务代码零侵入。

**再精确一层**:第 3 步那个"开着吗",判断其实是**分优先级的**(见 `langsmith/utils.py:tracing_is_enabled()`):

| 优先级 | 判断依据 | 场景 |
|---|---|---|
| 1 | 代码里显式设置(context var) | `with tracing_v2_enabled():` 最优先 |
| 2 | 当前是否已处在某个 trace 里 | 子步骤自动继承,不用重复判断 |
| 3 | 全局开关 ← **这里才落到环境变量** | `LANGSMITH_TRACING=true` |

→ 也就是说:**环境变量是"全局默认值",代码随时可以覆盖它。**

> 反过来说,这也解释了为什么"没填 key 但开着开关"会报错:第 3 步判断为"开",第 4 步真的去建 tracer,建的时候才发现没有客户端可连。

**想手动控制某一个块**(而不是全局开):

```python
from langchain_core.tracers.context import tracing_v2_enabled

with tracing_v2_enabled(project_name="my-project"):
    agent.invoke({"messages": [...]})   # 只有这一块被追踪
```

## 产出

- [x] 能跑通的 agent 脚本
- [x] 完整的消息链输出(上面已记录)
- [ ] LangSmith 上的一条 trace(等注册后补)
- [ ] commit 推送
