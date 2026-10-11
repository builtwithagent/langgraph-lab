# T1 · 不用框架,手搓一个 agent loop

## 做了什么

**完全不用 LangChain**,直接用 `openai` SDK 打智谱的兼容端点,手写一个 agent 循环。

核心就两样东西:

```
一个 while 循环  +  一张「工具名 → 函数」的映射表
```

就这么多。agent 没有魔法。

## 运行结果(2026-10-11 实测)

提问:`现在几点?帮我算 1234 + 5678,顺便数一下「人工智能」这几个字有几个字符`

```
---------- 第 1 轮 ----------
[模型要求] add({'a': 1234, 'b': 5678})
[工具返回] 6912
[模型要求] count_chars({'text': '人工智能'})
[工具返回] 4
[模型要求] get_current_time({})
[工具返回] 2026-10-11 09:58:41

---------- 第 2 轮 ----------
[终止条件达成] 模型没再要求调工具 → 循环结束

===== 最终回答 =====
- 现在时间:2026年10月11日 09:58:41
- 1234 + 5678 = 6912
- 「人工智能」有 4 个字符
```

**注意第 1 轮:模型一次性要求调用 3 个工具**(这三个不依赖彼此,所以它可以一次发起)。

> ⚠️ 但别以为"一定会一次发起"—— 同一个 prompt 换个时机跑,模型可能给你拆成三轮。原因见下面「并行工具调用」那一节。

## 这个任务真正要理解的

### ① 循环的终止条件是什么

`if not msg.tool_calls:` —— 模型**不再要求调工具**了,就说明它准备好给答案了。

这是整个 agent 唯一的分支判断。简单到有点反直觉。

### ② 消息历史必须完整回灌

```python
messages.append(msg)   # ⚠️ 模型的回复也要塞回历史
```

这一行最容易漏。漏了的话,下一轮模型不知道自己刚才调过什么工具、拿到什么结果。

而工具结果回去时必须带 `tool_call_id` 配对 —— 模型靠它知道"这是哪次工具调用的结果"。

### ③ agent = LLM 做分支决策的 while 循环

整个流程里**没有任何"智能"在你的代码里**。你的代码只是:
> 把消息发出去 → 看它要不要调工具 → 要就执行 → 塞回去 → 再来一遍

**"下一步干什么"完全由模型决定。** 这就是 agent 和普通程序最本质的区别。

---

## 对比 T0:框架到底替你干了什么

写完这份手搓版,回头看你就能列出 `create_agent` 替你做的事:

| 事项 | 手搓(T1) | 框架(T0) |
|---|---|---|
| 工具参数 schema | **手敲 JSON Schema**,一个个字段写 | 从**函数签名 + docstring 自动生成** |
| 工具映射表 | 手写 `TOOL_MAP`,忘同步就报错 | 自动维护 |
| 消息历史管理 | 手动 append + `tool_call_id` 配对 | 自动 |
| 循环与终止 | 自己写 `while` + 判断 | 内置 |
| 死循环保护 | 自己加 `max_turns` | 内置 |
| 并行工具调用 | 自己遍历编排 | 自动 |
| 错误重试 | 自己写 | 内置(部分) |
| 可观测(trace) | 完全没有 | 环境变量一开就有 |

**结论**:框架不是"更聪明",是**替你把上面这堆工程细节收进去了**。而你手写过一遍,以后它出问题时你知道该往哪查 —— 这是"会用"和"懂"的分水岭。

---

## 概念澄清:这就是 ReAct 范式吗?

**范式上——是。实现上——是"新一代"的。**

### ReAct 的出处

Yao et al., 2022,《**ReAct: Synergizing Reasoning and Acting in Language Models**》,arXiv:**2210.03629**(ICLR 2023)。项目页 https://react-lm.github.io/

核心思想:让 LLM 把「**推理轨迹**」和「**动作**」**交错(interleaved)** 地产生,形成 **Thought → Action → Observation** 循环。论文证明了"推理+行动交替"比单独的推理(如 CoT)或单独的行动效果都好。

### 但经典 ReAct 的实现方式和我们不一样

| | 经典 ReAct(2022) | 我们手搓的(现在) |
|---|---|---|
| 怎么驱动 | **纯 prompt 工程** + few-shot 人工示例 | 原生 **function calling** |
| "动作"怎么表达 | **文本**,如 `Action: search[weather]` | **结构化字段** `tool_calls` |
| 怎么执行 | 字符串/正则**解析文本** | 直接读结构化参数,无需解析 |
| Observation 怎么回灌 | 文本拼回 prompt | `role="tool"` 消息 + `tool_call_id` |
| Reasoning 在哪 | 显式写出来的 `Thought:` 文本 | **可能隐式**(模型不写出来) |

### 一个容易忽略的精确点

经典 ReAct 里,`Thought` **不产生任何外部效果** —— 它只是更新上下文;只有 `Action` 才会跟外界交互、拿到 `Observation`。

而我们手搓的 loop 里,如果模型没在 `content` 里写思考、直接返回 `tool_calls`,那"Reasoning"这一步就是**隐式**的(发生在模型内部的前向计算里)。所以严格讲:

> 我们写的是「**ReAct 范式的现代实现**」,通常叫 **tool-calling agent loop**。**思想同源,编码方式换了一代。**

### 为什么后来抛弃了文本解析

结构化输出比解析文本可靠得多:不用写正则、不会因为模型多说一句废话就解析失败。这是 function calling 带来的最大工程收益。

### 想看显式的 Reasoning 怎么办

两条路:
1. 让模型开 thinking 模式(智谱 `glm-4.7-flash` 支持 `thinking` 参数,会额外返回 `reasoning_content` 字段)
2. 在 system prompt 里明确要求"先写思考,再决定调不调工具"

> 📌 这部分属于 agent 的**架构模式**(学习计划里的 03 主题)。后面还会遇到 Plan-and-Execute、Reflexion 等变体 —— ReAct 只是其中最基础、最常用的一个。

### 演化线索:为什么说不是简单"更好"

```
2022  ReAct            显式 Thought,靠 prompt 工程 + 文本协议 + few-shot 示例
        ↓
2023–25  tool calling     推理隐式化;工程上更便捷可靠 —— 但"显式推理"这个优势丢了
        ↓
2025+  reasoning models  显式思考回来了,而且这次是模型训练出来的能力(thinking 模式),不靠 prompt
```

**所以更准确的说法:这是「演化 + 部分回归」,不是单纯替代。**

- **工程实现维度**:tool calling 确实全面胜出 —— 不用喂 few-shot 示例、不用正则解析、结构化输出更可靠
- **可解释性维度**:反而退了半步。ReAct 论文特意强调"推理轨迹让整个过程可读、可诊断、可人工纠错",而隐式推理把这个优势拿掉了

这也解释了为什么现在「思考模式」又成了标配 —— **本质上是在把 ReAct 的 Thought 找回来**,只不过换成了模型内化的能力。

### 那 Observation 呢?—— 就是你塞回去的那条消息

**对,`role="tool"` 那条消息 = ReAct 里的 Observation。** 角色完全对应:工具对动作的反馈,重新进入模型上下文。

两个差别:

| | 经典 ReAct | 我们的实现 |
|---|---|---|
| 格式 | 一段**文本**,拼进 prompt | 结构化消息 `{role:"tool", tool_call_id, content}` |
| 与 Action 怎么对应 | **靠文本顺序隐式对应** | **`tool_call_id` 显式绑定** |

**显式绑定这点很关键**:还记得 T0/T1 里模型一次并列调了 3 个工具吗?如果只靠文本顺序,模型根本分不清"`6912` 是 `add` 的结果还是 `count_chars` 的结果"。`tool_call_id` 就是解决这个的 —— 这是现代实现比经典 ReAct 严谨的地方,也是并行工具调用能成立的前提。

**一个边界情况**:严格说,`add()` 这种**纯计算函数**算不上真正的"环境"—— 它没跟外部世界交互,只是本地算了个数。经典 ReAct 的 Observation 来自真实环境(搜索 API、维基百科、游戏状态)。实践中大家统称 Observation,因为对模型来说没区别:**都是"外面给的信息"。**

---

## 并行工具调用:拆成三层才看得清(2026-10-11 实测)

「并行工具调用」这个词把三件不同的事混在一起了,分开看就清楚。

### 第二层 · 执行层:你的代码根本没并发

先看自己写的代码:

```python
for call in msg.tool_calls:      # ← 这是个 for 循环,一个一个跑
    ...
```

手动构造 3 个「各睡 1 秒」的工具调用,**完全绕过模型**,直接测执行耗时:

| 执行方式 | 耗时 |
|---|---|
| `for` 循环逐个跑(`agent_loop.py` 的写法) | **3.00s** |
| `ThreadPoolExecutor` 并发 | **1.00s** |

**「同一轮」不等于「同时跑」。** 模型一次发起了 3 个,你的代码还是排队执行完的。
LangChain / LangGraph 默认会真的并发跑同一轮的多个工具,手搓版不会 —— 这是框架替你干的又一件小事(对照上面那张 T0 vs T1 表里的"并行工具调用"行)。

### 第一层 · 决策层:一轮调几个,是模型说了算

同一句 prompt、`temperature=0`,连跑 3 次:

| 第几次 | 每轮的调用数 |
|---|---|
| 1 | `[1, 1, 1]` ← 拆成了三轮 |
| 2 | `[3]` ← 一轮并发发起 |
| 3 | `[3]` ← 一轮并发发起 |

**发给模型的消息一模一样,它给出的并行结构却不同。**

所以「并行还是串行」不是你在代码里拨的开关 —— 你的代码只是**如实执行模型要的那几个**:它要 3 个你跑 3 个,它要 1 个你跑 1 个。**决策权在模型手里。**

> ⚠️ 顺带一个工程结论:**`temperature=0` 不等于输出确定。** 别把可复现性建立在这上面,要复现得自己存断言、做回归(这就是 T8 评测的动机之一)。

### 第三层 · 依赖约束:结构上就并行不了

问:「北京现在气温多少?根据温度推荐一下穿什么。」

```
第 1 轮:调用 get_weather              → 拿到 temp=18
第 2 轮:调用 pack_clothes(temp=18)    → temp 只能来自上一步的返回值
第 3 轮:给出答案
```

每轮只有 1 个调用。`pack_clothes` 的 `temp` 参数**只能来自上一步的 Observation** —— 这不是模型"选择"了串行,而是**物理上不可能并行**。

> **判断能不能并行,只看一件事:后面的调用需不需要前面的结果。**
> 需要 → 必须等下一轮。不需要 → 模型才有机会一次发起。

### 三层合起来的意义

| 层 | 谁决定 | 你的代码要做什么 |
|---|---|---|
| 决策层 | **模型** | 干预不了,只能跟着走 |
| 执行层 | **你的代码** | 想真并发就自己写(框架帮你写了) |
| 依赖约束 | **任务本身** | 识别出来,别硬凑并行 |

**而并行真正的收益,不是省时间,是省 LLM 往返次数:**

```
3 个工具 · 并行:  1 次 LLM 调用(一次要 3 个) + 1 次 LLM 调用(汇总) = 2 次
3 个工具 · 串行:  3 次 LLM 调用(每次要 1 个) + 1 次 LLM 调用(汇总) = 4 次
```

每多一轮,就要把**整个消息历史**重新喂给模型一遍 —— 这正是后面 ReWOO / Plan-and-Execute 想解决的问题(一次性规划,避免反复重放历史)。详见 `ai-agent-notes/notes/03-Agent范式对比.md`。

### 那「并行工具调用」这个词,到底指哪一件?

**业界说「并行工具调用」(parallel tool calling / parallel function calling),通常指的是第一层 —— 模型的能力,而不是你的代码怎么执行。**

证据是它有自己的 API 参数:OpenAI 在 2024-06 加了 `parallel_tool_calls`,**默认 true**。官方原话:

> You can prevent this by setting `parallel_tool_calls` to false, which ensures exactly zero or one tool is called.

注意它管的是「**模型一次给几个调用**」—— 这个参数在**请求模型**的时候传,跟「你怎么执行工具」没有半点关系。

> 📎 https://community.openai.com/t/new-api-feature-disable-parallel-function-calling-via-parallel-tool-calls-false/805405

**所以:视频里讲的「并行工具调用」≈ 决策层能力;「开线程池并发跑」= 执行层实现。两件事,而且是串联的 —— 模型不给多个 call,你线程池就是空转。**

#### 实测:智谱端点忽略了这个参数

照着上面那个语义在 `glm-4.7-flash` 上试(每组跑 2 次,同一个问题):

| 配置 | 第 1 次 | 第 2 次 |
|---|---|---|
| 不传该参数 | `[3]` | `[3]` |
| `parallel_tool_calls=True` | `[3]` | `[3]` |
| **`parallel_tool_calls=False`** | **`[3]`** | **`[3]`** |

**传 `False` 照样一次给 3 个 —— 端点忽略了它**(不报错,也不生效)。

实践含义:**在智谱上你没法从 API 层面强制「一次只调一个」**。真要控制,只能 ① 在 system prompt 里明确要求;② 把有顺序要求的工具拆到不同的图节点里。

> ⚠️ 样本量小(每组 2 次),当参考。但「False 组两次都是 3」已经足够否掉「参数生效」这个假设。
> 复现:`./.venv/Scripts/python 02-raw-agent-loop/parallel_tool_calls_probe.py`(免费模型限流严重,会跑十几分钟)

### LangGraph 里不用你自己开线程池

手搓版得写 `ThreadPoolExecutor`,但框架里这一步**默认就做好了**。LangGraph `ToolNode` 官方文档原话:

> If multiple tool calls are requested, they will be run in parallel.

源码层面 —— 异步走 `asyncio.gather`,同步走线程/执行器:

```python
outputs = await asyncio.gather(
    *(self._arun_one(call, input_type, config) for call in tool_calls)
)
```

> 📎 文档 https://langchain-ai.github.io/langgraph/reference/prebuilt/
> 📎 源码 `libs/prebuilt/langgraph/prebuilt/tool_node.py`
> ⚠️ `reference.langchain.com` 这个域名连不上(curl 返回 000),所以文档链接换成了上面这个可访问的地址。

**注意它用的是 asyncio,不是线程池。** 现代 Python 框架基本都往异步走 —— 所以写工具时,异步工具(配异步客户端)会更划算。

#### 顺带分清:工具级并发 ≠ 图级并行

这俩容易混,是两个不同层级的东西:

| | 工具级并发 | 图级并行 |
|---|---|---|
| 发生在哪 | **一个节点内部**(ToolNode) | **图的多个分支之间** |
| 谁触发 | 模型一轮给了多个 tool_call | 你用 `Send` API 扇出 |
| 常见叫法 | 节点内部的 "mini-superstep" | fan-out / map-reduce |
| 学到哪一步 | 现在(T1–T2) | 阶段 4(并行 / 子图) |

LangGraph 官方论坛的说法很到位:**ToolNode 的并发调用是节点内部的 "mini-superstep";图级扇出才用 `Send`。** 这是两个层级的东西,别混。

### 安全提醒:有副作用的工具别让它并行

并行 = **不保证顺序,也不保证原子性**。查询类工具并行没问题(互不影响),但**扣款、下单、发邮件、删数据**这类工具并行就是事故。

真要并行有副作用的工具,常见做法两条:
- system prompt 里限制「有副作用的工具一次只调一个」
- 或者把这类工具单独放一个节点,用图的拓扑强制串行

> 做资金/交易类 agent 尤其要记住这条 —— **涉及钱的地方,顺序和幂等性没有「大概」。**

### 顺带:经典 ReAct 结构上不可能并行

2022 年的 ReAct 是**一次生成一段文本、从中解析出一个 Action** —— 一条消息只能装一个动作,想并行都没地方放。
**并行工具调用是 function calling 时代才有的能力**,不是 ReAct 范式自带的。这是现代实现相对经典 ReAct 的一个实打实的增量。

### 复现方式

```bash
cd /d/VibeCoding/langgraph-lab
./.venv/Scripts/python 02-raw-agent-loop/parallel_experiment.py
```

脚本分三段:执行层(秒出,不调模型)→ 决策层(同一 prompt 跑 3 次)→ 依赖约束。

---

## 工具失败了怎么办:四类错误 × 四种处理(2026-10-11 实测)

工具报错只有四种归宿,判断标准是**「这件事该谁来修」**。完整清单见 `ai-agent-notes/notes/02-工具调用的并发与安全.md` 第六节,这里是**能跑的代码**。

| 类别 | 谁修 | 处理方式 |
|---|---|---|
| ① 瞬时错误(429 / 超时) | **系统** | 工具层重试掉,**模型根本不知道出过错** |
| ② 工具执行失败 / 参数错 | **模型** | 把错误当 Observation 塞回 messages,它自己改 |
| ③ 缺信息 / 需要授权 | **人** | 停下来问,拿到答复再继续 |
| ④ 未预期的 bug | **开发者** | **不捕获,让它抛出来** |

### 核心设计:用异常类型分流

整个 demo 的关键就这一段 —— **执行器只捕获它明确知道该怎么处理的两种,其余一律冒泡**:

```python
try:
    result = call_with_retry(fn, args)        # ① 瞬时错误在这一层解决
    return str(result)

except NeedHumanInput as e:                   # ③ 知道怎么办 → 问人
    ...

except ToolExecutionError as e:               # ② 知道怎么办 → 还给模型
    return f"工具执行失败:{e}"

# ④ 故意不写 except Exception
#    写上去就等于把 bug 伪装成「工具失败」
```

这正是 PEP 8 的建议:**捕获异常要写明具体类型,别用裸 `except`**。

### 实测结果(四个场景全按预期)

**① 瞬时错误 —— 模型全程无感**
```
第 1 轮:模型要求调用 ['fetch_weather']
  [调用] fetch_weather({'city': '北京'})
  [系统重试] 第 1 次失败(上游返回 429),等 1s 再试
  [系统重试] 第 2 次失败(上游返回 429),等 2s 再试
  [结果] 北京:晴,18℃
第 2 轮:无工具调用 → 终止
```
✅ 重试两次成功。**模型到最后都不知道发生过错误** —— 这正是我们要的:它知道了也没用,它既不能等也不能改参数。

**② 参数错 —— 模型自己改对了(零干预)**
```
第 1 轮:[调用] search_order({'order_id': '#00012345'})
        [还给模型] 订单号 '#00012345' 格式不对 —— 必须是 8 位纯数字(不要带 #),例如 00012345
第 2 轮:[调用] search_order({'order_id': '00012345'})     ← 自己改对了!
        [结果] 已发货,预计后天送达
```
✅ 这一条最能说明「错误是信息不是终止信号」。**错误文案写成"必须是 8 位纯数字,例如 00012345",模型读完下一轮就改对了。**

**③ 需要授权 —— 停下来问人**
```
第 1 轮:[调用] issue_refund({'order_id': '00012345', 'amount': 299})
        [需要人工] 即将为订单 00012345 退款 299 元,确认执行吗?
        [人工答复] y
        [结果] 已受理退款:00012345 共 299 元
```
✅ 有副作用的操作先问人。前提是工具本来属于第 ④ 级(非幂等写)—— 见 `notes/02-...` 的「工具四级」表。

**④ bug —— 直接炸,没人接**
```
第 1 轮:[调用] get_customer_profile({})

❌ 未预期错误 → 第 ④ 类,由开发者修
   类型:KeyError
   信息:KeyError('level')

Traceback (most recent call last):
  ...
  File "error_handling_demo.py", line 117, in get_customer_profile
    return f"会员等级:{profile['level']}"
KeyError: 'level'
```
✅ **前三类都有人接,这一类没人接 —— 这就对了。** 如果这里被一句 `except Exception` 接住、返回「工具调用失败」,你的 bug 就永远埋起来了。

### ⚠️ 手搓版做不了真正的 ③

手搓 loop 没有 checkpointer,**做不了「暂停 → 几小时后从断点恢复」**。这里的场景 3 是用「阻塞式问人 + 记住授权」模拟同样的语义。

LangGraph 里的正规实现是 `interrupt()`,而它**必须配 checkpointer** —— 这就是学习计划里说的「**T4 必须在 T5 前面**」。

### 复现

```bash
cd /d/VibeCoding/langgraph-lab
./.venv/Scripts/python -u 02-raw-agent-loop/error_handling_demo.py       # 全部(限流下约 5 分钟)
./.venv/Scripts/python -u 02-raw-agent-loop/error_handling_demo.py 2 4   # 只跑指定场景
```

想单看执行器的分流逻辑,不用调模型也能验(见文件顶部注释)。

### 一个只有踩过才知道的细节

场景 1 的重试**放在工具调用这一层**(`call_with_retry`),不是放在 agent loop 层。区别很大:
- 放在**工具层** → 重试成功的话,模型连这次失败都不知道,消息历史干干净净
- 放在 **loop 层** → 每一轮都要重试整轮,而且失败会污染消息历史,模型会看到"我刚才失败了"

**能往底层放的重试,就别往上层放。**

---

## 踩坑记录

### 429 限流(免费模型必踩)

第一次运行直接报错:

```
openai.RateLimitError: Error code: 429
{'error': {'code': '1305', 'message': '该模型当前访问量过大,请您稍后再试'}}
```

智谱免费模型的并发限制比较紧,**高峰期会直接拒绝**。等十几秒重试就通过了。

**生产环境的标准做法是加指数退避重试**,可以自己加到 `run_agent` 里练手:

```python
import time

def call_with_retry(messages, retries: int = 3):
    for i in range(retries):
        try:
            return client.chat.completions.create(
                model=MODEL, messages=messages, tools=TOOLS, temperature=0
            )
        except Exception as e:
            if i == retries - 1:
                raise
            wait = 2 ** i          # 1s → 2s → 4s
            print(f"[重试] {e.__class__.__name__},{wait}s 后第 {i + 2} 次尝试")
            time.sleep(wait)
```

> 这也是为什么"就绪工程化"比"跑通 demo"值钱:demo 不考虑限流,生产必须考虑。

## 产出

- [x] 手搓的 agent loop 跑通(不用任何框架)
- [x] 完整的两轮运行日志
- [x] 与框架版的逐项对比
- [ ] commit 推送
