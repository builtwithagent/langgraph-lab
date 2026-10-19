"""
T3 · 第 3 步:摘要压缩(summarize)
==================================

【这个文件要回答的问题】
上一步的 `trim_messages` 是"丢掉老的" —— 简单、零成本,但**信息永久损失**。

`【官方】` 官方原文直说了这个问题:
    "The problem with trimming or removing messages, as shown above, is that **you may lose
     information from culling of the message queue**. Because of this, some applications benefit
     from a more sophisticated approach of **summarizing the message history using a chat model**."
    —— https://docs.langchain.com/oss/python/langgraph/add-memory

摘要的思路:**不是丢掉,而是压成一段话**。
    · 裁剪 = 有损 + 不可逆 + 零成本
    · 摘要 = 有损 + 但保留了语义 + 要调模型

`【官方】` 更正式的定义(Anthropic 叫它 **compaction**):
    "Compaction is the practice of taking a conversation **nearing the context window limit**,
     **summarizing its contents**, and **reinitiating a new context window with the summary**."
    "Compaction typically serves as **the first lever in context engineering** to drive better
     long-term coherence."
    —— https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents

【本文件的来源标注】(规则见仓库根目录 SOURCES.md)
    `【官方】` = 官方文档/博客原文,附片段    `【源码】` = 本机包里的真实代码
    `【实测】` = 本机跑出的数据              `【推论】`/`【建议】` = 我推的 / 我建议的,官方没说

【官方文档】
  Add memory(Summarize messages 一节)
    https://docs.langchain.com/oss/python/langgraph/add-memory

⚠️ 本文件会**真的调模型**(摘要必须靠模型做)。
   智谱免费额度限流严重(429 / code 1305、1302),所以:
     · 每次失败自动等待重试(见 _invoke_with_retry)
     · 如果实在打不通,会打印提示并跳过,不会让整个脚本崩掉
     · 想**不联网也把管道跑通**,加 `--offline`:
          .venv/Scripts/python.exe 04-state-memory/03_summarize_messages.py --offline
       离线模式用一个"本地假摘要"顶替模型 —— 它**不是真的摘要**,
       只是为了让 RemoveMessage / state 合并 / 条件边这套**机制**能被验证。
       真摘要必须调模型。

运行:
    cd D:/VibeCoding/langgraph-lab
    .venv/Scripts/python.exe 04-state-memory/03_summarize_messages.py            # 真调模型
    .venv/Scripts/python.exe 04-state-memory/03_summarize_messages.py --offline  # 不联网
"""

import os
import sys
import time
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from pydantic import SecretStr

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

LINE = "=" * 70

# 离线模式:用本地假摘要顶替模型,专供限流时验证"管道"
OFFLINE = "--offline" in sys.argv


# ══════════════════════════════════════════════════════════════════
# 模型 + 带重试的调用
# ══════════════════════════════════════════════════════════════════
_api_key = os.getenv("MODEL_API_KEY")
if not _api_key and not OFFLINE:
    raise RuntimeError("没找到 MODEL_API_KEY —— 请检查 .env 文件")

model = None
if _api_key:
    model = ChatOpenAI(
        model=os.getenv("MODEL_NAME", "glm-4.7-flash"),
        base_url=os.getenv("MODEL_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
        api_key=SecretStr(_api_key),
        temperature=0,
        max_retries=0,  # 关掉 SDK 自带重试,我们自己控制节奏
    )

CALLS = {"n": 0}


def _invoke_with_retry(messages, attempts: int = 3):
    """
    免费模型限流太狠,这里自己写个"等待 + 重试"。

    ⚠️ 注意这跟 T1 笔记里"瞬时错误交给框架重试"是一个道理,
       但这里是**手动挡** —— 之所以手动,是因为免费接口的限流是
       "每 N 秒只允许 1 次",退避系数要比默认的大得多。
       真上了有配额的付费接口,直接用 RetryPolicy 就行(见 02_agent_as_graph.py)。

    📌 实测:智谱免费档在连续调用时几乎必 429,报错长这样
         429 - {'error': {'code': '1302', 'message': '您的账户已达到速率限制…'}}
         429 - {'error': {'code': '1305', 'message': '该模型当前访问量过大…'}}
       1302 偏"你调太快了",1305 偏"服务端挤",两个都归到"瞬时错误"里,
       处理方式都是等待 + 重试,不要当时业务逻辑 bug 去改代码。
    """
    last_err = None
    for i in range(attempts):
        try:
            CALLS["n"] += 1
            resp = model.invoke(messages)  # type: ignore[union-attr]
            time.sleep(3)  # 成功后也歇一下,给后续请求留额度
            return resp
        except Exception as e:  # noqa: BLE001
            last_err = e
            wait = 8 * (i + 1)
            print(f"    ⏳ 第 {i + 1} 次失败({type(e).__name__}),等 {wait}s 再试…")
            time.sleep(wait)
    raise RuntimeError(f"模型连着 {attempts} 次没打通:{last_err}")


# ══════════════════════════════════════════════════════════════════
# 造一段"长对话"(纯手工,不调模型)
# ══════════════════════════════════════════════════════════════════
def build_long_history(assign_ids: bool = True) -> list:
    """
    模仿一个真实的多轮对话:前面几轮信息密度高(有偏好、有约束),
    后面几轮开始啰嗦(闲聊、重复确认)。

    这正是摘要最该发挥作用的场景:**把前面高密度的信息保住,把后面啰嗦的压掉。**

    ── assign_ids 这个参数是踩坑踩出来的 ──────────────────
    手工 `HumanMessage(content=...)` 造出来的消息,**id 是 None**。
    而 `RemoveMessage(id=...)` 是**完全靠 id 定位**的 ——
    没有 id 就删不掉,而且**不报错、静默失效**(见下面 part_3 的实测)。

    真实图里不会有这个问题:消息进 state 时会经过 add_messages reducer,
    reducer 会自动给没 id 的消息补一个 uuid。
    这里为了在脚本里模拟出真实效果,就先手动过一遍 reducer 把 id 补上。
    """
    msgs = [
        SystemMessage(content="你是一名电商客服助手。回答要简洁,不承诺时效,不报价。"),

        # ── 前 2 轮:关键信息(裁掉就出事)──────────────────
        HumanMessage(content="我叫小李,会员号是 M88231。"),
        AIMessage(content="好的,记下了:小李,会员号 M88231。"),

        HumanMessage(content="我只买蓝色那款,预售的不考虑。"),
        AIMessage(content="明白。筛选条件:蓝色,排除预售款。"),

        HumanMessage(content="还有,别给我打电话,只发短信就行。"),
        AIMessage(content="了解,后续只通过短信联系您。"),

        # ── 中间几轮:操作细节(压缩时可以丢细节)──────────────
        HumanMessage(content="帮我看一下订单 88345 到哪了。"),
        AIMessage(content="订单 88345 已从华东仓发出,目前在运输途中。"),
        HumanMessage(content="预计几号能到?"),
        AIMessage(content="预计后天下午送达。"),

        # ── 后面几轮:啰嗦、重复(最该被压掉的)──────────────
        HumanMessage(content="嗯,我再想想。"),
        AIMessage(content="好的,您慢慢考虑,有需要随时找我。"),
        HumanMessage(content="对了,刚才那个物流信息准吗?"),
        AIMessage(content="是的,和您刚才问的订单 88345 是同一单。"),
        HumanMessage(content="那再帮我确认一次我的会员号。"),
        AIMessage(content="您的会员号是 M88231。"),
    ]

    if assign_ids:
        # 过一遍 add_messages(就是 messages 字段的 reducer),让它自动补 id
        from langgraph.graph import add_messages

        return list(add_messages([], msgs))  # type: ignore[arg-type]
    return msgs


def _brief(msgs: list) -> None:
    if not msgs:
        print("      (空)")
        return
    for i, m in enumerate(msgs):
        tag = {
            "SystemMessage": "SYS ",
            "HumanMessage": "HUM ",
            "AIMessage": "AI  ",
            "ToolMessage": "TOOL",
            "RemoveMessage": "DEL ",
        }.get(type(m).__name__, "??  ")
        text = str(getattr(m, "content", ""))[:34]
        mid = getattr(m, "id", None)
        mid = f"#{mid[:6]}" if mid else "      "
        print(f"    [{i:>2}] {tag} {mid} {text}")


# ══════════════════════════════════════════════════════════════════
# 第 1 部分:先看清"裁剪"解决不了什么
# ══════════════════════════════════════════════════════════════════
def part_1_why_trim_is_not_enough() -> None:
    print("\n" + LINE)
    print("第 1 部分:为什么光裁剪不够 —— 裁掉的关键信息不会自己回来")
    print(LINE)

    history = build_long_history()
    print(f"\n  原始对话 {len(history)} 条:")
    _brief(history)

    print("""
  假设现在上下文满了,我们粗暴地"只留最后 4 条":

      保留: 物流那轮 + 两句闲聊 + 最后一次确认会员号
      丢掉: 「只买蓝色、不要预售」
            「别打电话,只发短信」
            「我叫小李,会员号 M88231」

  💥 后果:下一轮就算只留了最后 4 条,模型也**不知道**这些约束了。
     用户再问"帮我看看有没有合适的",模型可能会推预售款 ——
     它不是在敷衍你,它是**真的忘了**。

  👉 这就是"有损不可逆":消息一旦从 state 里删掉,就没有任何地方记着它了。
     要让它"记得但轻一点",就得换一招 —— 摘要。
""")


# ══════════════════════════════════════════════════════════════════
# 第 2 部分:摘要压缩的核心动作
# ══════════════════════════════════════════════════════════════════
SUMMARY_PROMPT = """请把上面的对话压缩成一段中文摘要,供后续对话继续使用。

要求:
  · 只保留「事实、约束、偏好、已执行的动作」,闲聊和重复确认全部删掉
  · 用户给出过的标识(会员号、订单号、地址等)必须原样保留
  · 不要写"用户问了…助手回答了…"这种流水账
  · 直接在 120 字以内说完,不要任何开场白"""


def _fake_summarize(messages: list, previous_summary: str = "") -> str:
    """
    ⚠️ 这**不是**摘要,只是一个"长得像摘要的占位符"。

    作用:在模型被限流打不通的时候,让脚本后面的**管道部分**
    (RemoveMessage 清空 / state 合并 / 条件边触发)仍然能被完整验证。

    实现:抽取出"含数字或含约束词"的消息,截断拼接 ——
          它连语义都不理解,纯粹是字符串筛选。
          所以别指望它压得准,它只负责**把形状做对**。
    """
    keep_words = ("只", "不", "会员", "订单", "短信", "蓝色", "预售", "叫")
    picked = []
    for m in messages:
        if type(m).__name__ != "HumanMessage":
            continue
        text = str(m.content)
        if any(w in text for w in keep_words):
            picked.append(text)
    body = ";".join(picked[:4])
    if previous_summary:
        return f"[旧摘要]{previous_summary}\n[新增]{body}"[:200]
    return f"[本地假摘要,非模型生成]{body}"[:200]


def summarize(messages: list, previous_summary: str = "") -> str:
    """
    把一串消息压成一段摘要。

    两种模式:
      · 首次压缩(previous_summary 为空):整段历史 -> 一段摘要
      · 增量压缩(previous_summary 非空):旧摘要 + 新增消息 -> 新摘要

    ⚠️ 为什么要区分?因为**摘要本身也会越来越长**。
       如果每次都是"全新压一遍",你会把已经压过的内容再压一次 —— 信息衰减很快。
       正确做法是"把旧摘要带上,让它接着补",这样旧信息不会被反复重压。
    """
    if OFFLINE:
        print("    (offline 模式:用本地假摘要顶替模型)")
        return _fake_summarize(messages, previous_summary)

    if previous_summary:
        prompt = (
            f"这是截至目前对话的摘要:\n{previous_summary}\n\n"
            "请结合下面的新增对话,把摘要**更新**一下(合并新旧内容,不要丢已有关键信息,"
            "仍然控制在 120 字以内):"
        )
    else:
        prompt = SUMMARY_PROMPT

    resp = _invoke_with_retry([*messages, HumanMessage(content=prompt)])
    return str(resp.content).strip()


def part_2_first_summary() -> str:
    print("\n" + LINE)
    print("第 2 部分:第一次摘要 —— 把整段历史压成一段话")
    print(LINE)

    history = build_long_history()
    print(f"\n  输入:{len(history)} 条消息")

    try:
        summary = summarize(history)
    except RuntimeError as e:
        print(f"\n  ❌ {e}")
        print("  (跳过本部分 —— 大概率是限流,过一会儿再跑)")
        return ""

    print("\n  📌 摘要结果:")
    print("  " + "-" * 64)
    for line in summary.splitlines():
        print(f"    {line}")
    print("  " + "-" * 64)
    print(f"\n  长度:{len(summary)} 字符(原始内容约 "
          f"{sum(len(str(m.content)) for m in history)} 字符)")

    print(f"""
  👉 关键点:摘要**不是**"把消息变短",而是"把消息换成另一种表示"。
     原来 {len(history)} 条消息结构复杂(有 role、有 tool_calls、有条数开销),
     现在变成 1 段纯文本。
     后续要发给模型时,它占的位置很小,但语义还在。
""")
    return summary


# ══════════════════════════════════════════════════════════════════
# 第 3 部分:把摘要"塞回 state" —— 用 REMOVE_ALL_MESSAGES
# ══════════════════════════════════════════════════════════════════
class SummaryState(MessagesState):
    """
    ⚠️ MessagesState 是 TypedDict,**可以直接被继承**。
       继承之后 messages 字段连带它的 add_messages reducer 一起拿来,
       我们只需要补一个 summary 字段。

       (T3 第 1 步 01_state_schema.py 里验证过:没写 Annotated 的字段
        用的是 LastValue —— 后写覆盖先写。摘要正好就需要这个语义:
        每次更新直接覆盖旧的,不需要累加。)
    """

    summary: str


def summarize_node(state: SummaryState) -> dict:
    """
    摘要节点。做三件事:

      1. 拿旧摘要(如果有),让模型"接着补"而不是"重压一遍"
      2. 把消息清掉 —— 用 RemoveMessage,而不是直接赋 []
      3. 把摘要存进 summary 字段

    ⚠️ 顺序很重要:先算摘要再删消息 —— 因为摘要要读消息。
       如果你先删了,模型就没东西可读了。

    📌 删消息有两种姿势,区别很大:

       姿势 A(精确删除,官方范例用的):
           delete = [RemoveMessage(id=m.id) for m in messages[:-2]]
           -> 只删老的,保留最后 2 条继续当上下文
           -> 适合"摘要 + 保留近期对话"的组合

       姿势 B(一键清空):
           RemoveMessage(id=REMOVE_ALL_MESSAGES)
           -> 全清,一条不留,只剩摘要
           -> 适合"这轮对话彻底翻篇,只留结论"

    本函数用 A。B 在下面 part_3 里单独演示。
    """
    old_summary = state.get("summary", "")

    # 只压"老的",保留最近 2 条给下一轮当上下文(官方范例就是这个套路)
    to_summarize = state["messages"][:-2]
    keep = state["messages"][-2:]
    new_summary = summarize(to_summarize, previous_summary=old_summary)

    return {
        "summary": new_summary,
        # 精准点名要删哪几条 —— add_messages 认 RemoveMessage 的 id,会逐条删掉
        "messages": [RemoveMessage(id=m.id) for m in to_summarize if m.id],
    }


def part_3_summary_into_state() -> None:
    print("\n" + LINE)
    print("第 3 部分:把摘要塞回 state —— 两种删消息的姿势")
    print(LINE)

    # ── 先踩一个坑:消息没有 id 时会静默删不掉 ──────────────
    print("\n  ⚠️ 先看一个坑:手工造的消息没有 id,RemoveMessage 就删不动它")

    from langgraph.graph import add_messages

    no_id = build_long_history(assign_ids=False)
    print(f"    手工构造的消息:第 1 条 id = {no_id[0].id}  ← None")
    print(f"    其中 id 不为 None 的有:{sum(1 for m in no_id if m.id)} 条 / 共 {len(no_id)} 条")

    bad = [RemoveMessage(id=m.id) for m in no_id[:-2] if m.id]
    after_bad = add_messages(no_id, bad)
    print(f"    尝试删掉前 {len(no_id) - 2} 条 → 实际剩 {len(after_bad)} 条"
          f"   💥 一条都没删掉,而且**没有任何报错**")

    print("""
    💥 这就是最阴的一类 bug:**静默失效**。
       `RemoveMessage` 是**完全靠 id 定位**的,id=None 它找不到目标,
       但它不会抛异常,只是默默地什么也没干。

       而真实图里不会出现这个问题 —— 因为消息进 state 时会经过
       add_messages reducer,reducer 会给没有 id 的消息自动补一个 uuid。
       所以:脚本里手工造数据时,记得先过一遍 add_messages 把 id 补上。
""")

    # ── 正式演示(带 id)────────────────────────────────
    history = build_long_history()  # assign_ids=True

    # 手动模拟一次 state(还没有 checkpointer,T4 才上)
    state: SummaryState = {"messages": history, "summary": ""}  # type: ignore[typeddict-item]

    print(f"\n  压缩前: messages = {len(state['messages'])} 条, summary = {state['summary']!r}")
    print(f"          第 1 条 id = {history[0].id}  ← 这次有了")

    try:
        update = summarize_node(state)
    except RuntimeError as e:
        print(f"\n  ❌ {e}")
        print("  (跳过本部分)")
        return

    print("\n  节点返回值(注意这只是 partial update,不是完整 state!):")
    print(f"    summary  = {update['summary'][:60]}…")
    print(f"    messages = {len(update['messages'])} 条 RemoveMessage"
          f"   ← 不是真的消息,是「删除指令」")
    for rm in update["messages"][:3]:
        print(f"      · RemoveMessage(id={str(rm.id)[:8]}…)")
    if len(update["messages"]) > 3:
        print(f"      · … 还有 {len(update['messages']) - 3} 条")

    # 手动把 partial update 合进 state,模拟框架做的事
    merged: SummaryState = dict(state)  # type: ignore[assignment]
    merged["summary"] = update["summary"]
    merged["messages"] = add_messages(merged["messages"], update["messages"])  # type: ignore[arg-type]

    print(f"\n  【姿势 A · 精确删除】合并后:")
    print(f"    messages = {len(merged['messages'])} 条(原 {len(history)} 条,"
          f"删掉 {len(history) - len(merged['messages'])} 条)")
    print(f"    summary  = {len(merged['summary'])} 字符")
    print("    留下的:")
    _brief(merged["messages"])

    # ── 姿势 B:一键清空 ────────────────────────────────
    print("\n  【姿势 B · 一键清空】REMOVE_ALL_MESSAGES:")
    cleared = add_messages(merged["messages"], [RemoveMessage(id=REMOVE_ALL_MESSAGES)])
    print(f"    messages = {len(cleared)} 条")

    print("""
  👉 这里暴露了四件事:

     ① 节点返回值是 **partial update**(只有被改的字段),
        不是完整 state —— 这跟 T2 学的一致。框架会拿它去逐个字段过 reducer。

     ② 删消息**不是** `state["messages"] = []` 或 `del state["messages"][0]`。
        因为节点的返回值**不能直接覆盖** state,
        只能"作为一次更新"交给该字段的 reducer 处理。
        所以要用 RemoveMessage 这种"删除指令"来表达意图:
          · 姿势 A:点名 id 删指定的几条
          · 姿势 B:REMOVE_ALL_MESSAGES 全清

        ⚠️ 顺便把这件事说死:**`return {"messages": []}` 也清不掉。**
           它的意思是"往 messages 里加 0 条",reducer 执行
           `add_messages(旧, [])` -> 旧列表原样返回,**不报错也不生效**。
           连"在节点里改完再 return dict(state)"都不行 ——
           返回值永远被当"更新指令",不是"最终状态"。
           完整实测见 `01_state_schema.py` 第 10 部分。

     ③ **没有 id 就删不掉,而且不报错**(上面那个坑)。

     ④ 姿势 A 保留的那 2 条,是下一次调模型的"上下文引子"。
        如果姿势 B 全清,下一轮模型手里就只剩一段摘要了 ——
        **它会不知道用户最后一句说了什么**,容易答非所问。
        所以实战里推荐 A,而不是图省事用 B。
""")


# ══════════════════════════════════════════════════════════════════
# 第 4 部分:组装成图 —— 让"该摘要了"自动触发
# ══════════════════════════════════════════════════════════════════
def build_graph():
    """
    形状:

        START --> [chat] ─┬─(消息数 <= 阈值)→ END
                          └─(消息数 >  阈值)→ [summarize] --> END

    阈值判断放在**条件边**里,而不是写在节点内部 ——
    这就是 T2 学的"条件边是图里唯一的分支来源"的实战用法。
    """

    def chat(state: SummaryState) -> dict:
        """普通对话节点:带上摘要 + 最近消息一起发给模型。"""
        msgs = []
        if state.get("summary"):
            msgs.append(SystemMessage(content=f"【前面对话的摘要】{state['summary']}"))
        msgs.extend(state["messages"][-4:])  # 只带最近几条,模拟"裁剪+摘要"组合拳

        if OFFLINE:
            return {"messages": [AIMessage(content="(offline 模式:这里本应调模型回答)")]}

        resp = _invoke_with_retry(msgs)
        return {"messages": [resp]}

    def should_summarize(state: SummaryState) -> str:
        """条件边:决定去 END 还是去摘要节点。"""
        if len(state["messages"]) > 6:
            return "summarize"
        return END

    b = StateGraph(SummaryState)  # type: ignore[arg-type]
    b.add_node("chat", chat)
    b.add_node("summarize", summarize_node)
    b.add_edge(START, "chat")
    b.add_conditional_edges("chat", should_summarize, {"summarize": "summarize", END: END})
    b.add_edge("summarize", END)
    return b.compile()


def part_4_graph() -> None:
    print("\n" + LINE)
    print("第 4 部分:组装成图 —— 条件边自动判断'该不该摘要了'")
    print(LINE)

    graph = build_graph()
    print("\n  图结构:")
    print(graph.get_graph().draw_mermaid())

    history = build_long_history()
    print(f"\n  塞进去 {len(history)} 条历史(> 6,应该触发摘要)…")

    try:
        result = graph.invoke(
            {"messages": history, "summary": ""},
            {"recursion_limit": 25},
        )
    except RuntimeError as e:
        print(f"\n  ❌ {e}")
        print("  (跳过本部分)")
        return

    print(f"\n  ▶ 跑完之后:")
    print(f"    messages = {len(result['messages'])} 条")
    _brief(result["messages"])
    print(f"\n    summary  = {result['summary'][:80]}…")

    print(f"""
  👉 看结果:messages 从 {len(history)} 条变成了 {len(result['messages'])} 条
     —— 留下的只有「原来的最后一条」和「chat 刚回的那条」。
     summary 里则留下了关键约束(会员号 M88231、只买蓝色、不要预售、只发短信)。
     —— 这就是"摘要 + 裁剪"的组合效果:体积下来了,约束还在。

  ⚠️ 但这里有个**当前还不够**的地方(留给 T4):
     这次 invoke 结束,state 就没了。下一轮再 invoke,又是从空白开始 ——
     因为这个图**没有 checkpointer**。
     摘要要真正发挥作用,必须让它"跨轮次存在",那正是 T4 的 checkpointer 要解决的。
""")


# ══════════════════════════════════════════════════════════════════
# 第 5 部分:三种记忆策略的取舍
# ══════════════════════════════════════════════════════════════════
def part_5_comparison() -> None:
    print("\n" + LINE)
    print("第 5 部分:裁剪 vs 摘要 vs 外部存储 —— 什么时候用哪个")
    print(LINE)
    print("""
  ┌────────────┬──────────┬──────────┬──────────┬─────────────────────────┐
  │ 手段        │ 信息损失 │ 成本      │ 可逆     │ 什么时候用               │
  ├────────────┼──────────┼──────────┼──────────┼─────────────────────────┤
  │ 不处理      │ 无       │ 0         │ —        │ 短对话、demo。长了必爆    │
  │ 裁剪(trim) │ 高(永久)│ 0(估算)  │ 不可逆   │ 上下文刚满、只是聊天记录  │
  │ 摘要(sum)  │ 中(有损) │ 要调模型  │ 不可逆   │ 有必须记住的偏好/约束     │
  │ 外部存储    │ 低       │ 读写一次  │ 可逆     │ **已发生的动作**必须用它  │
  └────────────┴──────────┴──────────┴──────────┴─────────────────────────┘

  实战里不是三选一,而是**叠着用**:

      每次调模型前:  trim(摘要 + 最近 N 条)     ← 控住体积
      消息数超阈值:  生成/更新摘要               ← 保语义
      写类动作发生时:写进外部存储(DB / checkpointer)→ 保事实

  ── 🛡️ 安全落点(这条对你最关键)─────────────────────
    **摘要绝对不能被当成"事实记录"。**

    模型摘要会出错 —— 会把数字记错、把"只是讨论过"写成"已经执行了"。
    所以凡是"钱、订单、是否已执行"这类,必须有**独立于模型的记录**:
      · 订单 / 资金状态:以订单系统、支付系统的实际状态为准,不能靠模型记着
      · 写操作类:执行前先查一次外部状态,而不是信任摘要里的说法
    摘要只用来**理解上下文**,不能用来**做事实判断**。

  ── 🎯 范式落点 ─────────────────────────────────────
    这套"裁剪 + 摘要"就是 **Reflexion / 长期记忆** 的地基:
      · Reflection 的核心是"把失败经验写下来,下一轮带上"——
        而"写下来带上"这个机制,就是本文件的 summary 字段;
      · 区别只在于 Reflexion 写的不是"对话摘要",而是**结构化的教训**
        (哪一步错了、为什么错、下次怎么改)。
    所以 T3 做完,你已经有了 Reflexion 的**存储机制**,
    缺的只是"写什么进去"这一层 —— 那是后面阶段的事。
""")


# ══════════════════════════════════════════════════════════════════
# 6. 对照官方范例 + 官方给的"该留什么、该丢什么"
# ══════════════════════════════════════════════════════════════════
def part_6_official_pattern() -> None:
    print("\n" + LINE)
    print("第 6 部分:对照官方范例 —— 我们写的和官方给的是一回事吗?")
    print(LINE)

    print("""
  `【官方】` 官方 add-memory 文档给的摘要节点原文(一字不改):

      def summarize_conversation(state: State):
          # First, we get any existing summary
          summary = state.get("summary", "")

          # Create our summarization prompt
          if summary:
              # A summary already exists
              summary_message = (
                  f"This is a summary of the conversation to date: {summary}\\n\\n"
                  "Extend the summary by taking into account the new messages above:"
              )
          else:
              summary_message = "Create a summary of the conversation above:"

          # Add prompt to our history
          messages = state["messages"] + [HumanMessage(content=summary_message)]
          response = model.invoke(messages)

          # Delete all but the 2 most recent messages
          delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
          return {"summary": response.content, "messages": delete_messages}

      —— https://docs.langchain.com/oss/python/langgraph/add-memory

  `【实测】` 逐条对照本文件的实现:

    ┌──────────────────────────┬────────────────────────────┬──────────┐
    │ 官方范例                  │ 本文件                     │ 一致?    │
    ├──────────────────────────┼────────────────────────────┼──────────┤
    │ state.get("summary","")  │ state.get("summary","")    │ ✅       │
    │ 有旧摘要就"Extend"        │ previous_summary 分支      │ ✅       │
    │ model.invoke(带提示)      │ _invoke_with_retry(...)    │ ✅       │
    │ 先 invoke 再构造删除       │ 先 summarize 再 deletes    │ ✅ 顺序同 │
    │ state["messages"][:-2]   │ state["messages"][:-2]     │ ✅       │
    │ 返回 summary + messages  │ 返回 summary + messages    │ ✅       │
    └──────────────────────────┴────────────────────────────┴──────────┘

  👉 **本文件没有自创写法** —— 就是官方范例的结构。
     唯一的差别是**工程增强**(都不改变语义):
       · 加了 429 重试(官方范例默认模型可用)
       · 加了 --offline 模式(为了在限流时也能验证管道)
       · 加了"盘点被删掉的都是哪些 id"的打印(为了看清机制)

  ── 官方还给了两条必须注意的 ⚠️ ────────────────────────

  `【官方】` 删消息时的**有效性警告**(原文):
      "When deleting messages, **make sure** that the resulting message history is valid.
       Check the limitations of the LLM provider you're using. For example:
         · Some providers expect message history to start with a `user` message
         · Most providers require `assistant` messages with tool calls to be followed by
           corresponding `tool` result messages."
      —— 同上

    💡 这就是第 2 步(裁剪)里那个"孤儿 ToolMessage"问题的**官方依据** ——
       官方在"删除"这一节也专门警告了一遍。**删除和裁剪是同一个坑。**

  ── Anthropic 给的"该留什么、该丢什么" ──────────────────

  `【官方】` Anthropic 讲了 Claude Code 的 compaction 具体保留/丢弃什么:
      "The model **preserves architectural decisions, unresolved bugs, and implementation
       details** while **discarding redundant tool outputs or messages**. The agent can then
       continue with this compressed context plus the **five most recently accessed files**."
      —— https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents

    翻成通用的说法(本项目场景):

    ┌─────────────┬──────────────────────────────┬──────────────────────┐
    │             │ 保 留                        │ 丢 弃                │
    ├─────────────┼──────────────────────────────┼──────────────────────┤
    │ 决策类       │ 定过的方案、做过的选择        │ 讨论过程中的反复      │
    │ 未完成的事   │ 还没解决的问题、待办          │ 已完成且无后续的      │
    │ 具体细节     │ 参数、路径、订单号等标识      │ 冗余的工具原始输出    │
    │ 格式类       │ 用户明确要求的格式偏好        │ 寒暄、重复确认        │
    └─────────────┴──────────────────────────────┴──────────────────────┘

  `【官方】` 而且官方明确警告了"压太狠"的后果:
      "The art of compaction lies in the selection of what to keep versus what to discard,
       as **overly aggressive compaction can result in the loss of subtle but critical context
       whose importance only becomes apparent later**."
      —— 同上

    💡 这句是"摘要不能当事实记录"的**权威依据** ——
       它丢的可能正是"当时看不出重要、后来才发现重要"的东西。
       所以钱、订单、是否已执行这类必须**独立于模型**记录。

  ── 一个很实用的技巧:tool result clearing ──────────────

  `【官方】` Anthropic 点名了**最安全的轻量压缩方式**:
      "An example of low-hanging superfluous content is **clearing tool calls and results** –
       once a tool has been called deep in the message history, why would the agent need to
       see the raw result again? One of the **safest lightest touch forms of compaction** is
       **tool result clearing**."
      —— 同上

    💡 这跟第 2 步讲的"别裁出孤儿 ToolMessage"是**互补的两面**:
       · 裁剪**单条** ToolMessage → 危险(配对断了)
       · **成对清除**老的 tool_calls + 它的 tool results → 安全且省最多 token
       因为工具返回往往是整个对话里最占地方的部分。

  ── 官方推荐的 prompt 调优顺序 ──────────────────────────

  `【官方】` 怎么把摘要提示词调好,官方给了方法:
      "For engineers implementing compaction systems, we recommend **carefully tuning your
       prompt on complex agent traces**. **Start by maximizing recall** to ensure your compaction
       prompt captures every relevant piece of information from the trace, **then iterate to
       improve precision** by eliminating superfluous content."
      —— 同上

    👉 **先保召回,再提精度**。也就是:
       第一版提示词应该"宁可多留",确认关键信息都抓到了,
       然后再逐步收紧、删废话。**别一上来就追求"压得短"。**

  ── 那有没有现成的轮子?有,但本文件不用 ─────────────────

  `【官方】` 官方范例里其实用了 `langmem` 包里的 `SummarizationNode`:
      from langmem.short_term import SummarizationNode, RunningSummary
      summarization_node = SummarizationNode(
          token_counter=count_tokens_approximately,
          model=summarization_model,
          max_tokens=256,
          max_tokens_before_summary=256,   # ← 注意:阈值按 **token** 算,不是按消息条数
          max_summary_tokens=128,
      )
      —— 同上

  `【实测】` 本项目**没有装它**,理由(2026-10-15 查 PyPI 元数据):

      · 它还是 **0.0.30** 版本 —— 早期版本,API 可能变
      · 依赖很重:langchain / langchain-anthropic / langchain-openai /
        langgraph / langsmith / trustcall 一整套
      · **本文件的目标是"看懂摘要机制"** —— 手搓版已经和官方 `summarize_conversation`
        范例逐条一致(见上面那张对照表)。
        套一层封装反而会让人分不清"哪些是框架给的、哪些是要自己写的"

  `【建议】` 真到生产里、又不想自己维护摘要逻辑时,再考虑 `pip install langmem` 用它。

    ⚠️ 但注意上面那个参数名 —— `max_tokens_before_summary` 是**按 token 设阈值**,
       这跟前面第 2 步说的"按比例而不是按条数"是同一个思路。
""")


# ══════════════════════════════════════════════════════════════════
def main() -> None:
    print(LINE)
    if OFFLINE:
        print("模式:offline —— 不联网。摘要那一格用「本地假摘要」顶替模型。")
        print("      跑这个模式是为了验证**管道**:RemoveMessage 清空 / state 合并 / 条件边。")
        print("      要看到真正的摘要,去掉 --offline 重跑(前提是模型没被限流)。")
    else:
        print("模式:online —— 会真的调模型做摘要。")
        print("      如果一直 429(智谱免费档很常见),加 --offline 先验证管道。")
    print(LINE)

    part_1_why_trim_is_not_enough()
    part_2_first_summary()
    part_3_summary_into_state()
    part_4_graph()
    part_5_comparison()
    part_6_official_pattern()

    print("\n" + LINE)
    print("本文件结论(背这 7 条就够)")
    print(LINE)
    print(f"""
  1. `【官方】` 摘要解决的是官方亲口说的那个问题:"trimming or removing messages …
     you may lose information from culling of the message queue"。
  2. `【官方】` 增量摘要(把旧摘要带上做 "Extend")是官方范例的做法,不是本项目自创。
  3. `【官方】` 官方范例就是"保留最近 2 条"(`state["messages"][:-2]`),
     并且**先算摘要、再构造删除** —— 顺序不能反。
  4. `【官方】` 删消息必须保证序列有效(官方 Warning):历史要以 user 开头、
     带 tool_calls 的 assistant 后面必须跟对应的 tool 结果。
  5. `【官方】` 摘要该保留"决策 / 未解决的问题 / 具体细节",丢弃"冗余的工具输出与消息";
     Anthropic 明确警告**压太狠会丢"当时看不出重要、后来才发现重要"的信息**。
  6. `【官方】` 最安全的轻量压缩是 **tool result clearing**(成对清掉老的工具调用与结果);
     `【官方】` 调提示词的顺序是**先最大化召回,再迭代提精度**。
  7. `【建议】` 摘要**不能当事实记录** —— 钱/订单/是否已执行必须独立于模型存。

  本次实际调用模型:{CALLS["n"]} 次。
""")
    print("  下一步(T4):上 checkpointer,让这份 state(含摘要)能跨轮次存活、能断点续跑。")


if __name__ == "__main__":
    main()
