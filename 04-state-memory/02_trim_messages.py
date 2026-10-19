"""
T3 · 第 2 步:消息裁剪(trim_messages)
=====================================

【这个文件要回答的问题】
对话越滚越长,迟早爆上下文。业界第一步的应对不是"上摘要",而是最朴素的一招:
—— 把老消息**直接裁掉**,只留最近的一段。

LangChain 提供的工具是 `trim_messages`。

【最重要的认知:裁剪不需要调模型】
`trim_messages` 需要一个"算 token"的东西(参数名 token_counter)。
你可以传 tiktoken、也可以传真实的 model,但**最常用的是 `count_tokens_approximately`**
—— 它是**纯字符估算**:不看内容语义,就是 `字符数 / 4 + 每条消息固定开销`。

    ⚠️ 这是估算,不是精确计数。但裁剪本来就要"多裁一点留余量",
       所以估算够用 —— 而且它**零成本、毫秒级、不发网络请求**。

本文件因此**不调任何模型,秒出**。

【官方文档】
  Add memory(含 Trim messages 一节)
    https://docs.langchain.com/oss/python/langgraph/add-memory
  trim_messages API 参考
    https://reference.langchain.com/python/langchain-core/messages/utils/trim_messages

运行:
    cd D:/VibeCoding/langgraph-lab
    .venv/Scripts/python.exe 04-state-memory/02_trim_messages.py
"""

import os
import time

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    trim_messages,
)
from langchain_core.messages.utils import count_tokens_approximately


# ══════════════════════════════════════════════════════════════════
# 0. 先看看 token_counter 到底是怎么算的
# ══════════════════════════════════════════════════════════════════
def part_0_how_counting_works() -> None:
    print("\n" + "=" * 70)
    print("第 0 部分:count_tokens_approximately 到底怎么算的")
    print("=" * 70)

    # 签名(打印出来给自己看,别背)
    # (messages, *, chars_per_token=4.0, extra_tokens_per_message=3.0,
    #  count_name=True, tokens_per_image=85,
    #  use_usage_metadata_scaling=False, tools=None)
    #
    # 参数含义:
    #   chars_per_token = 4.0          -> 大约 4 个字符算 1 个 token
    #   extra_tokens_per_message = 3.0 -> 每条消息额外算 3 个 token 的"信封开销"
    #   count_name = True              -> 把消息的 name 字段也算进去
    #   tokens_per_image = 85          -> 一张图固定算 85(对齐 OpenAI 低清图)
    #
    # ⚠️ 这是我一开始想当然写错的版本:
    #        [每条消息] = len(content) / 4 + 3
    #    实测三条内容长度分别是 10 / 11 / 5 个字符,结果却**都是 7**。
    #    内容和长度差一倍,估算值却一模一样 —— 说明这个公式漏了东西。
    #
    # ✅ 去读源码(langchain_core.messages.utils.count_tokens_approximately)后
    #    才看清真实算法,**content 只是其中一部分**:

    role_len = {
        "SystemMessage": 6,   # role = "system"
        "HumanMessage": 4,    # role = "user"
        "AIMessage": 9,       # role = "assistant"
        "ToolMessage": 4,     # role = "tool"
    }

    msgs = [
        SystemMessage(content="你是一个严谨的助手。"),   # content 10 字符
        HumanMessage(content="你好,请介绍一下自己。"),   # content 11 字符
        AIMessage(content="我是助手。"),                   # content 5 字符
    ]

    print("""
  【真实公式 —— 逐条】
      step1  chars  = len(content)
      step2  chars += len(AI 的 tool_calls 的 repr)      ← 仅 AI 且带工具调用时
      step3  chars += len(tool_call_id)                 ← 仅 ToolMessage
      step4  chars += len(role)                          ← 关键!我一开始就漏在这
      step5  chars += len(name)                          ← 仅当有 name 且 count_name=True
      step6  token  = ceil(chars / chars_per_token)
      step7  token += extra_tokens_per_message           ← 每条固定 +3

      role 的长度是常量,四种角色各不相同:
        system=6 / user=4 / assistant=9 / tool=4
      —— 这就是"内容 10 字符"和"内容 5 字符"最后都算出 7 的原因:
         加完 role 之后都是十几,ceil(十几/4) 都是 4,再 +3 还是 7。
""")

    print("  [手算 vs 实测,逐条对照]")
    print(f"  {'类型':<14} {'content':>8} {'role':>5} {'chars':>6} {'手算':>5} {'实测':>5}  对得上?")
    print("  " + "-" * 62)

    total_manual = 0
    for m in msgs:
        c = len(str(m.content))
        rl = role_len[type(m).__name__]
        chars = c + rl
        manual = -(-chars // 4) + 3          # ceil 除 + 3
        actual = count_tokens_approximately([m])
        total_manual += manual
        ok = "✅" if manual == actual else "❌"
        print(f"  {type(m).__name__:<14} {c:>8} {rl:>5} {chars:>6} "
              f"{manual:>5} {actual:>5}  {ok}")

    total_actual = count_tokens_approximately(msgs)
    print(f"\n  三条合计:手算 {total_manual} / 实测 {total_actual}  "
          f"{'✅ 一致' if total_manual == total_actual else '❌ 不一致'}")

    # 再演示两处"内容之外"的开销,这两处最容易估错
    print("\n  [再验两处额外开销]")

    plain = AIMessage(content="")
    with_tc = AIMessage(
        content="",
        tool_calls=[{"name": "get_weather", "args": {"city": "北京"},
                     "id": "call_001", "type": "tool_call"}],
    )
    print(f"    AIMessage 内容都是空串:")
    print(f"      不带 tool_calls -> {count_tokens_approximately([plain])} token")
    print(f"      带   tool_calls -> {count_tokens_approximately([with_tc])} token"
          f"   ← tool_calls 的 repr 也计费")

    t1 = ToolMessage(content="北京:晴", tool_call_id="c1")
    t2_id = "call_a1b2c3d4e5f6"
    t2 = ToolMessage(content="北京:晴", tool_call_id=t2_id)
    print(f"    ToolMessage 内容都是 5 字符:")
    print(f"      tool_call_id='c1'                -> {count_tokens_approximately([t1])} token")
    print(f"      tool_call_id='{t2_id}'({len(t2_id)}) -> {count_tokens_approximately([t2])} token"
          f"   ← id 越长越贵")

    print("""
  👉 结论:它是「content + role + tool 相关字段,再 ceil÷4,每条 +3」的粗估。
     零成本、不联网 —— 但它是**估算**,不要指望和某个模型的真账一分不差。
     裁剪场景本来就该留余量,所以够用。
""")


# ══════════════════════════════════════════════════════════════════
# 1. 造一段"带工具调用"的真实历史
# ══════════════════════════════════════════════════════════════════
def build_history() -> list:
    """
    造一段 9 条消息的历史,结构模仿真实的 agent 对话:
      系统提示 -> 人问 -> AI 决定调工具 -> 工具返回 -> AI 总结 -> 人追问 -> ...

    ⚠️ 为什么特意强调"带工具调用"?
       因为 ToolMessage 有一个硬约束:它必须**紧跟**在发起它的 AIMessage 后面,
       靠 `tool_call_id` 配对。裁掉 AI 那一半、留下 Tool 这一半,
       序列就非法了 —— 发给模型直接 400。
       这正是下面「坑 1」的成因。
    """
    return [
        # ① 系统提示
        SystemMessage(content="你是一个严谨的助手,回答要简洁。"),

        # ② 第一轮:人问
        HumanMessage(content="北京现在的天气怎么样?"),

        # ③ AI 决定调工具(AIMessage 带 tool_calls)
        AIMessage(
            content="",
            tool_calls=[{
                "name": "get_weather",
                "args": {"city": "北京"},
                "id": "call_001",
                "type": "tool_call",
            }],
        ),

        # ④ 工具返回(必须用同一个 tool_call_id 配对)
        ToolMessage(content="北京:晴,18℃,微风", tool_call_id="call_001"),

        # ⑤ AI 总结
        AIMessage(content="北京现在晴,18℃,微风,挺舒服的。"),

        # ⑥ 第二轮:人追问
        HumanMessage(content="那上海呢?"),

        # ⑦ AI 又调工具
        AIMessage(
            content="",
            tool_calls=[{
                "name": "get_weather",
                "args": {"city": "上海"},
                "id": "call_002",
                "type": "tool_call",
            }],
        ),

        # ⑧ 工具返回
        ToolMessage(content="上海:多云,22℃,东南风 3 级", tool_call_id="call_002"),

        # ⑨ AI 总结
        AIMessage(content="上海多云,22℃,东南风 3 级。比北京暖一点。"),
    ]


def _brief(msgs: list) -> str:
    """把消息列表压缩成一行,方便并排对比。"""
    if not msgs:
        return "(空列表)"
    out = []
    for m in msgs:
        tag = {
            "SystemMessage": "SYS",
            "HumanMessage": "HUM",
            "AIMessage": "AI",
            "ToolMessage": "TOOL",
        }.get(type(m).__name__, type(m).__name__[:4])
        if getattr(m, "tool_calls", None):
            tag += "+任"
        out.append(tag)
    return " | ".join(out)


def _first_type(msgs: list) -> str:
    return type(msgs[0]).__name__ if msgs else "(无)"


# ══════════════════════════════════════════════════════════════════
# 2. 主实验:四种参数搭配
# ══════════════════════════════════════════════════════════════════
def part_1_four_configs() -> None:
    print("\n" + "=" * 70)
    print("第 1 部分:四种参数搭配的实测结果")
    print("=" * 70)

    history = build_history()
    full_tokens = count_tokens_approximately(history)

    print(f"\n原始历史:{len(history)} 条,约 {full_tokens} token")
    print(f"  结构: {_brief(history)}")

    cases = [
        (dict(strategy="last", token_counter=count_tokens_approximately,
              max_tokens=40),
         "A. 不设 start_on, max_tokens=40"),

        (dict(strategy="last", token_counter=count_tokens_approximately,
              max_tokens=40, start_on="human"),
         "B. start_on='human', max_tokens=40"),

        (dict(strategy="last", token_counter=count_tokens_approximately,
              max_tokens=80, start_on="human"),
         "C. start_on='human', max_tokens=80"),

        (dict(strategy="last", token_counter=count_tokens_approximately,
              max_tokens=80, start_on="human", include_system=True),
         "D. C + include_system=True"),
    ]

    for kw, label in cases:
        result = trim_messages(history, **kw)
        tokens = count_tokens_approximately(result) if result else 0
        print(f"\n[{label}]")
        print(f"  保留 {len(result)} 条 / 约 {tokens} token")
        print(f"  结构: {_brief(result)}")
        print(f"  首条: {_first_type(result)}")


# ══════════════════════════════════════════════════════════════════
# 3. 坑 1:不设 start_on -> 裁出以 ToolMessage 打头的非法序列
# ══════════════════════════════════════════════════════════════════
def part_2_pitfall_orphan_tool() -> None:
    print("\n" + "=" * 70)
    print("第 2 部分:坑 1 —— 不设 start_on 会裁出「孤儿 ToolMessage」")
    print("=" * 70)

    history = build_history()

    result = trim_messages(
        history,
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=40,
        # ← 故意不写 start_on
    )

    print(f"\n原始: {_brief(history)}")
    print(f"裁剪后: {_brief(result)}")
    print(f"首条类型: {_first_type(result)}")

    print("""
  💥 问题在哪:
     裁剪结果的第一条是 ToolMessage,但它对应的 AIMessage(带 tool_calls 的那条)
     已经被裁掉了。ToolMessage 靠 tool_call_id 跟 AIMessage 配对,
     前面没有 AIMessage -> 这个 tool_call_id 就是"孤儿"。
     发给模型会直接报错,大意是:
        "An assistant message with 'tool_calls' must be followed by tool messages
         responding to each 'tool_call_id'"

  ✅ 修法:加 start_on="human"
     —— 告诉它"裁剪后的开头必须是 HumanMessage 或 SystemMessage",
        它就绝不会在工具调用链的中间下刀。
""")

    # 再验证一次:加了 start_on 就干净了
    fixed = trim_messages(
        history,
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=80,
        start_on="human",
    )
    print(f"  [加 start_on='human' 后] {_brief(fixed)}")
    print(f"  首条 = {_first_type(fixed)}  ✅ 干净")


# ══════════════════════════════════════════════════════════════════
# 4. 坑 2:start_on="human" + max_tokens 太小 -> 返回空列表
# ══════════════════════════════════════════════════════════════════
def part_3_pitfall_empty() -> None:
    print("\n" + "=" * 70)
    print("第 3 部分:坑 2 —— max_tokens 太小会返回「空列表」")
    print("=" * 70)

    history = build_history()

    print("\n  固定 start_on='human',只调 max_tokens,看返回条数:")
    print(f"  {'max_tokens':>12} | {'返回条数':>8} | 结果")
    print("  " + "-" * 48)

    for mt in [20, 30, 40, 50, 60, 80, 100, 200]:
        r = trim_messages(
            history,
            strategy="last",
            token_counter=count_tokens_approximately,
            max_tokens=mt,
            start_on="human",
        )
        flag = "  ⚠️ 空!" if not r else ""
        print(f"  {mt:>12} | {len(r):>8} | {_brief(r)}{flag}")

    print("""
  💥 为什么会空:
     start_on="human" 的意思是"从后往前收,一直收到某个 HumanMessage 才开始算起点"。
     如果 max_tokens 的额度**连最后一条 HumanMessage 后面的内容都装不下**,
     它就一条都留不了 —— 直接返回 []。

  💥 空列表的直接后果:
     如果你后续写 model.invoke(result),会变成 model.invoke([])。
     有些模型直接 400,有些则返回一句莫名其妙的空回复。
     更糟的是 state["messages"][0] 这类取值会抛 IndexError。
     ——**错误信息指向的地方,跟你真正的病因(额度太小)完全不搭边。**

  ✅ 修法(三道保险,建议都上):
     ① max_tokens 给足 —— 至少是你单条最长消息的好几倍,别抠;
     ② 裁剪后判空,空了就退回原始消息(或退回最后 N 条);
     ③ 别用 result[0] 这种硬索引,先判 len(result) > 0。

  ⚠️ 别抄具体数字:悬崖在 50/60 之间,换个历史就变了。
     至于**为什么额度一紧就"一条都不留"、而不是"少留几条"**,
     这跟 start_on 的真实语义有关 —— 见下面第 7 部分(回溯砍)。
""")


# ══════════════════════════════════════════════════════════════════
# 5. 安全封装:裁剪 + 判空兜底
# ══════════════════════════════════════════════════════════════════
def safe_trim(messages: list, max_tokens: int = 2000) -> list:
    """
    裁剪消息的"生产级"写法。五个动作:

      1. 传 start_on="human"      -> 不在工具调用链中间下刀(避开坑 1)
      2. 传 include_system=True   -> 系统提示单独摘出来保留(见第 6 部分)
      3. 判空                     -> 空了就退回原始消息(避开坑 2)
      4. 判"只剩系统提示"          -> 也算裁失败,同样退回原样(第 6 部分那个坑)
      5. 留足 max_tokens          -> 别抠到临界值
    """
    if not messages:
        return messages

    trimmed = trim_messages(
        messages,
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=max_tokens,
        start_on="human",
        include_system=True,
    )

    # 兜底 1:裁到一条不剩 —— 额度太小
    if not trimmed:
        return messages

    # 兜底 2:只活下来一条 SystemMessage,对话全被挤没了 —— 同样不能用。
    #         ⚠️ 这种情况**不会被"判空"拦住**,因为结果不是空的,
    #            是"看起来有一条消息、其实什么都不能回答"。
    if all(isinstance(m, SystemMessage) for m in trimmed):
        return messages

    return trimmed


def part_4_safe_wrapper() -> None:
    print("\n" + "=" * 70)
    print("第 4 部分:把它包成一个能上生产的 safe_trim()")
    print("=" * 70)

    history = build_history()

    print(f"\n原始: {len(history)} 条,约 {count_tokens_approximately(history)} token")

    for mt in [40, 55, 80, 2000]:
        raw = trim_messages(history, strategy="last", token_counter=count_tokens_approximately,
                            max_tokens=mt, start_on="human", include_system=True)
        r = safe_trim(history, max_tokens=mt)
        fb = "  ← 触发兜底,退回原始消息" if len(r) == len(history) else ""
        print(f"\n  max_tokens={mt}")
        print(f"    裸调 trim_messages -> {len(raw)} 条: {_brief(raw)}")
        print(f"    safe_trim()        -> {len(r)} 条: {_brief(r)}{fb}")

    print("""
  👉 看 max_tokens=55 那一组:裸调的结果是只剩一条 SYS,
     看起来"有结果",但它**完全没法回答用户任何问题**。
     这就是第 6 部分说的那个隐形坑 —— 它不算空,但一样是废的。
     safe_trim 的第二道兜底就是专门拦这个。

     至于 max_tokens=40 那一组:裸调返回**空列表**,
     拿空列表去 model.invoke([]) 是"未定义行为",比明确报错更难排查。
""")


# ══════════════════════════════════════════════════════════════════
# 6. 挂进图上长什么样(伪代码,不实际调模型)
# ══════════════════════════════════════════════════════════════════
def part_5_in_graph() -> None:
    print("\n" + "=" * 70)
    print("第 5 部分:这东西在图里怎么用(官方推荐写法)")
    print("=" * 70)
    print("""
  官方在 add-memory 文档里给的范式,就是"调模型之前先裁":

    from langchain_core.messages.utils import (
        trim_messages, count_tokens_approximately,
    )

    def call_model(state: MessagesState):
        messages = trim_messages(
            state["messages"],
            strategy="last",
            token_counter=count_tokens_approximately,
            max_tokens=128,
            start_on="human",
            end_on=("human", "tool"),
        )
        response = model.invoke(messages)
        return {"messages": [response]}

  ⚠️ 注意:官方这里的 max_tokens=128 是**教学用的极小值**(为了演示效果明显),
     真实项目里至少要几千。别照抄 128。

  ── end_on 是干嘛的 ────────────────────────────────
    start_on 管"从哪开始"(前面那个头),
    end_on  管"到哪结束"(后面那个尾)。
    end_on=("human", "tool") 的意思是:
      "裁剪后的最后一条,只能是 Human 或 Tool"
      -> 也就是**不能以一条没干的 AIMessage(只有话、没有动作)收尾**,
         否则模型会以为自己上一句还没说完。

  ── strategy="last" vs "first" ─────────────────────
    last  = 保留最新的(对话场景几乎都用这个)。
    first = 保留最早的(少用;一般用于你想保住原始指令的场景)。

  ── 🎯 范式落点 ────────────────────────────────────
    裁剪是"记忆系统"里最便宜的降级手段,它和 Reflexion / 摘要压缩是**互补**的:
      · 裁剪(本文件)= 丢掉老的,信息**永久损失**,但零成本零延迟;
      · 摘要(下一步)  = 把老的压成一段话,信息**有损保留**,但要调模型;
      · Reflexion    = 不只压缩,还要**把失败经验写成教训**参与下一轮。
    所以顺序上,先把裁剪做扎实,再上摘要 —— 否则你连"该留多少余量"都没数。

  ── 🛡️ 安全落点 ────────────────────────────────────
    裁剪 = **有损操作**,而且是不可逆的。
    两个安全习惯:
      ① 裁剪掉的消息如果涉及"已经发生的动作"(下过单、发过邮件、改过数据),
         裁掉后模型会忘记自己做过什么 -> 可能**重复执行**。
         资金/写入类场景,这类动作必须有**外部存储**(比如 checkpointer 或库表)兜底,
         不能只靠 message 里记着。
      ② max_tokens 别抠到临界。抠到临界的结果就是坑 2 那种空列表,
         而空列表喂给模型的行为是"未定义"的,比明确报错更危险。
""")


# ══════════════════════════════════════════════════════════════════
# 6. SystemMessage 会不会被裁掉?—— include_system 的真相
# ══════════════════════════════════════════════════════════════════
def part_6_system_message() -> None:
    print("\n" + "=" * 70)
    print("第 6 部分:SystemMessage 会不会被裁掉?")
    print("=" * 70)

    print("""
  【先问一句:为什么会担心这个】
    裁剪的套路是"从后往前尽量收,再从收进来的那批头上往回砍"。
    而排在**列表最开头**的那条 SystemMessage(通常写着"你是谁、要守什么规矩"),
    天然就是最容易被砍掉的那个。

    实测一下就知道担心是不是多余的 ——
""")

    history = build_history()
    print(f"  原始 {len(history)} 条 / 约 {count_tokens_approximately(history)} token")
    print(f"  结构: {_brief(history)}")
    print(f"  首条: {type(history[0]).__name__}({count_tokens_approximately([history[0]])} token)")

    print("\n  [场景 1] 默认(include_system=False)")
    r1 = trim_messages(history, strategy="last", token_counter=count_tokens_approximately,
                       max_tokens=80, start_on="human")
    print(f"    结果 {len(r1)} 条: {_brief(r1)}")
    print(f"    首条 = {_first_type(r1)}   "
          f"→ 系统提示{'还在' if r1 and isinstance(r1[0], SystemMessage) else '❌ 没了'}")

    print("\n  [场景 2] include_system=True")
    r2 = trim_messages(history, strategy="last", token_counter=count_tokens_approximately,
                       max_tokens=80, start_on="human", include_system=True)
    print(f"    结果 {len(r2)} 条: {_brief(r2)}")
    print(f"    首条 = {_first_type(r2)}   "
          f"→ 系统提示{'✅ 保住了' if r2 and isinstance(r2[0], SystemMessage) else '没了'}")

    print("\n  [场景 3] include_system=True,但额度抠紧 —— 系统提示是「要花钱」的")
    print(f"    {'max_tokens':>10} | {'总条数':>6} | {'其中对话':>8} | 结构")
    print("    " + "-" * 62)
    for mt in (30, 45, 50, 55, 60, 65, 70, 80, 120):
        r = trim_messages(history, strategy="last", token_counter=count_tokens_approximately,
                          max_tokens=mt, start_on="human", include_system=True)
        dialog = [m for m in r if not isinstance(m, SystemMessage)]
        flag = "  ← 对话一条不剩!" if not dialog else ""
        print(f"    {mt:>10} | {len(r):>6} | {len(dialog):>8} | {_brief(r)}{flag}")

    print("""
    💥 看那一行"对话一条不剩!" —— 这就是 include_system 的代价:

       ① 系统提示会被**先摘出来单独保管**,不参与裁剪;
       ② 但它要**先从 max_tokens 里扣掉自己的 token**,
          剩下的额度才轮到对话消息;
       ③ 如果剩下的额度连一条最小的消息都装不下,
          结果就变成"只剩一条系统提示",对话全丢 —— 而且同样**不报错**。

  【场景 4】系统提示不在第一条时,这个开关还管用吗?
""")

    # 把 SystemMessage 挪到中间
    h2 = list(history)
    sys_msg = h2.pop(0)
    h2.insert(3, sys_msg)
    print(f"    改造后 {len(h2)} 条: {_brief(h2)}")
    r4 = trim_messages(h2, strategy="last", token_counter=count_tokens_approximately,
                       max_tokens=80, start_on="human", include_system=True)
    print(f"    include_system=True → {_brief(r4)}")
    print(f"    结果里还有 SystemMessage 吗? "
          f"{'有' if any(isinstance(m, SystemMessage) for m in r4) else '❌ 没有'}")

    print("""
  ── 源码就三行,看完就彻底明白了 ────────────────────────
  (langchain_core.messages.utils._last_max_tokens)

      # ① 只有「列表第一条」是 SystemMessage 时才特殊照顾
      if include_system and len(messages) > 0 and isinstance(messages[0], SystemMessage):
          system_message = messages[0]     # 摘出来
          messages = messages[1:]          # 从待裁列表里拿掉

      # ② 它要占额度
      remaining_tokens = max(0, max_tokens - system_tokens)

      # ③ 裁完塞回最前面
      if system_message:
          result = [system_message, *result]

  ── 三个结论 ───────────────────────────────────────────
   ① **默认(False)确实会丢**。而且它是"悄无声息地丢" ——
      不报错、不警告,你只会发现模型突然不守规矩了。
   ② **只认第一条**。判断条件写的是 `messages[0]`,不是"任意一条"。
      系统提示如果不在最前面,这个开关对你毫无帮助。
   ③ **它抢额度**。极端情况下会把对话挤到一条不剩,只剩一条光杆系统提示。

  ── 实战怎么办 ─────────────────────────────────────────
   · 系统提示**永远放在 messages[0]**,这是硬规矩(几乎所有框架都这么假设);
   · 用 `include_system=True`,并且**留出系统提示那份额度**;
   · 系统提示本身尽量短 —— 它是每轮都要付的"固定成本",
     一份 2000 token 的系统提示,等于每轮先烧掉 2000。
""")


# ══════════════════════════════════════════════════════════════════
# 7. start_on 的真实语义:回溯砍(补第 3 部分那个坑的根因)
# ══════════════════════════════════════════════════════════════════
def part_7_backtracking() -> None:
    print("\n" + "=" * 70)
    print("第 7 部分:start_on 不是「找到就停」,是「一路往回砍」")
    print("=" * 70)

    print("""
  【先纠正一个很容易有的理解】
    直觉上,start_on="human" 好像是:
        "从后往前找,碰到第一条 HumanMessage 就停,从它开始往后全都保留。"

    不太对。真实过程是:

        第一步  额度内**尽量**从后往前收,能收多少收多少;
        第二步  检查收进来的这批,**头上那条**是不是 HumanMessage;
        第三步  不是 → 把头上那条**砍掉**,再看新的头上那条;
               是 → 停,把剩下的交出去。
        第四步  一路砍到空,就返回**空列表**。

    ——所以它是"**先收一批,再往回砍**",而不是"找到一个锚点就固定下来"。
""")

    history = build_history()
    print(f"  原始: {_brief(history)}  ({len(history)} 条 / "
          f"{count_tokens_approximately(history)} token)")

    print("\n  [额度从大到小,看开头被砍的过程]")
    print(f"    {'max_tokens':>10} | {'条数':>4} | 结构")
    print("    " + "-" * 62)
    for mt in (200, 120, 90, 80, 70, 60, 55, 50, 45, 30):
        r = trim_messages(history, strategy="last", token_counter=count_tokens_approximately,
                          max_tokens=mt, start_on="human")
        star = "  ← 空!" if not r else ""
        print(f"    {mt:>10} | {len(r):>4} | {_brief(r)}{star}")

    print("""
  👉 这张表最该看懂的两件事:

   ① 额度大的那几档(200、120)结果**不是全保留** ——
      头上那条 SystemMessage 照样被砍掉了。
      因为 start_on="human" 要求"头的第一条必须是 HumanMessage",
      系统提示不满足,所以**哪怕额度用不完也要砍**。

      ⚠️ 这就是最阴的地方:**你以为"额度够就不会有事",其实照样出事。**
         想要系统提示不被砍,必须**同时**加 include_system=True(见第 6 部分)。

   ② 额度一路缩小到某个点,结果不是"保留得更少",而是**突然变成空**。
      原因就是上面那第四步:收进来的那批头上都不是 HumanMessage,
      一刀一刀砍完,砍到一条不剩。

  ── 拿一个具体例子走一遍 ────────────────────────────────
    假设额度只够收进 [AI, TOOL, AI] 这三条(不含它们前面的那条 HUM):

       收进来: [AI, TOOL, AI]      头上是 AI     ✗ 砍掉
       砍一刀: [TOOL, AI]          头上是 TOOL   ✗ 砍掉
       砍一刀: [AI]                头上是 AI     ✗ 砍掉
       砍一刀: []                  → 返回空列表

    三个"看起来还能用"的消息,最后一条都没剩下。

  ── 所以正确的写法是 ────────────────────────────────────
     max_tokens 留足余量 + include_system=True + 裁完判空兜底(见第 4 部分)。
""")


# ══════════════════════════════════════════════════════════════════
# 8. 触发时机:到底什么时候该裁?
# ══════════════════════════════════════════════════════════════════
def part_8_trigger_timing() -> None:
    print("\n" + "=" * 70)
    print("第 8 部分:触发时机 —— 到底什么时候该裁?")
    print("=" * 70)

    print("""
  【先纠正一个常见误解】
    很多人以为裁剪是"防爆机制":等上下文快超过模型窗口了,才裁一刀救火。

    不对。**max_tokens 不是模型的上下文窗口,是你自己设的预算。**

    模型窗口(比如 128K)是"红线" —— 超了 API 直接报错。
    你设的 max_tokens(比如 4000)是"预算" —— 你只想留这么多。
    两个差了两个数量级,裁剪的界线画在**预算**上,离红线远得很。

    所以"每轮调模型前都裁"不是"不管三七二十一先砍一刀",
    而是"**始终把上下文压在预算以内**" —— 是常态控制,不是救火。
""")

    # ── 8.1 装得下的时候,它到底动不动手 ────────────────────
    print("  ── 8.1 关键实测:历史本来就装得下,trim 会不会照样动手? ──\n")

    small = [
        SystemMessage(content="你是一个助手。"),
        HumanMessage(content="你好"),
        AIMessage(content="你好,有什么可以帮你?"),
    ]
    print(f"    历史: {_brief(small)}   共 {count_tokens_approximately(small)} token")
    print(f"    给一个故意用不完的额度 max_tokens=9999\n")
    print(f"    {'传了哪些边界':<26} | {'结果':<28} | 判定")
    print("    " + "-" * 74)

    combos = [
        ("都不传", {}),
        ("只传 start_on='human'", {"start_on": "human"}),
        ("只传 end_on=('human','tool')", {"end_on": ("human", "tool")}),
        ("两个都传", {"start_on": "human", "end_on": ("human", "tool")}),
    ]
    for label, kw in combos:
        r = trim_messages(small, strategy="last", token_counter=count_tokens_approximately,
                          max_tokens=9999, **kw)
        same = "原样返回,什么都没做" if len(r) == len(small) else f"被砍了 {len(small)} -> {len(r)} 条"
        mark = "✅" if len(r) == len(small) else "❌"
        print(f"    {label:<26} | {_brief(r):<28} | {mark} {same}")

    print("""
    💥 看明白了吗 —— 这里是最容易踩的一处:

       · **不传 start_on / end_on** → 装得下就**原样返回**,一条不动。
         也就是说 trim_messages 内部有"够了就收手"的短路逻辑。

       · **只要传了任意一个边界** → 短路逻辑之后**仍然会做边界过滤**。
         哪怕额度根本用不完,它照样按规则砍。

       而官方文档给的例子 **两个边界都传了**(start_on="human" + end_on=("human","tool")),
       所以那个写法实际上是"**每一轮都会动手**"。

    ⚠️ 但这不是 bug —— end_on=("human","tool") 存在的意义就是:
       保证"发给模型的最后一条不是 AI 自己刚说的那句话"。
       它是**故意**要动刀的。
""")

    # ── 8.2 开销 ────────────────────────────────────────────
    print("  ── 8.2 那么「每轮都裁」会不会很慢? 实测 ──\n")

    big = [
        SystemMessage(content="你是一个助手。"),
        HumanMessage(content="北京今天天气怎么样?"),
        AIMessage(content="", tool_calls=[{"name": "get_weather", "args": {"city": "北京"},
                                            "id": "c1", "type": "tool_call"}]),
        ToolMessage(content="北京:晴,18℃,微风", tool_call_id="c1"),
        AIMessage(content="北京现在晴,18℃,微风。"),
    ] * 20   # 100 条
    N = 200

    t0 = time.perf_counter()
    for _ in range(N):
        trim_messages(big, strategy="last", token_counter=count_tokens_approximately,
                      max_tokens=300, start_on="human", include_system=True)
    trim_ms = (time.perf_counter() - t0) / N * 1000

    t0 = time.perf_counter()
    for _ in range(N):
        count_tokens_approximately(big)
    count_ms = (time.perf_counter() - t0) / N * 1000

    print(f"    规模:{len(big)} 条消息")
    print(f"    裁剪一次(含二分查找): 约 {trim_ms:.3f} ms")
    print(f"    光算一次 token:        约 {count_ms:.3f} ms")
    print(f"    → 裁剪内部要调用约 log2({len(big)}) ≈ "
          f"{len(big).bit_length() - 1} 次计数,倍数对得上")

    print("""
    👉 每轮给裁剪留 1 毫秒都嫌多。**所以"每轮都裁"在性能上完全不是问题。**

       真正的成本不是 CPU,是**它裁掉的东西** ——
       裁掉的消息就永久没了(这是第 3 部分要上摘要的原因)。
""")

    # ── 8.3 token_counter 的选择 ───────────────────────────
    print("  ── 8.3 能不能用「模型自带的精确计数」代替估算? ──\n")

    t0 = time.perf_counter()
    for _ in range(N):
        count_tokens_approximately(big)
    approx_ms = (time.perf_counter() - t0) / N * 1000
    print(f"    count_tokens_approximately : {approx_ms:.3f} ms   (纯字符串,本地,零依赖)")

    try:
        from langchain_openai import ChatOpenAI

        m = ChatOpenAI(model="gpt-4o", api_key="sk-x")
        m.get_num_tokens_from_messages(big)   # 预热(首次要加载编码表)
        t0 = time.perf_counter()
        for _ in range(N):
            m.get_num_tokens_from_messages(big)
        tiktoken_ms = (time.perf_counter() - t0) / N * 1000
        print(f"    tiktoken(模型 gpt-4o)      : {tiktoken_ms:.3f} ms   "
              f"(本地 tiktoken,慢 {tiktoken_ms / approx_ms:.0f} 倍)")
    except Exception as e:  # noqa: BLE001
        print(f"    tiktoken 计数不可用:{type(e).__name__}")

    try:
        from langchain_openai import ChatOpenAI

        m2 = ChatOpenAI(model=os.getenv("MODEL_NAME", "glm-4.7-flash"), api_key="sk-x")
        m2.get_num_tokens_from_messages(big)
        print(f"    {os.getenv('MODEL_NAME', 'glm-4.7-flash')} 计数: 可用")
    except NotImplementedError:
        print(f"    {os.getenv('MODEL_NAME', 'glm-4.7-flash')} 计数: "
              f"❌ NotImplementedError —— 该模型不支持精确计数")
    except Exception as e:  # noqa: BLE001
        print(f"    {os.getenv('MODEL_NAME', 'glm-4.7-flash')} 计数: "
              f"❌ {type(e).__name__}")

    print("""
    💥 最后一行是关键:**你想"精确计数"也做不到**,因为模型根本没实现这个接口。

       所以 `count_tokens_approximately` 不是"退而求其次的将就方案",
       而是**唯一可行的方案** —— 而且它快得可以忽略不计。

       ⚠️ 它是估算,可能有 10~20% 误差。但裁剪本来就要"留余量",
          留余量这件事刚好把误差吃掉了。
""")

    # ── 8.4 结论:两个触发点 ─────────────────────────────────
    print("  ── 8.4 完整答案:其实是「两个独立的触发点」,不是「一个」 ──\n")
    print("""
    ┌──────────────────────────────────────────────────────────────────┐
    │ 触发点 1 · 每一轮调模型之前                                        │
    │   动作:裁剪 trim                                                 │
    │   目的:保证这一次请求不超预算 / 不超窗口                           │
    │   特点:高频、零成本、**掉了就真没了**                              │
    ├──────────────────────────────────────────────────────────────────┤
    │ 触发点 2 · 消息数或 token 超过阈值                                 │
    │   动作:摘要 summarize                                            │
    │   目的:在"要丢东西"之前,先把结论压出来保住                         │
    │   特点:低频、要调模型、有损但保留语义                              │
    └──────────────────────────────────────────────────────────────────┘

    两个触发点的频率完全不同 —— 这正是最容易被搞混的地方:
      · 裁剪是"每轮都做"的日常动作;
      · 摘要是"攒够了才做"的定期动作。

    ── 那"阈值触发裁剪"这种写法呢? ────────────────────────
    也有人这么写(官方文档文字部分描述的就是这个):
        if 当前 token > 阈值: 才裁

    它跟"每轮都裁"的差别:
      ✅ 省下每次的 0.3ms(真的可以忽略)
      ❌ 风险:如果某一条消息特别大(用户粘贴一整篇文档),
         上一次检查时还不到阈值,这一次直接冲爆 —— 你得在**调用之后**
         才知道爆了,那时候请求已经失败了。

    所以现在的做法一般是:
      **每轮都裁(便宜的常态控制) + 超阈值时摘要(贵的效果补救)**。
    裁剪是"保证不炸",摘要是"保证不忘" —— 两件事,别指望一个机制干完。
""")


# ══════════════════════════════════════════════════════════════════
# 9. 固定额度 + 多轮对话 = 什么现象?(稳态)
# ══════════════════════════════════════════════════════════════════
def part_9_steady_state() -> None:
    print("\n" + "=" * 70)
    print("第 9 部分:max_tokens 是个固定值,那对话变长之后会怎样?")
    print("=" * 70)

    MAX = 150   # 故意设小,方便在几十行输出里看到"稳态"出现
    sys_msg = SystemMessage(content="你是一名电商客服助手,负责查订单与物流。")
    state_msgs: list = [sys_msg]

    print(f"""
  【先想一个问题】
    max_tokens 是个固定值。那:
      · 对话刚开始,消息很少 → 永远用不到这个额度
      · 越聊越多 → 逐渐逼近这个额度
      · 一旦到了 → 此后每来一轮新消息,都会超,都得裁一刀

    这个描述对不对?下面直接模拟 20 轮看。

    模拟设定:max_tokens = {MAX}(故意设小)、start_on="human"、include_system=True
""")

    print(f"    {'轮次':>4} | {'state 条数':>10} | {'state token':>11} | "
          f"{'请求条数':>8} | {'请求 token':>10} | 本次动作")
    print("    " + "-" * 86)

    first_cut_turn = None
    for turn in range(1, 21):
        state_msgs.append(HumanMessage(content=f"第 {turn} 轮:帮我查一下订单 {10000 + turn} 的物流。"))
        state_msgs.append(AIMessage(content=f"第 {turn} 轮:订单 {10000 + turn} 已发出,预计明天下午送达。"))

        raw_n = len(state_msgs)
        raw_t = count_tokens_approximately(state_msgs)

        trimmed = trim_messages(
            state_msgs,
            strategy="last",
            token_counter=count_tokens_approximately,
            max_tokens=MAX,
            start_on="human",
            include_system=True,
        )
        trimmed_n = len(trimmed)
        trimmed_t = count_tokens_approximately(trimmed) if trimmed else 0

        cut = raw_n - trimmed_n
        if cut == 0:
            action = "没裁,全留着"
        else:
            if first_cut_turn is None:
                first_cut_turn = turn
            # 裁剪后的窗口是"从第几轮开始"的
            first_kept = next((m for m in trimmed if isinstance(m, HumanMessage)), None)
            win = first_kept.content.split("轮")[0].strip() if first_kept else "?"
            action = f"砍 {cut} 条,窗口从{win}轮开始"
        print(f"    {turn:>4} | {raw_n:>10} | {raw_t:>11} | "
              f"{trimmed_n:>8} | {trimmed_t:>10} | {action}")

    print(f"""
  ── 你猜的没错,而且比你想的更"标准" ──────────────────────

    ★ 第 1 阶段(前几轮):**根本没裁** —— state 和请求一模一样。
      这就是你说的"消息少,永远达不到预算"。

    ★ 第 2 阶段(第 {first_cut_turn} 轮起):**开始裁了**。因为累计 token 越过了 {MAX}。

    ★ 第 3 阶段(之后每一轮):**每轮都在裁,而且裁掉的条数是恒定的**。

  ── 为什么会形成这个"恒定"? ────────────────────────────

    因为裁剪的目标是"裁到不超过 max_tokens"。
    一旦裁到位,下一轮新消息进来 → 又超了 → 再裁回 max_tokens 附近。

        进多少,出多少,总量卡住不动。

    这在工程上叫 **滑动窗口(sliding window)**:
        · 新消息从右边进来
        · 老消息从左边被挤出去
        · 窗口大小恒定

    所以你说的"不停地截不停的截" —— **完全正确,这就是设计如此,不是异常**。

  ── 但有个非常关键的区别,上面表里已经露出来了 ────────────

    看两列数字的走向:
      · **「请求 token」** 进入稳态后**卡住不动**(就在 {MAX} 附近)
      · **「state token」** 却**一直在涨**,永远不会自己回落

    为什么?因为官方推荐的写法(第 5 部分那个 call_model)只裁**发给模型的那份**,
    **根本没有动 state 里的 messages**。

    ⚠️ 这意味着两件事:
      ① state 里的消息会**无限增长** —— 上了 checkpointer 之后,
         数据库里这条 thread 的记录会一直变大;
      ② 每轮裁剪要处理的列表越来越长(虽然实测很快,可以忽略)。

  ── 所以完整的心智模型是这样的 ──────────────────────────

    ┌────────────────────────────────────────────────────────────┐
    │  state.messages  ← 一直在长,像一本完整的会议纪要          │
    │        │                                                    │
    │        ├── 裁剪 ──→ 只发最近的 N 条给模型(滑动窗口)        │
    │        │                                                    │
    │        └── 摘要 ──→ 把"即将被挤出去"的那批压成一段话        │
    │                     (这一步会真的删 state,见 03 那份脚本)   │
    └────────────────────────────────────────────────────────────┘

    **裁剪管的是"这一次发什么",摘要管的是"忘了什么还能捡回来"。**

  ── 现实里的两个推论 ────────────────────────────────────

    ① **进入稳态那一刻,模型就开始"失忆"了。**
       稳态之后,最早的消息一轮一轮被挤出去 ——
       用户第 1 轮说的"我叫 Bob"、第 2 轮给的订单号,迟早会被挤没。
       这就是为什么必须配一个**摘要触发点**。

    ② **稳态来得比你想象的早。**
       上面这个模拟里 {MAX} token 只装了 6 轮对话就满了 ——
       因为每条消息都带 role、每条都算"信封开销"(见第 0 部分),
       **你设的 token 预算,能装的实际轮数是打折扣的。**
""")


# ══════════════════════════════════════════════════════════════════
# 10. max_tokens 该设多大?跟模型窗口是什么关系?
# ══════════════════════════════════════════════════════════════════
def part_10_input_budget() -> None:
    from langchain_core.tools import tool

    print("\n" + "=" * 70)
    print("第 10 部分:max_tokens 该设多大?跟模型窗口是什么关系?")
    print("=" * 70)

    print("""
  【关键认知:窗口是"硬上限",不是"你的预算"】

    模型窗口 = 输入 + 输出 **共享**的一份空间。
    所以你不能把 max_tokens 贴着窗口设 —— 否则模型没有空间生成回复。

        窗口 200K
        ├──────────────────────────────────────────────┤
        │  你的输入(max_tokens)  │  留给输出  │ 余量   │
        └──────────────────────────────────────────────┘
                                  ↑
                        这块不给够,模型一开口就被截断

  【而且消耗方不止"对话消息"一个】

    一次请求实际占的 token 至少有四块:
      ① 系统提示
      ② 对话消息(裁剪管的就是这块)
      ③ **工具定义的 schema** ← 这个最容易被忘掉
      ④ 模型这次生成的输出(要预留)

    ③ 是真的占额度 —— count_tokens_approximately 的源码里就有这段:
        for tool in tools: tools_chars += len(json.dumps(tool_dict))
        token_count += math.ceil(tools_chars / chars_per_token)
      也就是说它连"工具描述文本"都算钱。下面实测一下。
""")

    # ── 实测工具 schema 的开销 ──────────────────────────────
    @tool
    def get_weather(city: str) -> str:
        """查询指定城市的实时天气,返回温度、天气状况和风力。"""
        return "晴"

    @tool
    def search_order(order_id: str) -> str:
        """根据订单号查询订单状态、发货时间和物流单号。"""
        return "已发货"

    @tool
    def refund(order_id: str, reason: str) -> str:
        """对指定订单发起退款申请。需要提供订单号和退款原因。"""
        return "已受理"

    TOOLS = [get_weather, search_order, refund]
    conv = [SystemMessage(content="你是一名电商客服助手。"),
            HumanMessage(content="帮我查一下订单 10001 的物流。")]
    without = count_tokens_approximately(conv)
    with_tools = count_tokens_approximately(conv, tools=TOOLS)

    print(f"    普通对话(2 条消息):            {without} token")
    print(f"    同样对话 + 挂上 3 个工具定义:  {with_tools} token")
    print(f"    → **光是工具定义就吃掉 {with_tools - without} token**")
    print("""
    ⚠️ 工具越多、docstring 写得越长,这项开销越大。
       一个挂了 20 个工具的 agent,光 schema 就可能上千 token ——
       而且**每一轮都要付这份钱**。
""")

    # ── 怎么算? ────────────────────────────────────────────
    def compute_max_tokens(
        context_window: int,
        want: int = 8000,
        reserve_output: int = 4000,
        reserve_tools: int = 1500,
        safety: float = 0.9,
    ) -> int:
        """
        两条约束取小的那个:

          · 硬约束(物理):不能超窗口,还要给输出和工具留位置
          · 软约束(业务):你真正想让模型记住多少

        context_window : 模型的上下文窗口(GLM-4.7-Flash 是 200_000)
        want           : 业务上你希望留多少历史(按"能放十几轮对话"估)
        reserve_output : 给模型生成回复留的空间
        reserve_tools  : 工具 schema 占的空间(上面实测那项)
        safety         : 安全系数,给估算误差和边界情况留的垫子
        """
        hard_limit = int((context_window - reserve_output - reserve_tools) * safety)
        return max(1000, min(want, hard_limit))

    print("  ── 两条约束取小:硬约束 vs 业务需求 ──\n")
    print(f"    {'模型窗口':>10} | {'业务想要':>8} | {'硬上限':>8} | {'最终 max_tokens':>14} | 谁在起作用")
    print("    " + "-" * 74)

    cases = [
        (200_000, 8_000, "GLM-4.7-Flash(200K 窗口)"),
        (128_000, 8_000, "典型中档模型(128K)"),
        (32_000, 8_000, "小窗口模型(32K)"),
        (8_000, 8_000, "老旧小模型(8K)"),
    ]
    for win, want, label in cases:
        final = compute_max_tokens(win, want=want)
        hard = int((win - 4000 - 1500) * 0.9)
        who = "业务需求" if final == want else "硬上限(窗口太小,被压下来了)"
        print(f"    {win:>10} | {want:>8} | {hard:>8} | {final:>14} | {who}")
        print(f"    {'':>10}   {label}")

    print("""
  ── 结论 ──────────────────────────────────────────────────

    **max_tokens 不是"贴着窗口设",而是两条线取小:**

        max_tokens = min( 业务上想留多少 , 窗口能给多少 )

    ① **窗口给了 200K,不代表你该设 200K。**
       设太大 = 每轮都烧钱 + 延迟高 + 模型"迷失在中间"。
       实测那 200K 窗口的模型,业务上设 8K 就够了。

    ② **窗口小了,你就被迫往下压。**
       看最后一行:8K 窗口的模型,想留 8K 历史根本不可能 ——
       扣掉输出和工具预留只剩 2K 左右,裁剪会被逼得非常狠,
       对话聊两轮就滑动一遍。

    ③ **换模型必须重新估一次。**
       从 200K 模型换到 8K 模型,如果 max_tokens 还是 8000,
       请求会**直接超窗口报错** —— 而且这个错误在开发时不一定测出来,
       上线换配置才炸。

  ── 🛡️ 安全落点 ─────────────────────────────────────────
    · **工具定义也是成本**。每轮都付,所以要精简工具数量和 docstring,
      不要把整个 API 文档塞进工具的说明里。
    · **输出预留不能省**。省了的表现是"模型回复被莫名截断",
      不是报错,很难查。
    · **换模型是配置变更,要走一遍测试**。窗口差异是这类事故的高发区。
""")


# ══════════════════════════════════════════════════════════════════
# 11. 「业务想要多少」到底怎么估?
# ══════════════════════════════════════════════════════════════════
def part_11_budget_estimate() -> None:
    from langchain_core.tools import tool

    print("\n" + "=" * 70)
    print("第 11 部分:「业务上想留多少」到底怎么估?")
    print("=" * 70)

    # ── 11.1 先分清:工程里有两个 max_tokens ────────────────
    print("""
  ⚠️ 11.1 先躲开一个混淆:工程里有两个"max_tokens",含义完全不同

     ┌──────────────────────────────┬──────────────────────────────────┐
     │ trim_messages(max_tokens=X)  │ 输入侧:裁完之后,最多带多少进模型│
     ├──────────────────────────────┼──────────────────────────────────┤
     │ model.bind(                  │ 输出侧:模型最多生成多少 token   │
     │   max_completion_tokens=Y)   │                                  │
     └──────────────────────────────┴──────────────────────────────────┘

     一个管"喂进去多少",一个管"吐出来多少"。**名字像,作用正好相反。**
""")

    import inspect

    from langchain_openai import ChatOpenAI

    m = ChatOpenAI(model="x", api_key="sk-x")
    tok_params = [p for p in inspect.signature(ChatOpenAI).parameters if "token" in p.lower()]
    print(f"    实测 ChatOpenAI 的构造参数里,跟 token 有关的是:{tok_params}")
    print(f"    实测 trim_messages 的输入侧参数是:['max_tokens', 'token_counter', ...]")
    print("""
    ⚠️ 注意:**ChatOpenAI 现在叫 `max_completion_tokens`,不再叫 `max_tokens` 了。**
       老文档和老代码里那个 `max_tokens` 已经改名。

    💥 而且有个坑:`bind()` **不做任何校验**,你写什么它都收下:
""")
    b_old = m.bind(max_tokens=128)
    b_new = m.bind(max_completion_tokens=128)
    print(f"    m.bind(max_tokens=128)            -> 收下了, kwargs = "
          f"{getattr(b_old, 'kwargs', None)}  ← 过时参数名,会被原样发给接口")
    print(f"    m.bind(max_completion_tokens=128) -> 收下了, kwargs = "
          f"{getattr(b_new, 'kwargs', None)}  ← 正确")

    print("""
    ⚠️ 官方 add-memory 文档那个例子里有一行 `summarization_model = model.bind(max_tokens=128)`,
       按现在这个版本,它属于"能跑但不一定生效"的写法 —— 别照抄, 用 `max_completion_tokens`。
""")

    # ── 11.2 估算算例 ──────────────────────────────────────
    @tool
    def search_order(order_id: str) -> str:
        """根据订单号查询订单状态、发货时间和物流单号。"""
        return "已发货"

    @tool
    def change_address(order_id: str, address: str) -> str:
        """修改订单的收货地址。需要订单号和新地址。"""
        return "已修改"

    @tool
    def apply_refund(order_id: str, reason: str) -> str:
        """对订单发起退款申请,需要订单号和退款原因。"""
        return "已受理"

    @tool
    def check_logistics(tracking_no: str) -> str:
        """根据物流单号查询最新物流轨迹节点。"""
        return "运输中"

    @tool
    def handoff_to_human(reason: str) -> str:
        """把当前会话转接给人工客服,需要说明转接原因。"""
        return "已转接"

    TOOLS = [search_order, change_address, apply_refund, check_logistics, handoff_to_human]
    SYS = SystemMessage(content="你是一名电商客服助手,负责处理订单查询、改地址、退款与转人工。"
                                "回答要简洁,不承诺时效,不报价。")

    sys_tok = count_tokens_approximately([SYS])
    tools_tok = count_tokens_approximately([SYS], tools=TOOLS) - sys_tok
    # 一条典型的一轮对话
    one_turn = [HumanMessage(content="你好,我上周买的那个蓝色杯子还没到,能帮我查一下吗?"
                                     "订单号是 88345,麻烦看下现在到哪了。"),
                AIMessage(content="好的,我帮您查一下订单 88345。"
                                  "这笔订单已经发货,目前在运输中,预计明天下午送达杭州。")]
    turn_tok = count_tokens_approximately(one_turn)

    rounds = 8   # 客服这类 agent:需要记住用户前面给的订单号、诉求,大概 8 轮
    conversation_budget = turn_tok * rounds
    trim_max = sys_tok + conversation_budget
    reserve_output = 1500
    reserve_extra = int((trim_max + tools_tok + reserve_output) * 0.15)   # 15% 余量

    print("  ── 11.2 算例:一个客服 agent 该设多少? ──\n")
    print(f"    {'项目':<34} | {'token':>7}")
    print("    " + "-" * 46)
    print(f"    {'① 系统提示':<34} | {sys_tok:>7}")
    print(f"    {'② 工具 schema(5 个工具)':<34} | {tools_tok:>7}")
    print(f"    {'③ 一轮典型对话(用户+AI)':<34} | {turn_tok:>7}")
    print(f"    {'④ 需要保留的轮数(客服类,估计)':<34} | {rounds:>7}")
    print("    " + "-" * 46)
    print(f"    {'→ 对话部分 = ③ × ④':<34} | {conversation_budget:>7}")
    print(f"    {'→ trim_messages(max_tokens=?) = ① + 对话部分':<34} | {trim_max:>7}")
    print()
    print(f"    {'输出预留(model 侧)':<34} | {reserve_output:>7}")
    print(f"    {'安全余量(15%)':<34} | {reserve_extra:>7}")
    print(f"    {'→ 总共要从窗口里扣掉':<34} | "
          f"{trim_max + tools_tok + reserve_output + reserve_extra:>7}")

    print(f"""
    💡 这个例子里有几个数字很值得看:

       · **工具 schema({tools_tok} token)比整个系统提示({sys_tok} token)还大 20 倍。**
         5 个工具的描述文字,比你精心写的系统提示贵得多 ——
         因为每个工具的 schema 是一整段 JSON(名字 + 描述 + 参数结构)。
         **精简工具,比精简系统提示划算。**

       · 一轮对话才 {turn_tok} token,所以"留 8 轮"其实很便宜({conversation_budget} token。
         真正贵的是**固定成本**(系统提示 + 工具),它们每轮都要付。

       ⚠️ 这里的数字是**这个例子专用的**。你的系统提示写多长、
         挂几个工具、一轮对话聊多长,都要拿你自己的真实内容去量 ——
         上面那几行代码就是量法,改内容重跑即可。
""")

    print(f"""
  ── 怎么反推"需要几轮"?(这才是问题的核心) ──────────

    不要问"我这个业务有多少 token",要问:

        **"回答用户当前这句话,模型至少需要回看多少历史?"**

    这个答案取决于 **agent 的类型**,不取决于业务领域:

    ┌────────────────────────────┬──────────────┬────────────────────────────┐
    │ Agent 类型                  │ 需要回看      │ 为什么                      │
    ├────────────────────────────┼──────────────┼────────────────────────────┤
    │ 单轮问答 / 分类 / 抽取      │ 0 ~ 1 轮      │ 每次独立,历史没用           │
    │ 客服对话(查单、改地址)     │ 3 ~ 10 轮     │ 要记住用户前面给的订单号/诉求│
    │ 多步任务(帮我把这活干完)   │ 几乎全部      │ 目标在开头,每步都依赖它      │
    │ 代码助手(改文件)           │ 全部 + 文件   │ 上下文本身就是工作区         │
    └────────────────────────────┴──────────────┴────────────────────────────┘

    所以"业务想要多少"翻译成可操作的话是:

        **"这个 agent 在第 N 轮回答时,还看不看得见第 1 轮说的话?"**

    ── ⚠️ 但它不是"越大越好" ─────────────────────────────

    三个理由:
      ① **成本**:每轮都按输入 token 付费,多留 = 多花钱
      ② **延迟**:输入越长,首个 token 出得越慢
      ③ **质量**:上下文越长,模型越容易"迷失在中间" ——
         无关的老对话会稀释它对当前问题的注意力。
         (这就是 "lost in the middle" 现象,业界有专门论文研究)

    所以目标不是"尽量多留",而是 **"刚好够用"**。

    ── 工程上怎么定?别拍脑袋,要观测 ────────────────────

      第一步:先按上面的算例拍一个保守值(比如 4K)
      第二步:上线跑真实流量,**统计每轮请求的实际 token 分布**
      第三步:看中位数和 P95 —— 你会发现**绝大多数请求远用不到那么多**,
             是少数长对话把额度撑起来的
      第四步:把 max_tokens 定在 P95 附近,剩下的长尾交给裁剪兜底

    🎯 **一句话**:max_tokens 的正确目标不是"最大能装多少",
       而是"**覆盖 95% 的请求刚好够用,极端长对话让它裁**"。
""")


# ══════════════════════════════════════════════════════════════════
def main() -> None:
    part_0_how_counting_works()
    part_1_four_configs()
    part_2_pitfall_orphan_tool()
    part_3_pitfall_empty()
    part_4_safe_wrapper()
    part_5_in_graph()
    part_6_system_message()
    part_7_backtracking()
    part_8_trigger_timing()
    part_9_steady_state()
    part_10_input_budget()
    part_11_budget_estimate()

    print("\n" + "=" * 70)
    print("本文件结论(背这 12 条就够)")
    print("=" * 70)
    print("""
  1. 算 token 不需要调模型 —— count_tokens_approximately 是纯字符估算,零成本。
     (而且很多模型压根没实现精确计数接口,想精确也算不了。)
  2. 必须传 start_on="human" —— 否则会裁出"孤儿 ToolMessage",序列非法直接 400。
  3. start_on 是"先收一批、再从头往回砍",不是"找到一个锚点就停" ——
     砍光了就是空列表。而且**额度充足时也照砍不误**。
  4. 系统提示默认会被砍掉(因为它在列表最开头),要靠 include_system=True 保住。
     但它只认 `messages[0]` 那一条,而且要**先占走一份额度**。
  5. ⚠️ **不传边界 -> 装得下就原样返回;传了任一边界 -> 装得下也照样动手。**
     官方例子两个边界都传了,所以它是"每轮都会动刀"的写法。
  6. max_tokens 是**你的预算**,不是模型的上下文窗口 ——
     所以"每轮调模型前都裁"是常态控制(控成本、控延迟、防中间迷失),
     不是等快爆了才救火。
  7. 裁剪单次约 0.25 ms(100 条消息),**每轮都做完全没负担**;
     真正的成本是"被裁掉的内容"。
  8. **触发点有两个,别混**:裁剪 = 每轮都做;摘要 = 超阈值才做。
     裁剪保证"不炸",摘要保证"不忘"。
  9. max_tokens 固定 → 对话必然进入**滑动窗口稳态**:
     新消息从右边进,老消息从左边被挤出去,请求 token 卡住不动。
     **进入稳态那一刻,模型就开始"失忆"了。**
 10. 只用 trim 不动 state 的话,**state.messages 会无限增长**
     (官方那个 call_model 就是这样)。真正缩小 state 要靠 RemoveMessage(见 03)。
 11. max_tokens = min(业务上想留多少, 窗口能给多少);
     **工具定义的 schema 也占额度且每轮都付** —— 实测 5 个工具 ≈ 399 token,
     比精心写的系统提示贵 20 倍。**精简工具比精简提示划算。**
 12. ⚠️ **两个 max_tokens 别搞混**:
     `trim_messages(max_tokens=X)` 管**输入**(带多少进去);
     `model.bind(max_completion_tokens=Y)` 管**输出**(生成多少)。
     `ChatOpenAI` 已改名,老的 `max_tokens` 不再生效,而 `bind()` 不校验 —— 会静默失效。
""")


if __name__ == "__main__":
    main()
