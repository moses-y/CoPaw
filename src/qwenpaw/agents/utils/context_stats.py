# -*- coding: utf-8 -*-
"""Token accounting helpers for ``/history`` and context inspection."""
from __future__ import annotations

from typing import TYPE_CHECKING

from .as_msg_handler import AsMsgHandler
from .estimate_token_counter import EstimatedTokenCounter

if TYPE_CHECKING:
    from agentscope.state import AgentState


async def estimate_context_tokens(
    state: "AgentState",
    token_counter: EstimatedTokenCounter,
    max_input_length: int,
) -> dict:
    """Compute the per-message and summary token breakdown."""
    handler = AsMsgHandler(token_counter)

    summary = state.summary if isinstance(state.summary, str) else ""
    summary_tokens = await handler.count_str_token(summary)

    messages_detail = [
        await handler.stat_message(msg) for msg in state.context
    ]
    messages_tokens = sum(stat.total_tokens for stat in messages_detail)
    estimated_tokens = messages_tokens + summary_tokens
    usage_ratio = (
        (estimated_tokens / max_input_length * 100)
        if max_input_length > 0
        else 0
    )

    return {
        "total_messages": len(state.context),
        "compressed_summary_tokens": summary_tokens,
        "messages_tokens": messages_tokens,
        "estimated_tokens": estimated_tokens,
        "max_input_length": max_input_length,
        "context_usage_ratio": usage_ratio,
        "messages_detail": messages_detail,
    }


async def format_history_str(
    state: "AgentState",
    token_counter: EstimatedTokenCounter,
    max_input_length: int,
    start: int | None = None,
    end: int | None = None,
    tail: int | None = None,
) -> str:
    """Render the ``/history`` reply text.

    The optional bounds narrow the listing so ``/history 20`` and
    ``/history :5-20`` can show a slice:

    * ``tail`` keeps the last N messages (``/history 20``);
    * ``start``/``end`` are 1-based inclusive bounds (``/history :5-20``).

    The summary header always describes the whole history, and rendered row
    indices stay absolute so ``/message <index>`` keeps working on a narrowed
    view.
    """
    stats = await estimate_context_tokens(
        state,
        token_counter,
        max_input_length,
    )

    all_messages = stats["messages_detail"]
    total = len(all_messages)

    if tail is not None and tail > 0:
        lo = max(1, total - tail + 1)
        hi = total
    else:
        # Normalise the requested window into inclusive 1-based bounds.
        lo = 1 if start is None else max(1, min(start, total or 1))
        hi = total if end is None else max(lo, min(end, total or 1))
    window = all_messages[lo - 1 : hi] if total else []

    lines = []
    for i, msg_stat in enumerate(window, lo):
        blocks_info = ""
        if msg_stat.content:
            block_strs = [
                f"{b.block_type}(tokens={b.token_count})"
                for b in msg_stat.content
            ]
            blocks_info = f"\n    content: [{', '.join(block_strs)}]"

        lines.append(
            f"[{i}] **{msg_stat.role}** "
            f"(total_tokens={msg_stat.total_tokens})"
            f"{blocks_info}\n    preview: {msg_stat.preview}",
        )

    body = "\n\n".join(lines)
    if total and (lo > 1 or hi < total):
        body += (
            f"\n\n_Showing messages {lo}-{hi} of {total}. "
            "Omit the range or widen it to see more._"
        )

    return (
        f"**Conversation History**\n\n"
        f"- Total messages: {stats['total_messages']}\n"
        f"- Estimated tokens: {stats['estimated_tokens']}\n"
        f"- Max input length: {stats['max_input_length']}\n"
        f"- Context usage: "
        f"{stats['context_usage_ratio']:.1f}%\n"
        f"- Compressed summary tokens: "
        f"{stats['compressed_summary_tokens']}\n\n" + body
    )
