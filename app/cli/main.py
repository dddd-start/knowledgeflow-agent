"""命令行入口 —— 不打开浏览器也能完整体验整套链路。

用法:
    python -m app.cli.main                    # 交互式(意图路由自动选择)
    python -m app.cli.main --mode quality     # 高质量检索(改写 + HyDE + 重排 + 自检)
    python -m app.cli.main --mode agent       # Agent 工具循环
    python -m app.cli.main --mode agentic     # Agentic RAG 迭代检索
    python -m app.cli.main -q "出差住宿标准是多少"   # 单次提问(脚本/评测用)
    python -m app.cli.main --no-stream        # 关闭流式

内置命令:/exit 退出、/clear 清空会话记忆、/memory 查看记忆与偏好、/kb 知识库统计。
"""

from __future__ import annotations

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console  # noqa: E402
from rich.markdown import Markdown  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.table import Table  # noqa: E402

from app.obs.logging import setup_logging  # noqa: E402
from app.services import ChatService  # noqa: E402

console = Console()
setup_logging()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="KnowledgeFlow CLI")
    parser.add_argument(
        "--mode", default="auto", choices=["auto", "fast", "quality", "chat", "agent", "agentic"]
    )
    parser.add_argument("-q", "--question", default=None, help="单次提问后退出")
    parser.add_argument("--user", default="cli_user")
    parser.add_argument("--session", default="cli_session")
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出(便于脚本解析)")
    return parser


def _print_meta(service: ChatService) -> None:
    meta = service.meta()
    llm = meta["llm"]
    emb = meta["embedding"]
    kb = meta["kb"]
    console.print(
        Panel(
            f"LLM  : {llm['backend']} / {llm['model']}\n"
            f"向量 : {emb['backend']} / {emb['model']} ({emb['dim']}d)\n"
            f"知识库: {kb['documents']} 篇文档 / {kb['chunks']} 个片段\n"
            f"意图 : {meta['intent']['backend']} (阈值 {meta['intent']['threshold']})\n"
            f"模式 : {', '.join(meta['modes'])}",
            title="KnowledgeFlow",
            border_style="cyan",
        )
    )
    if meta["offline_demo"]:
        console.print(
            "[yellow]当前为离线演示模式(未配置 LLM_API_KEY):回答由确定性 Mock 生成,"
            "检索与引用链路是真实的。[/yellow]"
        )


def _render_events(events, *, stream: bool) -> dict:
    """渲染事件流并返回收敛结果。"""
    answer_started = False
    result: dict = {"answer": "", "citations": [], "llm": {}}
    for event in events:
        if event.type == "meta":
            plan = event.data.get("plan", {})
            console.print(
                f"[dim]模式={event.data['mode']} 路径={event.data['path']} "
                f"意图={plan.get('intent')}({plan.get('confidence')})[/dim]"
            )
            result["meta"] = event.data
        elif event.type == "retrieval":
            trace = event.data.get("trace", {})
            stages = " → ".join(f"{s['name']}({s['ms']}ms)" for s in trace.get("stages", []))
            console.print(f"[dim]检索链路: {stages}[/dim]")
        elif event.type == "tools":
            for item in event.data.get("items", []):
                flag = "[red]失败[/red]" if item.get("is_error") else "[green]成功[/green]"
                console.print(
                    f"[dim]  工具 {item['name']}({item['arguments']}) → {flag} "
                    f"{item['output'][:60].replace(chr(10), ' ')}[/dim]"
                )
        elif event.type == "delta":
            if stream:
                if not answer_started:
                    console.print()
                    answer_started = True
                console.print(event.data.get("text", ""), end="", markup=False, highlight=False)
            result["answer"] += event.data.get("text", "")
        elif event.type == "replace":
            console.print(f"\n[yellow]自检修订: {event.data.get('reason', '')}[/yellow]")
            result["answer"] = event.data.get("text", "")
        elif event.type == "citations":
            result["citations"] = event.data.get("items", [])
        elif event.type == "usage":
            result["usage"] = event.data
        elif event.type == "error":
            console.print(f"[red]错误: {event.data.get('message')}[/red]")
            result["error"] = event.data.get("message")
        elif event.type == "done":
            result["answer"] = event.data.get("answer", result["answer"])
    if stream and answer_started:
        console.print()
    elif not stream and result.get("answer"):
        console.print()
        console.print(Markdown(result["answer"]))
    return result


def _print_citations(citations: list[dict]) -> None:
    if not citations:
        return
    table = Table(title="引用来源", show_lines=False, border_style="dim")
    table.add_column("#", width=3)
    table.add_column("文档", overflow="fold")
    table.add_column("相关度", width=8, justify="right")
    for item in citations[:5]:
        table.add_row(str(item["index"]), item["title"], f"{item['score']:.3f}")
    console.print(table)


def _print_usage(usage: dict) -> None:
    if not usage:
        return
    console.print(
        f"[dim]token: {usage.get('prompt_tokens')} in / {usage.get('completion_tokens')} out | "
        f"LLM 调用 {usage.get('llm_calls')} 次 | 成本 ${usage.get('cost_usd')} | "
        f"{usage.get('elapsed_ms')}ms[/dim]"
    )


def main() -> int:
    args = _build_parser().parse_args()
    service = ChatService()

    if args.question:
        events = service.answer(
            args.question,
            session_id=args.session,
            user_id=args.user,
            mode=args.mode,
            stream=not args.no_stream,
        )
        result = _render_events(events, stream=not args.no_stream)
        if args.json:
            console.print_json(data={k: result.get(k) for k in ("answer", "citations", "usage")})
        else:
            _print_citations(result["citations"])
            _print_usage(result.get("usage", {}))
        return 0

    service.ensure_index_ready()
    if service.cfg.warmup:
        service.warmup()
    _print_meta(service)
    console.print("[dim]/exit 退出 | /clear 清空记忆 | /memory 记忆状态 | /kb 知识库统计[/dim]")
    while True:
        try:
            question = console.input("[bold cyan]你 > [/bold cyan]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print()
            break
        if not question:
            continue
        if question in {"/exit", "/quit"}:
            break
        if question == "/clear":
            service.memory(args.session, args.user).short.reset()
            console.print("[green]会话记忆已清空[/green]")
            continue
        if question == "/memory":
            console.print_json(data=service.memory(args.session, args.user).status())
            continue
        if question == "/kb":
            console.print_json(data=service.pipeline.stats())
            continue

        events = service.answer(
            question,
            session_id=args.session,
            user_id=args.user,
            mode=args.mode,
            stream=not args.no_stream,
        )
        result = _render_events(events, stream=not args.no_stream)
        _print_citations(result.get("citations", []))
        _print_usage(result.get("usage", {}))
        console.print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
