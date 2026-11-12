"""Prompt 管理 —— 模板化 + 热加载 + 版本号。

为什么把 prompt 从代码里抽出来:
- 面试官会问"你怎么迭代 prompt 的"——答案是:模板文件 + 版本号 + 评测集回归,
  改模板不用改代码、不用重启(FileSystemLoader auto_reload),评测脚本直接对比新旧版本;
- system 里带 `prompt-version`,写进 trace 后可以回答"这条回答是哪个版本的 prompt 产出的"。
"""

from __future__ import annotations

import re
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, TemplateError, Undefined

from app.config import PROJECT_ROOT

_VERSION_RE = re.compile(r"#\s*prompt-version:\s*([\w.-]+)")


class _SilentUndefined(Undefined):
    """模板变量缺失时渲染成空串而不是 500,保证线上 prompt 改动不会打挂服务。"""

    def __str__(self) -> str:  # pragma: no cover - 兜底路径
        return ""


class PromptManager:
    def __init__(self, templates_dir: Path | None = None, versions_dir: Path | None = None):
        self.templates_dir = templates_dir or PROJECT_ROOT / "app" / "prompts" / "templates"
        self.versions_dir = versions_dir or PROJECT_ROOT / "app" / "prompts" / "versions"
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            auto_reload=True,
            trim_blocks=True,
            lstrip_blocks=True,
            undefined=_SilentUndefined,
        )

    def _resolve(self, name: str) -> str:
        if name.endswith(".j2"):
            return name
        return f"{name}.j2" if (self.templates_dir / f"{name}.j2").exists() else name

    def render(self, name: str, **variables) -> str:
        try:
            return self.env.get_template(self._resolve(name)).render(**variables)
        except TemplateError as exc:
            raise RuntimeError(f"prompt 模板 {name} 渲染失败: {exc}") from exc

    def get_version(self, name: str) -> str:
        path = self.templates_dir / self._resolve(name)
        if not path.exists():
            return "?"
        match = _VERSION_RE.search(path.read_text(encoding="utf-8"))
        return match.group(1) if match else "?"

    def bump_version(self, name: str) -> str:
        path = self.templates_dir / self._resolve(name)
        text = path.read_text(encoding="utf-8")
        match = _VERSION_RE.search(text)
        old = match.group(1) if match else "0.0"
        major, _, minor = old.partition(".")
        new = f"{major}.{int(minor or 0) + 1}"
        snapshot = self.versions_dir / f"{Path(name).stem}-v{old}.j2"
        if not snapshot.exists():
            snapshot.write_text(text, encoding="utf-8")
        path.write_text(
            _VERSION_RE.sub(f"# prompt-version: {new}", text, count=1), encoding="utf-8"
        )
        return new


_manager: PromptManager | None = None


def get_prompt_manager() -> PromptManager:
    global _manager
    if _manager is None:
        _manager = PromptManager()
    return _manager


def reset_prompt_manager() -> None:
    global _manager
    _manager = None
