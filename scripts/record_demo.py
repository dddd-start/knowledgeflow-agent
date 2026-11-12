"""录制演示视频(WebM -> MP4 + GIF),把真实后端跑一遍完整链路。

依赖(仅录制用,不在运行依赖里):
    uv pip install playwright imageio-ffmpeg
    python -m playwright install chromium

用法:
    python scripts/record_demo.py                # 需要本地服务已在 8000 端口运行
    python scripts/record_demo.py --url http://127.0.0.1:8000 --out docs/assets

产出:
    docs/assets/demo.mp4  (H.264, 适合发微信/简历附件/GitHub 链接)
    docs/assets/demo.gif  (压缩版,README 里可直接自动播放)
    docs/assets/demo-poster.png (封面图)

说明:录制时会在页面里注入"光标 + 点击涟漪 + 字幕条 + 片头卡",
让没有旁白的录屏也能看懂每一步在做什么。这些注入只影响录屏,不修改项目代码。
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# 注入到页面里的录屏装饰:光标 / 点击涟漪 / 字幕 / 片头
# ---------------------------------------------------------------------------
OVERLAY_JS = r"""
(() => {
  if (window.__kfOverlay) return;
  window.__kfOverlay = true;

  function boot() {
    const css = document.createElement('style');
    css.textContent = `
      #__kf-cursor{position:fixed;left:50%;top:50%;width:26px;height:26px;margin:-13px 0 0 -13px;
        border-radius:50%;border:2px solid #8b9cff;background:rgba(139,156,255,.22);
        box-shadow:0 0 0 1px rgba(0,0,0,.4),0 6px 18px rgba(99,102,241,.45);
        z-index:2147483647;pointer-events:none;
        transition:width .12s,height .12s,margin .12s,background .12s}
      #__kf-cursor.is-down{width:18px;height:18px;margin:-9px 0 0 -9px;
        background:rgba(139,156,255,.5)}
      .__kf-ripple{position:fixed;width:14px;height:14px;margin:-7px 0 0 -7px;border-radius:50%;
        border:2px solid rgba(139,156,255,.9);z-index:2147483646;pointer-events:none;
        animation:__kf-rip .55s ease-out forwards}
      @keyframes __kf-rip{0%{transform:scale(.5);opacity:.95}100%{transform:scale(4.2);opacity:0}}
      #__kf-cap{position:fixed;left:50%;bottom:118px;transform:translateX(-50%);max-width:min(72vw,760px);
        padding:10px 18px;border-radius:999px;background:rgba(9,11,17,.92);color:#eef1ff;
        border:1px solid rgba(139,156,255,.28);box-shadow:0 10px 30px rgba(0,0,0,.5);
        font:500 15px/1.5 -apple-system,"PingFang SC","Microsoft YaHei",sans-serif;
        z-index:2147483645;pointer-events:none;opacity:0;transition:opacity .25s,transform .25s;
        white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
      #__kf-cap.show{opacity:1;transform:translateX(-50%) translateY(-4px)}
      #__kf-cap b{color:#a8b4ff;font-weight:700;margin-right:8px}
      #__kf-title{position:fixed;inset:0;display:grid;place-items:center;z-index:2147483644;
        background:radial-gradient(1200px 600px at 50% 30%,#141a2e 0%,#08090c 70%);
        color:#fff;
        font:700 40px/1.3 -apple-system,"PingFang SC","Microsoft YaHei",sans-serif;
        transition:opacity .45s ease;text-align:center}
      #__kf-title .sub{margin-top:14px;font-size:16px;font-weight:500;
        color:#9aa4c8;letter-spacing:.5px}
      #__kf-title .logo{width:64px;height:64px;border-radius:16px;margin:0 auto 22px;
        background:linear-gradient(135deg,#6366f1,#22d3ee);display:grid;place-items:center;
        font-size:30px;box-shadow:0 18px 50px rgba(99,102,241,.45)}
    `;
    document.head.appendChild(css);

    const cursor = document.createElement('div');
    cursor.id = '__kf-cursor';
    const cap = document.createElement('div');
    cap.id = '__kf-cap';
    const title = document.createElement('div');
    title.id = '__kf-title';
    title.innerHTML = '<div><div class="logo">⌘</div>KnowledgeFlow' +
      '<div class="sub">企业知识库智能问答 Agent · 演示</div></div>';
    document.documentElement.append(title, cursor, cap);

    window.__caption = (text) => {
      if (!text) { cap.classList.remove('show'); return; }
      const [num, ...rest] = String(text).split('|');
      cap.innerHTML = num ? '<b>' + num + '</b>' + rest.join('|') : rest.join('|');
      cap.classList.add('show');
    };
    window.__hideTitle = () => {
      title.style.opacity = '0';
      setTimeout(() => title.remove(), 800);
    };

    let x = innerWidth / 2, y = innerHeight / 2;
    const place = (cx, cy) => {
      x = cx; y = cy;
      cursor.style.left = x + 'px';
      cursor.style.top = y + 'px';
    };
    place(x, y);
    document.addEventListener('mousemove', (e) => place(e.clientX, e.clientY), true);
    document.addEventListener('mousedown', () => {
      cursor.classList.add('is-down');
      const rip = document.createElement('div');
      rip.className = '__kf-ripple';
      rip.style.left = x + 'px';
      rip.style.top = y + 'px';
      document.documentElement.appendChild(rip);
      setTimeout(() => rip.remove(), 600);
    }, true);
    document.addEventListener('mouseup', () => cursor.classList.remove('is-down'), true);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
"""

# 字幕脚本:(动作名, 字幕, 动作函数)
TYPE_DELAY = 55


class Recorder:
    def __init__(self, url: str, out_dir: Path, width: int, height: int) -> None:
        self.url = url
        self.out_dir = out_dir
        self.width = width
        self.height = height
        self.errors: list[str] = []

    # ---------------------------------------------------------------- 工具
    def caption(self, page, text: str) -> None:
        page.evaluate("t => window.__caption(t)", text)

    def assistant_count(self, page) -> int:
        return page.locator(".msg-assistant").count()

    def wait_turn_done(self, page, before: int, timeout: int = 180_000) -> None:
        """等到新一轮助手消息完成。

        注意:只有 RAG 路径是流式的,agent/agentic 模式是"跑完一次性给答案",
        所以不能等 .is-streaming 出现,要等 msg-foot(完成后的脚注)显示出来。
        """
        page.wait_for_function(
            """(n) => {
                const els = Array.from(document.querySelectorAll('.msg-assistant'));
                if (els.length <= n) return false;
                const last = els[els.length - 1];
                if (last.querySelector('.error-banner')) return true;   // 出错也结束等待
                const foot = last.querySelector('.msg-foot');
                return !!(foot && !foot.hasAttribute('hidden'));
            }""",
            arg=before,
            timeout=timeout,
        )

    def ask(self, page, text: str, pause_after: int = 1200) -> None:
        before = self.assistant_count(page)
        page.locator("#input").click()
        page.type("#input", text, delay=TYPE_DELAY)
        page.wait_for_timeout(350)
        page.keyboard.press("Enter")
        self.wait_turn_done(page, before)
        page.wait_for_timeout(pause_after)

    def scroll_panel(self, page, selector: str, dy: int, pause: int = 900) -> None:
        page.evaluate(
            "(args) => { const el = document.querySelector(args[0]);"
            " if (el) el.scrollBy({top: args[1], behavior: 'smooth'}); }",
            [selector, dy],
        )
        page.wait_for_timeout(pause)

    # ---------------------------------------------------------------- 主流程
    def script(self, page) -> None:
        page.wait_for_timeout(2000)
        page.evaluate("() => window.__hideTitle()")
        page.wait_for_timeout(1000)

        # ① 关键词问题:回答 + 引用 + 检索链路
        self.caption(page, "① 自动路由|问一句,看它走了哪条链路")
        chip = page.locator("#demoChips .chip").first
        before = self.assistant_count(page)
        chip.hover()
        chip.click()
        page.wait_for_timeout(700)
        self.wait_turn_done(page, before)
        page.wait_for_timeout(1800)

        # ② 引用联动:点 [1],右侧证据高亮
        self.caption(page, "② 引用可点|点 [1],右侧就是命中的原文片段")
        cite = page.locator(".msg-assistant .cite-ref").first
        cite.scroll_into_view_if_needed()
        cite.hover()
        cite.click()
        page.wait_for_timeout(2200)

        # ② 展开检索阶段:候选 / 耗时 / 命中片段
        self.caption(page, "② 链路透明|每个阶段:候选数 / 耗时 / 命中片段")
        stage = page.locator(".stage-head").nth(1)
        stage.scroll_into_view_if_needed()
        stage.hover()
        stage.click()
        page.wait_for_timeout(2600)

        # ③ 高质量模式:多跳问题 + 幻觉自检
        page.click('.mode-btn[data-mode="quality"]')
        page.wait_for_timeout(600)
        self.caption(page, "③ 高质量模式|改写 + 自检:回答前先核对证据")
        self.ask(page, "出差住宿标准是多少？超标的怎么审批？", pause_after=1600)
        page.evaluate(
            "() => document.querySelector('.msg-assistant:last-child')"
            "?.scrollIntoView({block:'center'})"
        )
        page.wait_for_timeout(1800)

        # ④ Agent 模式:模型自主选工具
        page.click("#newSessionBtn")
        page.wait_for_timeout(800)
        page.locator("#modalConfirm").click()
        page.wait_for_timeout(1000)
        page.click('.mode-btn[data-mode="agent"]')
        page.wait_for_timeout(500)
        self.caption(page, "④ Agent 模式|模型自己决定调哪个工具")
        self.ask(page, "现在几点了？另外帮我算一下 (1280+320)*0.85", pause_after=1200)
        self.caption(page, "④ 工具卡片|入参 / 输出 / 耗时,全程可见")
        self.scroll_panel(page, "#traceBody", 620, 800)
        page.locator(".tool-card").first.scroll_into_view_if_needed()
        page.mouse.move(1140, 520, steps=18)
        page.wait_for_timeout(2400)

        # ⑤ 知识库:文档 / 分块 / 搜索
        page.click("#tab-kb")
        page.wait_for_timeout(2400)
        self.caption(page, "⑤ 知识库|入库、搜索、分块都能看见")
        page.locator("#docList .doc-row").first.click()
        page.wait_for_timeout(2600)
        self.scroll_panel(page, "#kbDetail", 420, 1600)
        page.locator("#kbSearch").click()
        page.type("#kbSearch", "报销", delay=TYPE_DELAY)
        page.wait_for_timeout(1800)

        # ⑥ 关于页:技术栈 / 链路 / 开关状态
        page.click("#tab-about")
        page.wait_for_timeout(2600)
        self.caption(page, "⑥ 关于|技术栈、链路、开关状态一屏说清")
        self.scroll_panel(page, "#view-about", 540, 2200)
        self.scroll_panel(page, "#view-about", 560, 2200)
        page.wait_for_timeout(1200)

        # ⑦ 收尾
        page.click("#tab-chat")
        page.wait_for_timeout(1200)
        self.caption(page, "⑦ 打开即用|有 Key 走真实模型,没 Key 自动离线演示")
        page.mouse.move(700, 760, steps=14)
        page.wait_for_timeout(3200)
        self.caption(page, "")

    def run(self) -> Path:
        from playwright.sync_api import sync_playwright

        video_dir = Path("/tmp/kf-demo-video")
        shutil.rmtree(video_dir, ignore_errors=True)
        video_dir.mkdir(parents=True, exist_ok=True)

        with sync_playwright() as p:
            browser = p.chromium.launch(args=["--no-sandbox", "--force-device-scale-factor=1"])
            context = browser.new_context(
                viewport={"width": self.width, "height": self.height},
                record_video_dir=str(video_dir),
                record_video_size={"width": self.width, "height": self.height},
                device_scale_factor=1,
                locale="zh-CN",
            )
            context.add_init_script(OVERLAY_JS)
            page = context.new_page()
            page.on("pageerror", lambda e: self.errors.append(f"pageerror: {e}"))
            page.on(
                "console",
                lambda m: self.errors.append(f"console: {m.text}") if m.type == "error" else None,
            )
            page.goto(self.url, wait_until="domcontentloaded", timeout=60_000)
            page.wait_for_selector("#demoChips .chip", timeout=60_000)
            self.script(page)
            video = page.video
            page.close()
            context.close()
            browser.close()
            raw = Path(video.path()) if video else None
        if raw is None or not raw.exists():
            raise RuntimeError("未生成录屏文件")
        return raw


def find_ffmpeg() -> str | None:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        return shutil.which("ffmpeg")


def transcode(ffmpeg: str, raw: Path, out_dir: Path) -> tuple[Path, Path, Path]:
    mp4 = out_dir / "demo.mp4"
    gif = out_dir / "demo.gif"
    poster = out_dir / "demo-poster.png"

    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(raw),
            "-vf",
            "fps=30,scale=1440:-2:flags=lanczos",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(mp4),
        ],
        check=True,
        capture_output=True,
    )
    # README 里自动播放的 GIF:720 宽、12fps、调色板量化
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(mp4),
            "-vf",
            "fps=12,scale=720:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=192[p];[b][p]paletteuse=dither=bayer:bayer_scale=4",
            "-loop",
            "0",
            str(gif),
        ],
        check=True,
        capture_output=True,
    )
    # 封面:取第 6 秒
    subprocess.run(
        [ffmpeg, "-y", "-ss", "6", "-i", str(mp4), "-frames:v", "1", str(poster)],
        check=True,
        capture_output=True,
    )
    return mp4, gif, poster


def main() -> int:
    parser = argparse.ArgumentParser(description="录制 KnowledgeFlow 演示视频")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--out", default="docs/assets")
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=900)
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    recorder = Recorder(args.url, out_dir, args.width, args.height)
    print(f"==> 开始录制 {args.url}")
    raw = recorder.run()
    print(f"==> 原始录屏 {raw} ({raw.stat().st_size / 1e6:.1f}MB)")
    if recorder.errors:
        print("!! 页面错误:", recorder.errors[:5])

    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        print("!! 未找到 ffmpeg,保留 WebM 原始文件;可 pip install imageio-ffmpeg")
        dest = out_dir / "demo.webm"
        shutil.copy2(raw, dest)
        print(f"==> {dest}")
        return 0

    mp4, gif, poster = transcode(ffmpeg, raw, out_dir)
    for path in (mp4, gif, poster):
        print(f"==> {path}  {path.stat().st_size / 1e6:.1f}MB")
    print(f"==> 完成,耗时 {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
