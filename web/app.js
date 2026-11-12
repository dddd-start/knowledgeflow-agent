/*!
 * KnowledgeFlow · 企业知识库智能问答 —— 演示前端主逻辑
 * ---------------------------------------------------------------------------
 * 纯原生 JavaScript,无第三方依赖、无构建、无 CDN,单文件承载:
 *   1) 系统状态与模式切换(数据全部来自 /api/meta、/api/health);
 *   2) 对话:fetch + ReadableStream 解析 SSE,逐字流式渲染;
 *   3) 自写 Markdown 渲染器(先转义再生成标签,防 XSS);
 *   4) [n] 引用角标 ↔ 右侧证据卡片联动;
 *   5) 右侧「检索链路」:阶段时间轴、候选命中与分数条、证据、Agent 工具卡、自检结论、用量;
 *   6) 知识库视图:文档列表 / 搜索 / 分类 / 分块详情 / 上传 / 文本入库 / 全量重建;
 *   7) 关于视图:技术栈、架构链路、模式说明、运行指标与面试讲解要点。
 *
 * 代码组织:常量 → 工具函数 → SSE 解析器 → Markdown 渲染器 → HTTP 层
 *          → 全局状态 → 状态徽章/模式/演示问题 → 提示与弹层
 *          → 对话渲染 → 检索链路渲染 → 知识库视图 → 关于视图 → 启动与事件绑定。
 * 约定:所有来自后端/用户的内容一律先 esc() 转义;仅本文件自己生成的标签才写入 innerHTML。
 */
(function () {
  'use strict';

  // ===========================================================================
  // 0. 常量
  // ===========================================================================

  /** 后端接口(集中一处,便于对照 docs/API.md)。 */
  var API = {
    meta: '/api/meta',
    health: '/api/health',
    demo: '/api/demo/questions',
    prompts: '/api/prompts',
    chat: '/api/chat',
    kbDocuments: '/api/kb/documents',
    kbDocument: function (docId) {
      return '/api/kb/documents/' + encodeURIComponent(docId);
    },
    kbUpload: '/api/kb/upload',
    kbIngest: '/api/kb/ingest',
    kbReindex: function (strategy) {
      return '/api/kb/reindex?strategy=' + encodeURIComponent(strategy || 'recursive');
    },
    memory: function (userId, sessionId) {
      return '/api/memory/' + encodeURIComponent(userId) + '/' + encodeURIComponent(sessionId);
    }
  };

  /** 回答模式:按钮上的 key + 中文名 + 底部提示 + 关于页解释。 */
  var MODE_META = {
    auto: {
      name: '自动路由',
      hint: '自动路由 · 按意图选择最快路径',
      desc: '意图分类先判断问题类型(确定性工具 / 闲聊 / 知识库 / 复杂任务),再选成本最低且够用的路径。'
    },
    fast: {
      name: '混合检索快路径',
      hint: '快路径 · 向量 + BM25 双路召回 + RRF 融合',
      desc: '双路并发召回后直接融合生成,不调用改写 / HyDE / 自检,延迟最低,适合演示现场。'
    },
    quality: {
      name: '高质量模式',
      hint: '高质量 · 查询改写 + HyDE + 重排 + 幻觉自检',
      desc: '先改写与 HyDE 提升召回,再重排取 top 证据,最后自检并在必要时修订答案。'
    },
    agent: {
      name: 'Agent 工具循环',
      hint: 'Agent · 模型自主选择工具,多轮循环至收敛',
      desc: '模型根据问题自行决定调用 kb_search / kb_read 等工具,多轮循环,每轮结果都会展示在右侧。'
    },
    agentic: {
      name: '迭代检索',
      hint: '迭代检索 · 检索 → 判断证据是否充分 → 补检',
      desc: '每轮检索后由模型判断证据是否充分,不足则生成补检查询继续检索,直到足够或达到轮数上限。'
    }
  };

  /**
   * 按后端 /api/meta 的 retrieval 开关动态修正模式说明。
   * 默认配置里 HyDE / 重排是关的,界面不能写死"改写 + HyDE + 重排"误导观众。
   */
  function applyRetrievalMeta(meta) {
    var r = (meta && meta.retrieval) || null;
    if (!r) return;
    var steps = [];
    if (r.rewrite !== false) steps.push('查询改写');
    if (r.hyde) steps.push('HyDE');
    steps.push('混合检索');
    if (r.rerank) steps.push('重排');
    if (r.self_check !== false) steps.push('幻觉自检');
    MODE_META.quality.hint = '高质量 · ' + steps.join(' + ');
    MODE_META.quality.desc = '在快路径之上叠加 ' + steps.join('、') +
      (r.hyde || r.rerank ? '' : '(HyDE/重排默认关闭,可配置开启)') + ',更准但更慢。';
    updateModeHint();
  }

  /** 路由 path 的中文名(meta 事件里的 path 字段)。 */
  var PATH_META = {
    rag: 'RAG 检索',
    fast: '工具直达',
    chat: '闲聊',
    agent: 'Agent 循环',
    agentic: '迭代检索',
    auto: '自动'
  };

  /** 检索阶段:后端 key → 中文名 + CSS data-kind(配色圆点由 style.css 决定)。 */
  var STAGE_META = {
    rewrite: { label: '查询改写', kind: 'rewrite' },
    vector: { label: '向量召回', kind: 'vector' },
    bm25: { label: 'BM25 召回', kind: 'bm25' },
    hyde: { label: 'HyDE 假设文档', kind: 'hyde' },
    fuse: { label: 'RRF 融合', kind: 'fuse' },
    rerank: { label: '重排', kind: 'rerank' },
    retrieve: { label: '检索(合并计时)', kind: 'retrieve' },
    generate: { label: '答案生成', kind: 'generate' },
    self_check: { label: '幻觉自检', kind: 'self_check' },
    agent_loop: { label: 'Agent 工具循环', kind: 'agent_loop' },
    agentic_rag: { label: '迭代检索', kind: 'agentic_rag' },
    fast_tool: { label: '确定性工具直达', kind: 'fast_tool' }
  };

  /** 自检结论样式与文案。 */
  var VERDICT_META = {
    supported: {
      cls: 'verdict-supported',
      icon: 'i-check',
      title: '自检通过 · 结论均有证据支撑',
      badgeCls: 'badge-ok',
      badge: '自检 supported'
    },
    partial: {
      cls: 'verdict-partial',
      icon: 'i-alert',
      title: '部分支持 · 存在证据不足的表述',
      badgeCls: 'badge-warn',
      badge: '自检 partial'
    },
    unsupported: {
      cls: 'verdict-unsupported',
      icon: 'i-alert',
      title: '证据不足 · 回答可能不可靠',
      badgeCls: 'badge-error',
      badge: '自检 unsupported'
    }
  };

  /** 演示问题的类型标签(与评测集 type 字段对应)。 */
  var QUESTION_TYPE_META = {
    keyword: '关键词直球',
    paraphrase: '口语化改写',
    multi_hop: '跨段落多跳',
    noisy: '带噪声长问',
    complex_agent: '复杂任务'
  };

  /** 前端内置的本地会话 id,只是让 /api/memory 有个键,不做鉴权。 */
  var USER_ID = 'web-demo';
  var DEFAULT_MODES = ['auto', 'fast', 'quality', 'agent', 'agentic'];
  var FETCH_TIMEOUT = 15000;        // 普通 JSON 请求超时
  var UPLOAD_TIMEOUT = 120000;      // 上传 / 重建索引可能较慢
  var STREAM_IDLE_TIMEOUT = 120000; // SSE 无数据看门狗
  var RENDER_THROTTLE_MS = 80;      // 流式 Markdown 重渲染节流

  // ===========================================================================
  // 1. 工具函数
  // ===========================================================================

  function $(selector, root) {
    return (root || document).querySelector(selector);
  }

  function $$(selector, root) {
    return Array.prototype.slice.call((root || document).querySelectorAll(selector));
  }

  function byId(id) {
    return document.getElementById(id);
  }

  /**
   * HTML 转义 —— 整个文件的安全底线。
   * 先转义再拼标签:凡是来自后端 API、用户输入、模型输出的文本都必须过一遍。
   */
  function esc(value) {
    return String(value === null || value === undefined ? '' : value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  /** 内联 SVG 图标(图标名只来自本文件常量,仍做一次转义)。 */
  function icon(name) {
    return '<svg class="icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false">' +
      '<use href="#i-' + esc(name) + '"/></svg>';
  }

  var numberFormatter = typeof Intl !== 'undefined' && Intl.NumberFormat
    ? new Intl.NumberFormat('en-US')
    : null;

  function fmtNum(value) {
    if (value === null || value === undefined || isNaN(Number(value))) return '—';
    return numberFormatter ? numberFormatter.format(Number(value)) : String(value);
  }

  /** token 数:1000 以上用 k 缩写,避免右侧面板一行放不下。 */
  function fmtTokens(value) {
    var n = Number(value);
    if (!isFinite(n)) return '—';
    if (n < 1000) return String(n);
    return (n / 1000).toFixed(n < 10000 ? 2 : 1) + 'k';
  }

  function fmtMs(ms) {
    var n = Number(ms);
    if (!isFinite(n)) return '—';
    if (n < 1000) return Math.round(n) + 'ms';
    return (n / 1000).toFixed(2) + 's';
  }

  function fmtCost(usd) {
    var n = Number(usd);
    if (!isFinite(n)) return '—';
    if (n === 0) return '$0';
    return '$' + (n < 0.01 ? n.toFixed(6) : n.toFixed(4));
  }

  function fmtScore(score) {
    var n = Number(score);
    if (!isFinite(n)) return '—';
    if (Math.abs(n) >= 100) return n.toFixed(0);
    if (Math.abs(n) >= 10) return n.toFixed(1);
    return n.toFixed(3);
  }

  function fmtUptime(seconds) {
    var s = Math.max(0, Math.round(Number(seconds) || 0));
    var d = Math.floor(s / 86400);
    var h = Math.floor((s % 86400) / 3600);
    var m = Math.floor((s % 3600) / 60);
    if (d > 0) return d + '天' + h + '小时';
    if (h > 0) return h + '小时' + m + '分';
    if (m > 0) return m + '分' + Math.floor(s % 60) + '秒';
    return s + '秒';
  }

  /** 模型名太长时只保留 / 之后的部分,例如 deepseek/deepseek-v4.1-flash → deepseek-v4.1-flash。 */
  function shortModel(model) {
    var text = String(model || '').trim();
    if (!text) return '未知';
    var parts = text.split('/');
    return parts[parts.length - 1] || text;
  }

  function clamp(value, lo, hi) {
    return Math.min(hi, Math.max(lo, value));
  }

  function isReducedMotion() {
    return typeof window !== 'undefined' && typeof window.matchMedia === 'function' &&
      window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  }

  /** 日期时间(右侧请求卡用),固定 zh-CN 便于讲解。 */
  function fmtClock(date) {
    try {
      return date.toLocaleTimeString('zh-CN', { hour12: false });
    } catch (err) {
      return '';
    }
  }

  /** 把任意对象序列化成可读 JSON(复制 trace / 工具输出用),失败时降级为 String()。 */
  function safeJson(value, space) {
    try {
      return JSON.stringify(value, null, space === undefined ? 2 : space);
    } catch (err) {
      return String(value);
    }
  }

  /** debounce:搜索框输入用。 */
  function debounce(fn, wait) {
    var timer = null;
    return function () {
      var args = arguments;
      var self = this;
      if (timer) clearTimeout(timer);
      timer = setTimeout(function () {
        timer = null;
        fn.apply(self, args);
      }, wait || 120);
    };
  }

  /** 复制文本:优先 Clipboard API,非安全上下文(局域网 http)回退 execCommand。 */
  function copyText(text) {
    var value = String(text === null || text === undefined ? '' : text);
    if (navigator.clipboard && window.isSecureContext) {
      return navigator.clipboard.writeText(value);
    }
    return new Promise(function (resolve, reject) {
      var area = document.createElement('textarea');
      area.value = value;
      area.setAttribute('readonly', 'readonly');
      area.style.position = 'fixed';
      area.style.top = '-1000px';
      area.style.opacity = '0';
      document.body.appendChild(area);
      area.select();
      var ok = false;
      try {
        ok = document.execCommand('copy');
      } catch (err) {
        ok = false;
      }
      document.body.removeChild(area);
      if (ok) resolve();
      else reject(new Error('copy 命令不可用'));
    });
  }

  /** 带提示的复制。 */
  function copyWithToast(text, successMessage) {
    return copyText(text).then(function () {
      toast(successMessage || '已复制到剪贴板', 'success');
    }).catch(function () {
      toast('复制失败,请手动选中文本复制', 'error');
    });
  }

  function scrollBehavior() {
    return isReducedMotion() ? 'auto' : 'smooth';
  }

  // ===========================================================================
  // 2. SSE 解析器
  // ===========================================================================
  //
  // 按 text/event-stream 规范解析:
  //   - 事件以「空行」分隔;
  //   - 一个事件可以有多行 `data:`(用 \n 拼接后交给 JSON.parse);
  //   - `:` 开头的行是注释,忽略(可用于心跳);
  //   - 未知字段(id / retry / 未知 event)不报错,直接忽略;
  //   - 兼容 \r\n / \r 换行,且块边界可能落在任意位置(网络分片)。
  //
  // 这样做的目的:后端把 JSON 放在单行 `data:` 里,但协议层要足够健壮,
  // 断线重连、代理追加注释、多行 JSON 都不会让前端解析错乱。
  // ---------------------------------------------------------------------------

  function createSSEParser(options) {
    var opts = options || {};
    var onEvent = typeof opts.onEvent === 'function' ? opts.onEvent : function () {};
    var onComment = typeof opts.onComment === 'function' ? opts.onComment : function () {};

    var buffer = '';
    var eventName = 'message';
    var dataLines = [];
    var lastEventId = '';

    function dispatch() {
      if (!dataLines.length) {
        eventName = 'message';
        return;
      }
      var raw = dataLines.join('\n');
      var data = raw;
      if (raw) {
        try {
          data = JSON.parse(raw);
        } catch (err) {
          // 不是 JSON 就原样交给上层(便于排查后端协议变更)
          data = { text: raw };
        }
      }
      var name = eventName || 'message';
      eventName = 'message';
      dataLines = [];
      onEvent(name, data, raw);
    }

    function handleLine(line) {
      if (line === '') {
        dispatch();
        return;
      }
      if (line.charAt(0) === ':') {
        onComment(line.slice(1));
        return;
      }
      var idx = line.indexOf(':');
      var field = idx === -1 ? line : line.slice(0, idx);
      var value = idx === -1 ? '' : line.slice(idx + 1);
      if (value.charAt(0) === ' ') value = value.slice(1);
      if (field === 'event') {
        eventName = value || 'message';
      } else if (field === 'data') {
        dataLines.push(value);
      } else if (field === 'id') {
        lastEventId = value;
      }
      // retry 等字段忽略即可
    }

    return {
      /** 喂入一段网络分片。 */
      feed: function (chunk) {
        if (!chunk) return;
        buffer += chunk;
        // 统一换行;末尾孤立 \r 先留着,等下一片到达再判断是否为 \r\n
        buffer = buffer.replace(/\r\n/g, '\n').replace(/\r(?!$)/g, '\n');
        var lines = buffer.split('\n');
        buffer = lines.pop();
        for (var i = 0; i < lines.length; i += 1) handleLine(lines[i]);
      },
      /** 流结束:处理残留行,并把尚未派发的事件补发一次。 */
      flush: function () {
        if (buffer) {
          handleLine(buffer);
          buffer = '';
        }
        dispatch();
      },
      lastEventId: function () {
        return lastEventId;
      }
    };
  }

  // ===========================================================================
  // 3. Markdown 渲染器(自写,无第三方库)
  // ===========================================================================
  //
  // 安全模型:
  //   1) 先把所有文本做 HTML 转义(esc),任何原始字符都不可能变成标签;
  //   2) 只在转义后的文本上识别 Markdown 标记,由本文件拼出固定白名单标签
  //      (p/h1-6/ul/ol/li/strong/em/del/code/pre/table/blockquote/a/button);
  //   3) 链接 href 再过 sanitizeUrl,只允许 http(s)/mailto/站内相对路径,
  //      javascript: / data: 一律降级为纯文本;
  //   4) 行内代码先抽成占位符,避免代码里的 * _ [n] 被误解析。
  //
  // 流式友好:未闭合的代码围栏会一直渲染到文本末尾(能边流边显示),
  // 表格 / 列表在文本还没写完时也尽可能按已到达的行渲染。
  // ---------------------------------------------------------------------------

  function sanitizeUrl(raw) {
    var url = String(raw || '').trim();
    if (!url) return '';
    if (/^(https?:|mailto:)/i.test(url)) return url;
    if (/^\/[^/]/.test(url)) return url;  // 站内相对路径
    if (/^#[\w-]*$/.test(url)) return url; // 页内锚点
    return '';
  }

  function renderInline(raw) {
    var text = String(raw === null || raw === undefined ? '' : raw);

    // (1) 行内代码先抽走:placeholder 使用不可见控制字符,不会与正文冲突
    var codes = [];
    text = text.replace(/`([^`\n]+)`/g, function (match, code) {
      codes.push(code);
      return '\u0000C' + (codes.length - 1) + '\u0000';
    });

    // (2) 转义 —— 从这里开始,所有字符都只是文本
    text = esc(text);

    // (3) 图片语法:演示环境不允许外链图片,只保留 alt 文本(避免出现裸 ! 与残留括号)
    text = text.replace(/!\[([^\]\n]*)\]\(((?:[^()\s]|\([^()\s]*\))+)\)/g, function (match, alt) {
      return alt;
    });

    // (4) 链接 [文字](href);href 允许一层成对括号,防止 javascript:alert(1) 的括号把语法切坏
    text = text.replace(/\[([^\]\n]+)\]\(((?:[^()\s]|\([^()\s]*\))+)\)/g, function (match, label, href) {
      var safe = sanitizeUrl(href);
      if (!safe) return label;
      // 注意:href 取自已经转义过的文本,这里不能再转义一次(否则 & 会变成 &amp;amp;)
      return '<a href="' + safe + '" target="_blank" rel="noopener noreferrer">' +
        label + '</a>';
    });

    // (5) 加粗 / 斜体 / 删除线(先后顺序很重要:先粗后斜)
    text = text.replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>');
    text = text.replace(/__([^_\n]+)__/g, '<strong>$1</strong>');
    text = text.replace(/(^|[\s(（\[{>，。；:：、!?！？])\*([^*\n]+)\*(?=$|[\s)）\]}，。；:：、!?！？])/g,
      '$1<em>$2</em>');
    text = text.replace(/~~([^~\n]+)~~/g, '<del>$1</del>');

    // (6) 引用角标 [n] → 可点击按钮(点击后由事件委托定位右侧证据卡)
    text = text.replace(/\[(\d{1,2})\]/g, function (match, num) {
      return '<button type="button" class="cite-ref" data-cite="' + num +
        '" title="引用 [' + num + ']" aria-label="查看引用 ' + num + '">' + num + '</button>';
    });

    // (7) 还原行内代码(内容同样转义)
    text = text.replace(/\u0000C(\d+)\u0000/g, function (match, index) {
      return '<code>' + esc(codes[Number(index)] || '') + '</code>';
    });

    // (8) 段内换行 → <br>,保留模型的排版意图
    text = text.replace(/\n/g, '<br>');
    return text;
  }

  function codeBlockHTML(lang, code) {
    return '<div class="code-block">' +
      '<div class="code-head"><span>' + esc(lang || 'text') + '</span>' +
      '<button type="button" class="code-copy" data-code-copy="1" aria-label="复制代码">复制</button></div>' +
      '<pre><code>' + esc(code) + '</code></pre>' +
      '</div>';
  }

  function indentWidth(line) {
    return line.replace(/\t/g, '    ').length - line.replace(/^\s*/, '').length;
  }

  var LIST_ITEM_RE = /^(\s*)([-*+]|\d{1,3}[.)])\s+([\s\S]*)$/;

  function listMatch(line) {
    var m = LIST_ITEM_RE.exec(line);
    if (!m) return null;
    return {
      indent: indentWidth(m[1]),
      ordered: /^\d/.test(m[2]),
      text: m[3]
    };
  }

  function isListLine(line) {
    return LIST_ITEM_RE.test(line);
  }

  function splitRow(line) {
    var text = line.trim();
    if (text.charAt(0) === '|') text = text.slice(1);
    if (text.charAt(text.length - 1) === '|') text = text.slice(0, -1);
    return text.split('|');
  }

  function isTableStart(lines, index) {
    if (index + 1 >= lines.length) return false;
    var head = lines[index];
    if (head.indexOf('|') === -1) return false;
    var sep = lines[index + 1];
    if (!sep || sep.indexOf('|') === -1) return false;
    var headCells = splitRow(head);
    var sepCells = splitRow(sep);
    if (!headCells.length || sepCells.length !== headCells.length) return false;
    for (var i = 0; i < sepCells.length; i += 1) {
      if (!/^\s*:?-{2,}:?\s*$/.test(sepCells[i])) return false;
    }
    return true;
  }

  function alignAttr(align) {
    return align ? ' style="text-align:' + align + '"' : '';
  }

  function renderTable(lines, start, out) {
    var headCells = splitRow(lines[start]);
    var sepCells = splitRow(lines[start + 1]);
    var aligns = sepCells.map(function (cell) {
      var text = cell.trim();
      if (/^:-+:$/.test(text)) return 'center';
      if (/^:-+$/.test(text)) return 'left';
      if (/^-+:$/.test(text)) return 'right';
      return '';
    });

    var html = '<div class="table-wrap"><table><thead><tr>';
    headCells.forEach(function (cell, index) {
      html += '<th' + alignAttr(aligns[index]) + '>' + renderInline(cell.trim()) + '</th>';
    });
    html += '</tr></thead><tbody>';

    var i = start + 2;
    while (i < lines.length && lines[i].indexOf('|') !== -1 && !/^\s*$/.test(lines[i])) {
      var row = splitRow(lines[i]);
      html += '<tr>';
      for (var c = 0; c < headCells.length; c += 1) {
        html += '<td' + alignAttr(aligns[c]) + '>' +
          renderInline((row[c] || '').trim()) + '</td>';
      }
      html += '</tr>';
      i += 1;
    }
    html += '</tbody></table></div>';
    out.push(html);
    return i;
  }

  /** 解析一个列表块(支持缩进嵌套、无序/有序混排),返回 HTML 与下一行下标。 */
  function parseListBlock(lines, start) {
    var first = listMatch(lines[start]);
    if (!first) return { html: '', next: start + 1 };
    var baseIndent = first.indent;
    var ordered = first.ordered;
    var items = [];
    var i = start;

    while (i < lines.length) {
      var line = lines[i];
      if (/^\s*$/.test(line)) {
        // 空行:若后面还有同级或更深层的列表项,视为 loose list 继续
        var j = i + 1;
        while (j < lines.length && /^\s*$/.test(lines[j])) j += 1;
        var peek = j < lines.length ? listMatch(lines[j]) : null;
        if (peek && peek.indent >= baseIndent) {
          i = j;
          continue;
        }
        break;
      }

      var item = listMatch(line);
      if (!item) break;
      if (item.indent < baseIndent) break;

      if (item.indent > baseIndent) {
        if (!items.length) break;
        var sub = parseListBlock(lines, i);
        items[items.length - 1].children += sub.html;
        i = sub.next;
        continue;
      }

      if (item.ordered !== ordered) break;

      items.push({ text: item.text, children: '' });
      i += 1;

      // 收集当前项的续行(缩进更深、且不是新的列表项)
      while (i < lines.length) {
        var cont = lines[i];
        if (/^\s*$/.test(cont)) break;
        if (listMatch(cont)) break;
        if (indentWidth(cont) <= baseIndent) break;
        items[items.length - 1].text += '\n' + cont.trim();
        i += 1;
      }
    }

    if (!items.length) return { html: '', next: Math.max(start + 1, i) };

    var tag = ordered ? 'ol' : 'ul';
    var html = '<' + tag + '>' + items.map(function (entry) {
      return '<li>' + renderInline(entry.text) + entry.children + '</li>';
    }).join('') + '</' + tag + '>';
    return { html: html, next: i };
  }

  function isBlockStart(lines, index) {
    var line = lines[index];
    if (/^\s*$/.test(line)) return true;
    if (/^\s*(```+|~~~+)/.test(line)) return true;
    if (/^\s{0,3}#{1,6}\s+/.test(line)) return true;
    if (/^\s{0,3}([-*_])(?:\s*\1){2,}\s*$/.test(line)) return true;
    if (/^\s{0,3}>/.test(line)) return true;
    if (isListLine(line)) return true;
    if (isTableStart(lines, index)) return true;
    return false;
  }

  /**
   * 主入口:把 Markdown 文本渲染成 HTML 字符串。
   * 支持:标题(#~######)、段落、无序/有序/嵌套列表、引用、表格、分隔线、
   *       加粗、斜体、删除线、行内代码、围栏代码块、链接、[n] 引用角标。
   */
  function renderMarkdown(source) {
    var text = String(source === null || source === undefined ? '' : source)
      .replace(/\r\n?/g, '\n');
    var lines = text.split('\n');
    var out = [];
    var i = 0;

    while (i < lines.length) {
      var line = lines[i];

      if (/^\s*$/.test(line)) {
        i += 1;
        continue;
      }

      // ---- 围栏代码块(未闭合时渲染到结尾,兼容流式) ----
      var fence = /^\s*(```+|~~~+)\s*([\w+#.\-]*)\s*$/.exec(line);
      if (fence) {
        var marker = fence[1].charAt(0);
        var markerLen = fence[1].length;
        var guard = new RegExp('^\\s*' + (marker === '`' ? '`' : '~') +
          '{' + markerLen + ',}\\s*$');
        var codeLines = [];
        i += 1;
        while (i < lines.length) {
          if (guard.test(lines[i])) {
            i += 1;
            break;
          }
          codeLines.push(lines[i]);
          i += 1;
        }
        out.push(codeBlockHTML(fence[2], codeLines.join('\n')));
        continue;
      }

      // ---- 标题 ----
      var heading = /^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$/.exec(line);
      if (heading) {
        var level = heading[1].length;
        out.push('<h' + level + '>' + renderInline(heading[2]) + '</h' + level + '>');
        i += 1;
        continue;
      }

      // ---- 分隔线 ----
      if (/^\s{0,3}([-*_])(?:\s*\1){2,}\s*$/.test(line)) {
        out.push('<hr>');
        i += 1;
        continue;
      }

      // ---- 表格 ----
      if (isTableStart(lines, i)) {
        i = renderTable(lines, i, out);
        continue;
      }

      // ---- 引用(递归渲染内部块) ----
      if (/^\s{0,3}>/.test(line)) {
        var quoteLines = [];
        while (i < lines.length && /^\s{0,3}>/.test(lines[i])) {
          quoteLines.push(lines[i].replace(/^\s{0,3}>\s?/, ''));
          i += 1;
        }
        out.push('<blockquote>' + renderMarkdown(quoteLines.join('\n')) + '</blockquote>');
        continue;
      }

      // ---- 列表 ----
      if (isListLine(line)) {
        var listBlock = parseListBlock(lines, i);
        if (listBlock.next > i) {
          if (listBlock.html) out.push(listBlock.html);
          i = listBlock.next;
        } else {
          i += 1; // 防御:解析器没有前进时强制推进,避免死循环
        }
        continue;
      }

      // ---- 段落 ----
      var paragraph = [];
      while (i < lines.length) {
        if (isBlockStart(lines, i)) break;
        paragraph.push(lines[i].trim());
        i += 1;
      }
      if (paragraph.length) {
        out.push('<p>' + renderInline(paragraph.join('\n')) + '</p>');
      } else {
        i += 1;
      }
    }

    return out.join('');
  }

  // ===========================================================================
  // 4. HTTP 层(JSON 请求 + SSE 流)
  // ===========================================================================

  class ApiError extends Error {
    constructor(message, status) {
      super(message);
      this.name = 'ApiError';
      this.status = status || 0;
    }
  }

  function extractError(data, status) {
    if (data) {
      const detail = data.detail !== undefined ? data.detail : data.message;
      if (typeof detail === 'string' && detail) return detail;
      if (detail) return safeJson(detail, 0);
    }
    if (status === 404) return 'HTTP 404(接口或资源不存在)';
    if (status === 422) return 'HTTP 422 请求参数不合法';
    return 'HTTP ' + status + ' 请求失败';
  }

  /** 普通 JSON 请求:统一超时、统一错误文案。 */
  async function httpJSON(url, options, timeoutMs) {
    const opts = options || {};
    const timeout = timeoutMs || FETCH_TIMEOUT;
    const controller = new AbortController();
    let timedOut = false;
    const timer = setTimeout(function () {
      timedOut = true;
      controller.abort();
    }, timeout);
    const headers = Object.assign({ Accept: 'application/json' }, opts.headers || {});
    try {
      const response = await fetch(url, Object.assign({}, opts, {
        headers: headers,
        signal: controller.signal
      }));
      const text = await response.text();
      let data = null;
      if (text) {
        try {
          data = JSON.parse(text);
        } catch (err) {
          data = null;
        }
      }
      if (!response.ok) throw new ApiError(extractError(data, response.status), response.status);
      if (data === null) throw new ApiError('服务返回了空响应(HTTP ' + response.status + ')', response.status);
      return data;
    } catch (err) {
      if (err instanceof ApiError) throw err;
      if (timedOut) throw new ApiError('请求超时(>' + Math.round(timeout / 1000) + 's),服务可能正忙', 0);
      if (err && err.name === 'AbortError') throw err;
      throw new ApiError('网络请求失败:' + ((err && err.message) || '无法连接服务'), 0);
    } finally {
      clearTimeout(timer);
    }
  }

  function isAbortError(err) {
    return !!err && (err.name === 'AbortError' || err.code === 20);
  }

  /**
   * 发起流式问答:fetch + ReadableStream。
   * 返回 Promise;事件通过 handlers.onEvent(name, data, raw) 逐个交给上层。
   * 超时用「无数据看门狗」而不是总时长(生成长答案本来就需要时间)。
   */
  async function streamAnswer(payload, handlers) {
    const controller = new AbortController();
    let idleFired = false;
    let idleTimer = null;

    function armIdle() {
      if (idleTimer) clearTimeout(idleTimer);
      idleTimer = setTimeout(function () {
        idleFired = true;
        controller.abort();
      }, STREAM_IDLE_TIMEOUT);
    }

    state.abort = {
      abort: function () {
        controller.abort();
      }
    };
    armIdle();

    try {
      const response = await fetch(API.chat, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
        body: JSON.stringify(payload),
        signal: controller.signal
      });

      const contentType = response.headers.get('content-type') || '';
      if (!response.ok || contentType.indexOf('text/event-stream') === -1) {
        // 非流式响应通常是参数错误(FastAPI 422)或服务异常,尽量把 detail 提取出来
        const raw = await response.text();
        let detail = '';
        try {
          detail = extractError(JSON.parse(raw), response.status);
        } catch (err) {
          detail = (raw || '').slice(0, 300) || ('HTTP ' + response.status);
        }
        throw new ApiError(detail, response.status);
      }
      if (!response.body) throw new ApiError('当前浏览器不支持流式响应(ReadableStream)', 0);

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      const parser = createSSEParser({
        onEvent: function (name, data, raw) {
          armIdle();
          handlers.onEvent(name, data, raw);
        }
      });

      while (true) {
        const step = await reader.read();
        if (step.done) break;
        armIdle();
        parser.feed(decoder.decode(step.value, { stream: true }));
      }
      parser.feed(decoder.decode());
      parser.flush();
    } catch (err) {
      if (idleFired) {
        throw new ApiError('生成超时(>' + Math.round(STREAM_IDLE_TIMEOUT / 1000) + 's 无数据),已中断', 0);
      }
      throw err;
    } finally {
      if (idleTimer) clearTimeout(idleTimer);
      state.abort = null;
    }
  }

  // ===========================================================================
  // 5. 全局状态与节点缓存
  // ===========================================================================

  function randomId(length) {
    const chars = 'abcdefghijkmnpqrstuvwxyz23456789';
    let out = '';
    for (let i = 0; i < (length || 6); i += 1) {
      out += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    return out;
  }

  var state = {
    meta: null,
    health: null,
    prompts: null,
    modes: DEFAULT_MODES.slice(),
    mode: 'auto',
    userId: USER_ID,
    sessionId: 'web-' + randomId(4),
    demoQuestions: [],
    streaming: false,
    abort: null,
    renderTimer: null,
    currentTurn: null,
    traceTurn: null,      // 右侧面板当前对应的那一轮
    citationsTurn: null,  // 右侧证据区当前展示的是哪一轮的引用
    turns: {},
    turnSeq: 0,
    lastTrace: null,
    memoryTurns: 0,
    kb: {
      loaded: false,
      loading: false,
      total: 0,
      chunks: 0,
      categories: {},
      documents: [],
      filterCat: '',
      filterText: '',
      selectedId: '',
      detail: null
    },
    aboutLoaded: false,
    aboutError: '',
    offlineNoticeShown: false
  };

  var nodeCache = {};

  /** 按 id 取节点并缓存(defer 脚本,启动时 DOM 已就绪)。 */
  function node(id) {
    if (!nodeCache[id]) nodeCache[id] = document.getElementById(id);
    return nodeCache[id];
  }

  // ===========================================================================
  // 6. 顶部状态徽章 / 模式切换 / 演示问题 / 会话标签
  // ===========================================================================

  function pillHTML(opts) {
    let cls = 'pill';
    if (opts.state === 'warn') cls += ' is-warn';
    if (opts.state === 'offline') cls += ' is-offline';
    if (opts.optional) cls += ' pill-optional';
    const title = opts.title ? ' title="' + esc(opts.title) + '"' : '';
    const label = opts.label ? '<span class="pill-label">' + esc(opts.label) + '</span>' : '';
    const value = opts.value ? '<span class="pill-value">' + esc(opts.value) + '</span>' : '';
    return '<span class="' + cls + '"' + title + '><span class="dot"></span>' + label + value + '</span>';
  }

  function renderStatusBadges() {
    const host = node('statusBadges');
    if (!host) return;
    const meta = state.meta;
    const health = state.health;

    // 启动阶段先占位,避免页面看起来是空的
    if (!meta && !health) {
      host.innerHTML = '<span class="pill skeleton-pill" aria-hidden="true"></span>' +
        '<span class="pill skeleton-pill" aria-hidden="true"></span>';
      return;
    }

    const pills = [];
    const llm = (meta && meta.llm) || {};
    const offline = !!(meta && meta.offline_demo);
    const degraded = !!llm.degraded;

    if (meta) {
      pills.push(pillHTML({
        label: 'LLM',
        value: offline ? '离线演示' : (degraded ? '已降级' : shortModel(llm.model)),
        state: offline ? 'offline' : (degraded ? 'warn' : 'ok'),
        title: '后端 ' + (llm.backend || '?') + ' · 模型 ' + (llm.model || '?')
      }));
      if (meta.embedding) {
        pills.push(pillHTML({
          label: 'Embedding',
          value: shortModel(meta.embedding.model) + (meta.embedding.dim ? ' · ' + meta.embedding.dim + 'd' : ''),
          title: '向量后端 ' + (meta.embedding.backend || '?') + ' · 维度 ' + (meta.embedding.dim || '?'),
          optional: true
        }));
      }
      if (meta.kb) {
        pills.push(pillHTML({
          label: 'KB',
          value: fmtNum(meta.kb.documents) + ' 篇 / ' + fmtNum(meta.kb.chunks) + ' 块',
          title: '向量库 ' + (meta.vector_store || '?') + ' · BM25 ' + (meta.kb.bm25 ? '开' : '关') +
            ' · 重排 ' + (meta.kb.rerank ? '开' : '关'),
          optional: true
        }));
      }
      if (offline || degraded) {
        pills.push(pillHTML({
          label: '模式',
          value: offline ? 'offline_demo' : 'degraded',
          state: offline ? 'offline' : 'warn',
          title: offline
            ? '后端处于离线演示模式:LLM 使用内置模拟输出,回答不代表真实模型能力'
            : 'LLM 调用失败后自动降级,部分回答可能来自模拟或缓存'
        }));
      }
    }
    if (health) {
      pills.push(pillHTML({
        label: 'uptime',
        value: fmtUptime(health.uptime_s),
        title: '服务 v' + (health.version || '?') + ' · 状态 ' + (health.status || '?'),
        optional: true
      }));
    }
    host.innerHTML = pills.join('');
  }

  /** 离线 / 降级要显著提示:在欢迎区插入一条常驻说明。 */
  function renderOfflineNotice() {
    const meta = state.meta;
    if (!meta || state.offlineNoticeShown) return;
    const offline = !!meta.offline_demo;
    const degraded = !!(meta.llm && meta.llm.degraded);
    if (!offline && !degraded) return;

    const welcome = node('welcome');
    if (!welcome) return;
    const note = document.createElement('div');
    note.className = 'note-line';
    note.setAttribute('role', 'status');
    note.innerHTML = icon('alert') + '<span>' + (offline
      ? '当前为离线演示模式(offline_demo):回答由内置模拟生成,不消耗真实模型额度,也不代表真实模型能力。'
      : 'LLM 已降级(degraded):部分回答可能来自模拟或缓存,真实调用恢复后会自动切回。') + '</span>';
    const demoBlock = welcome.querySelector('.demo-block');
    welcome.insertBefore(note, demoBlock || null);
    state.offlineNoticeShown = true;
  }

  function currentModeMeta(mode) {
    return MODE_META[mode] || { name: mode, hint: mode, desc: '' };
  }

  function renderModes() {
    const host = node('modeSegment');
    if (!host) return;
    // 重渲染会丢焦点:若用户正用键盘操作模式按钮,渲染后把焦点还回去
    const focusedMode = document.activeElement && document.activeElement.getAttribute
      ? document.activeElement.getAttribute('data-mode')
      : null;
    host.innerHTML = state.modes.map(function (mode) {
      const info = currentModeMeta(mode);
      const active = mode === state.mode;
      return '<button type="button" class="mode-btn' + (active ? ' is-active' : '') + '"' +
        ' data-mode="' + esc(mode) + '" aria-pressed="' + (active ? 'true' : 'false') + '"' +
        ' title="' + esc(info.name + ' — ' + info.hint) + '">' + esc(mode) + '</button>';
    }).join('');
    if (focusedMode) {
      const button = host.querySelector('.mode-btn[data-mode="' + focusedMode + '"]');
      if (button) button.focus();
    }
    updateModeHint();
  }

  function updateModeHint() {
    const hint = node('modeHint');
    if (hint) hint.textContent = currentModeMeta(state.mode).hint;
  }

  function setMode(mode) {
    if (!mode) return;
    if (state.modes.indexOf(mode) === -1) return;
    state.mode = mode;
    renderModes();
  }

  function renderSessionLabel() {
    const label = node('sessionLabel');
    if (!label) return;
    const turns = state.memoryTurns;
    label.textContent = '会话 ' + state.sessionId + (turns > 0 ? ' · ' + turns + ' 轮记忆' : ' · 新会话');
    label.setAttribute('title', 'user_id=' + state.userId + ',session_id=' + state.sessionId);
  }

  function questionTypeLabel(type) {
    return QUESTION_TYPE_META[type] || (type || '演示');
  }

  function chipHTML(question) {
    const type = question.type || '';
    return '<button type="button" class="chip" data-type="' + esc(type) + '"' +
      ' data-question="' + esc(question.text) + '" role="listitem"' +
      ' title="' + esc(question.note || question.text) + '">' +
      '<span class="chip-type">' + esc(questionTypeLabel(type)) + '</span>' +
      '<span>' + esc(question.text) + '</span></button>';
  }

  function renderDemoQuestions(questions) {
    state.demoQuestions = questions || [];
    const chips = node('demoChips');
    const hint = node('demoHint');
    const popoverBody = node('demoPopoverBody');

    if (!state.demoQuestions.length) {
      if (chips) chips.innerHTML = '';
      if (hint) hint.textContent = '暂无演示问题(评测集未就绪)';
      if (popoverBody) {
        popoverBody.innerHTML = '<div class="popover-group"><div class="popover-group-title">提示</div>' +
          '<div class="popover-item" style="cursor:default">/api/demo/questions 没有返回题目,' +
          '可以直接在输入框里提问。<small>服务端会从 data/eval/eval_queries.json 取题</small></div></div>';
      }
      return;
    }

    if (hint) hint.textContent = state.demoQuestions.length + ' 个 · 取自评测集,点了就能答上来';
    if (chips) chips.innerHTML = state.demoQuestions.map(chipHTML).join('');

    // 弹层按题型分组,面试时可以按「关键词 → 口语化 → 多跳」逐步加大难度
    if (popoverBody) {
      const groups = {};
      const order = [];
      state.demoQuestions.forEach(function (question) {
        const key = question.type || 'other';
        if (!groups[key]) {
          groups[key] = [];
          order.push(key);
        }
        groups[key].push(question);
      });
      popoverBody.innerHTML = order.map(function (type) {
        const items = groups[type].map(function (question) {
          return '<button type="button" class="popover-item" data-question="' + esc(question.text) + '">' +
            esc(question.text) +
            (question.note ? '<small>' + esc(question.note) + '</small>' : '') +
            '</button>';
        }).join('');
        return '<div class="popover-group"><div class="popover-group-title">' +
          esc(questionTypeLabel(type)) + '</div>' + items + '</div>';
      }).join('');
    }
  }

  function hideDemoPopover() {
    const popover = node('demoPopover');
    const button = node('demoBtn');
    if (popover) popover.hidden = true;
    if (button) button.setAttribute('aria-expanded', 'false');
  }

  function toggleDemoPopover() {
    const popover = node('demoPopover');
    const button = node('demoBtn');
    if (!popover || !button) return;
    const willOpen = popover.hidden;
    popover.hidden = !willOpen;
    button.setAttribute('aria-expanded', willOpen ? 'true' : 'false');
    if (willOpen) {
      const first = popover.querySelector('.popover-item');
      if (first && first.focus) first.focus();
    }
  }

  // ===========================================================================
  // 7. Toast 与确认弹层
  // ===========================================================================

  function toast(message, type) {
    const host = node('toasts');
    if (!host) return;
    const el = document.createElement('div');
    el.className = 'toast' + (type ? ' toast-' + type : '');
    el.setAttribute('role', 'status');
    const iconName = type === 'success' ? 'i-check' : type === 'error' ? 'i-alert' : type === 'warn' ? 'i-alert' : 'i-spark';
    el.innerHTML = '<svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><use href="#' +
      iconName + '"/></svg><span>' + esc(message) + '</span>';
    host.appendChild(el);
    setTimeout(function () {
      el.classList.add('is-leaving');
      setTimeout(function () {
        if (el.parentNode) el.parentNode.removeChild(el);
      }, 240);
    }, 4200);
  }

  var modalResolve = null;

  function confirmDialog(options) {
    const opts = options || {};
    const backdrop = node('modalBackdrop');
    if (!backdrop) return Promise.resolve(false);

    node('modalTitle').textContent = opts.title || '确认操作';
    node('modalBody').textContent = opts.body || '';
    const confirmBtn = node('modalConfirm');
    const cancelBtn = node('modalCancel');
    confirmBtn.textContent = opts.okText || '确认';
    cancelBtn.textContent = opts.cancelText || '取消';
    confirmBtn.className = 'btn ' + (opts.danger === false ? 'btn-primary' : 'btn-danger');

    backdrop.hidden = false;
    const previousFocus = document.activeElement;
    confirmBtn.focus();

    return new Promise(function (resolve) {
      function finish(value) {
        modalResolve = null;
        backdrop.hidden = true;
        confirmBtn.removeEventListener('click', onConfirm);
        cancelBtn.removeEventListener('click', onCancel);
        backdrop.removeEventListener('mousedown', onBackdrop);
        document.removeEventListener('keydown', onKey);
        if (previousFocus && previousFocus.focus) previousFocus.focus();
        resolve(value);
      }
      function onConfirm() {
        finish(true);
      }
      function onCancel() {
        finish(false);
      }
      function onBackdrop(event) {
        if (event.target === backdrop) finish(false);
      }
      function onKey(event) {
        if (event.key === 'Escape') {
          event.preventDefault();
          finish(false);
        }
      }
      modalResolve = { finish: finish };
      confirmBtn.addEventListener('click', onConfirm);
      cancelBtn.addEventListener('click', onCancel);
      backdrop.addEventListener('mousedown', onBackdrop);
      document.addEventListener('keydown', onKey);
    });
  }

  // ===========================================================================
  // 8. 对话:消息渲染 + SSE 事件处理
  // ===========================================================================

  function skeletonHTML() {
    return '<div class="msg-skeleton"><span></span><span></span><span></span></div>' +
      '<div class="thinking"><span>正在检索并组织答案</span>' +
      '<span class="dots"><i></i><i></i><i></i></span></div>';
  }

  function hideWelcome() {
    const welcome = node('welcome');
    if (welcome) welcome.hidden = true;
  }

  function scrollChatToBottom(force) {
    const scroll = node('chatScroll');
    if (!scroll) return;
    const distance = scroll.scrollHeight - scroll.scrollTop - scroll.clientHeight;
    if (force || distance < 180) {
      scroll.scrollTo({ top: scroll.scrollHeight, behavior: scrollBehavior() });
    }
  }

  function autoGrowInput() {
    const input = node('input');
    if (!input) return;
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 168) + 'px';
  }

  function appendUserMessage(text) {
    hideWelcome();
    const el = document.createElement('article');
    el.className = 'msg msg-user';
    el.innerHTML = '<div class="bubble">' + esc(text) + '</div>';
    node('messages').appendChild(el);
    scrollChatToBottom(true);
  }

  function createAssistantTurn() {
    const turn = {
      id: ++state.turnSeq,
      root: null,
      body: null,
      head: null,
      foot: null,
      markdown: null,
      raw: '',
      rendered: null,
      citations: [],
      citationsArrived: false,
      meta: null,
      usage: null,
      selfCheck: null,
      tools: [],
      done: null,
      error: null,
      notes: [],
      streaming: true,
      aborted: false,
      startedAt: Date.now()
    };
    const article = document.createElement('article');
    article.className = 'msg msg-assistant';
    article.setAttribute('data-msg', String(turn.id));
    article.innerHTML =
      '<div class="msg-avatar" aria-hidden="true"><svg viewBox="0 0 24 24"><use href="#i-spark"/></svg></div>' +
      '<div class="msg-main">' +
        '<div class="msg-head"><span class="msg-author">KnowledgeFlow</span></div>' +
        '<div class="msg-body">' + skeletonHTML() + '</div>' +
        '<div class="msg-foot" hidden></div>' +
      '</div>';
    node('messages').appendChild(article);
    turn.root = article;
    turn.body = article.querySelector('.msg-body');
    turn.head = article.querySelector('.msg-head');
    turn.foot = article.querySelector('.msg-foot');
    state.turns[turn.id] = turn;
    scrollChatToBottom(true);
    return turn;
  }

  function ensureMarkdown(turn) {
    if (!turn) return null;
    if (turn.markdown) return turn.markdown;
    turn.body.innerHTML = '<div class="markdown"></div>';
    turn.markdown = turn.body.firstChild;
    return turn.markdown;
  }

  /** 流式渲染节流:80ms 一次,避免每个 delta 都做 full Markdown 解析。 */
  function scheduleRender(turn, immediate) {
    if (!turn) return;
    if (immediate) {
      flushRender(turn);
      return;
    }
    if (state.renderTimer) return;
    state.renderTimer = setTimeout(function () {
      state.renderTimer = null;
      flushRender(state.currentTurn);
    }, RENDER_THROTTLE_MS);
  }

  function flushRender(turn) {
    if (!turn || !turn.markdown) return;
    if (turn.rendered === turn.raw) return;
    turn.rendered = turn.raw;
    turn.markdown.innerHTML = renderMarkdown(turn.raw);
    decorateCitations(turn);
    if (turn.streaming) scrollChatToBottom(false);
  }

  /** citations 到达后,给正文里的 [n] 角标补 title,并标出缺失的引用。 */
  function decorateCitations(turn) {
    if (!turn || !turn.markdown) return;
    const map = {};
    (turn.citations || []).forEach(function (item) {
      map[String(item.index)] = item;
    });
    $$('.cite-ref', turn.markdown).forEach(function (chip) {
      const index = chip.getAttribute('data-cite');
      const cite = map[index];
      if (cite) {
        chip.classList.remove('is-missing');
        chip.setAttribute('title', '[' + index + '] 《' + (cite.title || '无标题') + '》 · ' +
          (cite.doc_id || '') + ' · 相关度 ' + fmtScore(cite.score));
      } else if (!turn.citationsArrived) {
        // 流式过程中 citations 还没到,先不标记为缺失(否则整段角标都在虚线上)
        chip.classList.remove('is-missing');
        chip.setAttribute('title', '引用 [' + index + '] · 证据返回中…');
      } else {
        chip.classList.add('is-missing');
        chip.setAttribute('title', '引用 [' + index + '] 暂无对应证据');
      }
    });
  }

  function addTurnNote(turn, text, iconName) {
    if (!turn) return;
    turn.notes.push(text);
    const note = document.createElement('div');
    note.className = 'note-line';
    note.innerHTML = icon(iconName || 'alert') + '<span>' + esc(text) + '</span>';
    turn.body.appendChild(note);
  }

  function updateTurnHead(turn) {
    if (!turn) return;
    const meta = turn.meta || {};
    const plan = meta.plan || {};
    let html = '<span class="msg-author">KnowledgeFlow</span>';
    html += '<span class="badge badge-mode">' + esc(meta.mode || state.mode) + '</span>';
    if (meta.path) {
      html += '<span class="badge badge-path">' + esc(PATH_META[meta.path] || meta.path) + '</span>';
    }
    if (plan.intent) {
      const confidence = Number(plan.confidence);
      html += '<span class="badge">意图 ' + esc(plan.intent) +
        (isFinite(confidence) ? ' · ' + Math.round(confidence * 100) + '%' : '') + '</span>';
    }
    if (meta.llm && meta.llm.degraded) html += '<span class="badge badge-warn">LLM 降级</span>';
    if (turn.selfCheck && VERDICT_META[turn.selfCheck.verdict]) {
      const verdict = VERDICT_META[turn.selfCheck.verdict];
      html += '<span class="badge ' + verdict.badgeCls + '">' + esc(verdict.badge) + '</span>';
    }
    if (turn.error) html += '<span class="badge badge-error">出错</span>';
    turn.head.innerHTML = html;
  }

  function updateTurnFoot(turn) {
    if (!turn || !turn.foot) return;
    const parts = [];
    const meta = turn.meta || {};
    if (meta.request_id) parts.push('req ' + meta.request_id);
    if (turn.usage) {
      if (turn.usage.elapsed_ms != null) parts.push(fmtMs(turn.usage.elapsed_ms));
      if (turn.usage.total_tokens != null) parts.push(fmtTokens(turn.usage.total_tokens) + ' tokens');
      if (turn.usage.cost_usd != null) parts.push(fmtCost(turn.usage.cost_usd));
      if (turn.usage.llm_calls != null) parts.push(turn.usage.llm_calls + ' 次 LLM');
    }
    if (turn.citations.length) parts.push(turn.citations.length + ' 条引用');
    if (turn.tools.length) parts.push(turn.tools.length + ' 次工具调用');

    if (!parts.length) {
      turn.foot.hidden = true;
      return;
    }
    turn.foot.hidden = false;
    turn.foot.innerHTML = parts.join('<span class="sep">·</span>') +
      '<button type="button" class="foot-btn" data-act="copy-answer">' + icon('copy') + '复制回答</button>';
  }

  function showTurnError(turn, message) {
    if (!turn) return;
    turn.error = message;
    ensureMarkdown(turn);
    const banner = document.createElement('div');
    banner.className = 'error-banner';
    banner.innerHTML = icon('alert') + '<div><strong>生成失败:</strong>' + esc(message) + '</div>';
    turn.body.appendChild(banner);
    updateTurnHead(turn);
    updateTurnFoot(turn);
  }

  function finishTurn(turn) {
    if (!turn) return;
    turn.streaming = false;
    if (turn.body) turn.body.classList.remove('is-streaming');
    if (!turn.markdown && !turn.error && !turn.aborted && !turn.notes.length) {
      // 服务端正常结束但一个字都没返回:别把骨架动画留在页面上
      turn.body.innerHTML = '<div class="note-line">' + icon('alert') +
        '<span>没有收到回答内容,可以重试或换一种问法。</span></div>';
    }
    if (turn.markdown) flushRender(turn);
    updateTurnHead(turn);
    updateTurnFoot(turn);
    scrollChatToBottom(false);
  }

  function handleDelta(turn, text) {
    if (!turn || !text) return;
    ensureMarkdown(turn);
    turn.raw += String(text);
    turn.body.classList.add('is-streaming');
    scheduleRender(turn, false);
  }

  function handleReplace(turn, data) {
    if (!turn) return;
    ensureMarkdown(turn);
    turn.raw = String(data && data.text !== undefined ? data.text : '');
    scheduleRender(turn, true);
    addTurnNote(turn, '自检触发了答案修订' + (data && data.reason ? ':' + data.reason : ''), 'alert');
  }

  function handleCitations(turn, items) {
    if (!turn) return;
    turn.citations = items || [];
    turn.citationsArrived = true;
    state.citationsTurn = turn;
    if (turn.markdown) decorateCitations(turn);
    TraceView.setCitations(turn.citations);
  }

  function handleTools(turn, items) {
    if (!turn) return;
    turn.tools = items || [];
    TraceView.setTools(turn.tools);
    updateTurnFoot(turn);
  }

  function handleSelfCheck(turn, data) {
    if (!turn) return;
    turn.selfCheck = data || null;
    updateTurnHead(turn);
    TraceView.setSelfCheck(turn.selfCheck);
  }

  function handleUsage(turn, data) {
    if (!turn) return;
    turn.usage = data || null;
    updateTurnFoot(turn);
    TraceView.setUsage(turn.usage);
  }

  function handleDone(turn, data) {
    if (!turn) return;
    turn.done = data || null;
    state.lastTrace = data && data.trace ? data.trace : null;
    if (data && data.self_check) {
      turn.selfCheck = data.self_check;
      updateTurnHead(turn);
      TraceView.setSelfCheck(turn.selfCheck);
    }
    updateTurnFoot(turn);
    TraceView.setDone(data);
    setTimeout(refreshMemory, 250);
  }

  function handleStreamError(turn, message) {
    if (!turn) return;
    showTurnError(turn, message || '未知错误');
    TraceView.setError(message || '未知错误');
    toast('对话失败:' + (message || '未知错误'), 'error');
  }

  function handleEvent(turn, name, data) {
    switch (name) {
      case 'meta':
        turn.meta = data || {};
        updateTurnHead(turn);
        TraceView.setMeta(turn.meta);
        break;
      case 'memory':
        TraceView.setMemory(data || {});
        break;
      case 'retrieval':
        TraceView.setRetrieval(data && data.trace ? data.trace : data);
        break;
      case 'delta':
        handleDelta(turn, data && data.text);
        break;
      case 'replace':
        handleReplace(turn, data || {});
        break;
      case 'citations':
        handleCitations(turn, (data && data.items) || []);
        break;
      case 'tools':
        handleTools(turn, (data && data.items) || []);
        break;
      case 'self_check':
        handleSelfCheck(turn, data || null);
        break;
      case 'usage':
        handleUsage(turn, data || null);
        break;
      case 'done':
        handleDone(turn, data || null);
        break;
      case 'error':
        handleStreamError(turn, data && data.message);
        break;
      default:
        // 未知事件按协议忽略,保证后端新增事件不会打断前端
        break;
    }
  }

  function setStreaming(streaming) {
    state.streaming = streaming;
    const input = node('input');
    const sendBtn = node('sendBtn');
    const stopBtn = node('stopBtn');
    if (input) input.disabled = streaming;
    if (sendBtn) sendBtn.disabled = streaming;
    if (stopBtn) stopBtn.hidden = !streaming;
  }

  function abortStreaming() {
    if (!state.streaming) return;
    const turn = state.currentTurn;
    if (state.abort && typeof state.abort.abort === 'function') {
      state.abort.abort();
    }
    if (turn) {
      turn.aborted = true;
      addTurnNote(turn, '已手动停止生成,以上为已生成的部分回答。', 'alert');
    }
    toast('已停止生成', 'warn');
  }

  /** 发送一条问题:用户气泡 → 助手占位 → SSE 流 → 收尾。 */
  async function sendMessage(textOverride) {
    const input = node('input');
    const message = String(textOverride === undefined ? (input ? input.value : '') : textOverride).trim();

    if (!message) {
      toast('先输入一个问题再发送', 'warn');
      if (input) input.focus();
      return;
    }
    if (state.streaming) {
      toast('正在生成中,可先点「停止」或等这一轮结束', 'warn');
      return;
    }
    if (message.length > 2000) {
      toast('问题太长(最多 2000 字)', 'error');
      return;
    }

    hideDemoPopover();
    if (input) {
      input.value = '';
      autoGrowInput();
    }

    appendUserMessage(message);
    const turn = createAssistantTurn();
    state.currentTurn = turn;
    state.traceTurn = turn;
    state.citationsTurn = null;
    setStreaming(true);
    TraceView.start();

    const payload = {
      message: message,
      session_id: state.sessionId,
      user_id: state.userId,
      mode: state.mode,
      stream: true
    };

    try {
      await streamAnswer(payload, {
        onEvent: function (name, data) {
          handleEvent(turn, name, data);
        }
      });
      if (!turn.done && !turn.error && !turn.aborted) {
        addTurnNote(turn, '连接已结束但没有收到 done 事件,回答可能不完整。', 'alert');
        TraceView.setNote('连接提前结束:未收到 done 事件,链路可能不完整。');
      }
    } catch (err) {
      if (turn.aborted || isAbortError(err)) {
        if (!turn.aborted) addTurnNote(turn, '请求已中断,以上为已生成的部分回答。', 'alert');
        TraceView.setNote('请求被中断,链路记录可能不完整。');
      } else {
        handleStreamError(turn, (err && err.message) || String(err));
      }
    } finally {
      setStreaming(false);
      finishTurn(turn);
      if (input && !state.streaming) input.focus();
    }
  }

  async function refreshMemory() {
    try {
      const data = await httpJSON(API.memory(state.userId, state.sessionId), {}, 8000);
      state.memoryTurns = Number(data.turns) || 0;
      renderSessionLabel();
    } catch (err) {
      // 记忆接口不可用时不影响主流程,静默失败
    }
  }

  async function newSession() {
    if (state.streaming) {
      toast('正在生成中,请先停止再开新会话', 'warn');
      return;
    }
    const hasMessages = node('messages') && node('messages').children.length > 0;
    if (hasMessages) {
      const ok = await confirmDialog({
        title: '开始新会话',
        body: '将清空当前对话、引用与右侧检索链路,并重置 session_id(服务端记忆按 session 区分)。确定继续吗?',
        okText: '开始新会话',
        danger: true
      });
      if (!ok) return;
    }

    node('messages').innerHTML = '';
    node('welcome').hidden = false;
    state.currentTurn = null;
    state.turns = {};
    state.lastTrace = null;
    state.sessionId = 'web-' + randomId(4);
    state.memoryTurns = 0;
    TraceView.reset();
    renderSessionLabel();
    if (node('input')) {
      node('input').value = '';
      autoGrowInput();
    }
    toast('已开始新会话', 'success');
    if (node('input')) node('input').focus();
  }

  // ===========================================================================
  // 9. 右侧「检索链路」渲染
  // ===========================================================================

  function sectionHeadHTML(title, metaText) {
    return '<div class="trace-section-head">' +
      '<div class="trace-section-title">' + icon('activity') + '<span>' + esc(title) + '</span></div>' +
      (metaText ? '<span class="trace-section-meta">' + esc(metaText) + '</span>' : '') +
      '</div>';
  }

  function traceSection(name) {
    TraceView.ensureSections();
    const content = node('traceContent');
    return content ? content.querySelector('[data-sec="' + name + '"]') : null;
  }

  function writeSection(name, bodyHTML, title, metaText) {
    const box = traceSection(name);
    if (!box) return;
    if (!bodyHTML) {
      box.hidden = true;
      box.innerHTML = '';
      return;
    }
    box.hidden = false;
    box.innerHTML = sectionHeadHTML(title, metaText) + bodyHTML;
  }

  function metricCell(label, value, unit, accent) {
    return '<div class="metric' + (accent ? ' metric-accent' : '') + '">' +
      '<div class="metric-label">' + esc(label) + '</div>' +
      '<div class="metric-value">' + esc(value) +
      (unit ? '<small>' + esc(unit) + '</small>' : '') + '</div></div>';
  }

  function summaryHTML(meta) {
    const plan = meta.plan || {};
    const llm = meta.llm || {};
    const memory = meta.memory || {};
    const confidence = Number(plan.confidence);
    const hasConfidence = isFinite(confidence);
    const percent = hasConfidence ? Math.round(confidence * 100) : null;

    let html = '<div class="req-card">';

    html += '<div class="req-row"><span class="req-label">请求 ID</span>' +
      '<button type="button" class="req-id" data-copy="' + esc(meta.request_id || '') +
      '" aria-label="复制请求 ID">' + esc(meta.request_id || '—') + icon('copy') + '</button></div>';

    html += '<div class="req-row"><span class="req-label">模式 / 路径</span>' +
      '<span class="badges-row">' +
      '<span class="badge badge-mode">' + esc(meta.mode || '—') + '</span>' +
      '<span class="badge badge-path">' + esc(PATH_META[meta.path] || meta.path || '—') + '</span>' +
      (llm.degraded ? '<span class="badge badge-warn">LLM 降级</span>' : '') +
      '</span></div>';

    html += '<div class="req-row"><span class="req-label">意图</span>' +
      '<span class="req-value mono">' + esc(plan.intent || '—') +
      (percent !== null ? ' · ' + percent + '%' : '') + '</span></div>';

    if (percent !== null) {
      html += '<div class="confidence">' +
        '<div class="confidence-head"><span>意图置信度</span><b>' + percent + '%</b></div>' +
        '<div class="meter"><span style="width:' + clamp(percent, 0, 100) + '%"></span></div>' +
        '</div>';
    }

    html += '<div class="req-row"><span class="req-label">模型 / 知识库</span>' +
      '<span class="req-value mono">' + esc(shortModel(llm.model)) + ' · ' +
      fmtNum(meta.kb_chunks) + ' 块</span></div>';

    html += '<div class="req-row"><span class="req-label">会话记忆</span>' +
      '<span class="req-value mono">' + fmtNum(memory.turns) + ' 轮 · ≈' +
      fmtTokens(memory.estimated_tokens) + ' tok</span></div>';

    if (plan.reason) {
      html += '<p class="reason-text">路由理由:' + esc(plan.reason) + '</p>';
    }
    return html + '</div>';
  }

  function formatInfoValue(key, value) {
    if (value === null || value === undefined) return '—';
    if (typeof value === 'boolean') {
      if (key === 'sufficient') return value ? '证据充分' : '证据不足';
      return value ? '是' : '否';
    }
    if (typeof value === 'object') return safeJson(value, 0);
    return String(value);
  }

  function stageInfoHTML(info) {
    const source = info || {};
    const keys = Object.keys(source).filter(function (key) {
      const value = source[key];
      if (value === null || value === undefined || value === '') return false;
      if (Array.isArray(value) && value.length === 0) return false;
      return true;
    });
    if (!keys.length) return '';
    let html = '<dl class="stage-info">';
    keys.forEach(function (key) {
      html += '<dt>' + esc(key) + '</dt><dd>' + esc(formatInfoValue(key, source[key])) + '</dd>';
    });
    return html + '</dl>';
  }

  /** 归一化分数条:同组内相对最大值,保证「谁是第一」一眼可见。 */
  function scoreWidth(score, maxScore) {
    const value = Number(score) || 0;
    const max = Number(maxScore) || 0;
    if (max <= 0) return 3;
    return clamp((value / max) * 100, 3, 100);
  }

  function hitsHTML(hits) {
    const maxScore = (hits || []).reduce(function (max, hit) {
      return Math.max(max, Number(hit.score) || 0);
    }, 0);
    return (hits || []).map(function (hit, index) {
      const score = Number(hit.score) || 0;
      let html = '<div class="hit">';
      html += '<div class="hit-head">' +
        '<span class="hit-rank">#' + esc(hit.rank || index + 1) + '</span>' +
        '<span class="hit-title" title="' + esc(hit.title || '') + '">' +
        esc(hit.title || hit.doc_id || '未命名片段') + '</span>' +
        (hit.channel ? '<span class="hit-channel">' + esc(hit.channel) + '</span>' : '') +
        '<span class="hit-score">' + fmtScore(score) + '</span>' +
        '</div>';
      html += '<div class="score-track"><span style="width:' +
        scoreWidth(score, maxScore).toFixed(1) + '%"></span></div>';
      if (hit.text) {
        html += '<div class="hit-text">' + esc(hit.text) + '</div>';
        html += '<button type="button" class="hit-more" data-act="toggle-hit">展开全文</button>';
      }
      if (hit.doc_id) {
        html += '<div class="evidence-docid" style="margin-top:2px">' + esc(hit.doc_id) + '</div>';
      }
      return html + '</div>';
    }).join('');
  }

  function stageHTML(stage) {
    const info = stage.info || {};
    const hits = stage.hits || [];
    const meta = STAGE_META[stage.name] || (/^第\d+轮检索$/.test(stage.name || '')
      ? { label: stage.name, kind: 'agentic_round' }
      : { label: stage.name || '阶段', kind: 'misc' });
    let count = null;
    if (info.candidates !== undefined && info.candidates !== null) count = info.candidates;
    else if (info.count !== undefined && info.count !== null) count = info.count;
    else count = hits.length;

    let html = '<div class="stage" data-kind="' + esc(meta.kind) + '">';
    html += '<button type="button" class="stage-head" aria-expanded="false">' +
      '<span class="stage-name">' + esc(meta.label) + '</span>' +
      '<span class="stage-key">' + esc(stage.name || '') + '</span>' +
      '<span class="stage-spacer"></span>' +
      (count !== null ? '<span class="stage-count">' + fmtNum(count) + ' 候选</span>' : '') +
      '<span class="stage-ms">' + fmtMs(stage.ms) + '</span>' +
      icon('chevron').replace('class="icon"', 'class="icon stage-chevron"') +
      '</button>';
    html += '<div class="stage-body" hidden>' + stageInfoHTML(info) +
      (hits.length ? hitsHTML(hits) : '') + '</div>';
    return html + '</div>';
  }

  function timelineHTML(trace) {
    const stages = (trace && trace.stages) || [];
    if (!stages.length) {
      return '<div class="trace-note">' + icon('alert') +
        '<span>本次请求没有检索阶段(可能是闲聊、工具直达或意图路由跳过了检索)。</span></div>';
    }
    return '<div class="timeline">' + stages.map(stageHTML).join('') + '</div>';
  }

  function evidenceHTML(items) {
    const list = items || [];
    if (!list.length) {
      return '<div class="trace-note">' + icon('alert') +
        '<span>本次回答没有可展示的引用证据(闲聊或工具直达路径不会检索知识库)。</span></div>';
    }
    return list.map(function (item) {
      let html = '<div class="evidence-card" data-evidence="' + esc(item.index) + '">';
      html += '<div class="evidence-head">' +
        '<span class="evidence-idx">' + esc(item.index) + '</span>' +
        '<span class="evidence-title" title="' + esc(item.title || '') + '">' +
        esc(item.title || '(无标题)') + '</span>' +
        '<span class="evidence-score">' + fmtScore(item.score) + '</span>' +
        '</div>';
      html += '<div class="evidence-docid">' + esc(item.doc_id || '') +
        (item.channel ? ' · ' + esc(item.channel) : '') + '</div>';
      if (item.text) {
        html += '<div class="evidence-text">' + esc(item.text) + '</div>';
        html += '<button type="button" class="evidence-more" data-act="toggle-evidence">展开证据</button>';
      }
      return html + '</div>';
    }).join('');
  }

  function toolCardHTML(tool) {
    const isError = !!tool.is_error;
    const args = tool.arguments && typeof tool.arguments === 'object'
      ? safeJson(tool.arguments, 2)
      : String(tool.arguments || '');
    const output = String(tool.output || '');
    let html = '<div class="tool-card' + (isError ? ' is-error' : '') + '">';
    html += '<div class="tool-head">' +
      '<span class="tool-name">' + icon('terminal') + esc(tool.name || 'tool') + '</span>' +
      (isError ? '<span class="badge badge-error">失败</span>'
        : '<span class="badge badge-ok">成功</span>') +
      (tool.ms !== undefined && tool.ms !== null
        ? '<span class="tool-ms">' + fmtMs(tool.ms) + '</span>' : '') +
      '</div>';
    if (args) {
      html += '<div class="tool-block"><div class="tool-block-label">参数</div>' +
        '<pre class="tool-pre">' + esc(args) + '</pre></div>';
    }
    if (output) {
      html += '<div class="tool-block"><div class="tool-block-label">输出</div>' +
        '<pre class="tool-pre">' + esc(output) + '</pre>' +
        '<button type="button" class="tool-toggle" data-act="toggle-tool">展开全部</button></div>';
    }
    return html + '</div>';
  }

  function verdictHTML(result) {
    if (!result) return '';
    const meta = VERDICT_META[result.verdict] || {
      cls: '',
      icon: 'i-alert',
      title: '自检结论:' + (result.verdict || '未知')
    };
    return '<div class="verdict ' + meta.cls + '">' +
      '<div class="verdict-icon">' + icon(meta.icon.replace(/^i-/, '')) + '</div>' +
      '<div class="verdict-main">' +
      '<div class="verdict-title">' + esc(meta.title) + '</div>' +
      (result.reason ? '<div class="verdict-desc">' + esc(result.reason) + '</div>' : '') +
      '</div></div>';
  }

  var TraceView = {
    values: {},
    lastRetrieval: null,
    lastCitations: [],
    lastTools: [],
    lastExport: null,

    /** 清空右侧面板,回到空状态。 */
    reset: function () {
      this.values = {};
      this.lastRetrieval = null;
      this.lastCitations = [];
      this.lastTools = [];
      this.lastExport = null;
      const content = node('traceContent');
      if (content) {
        content.innerHTML = '';
        content.hidden = true;
        content.removeAttribute('data-ready');
      }
      const empty = node('traceEmpty');
      if (empty) empty.hidden = false;
      const copyBtn = node('copyTraceBtn');
      if (copyBtn) copyBtn.disabled = true;
      this.setSub('等待请求');
    },

    /** 首次写入前构造 7 个固定分区的骨架(之后只替换各自内容)。 */
    ensureSections: function () {
      const content = node('traceContent');
      if (!content) return;
      if (content.getAttribute('data-ready') === '1') return;
      content.innerHTML =
        '<div class="trace-section" data-sec="summary" hidden></div>' +
        '<div class="trace-section" data-sec="metrics" hidden></div>' +
        '<div class="trace-section" data-sec="timeline" hidden></div>' +
        '<div class="trace-section" data-sec="evidence" hidden></div>' +
        '<div class="trace-section" data-sec="tools" hidden></div>' +
        '<div class="trace-section" data-sec="verdict" hidden></div>' +
        '<div class="trace-section" data-sec="notes" hidden></div>';
      content.setAttribute('data-ready', '1');
    },

    setSub: function (text) {
      const sub = node('traceSub');
      if (sub) sub.textContent = text;
    },

    /** 请求开始:切到加载骨架。 */
    start: function () {
      this.values = {};
      this.lastRetrieval = null;
      this.lastCitations = [];
      this.lastTools = [];
      this.lastExport = null;
      const empty = node('traceEmpty');
      if (empty) empty.hidden = true;
      const content = node('traceContent');
      if (content) {
        content.hidden = false;
        content.removeAttribute('data-ready');
        content.innerHTML = '<div class="trace-loading"><span></span><span></span>' +
          '<span></span><span></span></div>';
      }
      const copyBtn = node('copyTraceBtn');
      if (copyBtn) copyBtn.disabled = true;
      this.setSub('请求进行中…');
    },

    setMeta: function (meta) {
      if (!meta) return;
      this.values = {
        prompt_tokens: null,
        completion_tokens: null,
        llm_calls: null,
        elapsed_ms: null,
        candidates: null,
        cost_usd: null
      };
      writeSection('summary', summaryHTML(meta), '请求概要', 'req ' + (meta.request_id || '—'));
      this.renderMetrics();
      this.setSub((PATH_META[meta.path] || meta.path || '—') + ' · ' + (meta.mode || ''));
    },

    renderMetrics: function () {
      const values = this.values || {};
      writeSection('metrics', '<div class="metrics-grid">' +
        metricCell('Prompt', fmtTokens(values.prompt_tokens), 'tok') +
        metricCell('Completion', fmtTokens(values.completion_tokens), 'tok') +
        metricCell('LLM 调用', values.llm_calls === null || values.llm_calls === undefined
          ? '—' : String(values.llm_calls), '次') +
        metricCell('总耗时', fmtMs(values.elapsed_ms), '', true) +
        metricCell('候选片段', fmtNum(values.candidates), '') +
        metricCell('成本', fmtCost(values.cost_usd), '') +
        '</div>', '用量与耗时', '');
    },

    setRetrieval: function (trace) {
      if (!trace) return;
      this.lastRetrieval = trace;
      const stages = trace.stages || [];
      const candidates = trace.candidates !== undefined && trace.candidates !== null
        ? trace.candidates
        : stages.reduce(function (sum, stage) {
          return sum + ((stage.hits && stage.hits.length) || 0);
        }, 0);
      this.values.candidates = candidates;
      this.renderMetrics();
      writeSection('timeline', timelineHTML(trace), '检索链路',
        (trace.mode ? trace.mode + ' · ' : '') + fmtMs(trace.total_ms) + ' / ' + stages.length + ' 阶段');
      this.setSub((trace.mode || '检索') + ' · ' + fmtMs(trace.total_ms));
    },

    setCitations: function (items) {
      const list = items || [];
      this.lastCitations = list;
      writeSection('evidence', evidenceHTML(list), '引用证据',
        list.length ? list.length + ' 条' : '无引用');
    },

    /** 点历史消息里的 [n] 时,临时把那一轮的引用切到证据区展示。 */
    showCitations: function (items, subText) {
      this.setCitations(items || []);
      if (subText) this.setSub(subText);
    },

    setTools: function (items) {
      const list = items || [];
      this.lastTools = list;
      if (!list.length) {
        writeSection('tools', '', 'Agent 工具调用', '');
        return;
      }
      const failed = list.filter(function (tool) {
        return !!tool.is_error;
      }).length;
      writeSection('tools', list.map(toolCardHTML).join(''), 'Agent 工具调用',
        list.length + ' 次' + (failed ? ' · ' + failed + ' 失败' : ''));
    },

    setSelfCheck: function (result) {
      if (!result) {
        writeSection('verdict', '', '幻觉自检', '');
        return;
      }
      writeSection('verdict', verdictHTML(result), '幻觉自检', result.verdict || '');
    },

    setUsage: function (usage) {
      if (!usage) return;
      Object.assign(this.values, usage);
      this.renderMetrics();
      this.setSub('完成 · ' + fmtMs(usage.elapsed_ms));
    },

    setMemory: function (data) {
      // 记忆维护是可选事件:摘要压缩 / 话题切换。这里只在发生了维护时提示,
      // 具体轮数与摘要已经显示在「请求概要」卡片里。
      const info = data || {};
      if (!info.compressed_messages && !info.topic_changed) return;
      const text = info.topic_changed
        ? '检测到话题切换,长记忆已按新话题重新聚焦。'
        : '历史对话已压缩 ' + fmtNum(info.compressed_messages) + ' 条,节省上下文 token。';
      this.setSub(text);
    },

    setDone: function (done) {
      state.lastTrace = done && done.trace ? done.trace : null;
      const copyBtn = node('copyTraceBtn');
      if (copyBtn) copyBtn.disabled = !state.lastTrace;
      if (done) {
        if (done.elapsed_ms !== undefined && done.elapsed_ms !== null) {
          this.values.elapsed_ms = done.elapsed_ms;
        }
        this.renderMetrics();
        this.setSub((PATH_META[done.path] || done.path || '—') + ' · ' + fmtMs(done.elapsed_ms));
      }
      if (!state.lastTrace) return;
      // 后端 done.trace 是请求级轨迹;把检索明细、引用与工具调用补挂上去,
      // 复制的 JSON 才是「一次请求的完整复盘材料」。
      this.lastExport = Object.assign({}, state.lastTrace, {
        retrieval: this.lastRetrieval,
        citations: this.lastCitations,
        tools: this.lastTools
      });
      const json = safeJson(this.lastExport, 2);
      const lineCount = json.split('\n').length;
      writeSection('notes',
        '<div class="trace-note">' + icon('terminal') +
        '<span>完整链路 trace 已生成(' + lineCount + ' 行 JSON,已把 retrieval / citations / tools 补挂到 done.trace 上),' +
        '点右上角复制按钮可整段带走,便于复盘与写进面试材料。</span></div>' +
        '<button type="button" class="hit-more" data-act="toggle-trace" aria-expanded="false">展开完整 trace JSON</button>' +
        '<pre class="tool-pre" data-role="trace-raw" hidden style="max-height:280px;margin-top:6px">' +
        esc(json) + '</pre>',
        '完整 Trace', 'done');
    },

    setNote: function (message) {
      if (!message) return;
      writeSection('notes', '<div class="trace-note">' + icon('alert') +
        '<span>' + esc(message) + '</span></div>', '提示', '');
    },

    setError: function (message) {
      writeSection('notes', '<div class="trace-alert">' + icon('alert') +
        '<span>' + esc(message || '请求失败') + '</span></div>', '异常', 'error');
    }
  };

  /** 右侧面板副标题文案:路径 + 耗时(切回最新一轮时恢复用)。 */
  function traceSubText(turn) {
    if (!turn) return '';
    const meta = turn.meta || {};
    const elapsed = turn.usage && turn.usage.elapsed_ms !== undefined && turn.usage.elapsed_ms !== null
      ? turn.usage.elapsed_ms
      : (turn.done && turn.done.elapsed_ms);
    return (PATH_META[meta.path] || meta.path || '—') +
      (elapsed !== undefined && elapsed !== null ? ' · ' + fmtMs(elapsed) : '');
  }

  /** 点击 [n]:高亮右侧对应证据卡并滚动到可见处。 */
  function focusEvidence(index, sourceChip) {
    const content = node('traceContent');
    if (!content) return;

    // 多轮对话时,右侧面板默认显示最新一轮;点历史消息的 [n] 就切到那一轮的引用,
    // 避免「点了旧回答却高亮新证据」。
    const article = sourceChip ? sourceChip.closest('.msg-assistant') : null;
    const turn = article ? state.turns[article.getAttribute('data-msg')] : state.currentTurn;
    if (turn && state.citationsTurn !== turn) {
      if (turn.citationsArrived) {
        TraceView.showCitations(turn.citations, turn === state.traceTurn
          ? traceSubText(turn)
          : '历史请求 · ' + ((turn.meta && turn.meta.request_id) || ''));
        state.citationsTurn = turn;
      } else {
        toast('引用 [' + index + '] 还在返回中,稍等一下', 'warn');
        return;
      }
    }

    const card = content.querySelector('[data-evidence="' + index + '"]');

    $$('.cite-ref', node('messages')).forEach(function (chip) {
      chip.classList.toggle('is-active', chip === sourceChip);
    });

    if (!card) {
      toast('没有找到引用 [' + index + '] 对应的证据', 'warn');
      return;
    }

    $$('.evidence-card', content).forEach(function (item) {
      item.classList.toggle('is-active', item === card);
    });
    card.scrollIntoView({ block: 'nearest', behavior: scrollBehavior() });

    // 窄屏时右侧面板堆叠在下方,把面板整体带到视口里
    const panel = node('tracePanel');
    if (panel && panel.getBoundingClientRect().top > window.innerHeight * 0.6) {
      panel.scrollIntoView({ behavior: scrollBehavior(), block: 'start' });
    }
  }

  /** 通用折叠:阶段 / 分块卡共用 is-open + aria-expanded + body.hidden。 */
  function toggleCollapsible(button) {
    if (!button) return;
    const container = button.closest('.stage, .chunk-card');
    if (!container) return;
    const open = !container.classList.contains('is-open');
    container.classList.toggle('is-open', open);
    button.setAttribute('aria-expanded', open ? 'true' : 'false');
    const body = container.querySelector('.stage-body, .chunk-text');
    if (body) body.hidden = !open;
  }

  // ===========================================================================
  // 10. 知识库视图
  // ===========================================================================

  function emptyStateHTML(iconName, title, desc) {
    return '<div class="empty-state"><div class="empty-icon" aria-hidden="true">' +
      '<svg viewBox="0 0 24 24"><use href="#i-' + esc(iconName || 'doc') + '"/></svg></div>' +
      '<p class="empty-title">' + esc(title) + '</p>' +
      (desc ? '<p class="empty-desc">' + esc(desc) + '</p>' : '') + '</div>';
  }

  function loadingHTML(count) {
    let spans = '';
    for (let i = 0; i < (count || 4); i += 1) spans += '<span></span>';
    return '<div class="trace-loading">' + spans + '</div>';
  }

  function renderKbStats() {
    const kb = state.kb;
    const catCount = Object.keys(kb.categories || {}).length;
    const cells = [
      { label: '文档', value: fmtNum(kb.total), unit: '篇' },
      { label: '文本块', value: fmtNum(kb.chunks), unit: '块' },
      { label: '分类', value: fmtNum(catCount), unit: '类' }
    ];
    const host = node('kbStats');
    if (host) {
      host.innerHTML = cells.map(function (cell) {
        return '<div class="kb-stat">' +
          '<div class="kb-stat-label">' + esc(cell.label) + '</div>' +
          '<div class="kb-stat-value">' + esc(cell.value) +
          '<small>' + esc(cell.unit) + '</small></div></div>';
      }).join('');
    }
    const sub = node('kbSub');
    if (sub) sub.textContent = kb.total + ' 篇 · ' + kb.chunks + ' 块';
  }

  function renderKbCats() {
    const host = node('kbCats');
    if (!host) return;
    const kb = state.kb;
    const categories = kb.categories || {};
    const chips = ['<button type="button" class="cat-chip' +
      (kb.filterCat ? '' : ' is-active') + '" data-cat="">全部</button>'];
    Object.keys(categories).sort().forEach(function (name) {
      chips.push('<button type="button" class="cat-chip' +
        (kb.filterCat === name ? ' is-active' : '') + '" data-cat="' + esc(name) + '">' +
        esc(name) + ' ' + fmtNum(categories[name]) + '</button>');
    });
    host.innerHTML = chips.join('');
  }

  function renderKbDatalist() {
    const host = node('kbCategories');
    if (!host) return;
    host.innerHTML = Object.keys(state.kb.categories || {}).map(function (name) {
      return '<option value="' + esc(name) + '"></option>';
    }).join('');
  }

  function filteredDocs() {
    const kb = state.kb;
    const keyword = (kb.filterText || '').trim().toLowerCase();
    return (kb.documents || []).filter(function (doc) {
      if (kb.filterCat && doc.category !== kb.filterCat) return false;
      if (!keyword) return true;
      return String(doc.title || '').toLowerCase().indexOf(keyword) !== -1 ||
        String(doc.doc_id || '').toLowerCase().indexOf(keyword) !== -1 ||
        String(doc.category || '').toLowerCase().indexOf(keyword) !== -1;
    });
  }

  function renderDocList() {
    const host = node('docList');
    if (!host) return;
    const kb = state.kb;
    if (kb.loading && !kb.documents.length) {
      host.innerHTML = loadingHTML(5);
      return;
    }
    if (!kb.documents.length) {
      host.innerHTML = emptyStateHTML('db', '知识库还是空的',
        '在下方「上传文件」或「粘贴文本入库」添加文档,再点「重建全部索引」即可开始检索。');
      return;
    }
    const docs = filteredDocs();
    if (!docs.length) {
      host.innerHTML = emptyStateHTML('search', '没有匹配的文档',
        '换个关键词试试,或点「全部」清除分类过滤。');
      return;
    }
    host.innerHTML = docs.map(function (doc) {
      const active = doc.doc_id === kb.selectedId;
      return '<button type="button" class="doc-row' + (active ? ' is-active' : '') + '"' +
        ' data-doc="' + esc(doc.doc_id) + '"' + (active ? ' aria-current="true"' : '') + '>' +
        '<span class="doc-row-top">' +
        '<span class="doc-row-title" title="' + esc(doc.title || '') + '">' +
        esc(doc.title || doc.doc_id) + '</span>' +
        '<span class="doc-row-cat">' + esc(doc.category || '未分类') + '</span>' +
        '</span>' +
        '<span class="doc-row-meta"><span>' + fmtNum(doc.chunks) + ' 块</span>' +
        '<span class="sep">·</span><span>' + fmtNum(doc.chars) + ' 字</span></span>' +
        '<span class="doc-row-id">' + esc(doc.doc_id) + '</span>' +
        '</button>';
    }).join('');
  }

  function renderKbDetail(doc) {
    const chunks = doc.chunks || [];
    const sub = node('kbDetailSub');
    if (sub) sub.textContent = (doc.category ? doc.category + ' · ' : '') + chunks.length + ' 块';
    const copyBtn = node('kbCopyChunksBtn');
    if (copyBtn) copyBtn.disabled = !chunks.length;
    const host = node('kbDetail');
    if (!host) return;

    let html = '<div class="doc-detail-head"><div>' +
      '<div class="doc-detail-title">' + esc(doc.title || doc.doc_id) + '</div>' +
      '<div class="doc-detail-meta"><span>' + esc(doc.doc_id) + '</span>' +
      (doc.category ? '<span>' + esc(doc.category) + '</span>' : '') +
      '<span>' + fmtNum(chunks.length) + ' 块</span></div></div></div>';

    if (!chunks.length) {
      html += emptyStateHTML('doc', '这篇文档没有分块',
        '可能入库时切分失败,可以在下方选择策略后重新入库或全量重建。');
    } else {
      html += chunks.map(function (chunk, index) {
        const open = index === 0; // 默认展开第一块
        const length = String(chunk.text || '').length;
        return '<div class="chunk-card' + (open ? ' is-open' : '') + '">' +
          '<button type="button" class="chunk-head" aria-expanded="' + (open ? 'true' : 'false') + '">' +
          '<span class="chunk-index">#' + esc(chunk.index !== undefined ? chunk.index : index) + '</span>' +
          '<span class="chunk-id" title="' + esc(chunk.chunk_id || '') + '">' +
          esc(chunk.chunk_id || '') + '</span>' +
          '<span class="chunk-len">' + fmtNum(length) + ' 字</span>' +
          icon('chevron').replace('class="icon"', 'class="icon chunk-chevron"') +
          '</button>' +
          '<pre class="chunk-text"' + (open ? '' : ' hidden') + '>' + esc(chunk.text || '') + '</pre>' +
          '</div>';
      }).join('');
    }
    host.innerHTML = html;
  }

  async function loadKbDocuments(force) {
    const kb = state.kb;
    if (kb.loading) return;
    if (kb.loaded && !force) return;
    kb.loading = true;
    renderDocList();
    try {
      const data = await httpJSON(API.kbDocuments, {}, 15000);
      kb.documents = data.documents || [];
      kb.total = data.total !== undefined ? data.total : kb.documents.length;
      kb.chunks = data.chunks !== undefined ? data.chunks : 0;
      kb.categories = data.categories || {};
      kb.loaded = true;
      renderKbStats();
      renderKbCats();
      renderKbDatalist();
    } catch (err) {
      kb.loaded = false;
      const host = node('docList');
      if (host) {
        host.innerHTML = '<div class="empty-state"><div class="empty-icon" aria-hidden="true">' +
          icon('alert') + '</div><p class="empty-title">文档列表加载失败</p>' +
          '<p class="empty-desc">' + esc(err.message) + '</p></div>';
      }
      const sub = node('kbSub');
      if (sub) sub.textContent = '加载失败';
      toast('知识库加载失败:' + err.message, 'error');
    } finally {
      kb.loading = false;
      renderDocList();
    }
  }

  async function loadKbDocument(docId) {
    if (!docId) return;
    const kb = state.kb;
    kb.selectedId = docId;
    renderDocList();

    const host = node('kbDetail');
    const sub = node('kbDetailSub');
    const copyBtn = node('kbCopyChunksBtn');
    if (copyBtn) copyBtn.disabled = true;
    if (sub) sub.textContent = '加载中…';
    if (host) host.innerHTML = loadingHTML(4);

    try {
      const doc = await httpJSON(API.kbDocument(docId), {}, 15000);
      kb.detail = doc;
      renderKbDetail(doc);
    } catch (err) {
      kb.detail = null;
      if (host) host.innerHTML = emptyStateHTML('alert', '分块加载失败', err.message);
      if (sub) sub.textContent = '加载失败';
      toast('文档分块加载失败:' + err.message, 'error');
    }
  }

  function chunksToText(doc) {
    if (!doc) return '';
    const lines = ['# ' + (doc.title || doc.doc_id), 'doc_id: ' + doc.doc_id +
      (doc.category ? ' · ' + doc.category : ''), ''];
    (doc.chunks || []).forEach(function (chunk, index) {
      lines.push('## 片段 #' + (chunk.index !== undefined ? chunk.index : index) +
        ' (' + (chunk.chunk_id || '') + ')');
      lines.push(chunk.text || '');
      lines.push('');
    });
    return lines.join('\n');
  }

  function summarizeIngest(result) {
    const data = result || {};
    const documents = data.documents !== undefined ? data.documents : '?';
    const chunks = data.chunks !== undefined ? data.chunks : (data.total_chunks !== undefined ? data.total_chunks : '?');
    return documents + ' 篇文档 / ' + chunks + ' 块' +
      (data.strategy ? ' · ' + data.strategy : '');
  }

  function setButtonBusy(button, busy, busyText) {
    if (!button) return;
    if (busy) {
      if (!button.dataset.originalLabel) button.dataset.originalLabel = button.innerHTML;
      button.disabled = true;
      if (busyText) button.textContent = busyText;
    } else {
      button.disabled = false;
      if (button.dataset.originalLabel) {
        button.innerHTML = button.dataset.originalLabel;
        delete button.dataset.originalLabel;
      }
    }
  }

  async function refreshKnowledge() {
    await Promise.allSettled([loadKbDocuments(true), loadMeta(true), loadHealth(true)]);
    const kb = state.kb;
    if (kb.selectedId && kb.documents.some(function (doc) {
      return doc.doc_id === kb.selectedId;
    })) {
      await loadKbDocument(kb.selectedId);
    }
  }

  async function handleUpload(event) {
    event.preventDefault();
    const form = node('uploadForm');
    const fileInput = node('uploadFile');
    const file = fileInput && fileInput.files && fileInput.files[0];
    if (!file) {
      toast('先选择一个 .md / .txt 文件', 'warn');
      return;
    }
    if (file.size > 5 * 1024 * 1024) {
      toast('文件超过 5MB,请拆分后再上传', 'error');
      return;
    }
    if (!/\.(md|txt|markdown)$/i.test(file.name || '')) {
      toast('只支持 .md / .txt 文件', 'warn');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('title', (node('uploadTitle').value || '').trim());
    formData.append('category', (node('uploadCategory').value || '').trim() || '未分类');
    formData.append('strategy', node('uploadStrategy').value || 'recursive');

    const submit = node('uploadSubmit');
    setButtonBusy(submit, true, '上传中…');
    try {
      const result = await httpJSON(API.kbUpload, { method: 'POST', body: formData }, UPLOAD_TIMEOUT);
      toast('上传入库成功:' + summarizeIngest(result), 'success');
      if (form) form.reset();
      await refreshKnowledge();
    } catch (err) {
      toast('上传失败:' + err.message, 'error');
    } finally {
      setButtonBusy(submit, false);
    }
  }

  async function handleIngest(event) {
    event.preventDefault();
    const title = (node('ingestTitle').value || '').trim();
    const text = (node('ingestText').value || '').trim();
    if (!title) {
      toast('先填写标题', 'warn');
      return;
    }
    if (text.length < 20) {
      toast('正文至少 20 字', 'warn');
      return;
    }

    const submit = node('ingestSubmit');
    setButtonBusy(submit, true, '写入中…');
    try {
      const result = await httpJSON(API.kbIngest, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          title: title,
          category: (node('ingestCategory').value || '').trim() || '未分类',
          text: text,
          strategy: node('ingestStrategy').value || 'recursive',
          rebuild: false
        })
      }, UPLOAD_TIMEOUT);
      toast('文本入库成功:' + summarizeIngest(result), 'success');
      node('ingestForm').reset();
      await refreshKnowledge();
    } catch (err) {
      toast('文本入库失败:' + err.message, 'error');
    } finally {
      setButtonBusy(submit, false);
    }
  }

  async function handleReindex() {
    const strategySelect = node('reindexStrategy');
    const strategy = (strategySelect && strategySelect.value) || 'recursive';
    const ok = await confirmDialog({
      title: '重建全部索引',
      body: '将重新读取 data/raw 下的全部原始文档,按「' + strategy +
        '」策略重新切分并重建向量与 BM25 索引。演示语料很少,通常几秒完成,期间检索会短暂变慢。确定继续吗?',
      okText: '开始重建',
      danger: true
    });
    if (!ok) return;

    const button = node('reindexBtn');
    setButtonBusy(button, true, '重建中…');
    try {
      const result = await httpJSON(API.kbReindex(strategy), { method: 'POST' }, UPLOAD_TIMEOUT);
      toast('索引重建完成:' + summarizeIngest(result), 'success');
      await refreshKnowledge();
    } catch (err) {
      toast('重建失败:' + err.message, 'error');
    } finally {
      setButtonBusy(button, false);
    }
  }

  // ===========================================================================
  // 11. 关于视图
  // ===========================================================================

  function aboutInfoRow(key, value, valueClass) {
    return '<div class="info-row"><span class="info-key">' + esc(key) + '</span>' +
      '<span class="info-val' + (valueClass ? ' ' + valueClass : '') + '">' +
      esc(value) + '</span></div>';
  }

  function aboutTag(text) {
    return '<span class="feat">' + esc(text) + '</span>';
  }

  function pipeNode(kicker, title, desc, extraClass) {
    return '<div class="pipe-node' + (extraClass ? ' ' + extraClass : '') + '">' +
      '<div class="pipe-kicker">' + esc(kicker) + '</div>' +
      '<div class="pipe-title">' + esc(title) + '</div>' +
      '<div class="pipe-desc">' + esc(desc) + '</div></div>';
  }

  function pipeArrow() {
    return '<div class="pipe-arrow" aria-hidden="true">' + icon('arrow') + '</div>';
  }

  function renderAbout() {
    const host = node('aboutContent');
    if (!host) return;
    const meta = state.meta;
    const health = state.health;
    const prompts = state.prompts;

    if (!meta) {
      if (state.aboutError) {
        host.innerHTML = emptyStateHTML('alert', '系统信息加载失败',
          state.aboutError + ' —— 请确认 http://127.0.0.1:8000 服务正常运行后刷新页面。');
      } else {
        host.innerHTML = emptyStateHTML('layers', '加载系统信息…',
          '正在读取 /api/meta、/api/health 与 /api/prompts。');
      }
      return;
    }

    const llm = meta.llm || {};
    const embedding = meta.embedding || {};
    const rerank = meta.rerank || {};
    const intent = meta.intent || {};
    const kb = meta.kb || {};
    const promptVersions = (prompts && prompts.versions) || meta.prompts || {};
    const pipeline = [];

    // ---- Hero ----
    pipeline.push('<section class="about-hero">' +
      '<h2>KnowledgeFlow · 企业知识库智能问答 Agent</h2>' +
      '<p>面向面试演示的完整 RAG 系统:意图路由决定走哪条链路,混合检索(向量 + BM25 + RRF)负责召回,' +
      '幻觉自检负责质量兜底(重排可按需开启),Agent 工具循环与迭代检索负责复杂问题。' +
      '右侧「检索链路」把每一次召回、融合、重排、自检都实时画出来,回答里的 [n] 可以逐条回溯到原文片段。</p>' +
      '<div class="about-hero-tags">' +
      aboutTag('原生 JS · 无构建 · 无 CDN') +
      aboutTag('FastAPI · SSE 流式') +
      aboutTag('DeepSeek 真实模型') +
      aboutTag('向量 + BM25 + RRF') +
      aboutTag('幻觉自检 + 引用溯源') +
      aboutTag('Prometheus /metrics') +
      '</div></section>');

    // ---- 技术栈卡片 ----
    pipeline.push('<div class="grid-cards">' +
      '<div class="info-card"><h3>模型与向量</h3>' +
      aboutInfoRow('LLM 后端', llm.backend || '—') +
      aboutInfoRow('LLM 模型', shortModel(llm.model)) +
      aboutInfoRow('Embedding', shortModel(embedding.model)) +
      aboutInfoRow('向量维度', embedding.dim ? embedding.dim + ' 维' : '—') +
      aboutInfoRow('向量库', meta.vector_store || '—') +
      '</div>' +
      '<div class="info-card"><h3>检索与重排</h3>' +
      aboutInfoRow('混合检索', '向量 + BM25 双路') +
      aboutInfoRow('融合算法', 'RRF(倒数排名融合)') +
      aboutInfoRow('重排', rerank.enabled ? ('开 · ' + shortModel(rerank.model)) : '关(可配置)',
        rerank.enabled ? 'is-ok' : 'is-off') +
      aboutInfoRow('候选上限', '每路召回后按分数截断') +
      aboutInfoRow('引用粒度', 'chunk(文本块)级') +
      '</div>' +
      '<div class="info-card"><h3>意图路由</h3>' +
      aboutInfoRow('路由开关', intent.enabled ? '开' : '关', intent.enabled ? 'is-ok' : 'is-off') +
      aboutInfoRow('分类后端', intent.backend || 'rule') +
      aboutInfoRow('置信度阈值', intent.threshold !== undefined ? String(intent.threshold) : '—') +
      aboutInfoRow('快路径', 'weather / calculator / datetime 零 LLM 直达') +
      aboutInfoRow('低置信兜底', '转 Agent 工具循环') +
      '</div>' +
      '<div class="info-card"><h3>知识库</h3>' +
      aboutInfoRow('文档数', fmtNum(kb.documents) + ' 篇') +
      aboutInfoRow('文本块数', fmtNum(kb.chunks) + ' 块') +
      aboutInfoRow('BM25', kb.bm25 ? '已建索引' : '未启用', kb.bm25 ? 'is-ok' : 'is-off') +
      aboutInfoRow('切分策略', 'recursive / fixed / semantic') +
      aboutInfoRow('上传格式', '.md / .txt(UTF-8)') +
      '</div>' +
      '<div class="info-card"><h3>Prompt 版本</h3>' +
      Object.keys(promptVersions).map(function (name) {
        return aboutInfoRow(name, 'v' + promptVersions[name]);
      }).join('') +
      '</div>' +
      '<div class="info-card"><h3>运行时</h3>' +
      aboutInfoRow('服务状态', health && health.status ? health.status : '—',
        health && health.status === 'ok' ? 'is-ok' : 'is-warn') +
      aboutInfoRow('版本', health && health.version ? 'v' + health.version : (meta.app || '—')) +
      aboutInfoRow('已运行', health ? fmtUptime(health.uptime_s) : '—') +
      aboutInfoRow('演示模式', meta.offline_demo ? '离线演示(offline_demo)' : '真实模型调用',
        meta.offline_demo ? 'is-warn' : 'is-ok') +
      aboutInfoRow('接口文档', '/docs · /metrics') +
      '</div>' +
      '</div>');

    // ---- 架构链路 ----
    pipeline.push('<section class="section-card"><h3>请求处理链路</h3>' +
      '<div class="pipeline">' +
      pipeNode('01 · route', '意图路由', '规则/模型判断问题类型,决定走工具直达、闲聊、检索还是 Agent。', 'is-accent') +
      pipeArrow() +
      pipeNode('02 · rewrite', '查询改写 / HyDE', '把口语问题改写成检索友好查询,必要时生成假设答案提升召回。') +
      pipeArrow() +
      pipeNode('03 · recall', '向量召回', 'embedding 相似度检索,负责语义相近的表述。', 'is-teal') +
      pipeArrow() +
      pipeNode('04 · recall', 'BM25 召回', '关键词倒排检索,负责专有名词与数字。', 'is-teal') +
      pipeArrow() +
      pipeNode('05 · fuse', 'RRF 融合', '按名次融合两路结果,避免不同分数量纲不可比。') +
      pipeArrow() +
      pipeNode('06 · rerank', '重排', '交叉编码器精排,把真正能回答问题的片段顶上来。') +
      pipeArrow() +
      pipeNode('07 · generate', 'LLM 生成', '用 top 证据拼上下文,要求逐句标注 [n] 引用。', 'is-accent') +
      pipeArrow() +
      pipeNode('08 · verify', '幻觉自检', '校验结论是否被证据支持,必要时修订答案或拒答。', 'is-teal') +
      '</div>' +
      '<p class="arch-note">同一条链路上,<code>fast</code> 只走 03→05→07,<code>quality</code> 走全链路,' +
      '<code>agent</code> 由模型自主决定调用哪些工具,<code>agentic</code> 则把「检索 → 判断证据是否充分 → 补检」循环多轮。' +
      '</p></section>');

    // ---- 模式说明表 ----
    const modeRows = [
      ['auto', '自动', '意图分类 + 置信度阈值,自动选择最省成本的路径。'],
      ['fast', 'rag', '混合检索 + RRF 融合直接生成,延迟最低,演示现场首选。'],
      ['quality', 'rag', MODE_META.quality.desc],
      ['agent', 'agent', '工具循环:模型自主调用 kb_search / kb_read 等,最多若干轮。'],
      ['agentic', 'agentic', '迭代检索:每轮判断证据充分性,不足则生成补检查询继续检索。']
    ];
    pipeline.push('<section class="section-card"><h3>五种回答模式</h3>' +
      '<div style="overflow-x:auto"><table class="mode-table"><thead><tr>' +
      '<th>模式</th><th>路径</th><th>说明</th></tr></thead><tbody>' +
      modeRows.map(function (row) {
        return '<tr><td class="mono">' + esc(row[0]) + '</td><td>' + esc(row[1]) +
          '</td><td>' + esc(row[2]) + '</td></tr>';
      }).join('') +
      '</tbody></table></div>' +
      '<p class="arch-note">模式不是写死在前端的:按钮来自 <code>/api/meta</code> 的 <code>modes</code> 字段,' +
      '后端新增模式后前端会自动出现对应按钮。</p></section>');

    // ---- 运行指标 ----
    pipeline.push('<section class="section-card"><h3>当前运行指标</h3>' +
      '<div class="about-stats">' +
      '<div class="big-stat"><div class="big-stat-label">知识库文档</div>' +
      '<div class="big-stat-value">' + fmtNum(kb.documents) + '<small>篇</small></div></div>' +
      '<div class="big-stat"><div class="big-stat-label">可检索文本块</div>' +
      '<div class="big-stat-value">' + fmtNum(kb.chunks) + '<small>块</small></div></div>' +
      '<div class="big-stat"><div class="big-stat-label">服务连续运行</div>' +
      '<div class="big-stat-value">' + esc(health ? fmtUptime(health.uptime_s) : '—') + '</div></div>' +
      '<div class="big-stat"><div class="big-stat-label">意图置信阈值</div>' +
      '<div class="big-stat-value">' + esc(intent.threshold !== undefined ? String(intent.threshold) : '—') +
      '</div></div>' +
      '</div>' +
      '<p class="arch-note">指标接口:<code>/metrics</code> 暴露 Prometheus 文本,按 mode / path 维度记录请求数、' +
      '耗时、token 与成本,可以直接接到 Grafana 面板;右侧面板展示的是单次请求的实时采样。</p></section>');

    // ---- 面试讲解要点 ----
    pipeline.push('<section class="section-card"><h3>面试讲解要点</h3>' +
      '<div class="arch-note">1. <b>为什么不是「向量一把梭」</b>:向量擅长语义相近,BM25 擅长专有名词、编号和数字;' +
      '两路召回用 RRF 按名次融合,避免余弦分数与 BM25 分数不同量纲无法直接加权。</div>' +
      '<div class="arch-note">2. <b>为什么要有意图路由</b>:不是所有问题都值得走完整 RAG。天气/算式/时间走零 LLM 工具直达,' +
      '闲聊走小模型,只有知识库问题才检索——延迟和成本差一个数量级,这在企业里是硬需求。</div>' +
      '<div class="arch-note">3. <b>引用与自检为什么重要</b>:RAG 最大的风险是「看起来很像但没依据」。' +
      '强制答案标注 [n]、自检结论分 supported / partial / unsupported 三档,前端还能点 [n] 跳到证据,' +
      '这样面试官可以当场验证每一句的依据。</div>' +
      '<div class="arch-note">4. <b>可观测性</b>:每次请求都有 request_id、阶段耗时、检索候选与分数、' +
      'LLM 调用次数与 token 成本,右侧面板就是一条完整的 trace;这套数据同时进 Prometheus,可做回归对比。</div>' +
      '<div class="arch-note">5. <b>评测闭环</b>:data/eval 下有检索评测集与 out-of-scope 拒答集,' +
      '改切分策略或换 embedding 后可以跑同一批 query 对比召回率与拒答率,而不是靠感觉调参。</div>' +
      '</section>');

    host.innerHTML = pipeline.join('');
  }

  // ===========================================================================
  // 12. 视图切换、启动流程与事件绑定
  // ===========================================================================

  var VIEWS = ['chat', 'kb', 'about'];

  function switchView(name) {
    if (VIEWS.indexOf(name) === -1) return;
    $$('.tab').forEach(function (tab) {
      const active = tab.getAttribute('data-view') === name;
      tab.classList.toggle('is-active', active);
      tab.setAttribute('aria-selected', active ? 'true' : 'false');
      tab.setAttribute('tabindex', active ? '0' : '-1');
    });
    VIEWS.forEach(function (view) {
      const section = node('view-' + view);
      if (!section) return;
      const active = view === name;
      section.classList.toggle('is-active', active);
      section.hidden = !active;
    });
    if (name === 'kb') loadKbDocuments(false);
    if (name === 'about') loadAbout();
  }

  async function loadMeta(force) {
    if (state.meta && !force) return state.meta;
    try {
      const meta = await httpJSON(API.meta, {}, FETCH_TIMEOUT);
      state.meta = meta || null;
      applyRetrievalMeta(state.meta);
      if (meta && Array.isArray(meta.modes) && meta.modes.length) {
        state.modes = meta.modes.slice();
        if (state.modes.indexOf(state.mode) === -1) state.mode = state.modes[0];
      }
      if (meta && meta.kb) {
        if (meta.kb.documents !== undefined) state.kb.total = meta.kb.documents;
        if (meta.kb.chunks !== undefined) state.kb.chunks = meta.kb.chunks;
      }
      renderStatusBadges();
      renderModes();
      renderOfflineNotice();
      if (state.aboutLoaded) renderAbout();
      return state.meta;
    } catch (err) {
      renderStatusBadges();
      toast('系统信息加载失败:' + err.message, 'error');
      return null;
    }
  }

  async function loadHealth(force) {
    if (state.health && !force) return state.health;
    try {
      state.health = await httpJSON(API.health, {}, 8000);
    } catch (err) {
      state.health = null;
    }
    renderStatusBadges();
    return state.health;
  }

  async function loadDemoQuestions() {
    try {
      const data = await httpJSON(API.demo, {}, FETCH_TIMEOUT);
      renderDemoQuestions((data && data.questions) || []);
    } catch (err) {
      renderDemoQuestions([]);
      const hint = node('demoHint');
      if (hint) hint.textContent = '演示问题加载失败';
    }
  }

  async function loadPrompts() {
    try {
      state.prompts = await httpJSON(API.prompts, {}, FETCH_TIMEOUT);
    } catch (err) {
      state.prompts = null;
    }
    return state.prompts;
  }

  async function loadAbout() {
    if (state.aboutLoaded) {
      renderAbout();
      return;
    }
    state.aboutLoaded = true;
    await Promise.allSettled([loadMeta(false), loadHealth(false), loadPrompts()]);
    if (!state.meta) state.aboutError = '无法读取 /api/meta';
    renderAbout();
  }

  // ---- 事件处理(集中在底部,便于对照 index.html 的结构) ----

  function onComposerKeydown(event) {
    if (event.key !== 'Enter') return;
    if (event.shiftKey) return;
    // 中文输入法正在选词时按 Enter 只是确认候选词,不能当成发送
    if (event.isComposing || event.keyCode === 229) return;
    event.preventDefault();
    sendMessage();
  }

  function onQuestionPick(event) {
    const target = event.target.closest('[data-question]');
    if (!target) return;
    const question = target.getAttribute('data-question');
    hideDemoPopover();
    sendMessage(question);
  }

  function onMessagesClick(event) {
    const cite = event.target.closest('.cite-ref');
    if (cite) {
      focusEvidence(cite.getAttribute('data-cite'), cite);
      return;
    }
    const codeCopy = event.target.closest('[data-code-copy]');
    if (codeCopy) {
      const block = codeCopy.closest('.code-block');
      const code = block ? block.querySelector('pre code') : null;
      if (code) copyWithToast(code.textContent, '代码已复制');
      return;
    }
    const copyAnswer = event.target.closest('[data-act="copy-answer"]');
    if (copyAnswer) {
      const article = copyAnswer.closest('.msg-assistant');
      const turn = article ? state.turns[article.getAttribute('data-msg')] : null;
      if (turn) copyWithToast(turn.raw, '回答已复制');
    }
  }

  function onTraceClick(event) {
    const stageHead = event.target.closest('.stage-head');
    if (stageHead) {
      toggleCollapsible(stageHead);
      return;
    }

    const hitMore = event.target.closest('[data-act="toggle-hit"]');
    if (hitMore) {
      const hit = hitMore.closest('.hit');
      if (hit) {
        const expanded = hit.classList.toggle('is-expanded');
        hitMore.textContent = expanded ? '收起' : '展开全文';
      }
      return;
    }

    const evidenceMore = event.target.closest('[data-act="toggle-evidence"]');
    if (evidenceMore) {
      const card = evidenceMore.closest('.evidence-card');
      if (card) {
        const expanded = card.classList.toggle('is-expanded');
        evidenceMore.textContent = expanded ? '收起' : '展开证据';
      }
      return;
    }

    const toolToggle = event.target.closest('[data-act="toggle-tool"]');
    if (toolToggle) {
      const block = toolToggle.closest('.tool-block');
      const pre = block ? block.querySelector('.tool-pre') : null;
      if (pre) {
        const expanded = pre.classList.toggle('is-expanded');
        toolToggle.textContent = expanded ? '收起' : '展开全部';
      }
      return;
    }

    const traceToggle = event.target.closest('[data-act="toggle-trace"]');
    if (traceToggle) {
      const pre = node('traceContent').querySelector('[data-role="trace-raw"]');
      if (pre) {
        pre.hidden = !pre.hidden;
        traceToggle.textContent = pre.hidden ? '展开完整 trace JSON' : '收起完整 trace JSON';
        traceToggle.setAttribute('aria-expanded', pre.hidden ? 'false' : 'true');
      }
      return;
    }

    const copyTrigger = event.target.closest('[data-copy]');
    if (copyTrigger) {
      copyWithToast(copyTrigger.getAttribute('data-copy'), '请求 ID 已复制');
    }
  }

  function onTabKeydown(event) {
    if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].indexOf(event.key) === -1) return;
    event.preventDefault();
    const tabs = $$('.tab');
    let index = tabs.indexOf(event.currentTarget);
    if (event.key === 'ArrowRight') index = (index + 1) % tabs.length;
    else if (event.key === 'ArrowLeft') index = (index - 1 + tabs.length) % tabs.length;
    else if (event.key === 'Home') index = 0;
    else index = tabs.length - 1;
    const target = tabs[index];
    if (!target) return;
    target.focus();
    switchView(target.getAttribute('data-view'));
  }

  function onDocumentClick(event) {
    const popover = node('demoPopover');
    if (!popover || popover.hidden) return;
    if (event.target.closest('.composer-tools')) return;
    hideDemoPopover();
  }

  function onGlobalKeydown(event) {
    if (event.key !== 'Escape') return;
    const backdrop = node('modalBackdrop');
    if (backdrop && !backdrop.hidden) return; // 弹层自己处理 Esc
    const popover = node('demoPopover');
    if (popover && !popover.hidden) {
      hideDemoPopover();
      if (node('demoBtn')) node('demoBtn').focus();
    }
  }

  function copyTrace() {
    const data = TraceView.lastExport || state.lastTrace;
    if (!data) {
      toast('还没有可复制的链路 trace', 'warn');
      return;
    }
    copyWithToast(safeJson(data, 2), '完整 trace 已复制');
  }

  function bindEvents() {
    // 视图切换(Tab 头部 + 键盘左右键)
    $$('.tab').forEach(function (tab) {
      tab.addEventListener('click', function () {
        switchView(tab.getAttribute('data-view'));
      });
      tab.addEventListener('keydown', onTabKeydown);
    });

    // 模式切换
    const modeSegment = node('modeSegment');
    if (modeSegment) {
      modeSegment.addEventListener('click', function (event) {
        const button = event.target.closest('.mode-btn');
        if (!button) return;
        setMode(button.getAttribute('data-mode'));
      });
    }

    // 输入区
    const input = node('input');
    if (input) {
      input.addEventListener('input', autoGrowInput);
      input.addEventListener('keydown', onComposerKeydown);
    }
    if (node('sendBtn')) {
      node('sendBtn').addEventListener('click', function () {
        sendMessage();
      });
    }
    if (node('stopBtn')) node('stopBtn').addEventListener('click', abortStreaming);
    if (node('newSessionBtn')) node('newSessionBtn').addEventListener('click', newSession);

    // 演示问题
    if (node('demoBtn')) node('demoBtn').addEventListener('click', toggleDemoPopover);
    if (node('demoPopoverClose')) node('demoPopoverClose').addEventListener('click', hideDemoPopover);
    if (node('demoChips')) node('demoChips').addEventListener('click', onQuestionPick);
    if (node('demoPopoverBody')) node('demoPopoverBody').addEventListener('click', onQuestionPick);

    // 消息区(引用角标 / 代码复制 / 复制回答)
    if (node('messages')) node('messages').addEventListener('click', onMessagesClick);

    // 检索链路面板
    if (node('traceContent')) node('traceContent').addEventListener('click', onTraceClick);
    if (node('copyTraceBtn')) node('copyTraceBtn').addEventListener('click', copyTrace);

    // 知识库
    if (node('kbRefreshBtn')) {
      node('kbRefreshBtn').addEventListener('click', function () {
        loadKbDocuments(true).then(function () {
          if (state.kb.selectedId) loadKbDocument(state.kb.selectedId);
        });
      });
    }
    if (node('kbSearch')) {
      node('kbSearch').addEventListener('input', debounce(function () {
        state.kb.filterText = node('kbSearch').value || '';
        renderDocList();
      }, 120));
    }
    if (node('kbCats')) {
      node('kbCats').addEventListener('click', function (event) {
        const chip = event.target.closest('.cat-chip');
        if (!chip) return;
        const cat = chip.getAttribute('data-cat') || '';
        state.kb.filterCat = state.kb.filterCat === cat ? '' : cat;
        renderKbCats();
        renderDocList();
      });
    }
    if (node('docList')) {
      node('docList').addEventListener('click', function (event) {
        const row = event.target.closest('.doc-row');
        if (!row) return;
        loadKbDocument(row.getAttribute('data-doc'));
      });
    }
    if (node('kbDetail')) {
      node('kbDetail').addEventListener('click', function (event) {
        const head = event.target.closest('.chunk-head');
        if (head) toggleCollapsible(head);
      });
    }
    if (node('kbCopyChunksBtn')) {
      node('kbCopyChunksBtn').addEventListener('click', function () {
        if (!state.kb.detail) {
          toast('还没有可复制的文档', 'warn');
          return;
        }
        copyWithToast(chunksToText(state.kb.detail), '全部分块已复制');
      });
    }
    if (node('uploadForm')) node('uploadForm').addEventListener('submit', handleUpload);
    if (node('ingestForm')) node('ingestForm').addEventListener('submit', handleIngest);
    if (node('reindexBtn')) node('reindexBtn').addEventListener('click', handleReindex);

    // 全局:点空白关弹层、Esc 关闭
    document.addEventListener('click', onDocumentClick);
    document.addEventListener('keydown', onGlobalKeydown);
  }

  async function init() {
    renderStatusBadges();
    renderModes();
    renderSessionLabel();
    bindEvents();
    autoGrowInput();
    TraceView.reset();

    // 三个启动接口独立请求:任何一个失败都不该让页面白屏
    await Promise.allSettled([loadMeta(false), loadHealth(false), loadDemoQuestions()]);
    renderAbout();
    loadPrompts().then(function () {
      if (state.aboutLoaded) renderAbout();
    });
    refreshMemory();

    // 支持通过 #kb / #about 直接打开对应视图
    const hash = (window.location.hash || '').replace('#', '');
    if (hash === 'kb' || hash === 'about') switchView(hash);

    if (state.meta && state.meta.offline_demo) {
      toast('后端处于离线演示模式(offline_demo):回答由内置模拟生成', 'warn');
    }
    if (node('input')) node('input').focus();
  }

  // ===========================================================================
  // 13. 纯函数导出(Node 自测用);浏览器不会走到这里
  // ===========================================================================
  //
  // 用 node 加载本文件即可拿到 SSE 解析器与 Markdown 渲染器做离线单测,
  // 临时脚本放 /tmp,不在仓库里留测试文件:
  //   node -e "const kf=require('./web/app.js'); ..."
  // ---------------------------------------------------------------------------
  if (typeof window === 'undefined' && typeof module !== 'undefined' && module.exports) {
    module.exports = {
      createSSEParser: createSSEParser,
      renderMarkdown: renderMarkdown,
      renderInline: renderInline,
      esc: esc,
      sanitizeUrl: sanitizeUrl
    };
    return;
  }

  // 启动
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function () {
      init();
    });
  } else {
    init();
  }

  // 供面试演示时在控制台里手动调用(只读调试入口,不参与业务逻辑)
  window.KnowledgeFlow = {
    state: state,
    sendMessage: sendMessage,
    switchView: switchView,
    setMode: setMode,
    resetTrace: function () {
      TraceView.reset();
    }
  };
})();
