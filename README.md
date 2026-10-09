# Paper-Hallucination-Detect (PHD)

面向计算机论文的参考文献审计工具：使用 MinerU 解析 PDF，并通过 DBLP、arXiv 等学术信源核验引用，帮助发现缺失、错配或疑似幻觉引用。

> 输入一篇 PDF，即可得到可复核的 HTML、JSON 或 Markdown 报告。

## 快速开始

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
paperaudit check paper.pdf --format html
```

生成的网页报告位于 `outputs/<文件名>_paperaudit_basic_report.html`。

## 怎么用

### 安装

需要 Python 3.9 或更高版本：

```bash
git clone https://github.com/horizon0320/Paper-Hallucination-Detect.git
cd Paper-Hallucination-Detect
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### 检查 PDF

```bash
paperaudit check paper.pdf
```

程序会自动定位文末参考文献、拆分条目、提取标题/作者/年份等字段，然后查询学术信源并生成终端结果。

### 检查文献文件

支持 BibTeX、RIS、CSL-JSON、NBIB、DOI 列表和普通文本：

```bash
paperaudit check refs.bib
paperaudit check refs.ris
paperaudit check refs.json
paperaudit check refs.txt
```

也可以检查单条引用：

```bash
paperaudit check-one "Vaswani et al. (2017). Attention is all you need. NeurIPS."
```

输出格式默认为终端表格，也可以保存为 JSON 或 Markdown：

```bash
paperaudit check paper.pdf --format json > result.json
paperaudit check paper.pdf --format md > report.md
```

如果希望验证后直接生成网页报告，只需一条命令：

```bash
paperaudit check paper.pdf --format html
```

该命令会同时保存 `outputs/<文件名>_paperaudit_basic.json` 和
`outputs/<文件名>_paperaudit_basic_report.html`，并在终端打印文件路径。

也可以显式使用当前默认配置：

```bash
paperaudit check paper.pdf --format html --domain auto --concurrency 8
```

`--domain auto` 针对本项目范围内的计算机论文，对每条参考文献查询 DBLP 和 arXiv，再结合 Crossref、OpenAlex、Semantic Scholar 等信源交叉匹配。URL 存活检查、LLM 内容相关性检查和 SQLite 缓存默认不启用，因此不需要主动填写 `--no-url`、`--no-llm`、`--no-cache`；并发默认是 8。

### PDF 解析

PDF 默认使用 MinerU 3.4.5（Python 3.12）提取版面和参考文献，不使用旧版 PDF 解析器。当前环境的 MinerU 可执行文件为：

```bash
/Users/acc/Documents/ChatGPT/paper-ethical-audit/.venv-mineru/bin/mineru
```

MinerU 不可用时程序才会回退到兼容性解析路径。MinerU 输出后会保留版面结构，并用 PDF 文本层做保守字符校正，处理双栏、跨页、连字和 ACM 字体漏字问题。

### 浏览器上传页面

如果希望在网页中选择 PDF 并查看进度条：

```bash
paper-audit-web
```

然后打开 <http://127.0.0.1:8765>。上传和验证都在本机进行；任务完成后页面会直接提供 HTML 报告链接。

演示时可以打开 <http://127.0.0.1:8765/demo>。该页面使用演示进度并在同一页展示预先生成的报告，不会等待外部接口。

## API 限流与可选配置

DBLP 没有 token 认证；程序按单请求队列访问，默认间隔 1.5 秒。Semantic Scholar 官方 API key 的 introductory limit 是 1 request/second；程序默认按 1.1 秒串行访问，并尊重服务端 `Retry-After`。有 key 时可配置：

```bash
export SEMANTIC_SCHOLAR_API_KEY="your-api-key"
export PAPERAUDIT_DBLP_MIN_INTERVAL=2.0
export PAPERAUDIT_SEMANTIC_SCHOLAR_MIN_INTERVAL=2.0
```

没有 key 也可以运行；如果信源被限流、反爬或暂时不可达，报告会标记为 `UNRESOLVABLE`，不会把网络故障直接判成“疑似幻觉”。

## 示例报告

- [CoDe_SEC 最新示例报告](outputs/CoDe_SEC_camera_ready_paperaudit_basic_report.html)
- [Principled Policy Optimization 示例报告](outputs/24114_Principled_Policy_Optimi_paperaudit_basic_report.html)

## 当前功能

- 从 PDF 自动识别参考文献区域，支持双栏排版和跨页文献。
- 自动识别 IEEE、APA、LNCS/Springer、Vancouver、MLA、Chicago 等常见格式。
- 你的输入范围默认是计算机论文；`auto` 模式对每条参考文献无条件查询 DBLP 和 arXiv，再结合 Crossref、OpenAlex、Semantic Scholar 等学术信源进行交叉匹配。
- 识别会议论文、预印本和期刊记录之间的同一研究工作关系。
- 比较原始引用与搜索到的权威记录，包括标题、作者、年份、载体和 DOI。
- 生成 HTML、JSON 和 Markdown 结果，便于人工复核。

## 结果含义

| 结果 | 含义 |
|---|---|
| `VERIFIED` | 找到与引用信息一致的学术记录。 |
| `METADATA_MISMATCH` | 找到相关记录，但标题、作者或年份存在差异。 |
| `NOT_FOUND` | 当前查询的学术信源没有形成可确认记录，不等同于文献一定不存在。 |
| `UNRESOLVABLE` | 引用缺少足够字段，暂时无法建立可靠查询。 |

HTML 报告会列出原始引用、搜索到的记录、字段对照和各信源返回情况。作者列表确实不一致时，会单独标记为“疑似幻觉”；作者缩写、大小写和正常的预印本/会议版本差异不会仅凭这些差异判错。

## 环境要求

- Python 3.9+
- 可访问学术数据库接口的网络环境
- PDF 输入默认使用 MinerU 3.4.5（Python 3.12）；请准备 `/Users/acc/Documents/ChatGPT/paper-ethical-audit/.venv-mineru/bin/mineru`。MinerU 不可用时才回退到项目自带的 PDF 文本解析器。

可选设置：

```bash
export PAPERAUDIT_MAILTO="your-email@example.com"
```

该邮箱仅用于向部分学术接口提供礼貌联系信息，不是必需项。

## 退出状态

当存在 `NOT_FOUND` 结果时，命令会返回非零退出状态，适合在 CI 中使用。
