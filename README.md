# Paper-Hallucination-Detect

面向计算机论文的参考文献审计工具。项目使用 MinerU 解析 PDF 版面与文献条目，再通过 DBLP、arXiv、Crossref、OpenAlex 和 Semantic Scholar 等学术信源交叉核验引用，帮助发现缺失、错配和疑似幻觉引用。

<p align="center">
  <strong>PDF → 参考文献解析 → 多源检索 → 可复核报告</strong>
</p>

## 项目特点

- **MinerU 解析**：优先使用 MinerU 3.4.5 处理双栏、跨页、连字和复杂排版。
- **计算机论文保底检索**：`auto` 模式对每条引用查询 DBLP 和 arXiv，再结合其他学术信源交叉匹配。
- **多种输入输出**：支持 PDF、BibTeX、RIS、CSL-JSON、NBIB、DOI 列表和普通文本；可输出终端表格、JSON、Markdown 或 HTML。
- **结果可复核**：报告保留原始引用、候选记录、标题/作者/年份/载体对照以及各信源状态。
- **本地网页演示**：浏览器中上传 PDF，实时查看进度，并在同一页面打开生成的报告。

## 演示

观看项目讲解视频：[`docs/paper-hallucination-detect-demo.mp4`](docs/paper-hallucination-detect-demo.mp4)

也可以直接打开内置演示页面：

```bash
paper-audit-web
```

然后访问 <http://127.0.0.1:8765/demo>。演示页面使用仓库中的预生成结果，不会等待外部接口。

## 快速开始

需要 Python 3.9 或更高版本：

```bash
git clone https://github.com/horizon0320/Paper-Hallucination-Detect.git
cd Paper-Hallucination-Detect

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

检查一篇 PDF 并直接生成 HTML：

```bash
paperaudit check paper.pdf --format html
```

报告会保存到：

```text
outputs/<文件名>_paperaudit_basic_report.html
outputs/<文件名>_paperaudit_basic.json
```

## 命令行用法

检查 PDF、参考文献文件或单条引用：

```bash
paperaudit check paper.pdf
paperaudit check refs.bib
paperaudit check refs.ris
paperaudit check refs.json
paperaudit check-one "Vaswani et al. (2017). Attention is all you need. NeurIPS."
```

选择输出格式：

```bash
paperaudit check paper.pdf --format html
paperaudit check paper.pdf --format json > result.json
paperaudit check paper.pdf --format md > report.md
```

当前默认配置为：

```text
--domain auto
--concurrency 8
```

在 `auto` 模式下，项目范围默认是计算机论文：每条参考文献都会查询 DBLP 和 arXiv，并结合 Crossref、OpenAlex、Semantic Scholar 等来源进行交叉匹配。URL 存活检查、LLM 内容相关性检查和 SQLite 缓存默认关闭，无需额外填写对应的 `--no-*` 参数。

## MinerU 配置

PDF 解析优先使用 MinerU 3.4.5。Paper-audit 本身仍可在 Python 3.9 上安装运行，MinerU 使用独立的 Python 3.12 环境，通过子进程调用。

默认路径是：

```text
/Users/acc/Documents/ChatGPT/paper-ethical-audit/.venv-mineru/bin/mineru
```

在其他机器上可显式指定 MinerU：

```bash
export PAPERAUDIT_MINERU_CLI=/path/to/.venv-mineru/bin/mineru
export PAPERAUDIT_MINERU_PROJECT=/path/to/paper-ethical-audit
```

运行时会使用 CPU pipeline、`modelscope` 模型源，并保留解析后的版面结构。MinerU 不可用时才回退到兼容性解析路径。

## 浏览器页面

启动本地上传页面：

```bash
paper-audit-web
```

打开 <http://127.0.0.1:8765>，选择 PDF 后即可看到解析、检索和报告生成进度。任务完成后，报告会直接显示在当前页面中。

## API 限流

项目对外部学术接口使用串行限流和 `Retry-After` 处理，避免把暂时的限流误判为论文不存在。可按网络环境调整间隔：

```bash
export SEMANTIC_SCHOLAR_API_KEY="your-api-key"
export PAPERAUDIT_DBLP_MIN_INTERVAL=2.0
export PAPERAUDIT_SEMANTIC_SCHOLAR_MIN_INTERVAL=2.0
```

没有 Semantic Scholar token 也可以运行；当信源被限流、反爬或暂时不可达时，报告会标记为 `UNRESOLVABLE`。

## 示例报告

- [CoDe_SEC 示例报告](outputs/CoDe_SEC_camera_ready_paperaudit_basic_report.html)
- [Principled Policy Optimization 示例报告](outputs/24114_Principled_Policy_Optimi_paperaudit_basic_report.html)

## 结果含义

| 状态 | 含义 |
| --- | --- |
| `VERIFIED` | 找到与引用信息基本一致的学术记录。 |
| `METADATA_MISMATCH` | 找到相关记录，但标题、作者、年份或载体存在差异。 |
| `NOT_FOUND` | 当前信源未形成可确认记录，不等同于文献一定不存在。 |
| `UNRESOLVABLE` | 引用字段不足，或外部信源暂时不可用。 |

作者缩写、大小写差异以及正常的预印本/会议版本差异不会仅凭单一字段直接判定为幻觉；明确的作者冲突会单独进入疑似幻觉提示。

## 项目结构

```text
paperaudit/
├── parse/       # MinerU 与兼容解析路径
├── sources/     # DBLP、arXiv、Crossref、OpenAlex 等信源
├── core/        # 多源验证与判定逻辑
├── web/         # 上传页与演示页
└── webapp.py    # 本地网页服务
outputs/         # 示例 JSON / HTML 报告
docs/            # 项目演示视频
tests/           # 单元测试与解析测试
```

## 开发与测试

```bash
python -m pytest tests/ -q
```

当前测试覆盖解析、文献拆分、信源适配、限流、判定逻辑、报告生成和网页接口。

## 许可证

本项目采用 [MIT License](LICENSE)。
