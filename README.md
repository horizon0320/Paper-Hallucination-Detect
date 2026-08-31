# Paper-audit

Paper-audit 用于检查论文参考文献是否能够在学术数据库中找到，并比较引用信息与数据库记录是否一致。

## 怎么用

### 安装

需要 Python 3.9 或更高版本：

```bash
git clone https://github.com/horizon0320/Paper-audit.git
cd Paper-audit
pip install -e ".[dev]"
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

### 浏览器上传页面

如果希望在网页中选择 PDF 并查看进度条：

```bash
paper-audit-web
```

然后打开 <http://127.0.0.1:8765>。上传和验证都在本机进行；任务完成后页面会直接提供 HTML 报告链接。

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
- PDF 输入需要 PyMuPDF（安装项目依赖时会自动安装）

可选设置：

```bash
export PAPERAUDIT_MAILTO="your-email@example.com"
```

该邮箱仅用于向部分学术接口提供礼貌联系信息，不是必需项。

## 退出状态

当存在 `NOT_FOUND` 结果时，命令会返回非零退出状态，适合在 CI 中使用。
