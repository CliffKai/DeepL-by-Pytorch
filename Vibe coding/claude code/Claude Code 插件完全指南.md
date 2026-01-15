# Claude Code 插件、MCP 与 Skills 完全指南

> 本文档详细介绍 Claude Code 的三大扩展机制：**Plugins（插件）**、**MCP（Model Context Protocol，模型上下文协议）**、以及 **Skills（技能）** 的使用方法、组件查找与自定义构建指南。
>
> **参考官方文档：**
> - [Agent Skills](https://code.claude.com/docs/en/skills)
> - [MCP](https://code.claude.com/docs/en/mcp)
> - [Plugins](https://code.claude.com/docs/en/plugins)

---

## 目录

1. [概述](#概述)
2. [/plugin 命令详解](#plugin-命令详解)
   - [什么是 Plugin](#什么是-plugin)
   - [Plugin 的组成结构](#plugin-的组成结构)
   - [使用 /plugin 命令](#使用-plugin-命令)
   - [查找和安装特定插件](#查找和安装特定插件)
   - [自定义构建 Plugin](#自定义构建-plugin)
3. [/mcp 命令详解](#mcp-命令详解)
   - [什么是 MCP](#什么是-mcp)
   - [MCP 的使用场景](#mcp-的使用场景)
   - [使用 /mcp 命令](#使用-mcp-命令)
   - [安装 MCP Server 的三种方式](#安装-mcp-server-的三种方式)
   - [MCP 安装范围](#mcp-安装范围)
   - [查找和安装 MCP Server](#查找和安装-mcp-server)
   - [自定义构建 MCP Server](#自定义构建-mcp-server)
4. [Skills 系统详解](#skills-系统详解)
   - [什么是 Skills](#什么是-skills)
   - [Skills 的工作原理](#skills-的工作原理)
   - [Skills 存放位置](#skills-存放位置)
   - [创建你的第一个 Skill](#创建你的第一个-skill)
   - [SKILL.md 配置详解](#skillmd-配置详解)
   - [Skills 高级功能](#skills-高级功能)
   - [查找现有 Skills](#查找现有-skills)
   - [故障排除](#故障排除)
5. [三者的对比与选择](#三者的对比与选择)
6. [参考资源](#参考资源)

---

## 概述

Claude Code 提供了三种扩展机制，让你能够根据自己的工作流程定制 AI 助手的能力：

| 扩展机制   | 定位                                            | 作用范围           | 触发方式                                |
| ---------- | ----------------------------------------------- | ------------------ | --------------------------------------- |
| **Plugin** | 功能包（包含命令、代理、技能、钩子、MCP服务器） | 跨项目可分享       | 显式调用斜杠命令 `/plugin-name:command` |
| **MCP**    | 外部工具连接（数据库、API、服务）               | 用户/项目/本地级别 | Claude 自动调用                         |
| **Skills** | 专业知识模块                                    | 项目/用户级别      | Claude 自动识别或手动 `/skill-name`     |

---

## /plugin 命令详解

### 什么是 Plugin

**Plugin（插件）** 是 Claude Code 的功能扩展包，将多种组件打包成可分享、可安装的单元。

#### 何时使用 Plugin vs 独立配置

| 场景                   | 推荐方式                       |
| ---------------------- | ------------------------------ |
| 为单个项目定制         | 使用 `.claude/` 目录的独立配置 |
| 个人配置，不需分享     | 独立配置                       |
| 需要短命令如 `/hello`  | 独立配置                       |
| 分享给团队或社区       | **Plugin**                     |
| 跨多个项目使用相同命令 | **Plugin**                     |
| 通过 Marketplace 分发  | **Plugin**                     |

> **注意**：Plugin 命令会使用命名空间，如 `/my-plugin:hello`，以避免不同插件间的冲突。

### Plugin 的组成结构

一个标准的 Claude Code 插件具有以下目录结构：

```
my-plugin/
├── .claude-plugin/
│   └── plugin.json          # 插件清单文件（必需）
├── commands/                 # 斜杠命令目录
│   └── hello.md             # 命令定义文件
├── agents/                   # 代理目录
│   └── reviewer.md          # 代理定义文件
├── skills/                   # 技能目录
│   └── code-review/
│       └── SKILL.md         # 技能定义文件
├── hooks/                    # 钩子目录
│   └── hooks.json           # 钩子配置
├── .mcp.json                 # MCP 服务器配置（可选）
├── .lsp.json                 # LSP 服务器配置（可选）
└── README.md                 # 插件说明文档
```

#### plugin.json 配置示例

```json
{
  "name": "my-first-plugin",
  "description": "A greeting plugin to learn the basics",
  "version": "1.0.0",
  "author": {
    "name": "Your Name"
  }
}
```

| 字段          | 说明                                              |
| ------------- | ------------------------------------------------- |
| `name`        | 插件名称，决定命令前缀如 `/my-first-plugin:hello` |
| `description` | 插件描述，出现在 Marketplace 中                   |
| `version`     | 语义化版本号                                      |
| `author`      | 作者信息                                          |
| `homepage`    | （可选）插件主页                                  |
| `repository`  | （可选）代码仓库                                  |
| `license`     | （可选）许可证                                    |

### 使用 /plugin 命令

#### 基本操作

```bash
# 打开插件管理菜单
/plugin

# 安装插件
/plugin install <plugin-name>@<marketplace-name>

# 查看已安装插件
/plugin list

# 卸载插件
/plugin uninstall <plugin-name>
```

#### 测试本地插件

```bash
# 使用 --plugin-dir 参数加载本地插件进行测试
claude --plugin-dir ./my-plugin

# 同时加载多个插件
claude --plugin-dir ./plugin-one --plugin-dir ./plugin-two
```

### 查找和安装特定插件

#### 官方与社区资源

| 资源名称                | 网址/方式                                                                        | 说明           |
| ----------------------- | -------------------------------------------------------------------------------- | -------------- |
| **发现插件文档**        | [Discover and install plugins](https://code.claude.com/docs/en/discover-plugins) | 官方安装指南   |
| **Plugin Marketplaces** | [Plugin marketplaces](https://code.claude.com/docs/en/plugin-marketplaces)       | 创建和分发市场 |
| **GitHub**              | 搜索 `.claude-plugin`                                                            | 社区插件       |

#### 常用插件类别

- **开发工作流**：代码提交、PR 审核、功能开发
- **文档生成**：代码库分析、变更日志、API 文档
- **LSP 集成**：TypeScript、Python、Go、Rust 等语言服务器
- **DevOps**：CI/CD、Docker、云平台部署

### 自定义构建 Plugin

#### Step 1: 创建插件目录结构

```bash
mkdir my-first-plugin
mkdir my-first-plugin/.claude-plugin
mkdir my-first-plugin/commands
```

#### Step 2: 编写 plugin.json

创建 `my-first-plugin/.claude-plugin/plugin.json`：

```json
{
  "name": "my-first-plugin",
  "description": "A greeting plugin to learn the basics",
  "version": "1.0.0",
  "author": {
    "name": "Your Name"
  }
}
```

#### Step 3: 创建斜杠命令

创建 `my-first-plugin/commands/hello.md`：

```markdown
---
description: Greet the user with a personalized message
---

# Hello Command

Greet the user named "$ARGUMENTS" warmly and ask how you can help them today.
Make the greeting personal and encouraging.
```

> **参数说明**：
> - `$ARGUMENTS` - 用户输入的所有参数
> - `$1`, `$2` - 第一个、第二个参数

#### Step 4: 添加 Skills（可选）

创建 `my-first-plugin/skills/code-review/SKILL.md`：

```markdown
---
name: code-review
description: Reviews code for best practices and potential issues. Use when reviewing code, checking PRs, or analyzing code quality.
---

When reviewing code, check for:
1. Code organization and structure
2. Error handling
3. Security concerns
4. Test coverage
```

#### Step 5: 添加 Hooks（可选）

创建 `my-first-plugin/hooks/hooks.json`：

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Write|Edit",
        "hooks": [{
          "type": "command",
          "command": "npm run lint:fix $FILE"
        }]
      }
    ]
  }
}
```

#### Step 6: 添加 MCP 服务器（可选）

创建 `my-first-plugin/.mcp.json`：

```json
{
  "database-tools": {
    "command": "${CLAUDE_PLUGIN_ROOT}/servers/db-server",
    "args": ["--config", "${CLAUDE_PLUGIN_ROOT}/config.json"],
    "env": {
      "DB_URL": "${DB_URL}"
    }
  }
}
```

> `${CLAUDE_PLUGIN_ROOT}` 会自动替换为插件根目录路径。

#### Step 7: 测试与调试

```bash
# 加载并测试插件
claude --plugin-dir ./my-first-plugin

# 测试命令
/my-first-plugin:hello Alex

# 查看可用命令
/help
```

**调试技巧**：
1. 确保目录结构正确（组件目录应在插件根目录，不是 `.claude-plugin/` 内）
2. 逐个测试各组件
3. 参考 [Debugging and development tools](https://code.claude.com/docs/en/plugins-reference#debugging-and-development-tools)

#### Step 8: 分发插件

1. 添加 `README.md` 说明安装和使用方法
2. 使用语义化版本更新 `plugin.json`
3. 发布到 [Plugin Marketplace](https://code.claude.com/docs/en/plugin-marketplaces)
4. 让团队成员测试

---

## /mcp 命令详解

### 什么是 MCP

**MCP（Model Context Protocol，模型上下文协议）** 是 Anthropic 开发的开放协议，用于标准化 AI 模型与外部工具、服务和数据源之间的连接。

### MCP 的使用场景

MCP 让 Claude Code 能够：

- **从问题跟踪器实现功能**：*"实现 JIRA issue ENG-4521 中描述的功能并创建 GitHub PR"*
- **分析监控数据**：*"检查 Sentry 和 Statsig 查看 ENG-4521 功能的使用情况"*
- **查询数据库**：*"基于 PostgreSQL 数据库，找出使用 ENG-4521 功能的 10 个随机用户的邮箱"*
- **集成设计**：*"根据 Slack 中发布的新 Figma 设计更新邮件模板"*
- **自动化工作流**：*"创建 Gmail 草稿邀请这 10 位用户参加新功能反馈会议"*

### 使用 /mcp 命令

在 Claude Code 中输入 `/mcp` 可以：
- 查看已配置的 MCP 服务器状态
- 认证需要 OAuth 的远程服务器
- 清除认证信息

### 安装 MCP Server 的三种方式

#### 方式一：添加远程 HTTP 服务器

```bash
# 基本语法
claude mcp add --transport http <name> <url>

# 示例：连接到 Notion
claude mcp add --transport http notion https://mcp.notion.com/mcp

# 带 Bearer Token 认证
claude mcp add --transport http secure-api https://api.example.com/mcp \
  --header "Authorization: Bearer your-token"
```

#### 方式二：添加远程 SSE 服务器

```bash
# 基本语法
claude mcp add --transport sse <name> <url>

# 示例：连接到 Asana
claude mcp add --transport sse asana https://mcp.asana.com/sse

# 带 API Key 认证
claude mcp add --transport sse private-api https://api.company.com/sse \
  --header "X-API-Key: your-key-here"
```

#### 方式三：添加本地 stdio 服务器

```bash
# 基本语法
claude mcp add [options] <name> -- <command> [args...]

# 示例：添加 Airtable 服务器
claude mcp add --transport stdio --env AIRTABLE_API_KEY=YOUR_KEY airtable \
  -- npx -y airtable-mcp-server
```

**参数说明**：
- `--transport`：传输类型（http/sse/stdio）
- `--env`：设置环境变量，如 `--env KEY=value`
- `--scope`：安装范围
- `--header`：HTTP 头（用于认证）
- `--`：分隔符，之后是实际执行的命令

#### 管理已安装的服务器

```bash
# 列出所有配置的服务器
claude mcp list

# 获取特定服务器详情
claude mcp get github

# 移除服务器
claude mcp remove github

# 在 Claude Code 内检查状态
/mcp
```

### MCP 安装范围

| Scope             | 说明                              | 存储位置               |
| ----------------- | --------------------------------- | ---------------------- |
| **local**（默认） | 仅对当前项目中的你可用            | `~/.claude.json`       |
| **project**       | 通过 `.mcp.json` 共享给项目所有人 | 项目根目录 `.mcp.json` |
| **user**          | 对你的所有项目可用                | `~/.claude.json`       |

```bash
# 指定安装范围
claude mcp add --transport http stripe --scope local https://mcp.stripe.com
claude mcp add --transport http paypal --scope project https://mcp.paypal.com/mcp
claude mcp add --transport http hubspot --scope user https://mcp.hubspot.com/anthropic
```

#### 选择合适的范围

- **Local**：个人服务器、实验配置、特定项目的敏感凭证
- **Project**：团队共享的服务器、项目特定工具、协作所需的服务
- **User**：跨多项目的个人工具、开发工具、常用服务

### 查找和安装 MCP Server

#### 官方与社区资源

| 资源名称         | 网址                                              | 说明                  |
| ---------------- | ------------------------------------------------- | --------------------- |
| **MCP 官方仓库** | https://github.com/modelcontextprotocol/servers   | 官方 MCP Server 集合  |
| **MCP SDK**      | https://modelcontextprotocol.io/quickstart/server | 构建自己的 MCP Server |

#### 常用 MCP Server 示例

**监控错误 - Sentry**：
```bash
# 1. 添加 Sentry MCP 服务器
claude mcp add --transport http sentry https://mcp.sentry.dev/mcp

# 2. 使用 /mcp 进行 OAuth 认证
> /mcp

# 3. 开始使用
> "过去 24 小时最常见的错误是什么？"
> "显示错误 ID abc123 的堆栈跟踪"
```

**代码审核 - GitHub**：
```bash
# 1. 添加 GitHub MCP 服务器
claude mcp add --transport http github https://api.githubcopilot.com/mcp/

# 2. 认证
> /mcp

# 3. 使用
> "审核 PR #456 并提出改进建议"
> "为我们刚发现的 bug 创建一个新 issue"
```

**数据库查询 - PostgreSQL**：
```bash
# 添加数据库服务器
claude mcp add --transport stdio db -- npx -y @bytebase/dbhub \
  --dsn "postgresql://readonly:password@localhost:5432/analytics"

# 自然语言查询
> "这个月的总收入是多少？"
> "显示 orders 表的 schema"
```

### 自定义构建 MCP Server

#### 使用 JSON 配置添加

```bash
# 基本语法
claude mcp add-json <name> '<json>'

# HTTP 服务器示例
claude mcp add-json weather-api '{"type":"http","url":"https://api.weather.com/mcp","headers":{"Authorization":"Bearer token"}}'

# stdio 服务器示例
claude mcp add-json local-weather '{"type":"stdio","command":"/path/to/weather-cli","args":["--api-key","abc123"],"env":{"CACHE_DIR":"/tmp"}}'
```

#### 使用 Python + FastMCP 构建

```python
# server.py
from datetime import datetime
from fastmcp import FastMCP

mcp = FastMCP(
    name="my-custom-server",
    description="我的自定义 MCP Server"
)

@mcp.tool()
def get_current_time() -> str:
    """获取当前时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@mcp.tool()
def add_numbers(a: int, b: int) -> int:
    """将两个数字相加
    
    Args:
        a: 第一个数字
        b: 第二个数字
    
    Returns:
        两数之和
    """
    return a + b

if __name__ == "__main__":
    mcp.run()
```

注册到 Claude Code：
```bash
claude mcp add --transport stdio my-server -- python /path/to/server.py
```

#### 从 Claude Desktop 导入

```bash
# 导入 Claude Desktop 的 MCP 配置
claude mcp add-from-claude-desktop

# 验证导入
claude mcp list
```

> 此功能仅支持 macOS 和 WSL

#### 将 Claude Code 作为 MCP Server

```bash
# 启动 Claude 作为 MCP 服务器
claude mcp serve
```

在其他客户端配置：
```json
{
  "mcpServers": {
    "claude-code": {
      "type": "stdio",
      "command": "claude",
      "args": ["mcp", "serve"],
      "env": {}
    }
  }
}
```

---

## Skills 系统详解

### 什么是 Skills

**Skills（技能）** 是 Claude Code 的专业知识模块，它们像"便携式说明书"一样，教会 Claude 特定的工作流程和专业知识。

### Skills 的工作原理

1. **发现（Discovery）**：Claude 在启动时发现可用的 Skills
2. **激活（Activation）**：根据 `description` 判断是否与当前任务相关
3. **执行（Execution）**：加载 SKILL.md 内容并按指示执行

### Skills 存放位置

| 位置       | 路径                | 作用范围         |
| ---------- | ------------------- | ---------------- |
| **用户级** | `~/.claude/skills/` | 当前用户所有项目 |
| **项目级** | `.claude/skills/`   | 当前项目所有用户 |
| **插件内** | `plugin/skills/`    | 随插件安装       |

### 创建你的第一个 Skill

#### Step 1: 检查可用 Skills

```
What Skills are available?
```

#### Step 2: 创建 Skill 目录

```bash
mkdir -p ~/.claude/skills/explaining-code
```

#### Step 3: 编写 SKILL.md

创建 `~/.claude/skills/explaining-code/SKILL.md`：

```markdown
---
name: explaining-code
description: Explains code with visual diagrams and analogies. Use when explaining how code works, teaching about a codebase, or when the user asks "how does this work?"
---

When explaining code, always include:

1. **Start with an analogy**: Compare the code to something from everyday life
2. **Draw a diagram**: Use ASCII art to show the flow, structure, or relationships
3. **Walk through the code**: Explain step-by-step what happens
4. **Highlight a gotcha**: What's a common mistake or misconception?

Keep explanations conversational. For complex concepts, use multiple analogies.
```

#### Step 4: 验证并测试

```
What Skills are available?
```

应该能看到 `explaining-code` 出现在列表中。

测试：
```
How does this code work?
```

Claude 应该会使用 `explaining-code` Skill 来解释代码。

### SKILL.md 配置详解

#### 基本格式

```markdown
---
name: your-skill-name
description: Brief description of what this Skill does and when to use it
---

# Your Skill Name

## Instructions
Provide clear, step-by-step guidance for Claude.

## Examples
Show concrete examples of using this Skill.
```

#### 完整元数据字段

| 字段                       | 说明             | 示例                                 |
| -------------------------- | ---------------- | ------------------------------------ |
| `name`                     | Skill 名称       | `code-review`                        |
| `description`              | 描述及触发条件   | `Review code for best practices...`  |
| `allowed-tools`            | 限制可用工具     | `Read, Grep, Glob`                   |
| `model`                    | 指定使用的模型   | `claude-sonnet-4-20250514`           |
| `context`                  | 执行上下文       | `fork`（在独立上下文执行）           |
| `agent`                    | 代理类型         | `Explore`, `Plan`, `general-purpose` |
| `hooks`                    | 定义钩子         | `PreToolUse`, `PostToolUse`, `Stop`  |
| `user-invocable`           | 是否允许手动调用 | `true`/`false`                       |
| `disable-model-invocation` | 禁止模型自动调用 | `true`/`false`                       |

### Skills 高级功能

#### 限制工具访问

使用 `allowed-tools` 创建只读 Skill：

```markdown
---
name: reading-files-safely
description: Read files without making changes. Use when you need read-only file access.
allowed-tools:
  - Read
  - Grep
  - Glob
---
```

#### 在独立上下文中运行

使用 `context: fork` 隔离执行环境：

```markdown
---
name: code-analysis
description: Analyze code quality and generate detailed reports
context: fork
---
```

#### 定义钩子

```markdown
---
name: secure-operations
description: Perform operations with additional security checks
hooks:
  PreToolUse:
    - matcher: "Bash"
      hooks:
        - type: command
          command: "./scripts/security-check.sh $TOOL_INPUT"
          once: true
---
```

#### 控制可见性

Skills 可以通过三种方式被调用：
1. **手动调用**：在提示中输入 `/skill-name`
2. **程序调用**：Claude 通过 Skill 工具调用
3. **自动发现**：Claude 根据 description 判断是否加载

```markdown
# 仅允许模型调用，不允许手动调用
---
name: internal-review-standards
description: Apply internal code review standards when reviewing pull requests
user-invocable: false
---
```

#### 多文件 Skill 结构

```
pdf-processing/
├── SKILL.md              # 概述和导航（必需）
├── FORMS.md              # 表单字段映射和填充说明
├── REFERENCE.md          # pypdf 和 pdfplumber API 详情
└── scripts/
    ├── fill_form.py      # 填充表单字段的工具
    └── validate.py       # 检查 PDF 必需字段
```

`SKILL.md` 内容：

```markdown
---
name: pdf-processing
description: Extract text, fill forms, merge PDFs. Use when working with PDF files, forms, or document extraction. Requires pypdf and pdfplumber packages.
allowed-tools: Read, Bash(python:*)
---

# PDF Processing

## Quick start
Extract text:
```python
import pdfplumber
with pdfplumber.open("doc.pdf") as pdf:
    text = pdf.pages[0].extract_text()
```

For form filling, see [FORMS.md](FORMS.md).
For detailed API reference, see [REFERENCE.md](REFERENCE.md).

## Requirements
Packages must be installed in your environment:
```bash
pip install pypdf pdfplumber
```
```

### 查找现有 Skills

#### 官方资源

| 资源                                                                                            | 说明         |
| ----------------------------------------------------------------------------------------------- | ------------ |
| [Agent Skills 概述](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview) | 官方概念说明 |
| [最佳实践指南](https://docs.claude.com/en/docs/agents-and-tools/agent-skills/best-practices)    | 编写指南     |
| [Agent SDK 中使用 Skills](https://docs.claude.com/en/docs/agents-and-tools/agent-skills/sdk)    | SDK 集成     |

#### 检查已安装的 Skills

```bash
# 查看用户级 Skills
ls ~/.claude/skills/

# 查看项目级 Skills
ls .claude/skills/
```

或在 Claude Code 中询问：
```
What Skills are available?
```

### 故障排除

#### Skill 不触发

检查 `description` 是否清晰说明：
1. 这个 Skill 做什么？
2. 什么时候 Claude 应该使用它？

**示例**：
```markdown
description: Extract text and tables from PDF files, fill forms, merge documents. Use when working with PDF files or when the user mentions PDFs, forms, or document extraction.
```

#### Skill 不加载

1. 检查 `SKILL.md` 是否存在且路径正确：
   - `~/.claude/skills/my-skill/SKILL.md`
   - `.claude/skills/my-skill/SKILL.md`
2. 检查 YAML frontmatter 格式：必须以 `---` 开始和结束
3. 使用 `claude --debug` 查看详细日志

#### Skill 有错误

1. 检查脚本权限：`chmod +x scripts/*.py`
2. 注意路径分隔符：Unix 使用 `/`，Windows 使用 `\`

#### Plugin Skills 不出现

```bash
# 清除插件缓存
rm -rf ~/.claude/plugins/cache

# 重新安装插件
/plugin install plugin-name@marketplace-name
```

确保插件结构正确：
```
my-plugin/
├── .claude-plugin/
│   └── plugin.json
└── skills/
    └── my-skill/
        └── SKILL.md
```

---

## 三者的对比与选择

### 功能对比表

| 特性           | Plugin                      | MCP               | Skills          |
| -------------- | --------------------------- | ----------------- | --------------- |
| **主要用途**   | 打包和分发功能              | 连接外部工具/服务 | 封装专业知识    |
| **触发方式**   | 显式斜杠命令                | Claude 自动调用   | Claude 自动判断 |
| **包含内容**   | 命令、代理、钩子、技能、MCP | 工具定义          | 知识和流程      |
| **技术复杂度** | 中等                        | 较高              | **最低**        |
| **分发方式**   | Marketplace                 | 配置文件/CLI      | 目录复制        |

### 选择决策树

```
需要扩展 Claude Code？
    │
    ├── 需要访问外部服务/API？
    │   └── 是 → 使用 MCP
    │
    ├── 只需要知识性指导？
    │   └── 是 → 使用 Skills
    │
    ├── 需要打包多个组件分发？
    │   └── 是 → 使用 Plugin
    │
    └── 只在当前项目使用？
        └── 是 → 使用 .claude/ 独立配置
```

---

## 参考资源

### 官方文档

| 文档                    | 链接                                                |
| ----------------------- | --------------------------------------------------- |
| **Agent Skills**        | https://code.claude.com/docs/en/skills              |
| **MCP**                 | https://code.claude.com/docs/en/mcp                 |
| **Plugins**             | https://code.claude.com/docs/en/plugins             |
| **Plugins Reference**   | https://code.claude.com/docs/en/plugins-reference   |
| **Discover Plugins**    | https://code.claude.com/docs/en/discover-plugins    |
| **Plugin Marketplaces** | https://code.claude.com/docs/en/plugin-marketplaces |
| **MCP GitHub 仓库**     | https://github.com/modelcontextprotocol/servers     |
| **MCP 官方网站**        | https://modelcontextprotocol.io                     |

### 相关功能

| 功能                   | 链接                                           |
| ---------------------- | ---------------------------------------------- |
| **Slash Commands**     | https://code.claude.com/docs/en/slash-commands |
| **Subagents**          | https://code.claude.com/docs/en/sub-agents     |
| **Hooks**              | https://code.claude.com/docs/en/hooks          |
| **Memory (CLAUDE.md)** | https://code.claude.com/docs/en/memory         |

---

> 📝 **最后更新**: 2026-01-14  
> 📚 **数据来源**: Claude Code 官方文档
