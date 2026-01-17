# Claude Code 扩展资源大全：Skills、Plugins、MCP

> 本文档全面整理了 Claude Code 的三大扩展机制 **Skills（技能）**、**Plugins（插件）**、**MCP（模型上下文协议）** 的优质资源，包含官方和社区资源的详细介绍、安装方法和使用场景。
>
> **最后更新**: 2026-01-17

---

## 目录

1. [资源概览](#资源概览)
2. [MCP 服务器资源](#mcp-服务器资源)
   - [官方核心 MCP 服务器](#官方核心-mcp-服务器)
   - [云平台与基础设施](#云平台与基础设施)
   - [开发工具集成](#开发工具集成)
   - [项目管理与协作](#项目管理与协作)
   - [监控与调试](#监控与调试)
   - [MCP 资源汇总网站](#mcp-资源汇总网站)
3. [Plugins 插件资源](#plugins-插件资源)
   - [官方插件市场](#官方插件市场)
   - [社区插件市场](#社区插件市场)
   - [开发工具类插件](#开发工具类插件)
   - [DevOps 与 CI/CD 插件](#devops-与-cicd-插件)
   - [测试与质量类插件](#测试与质量类插件)
   - [文档与项目管理插件](#文档与项目管理插件)
   - [安全类插件](#安全类插件)
4. [Skills 技能资源](#skills-技能资源)
   - [官方技能库](#官方技能库)
   - [社区技能合集](#社区技能合集)
   - [开发与技术类技能](#开发与技术类技能)
   - [文档处理类技能](#文档处理类技能)
   - [创意与设计类技能](#创意与设计类技能)
   - [安全与研究类技能](#安全与研究类技能)
5. [资源发现平台](#资源发现平台)
6. [参考链接](#参考链接)

---

## 资源概览

| 扩展类型    | 数量级  | 主要来源                        | 安装方式                  |
| ----------- | ------- | ------------------------------- | ------------------------- |
| **MCP**     | 100+    | modelcontextprotocol/servers    | `claude mcp add`          |
| **Plugins** | 200+    | 各大社区 Marketplace            | `/plugin install`         |
| **Skills**  | 50+     | anthropics/skills、社区贡献     | `/plugin marketplace add` |

---

## MCP 服务器资源

MCP（Model Context Protocol）服务器让 Claude Code 能够连接外部工具、服务和数据源，极大扩展 AI 的能力边界。

### 官方核心 MCP 服务器

这些是由 Model Context Protocol 官方团队维护的参考服务器：

| 服务器名称             | 功能描述                                 | 适用场景                             |
| ---------------------- | ---------------------------------------- | ------------------------------------ |
| **Filesystem**         | 安全的本地文件操作，支持读写编辑搜索     | 代码重构、文件管理、项目操作         |
| **Git**                | Git 仓库的读取、搜索和操作               | 版本控制、代码历史分析               |
| **Memory**             | 基于知识图谱的持久化记忆系统             | 跨会话保持项目上下文                 |
| **Fetch**              | 网页内容抓取和转换                       | 文档获取、网页数据提取               |
| **Sequential Thinking**| 结构化的反思式问题解决                   | 复杂架构决策、系统设计               |
| **Time**               | 时间和时区转换功能                       | 国际化开发、时间处理                 |

#### 安装示例

```bash
# Sequential Thinking - 结构化思考
claude mcp add sequential-thinking npx -- -y @modelcontextprotocol/server-sequential-thinking

# Filesystem - 文件系统操作
claude mcp add filesystem -- npx -y @modelcontextprotocol/server-filesystem /path/to/allowed/dir
```

**官方仓库**: https://github.com/modelcontextprotocol/servers

---

### 云平台与基础设施

| 服务器名称        | 功能描述                                      | 安装命令                                                                                   |
| ----------------- | --------------------------------------------- | ------------------------------------------------------------------------------------------ |
| **AWS**           | AWS 服务访问（Lambda、DynamoDB、CloudFormation）| `claude mcp add aws-api -e AWS_REGION=us-east-1 -- uvx awslabs.aws-api-mcp-server@latest` |
| **Cloudflare**    | Edge 计算平台管理（Workers、R2、D1）          | `claude mcp add --transport sse cloudflare-workers https://bindings.mcp.cloudflare.com/sse` |
| **GCP**           | Google Cloud 服务集成                         | `claude mcp add gcp -e GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json -- npx @eniayomi/gcp-mcp-server` |

**使用场景**：
- 基础设施自动化配置
- Lambda 函数调试
- 边缘计算部署
- 云资源管理和查询

---

### 开发工具集成

| 服务器名称      | 功能描述                                   | 安装命令                                                                           |
| --------------- | ------------------------------------------ | ---------------------------------------------------------------------------------- |
| **GitHub**      | GitHub 仓库、PR、Issue、CI/CD 工作流管理   | `claude mcp add --transport http github https://api.githubcopilot.com/mcp/`        |
| **Playwright**  | Web 自动化测试，使用可访问性树             | `claude mcp add playwright npx -- @playwright/mcp@latest`                          |
| **Puppeteer**   | 浏览器控制，自动化测试和爬虫               | `claude mcp add puppeteer -- npx -y @anthropic/puppeteer-mcp-server`               |
| **Apidog**      | API 规范访问，类型安全代码生成             | `claude mcp add apidog -- npx -y apidog-mcp-server@latest --oas=<url-or-path>`     |
| **Context7**    | 实时获取最新框架文档                       | `claude mcp add --transport http context7 https://mcp.context7.com/mcp`            |
| **PostgreSQL**  | 自然语言数据库查询                         | `claude mcp add db -- npx -y @bytebase/dbhub --dsn "postgresql://..."`             |

**使用场景示例**：
```bash
# 安装 GitHub MCP 后的使用
> "审核 PR #456 并提出改进建议"
> "为我们刚发现的 bug 创建一个新 issue"

# 安装 Context7 后的使用
> "获取 React 19 的最新 API 文档"
> "Next.js 15 的服务端组件如何使用？"
```

---

### 项目管理与协作

| 服务器名称     | 功能描述                                | 安装命令                                                                              |
| -------------- | --------------------------------------- | ------------------------------------------------------------------------------------- |
| **Notion**     | 知识库访问和文档管理                    | `claude mcp add notion -e NOTION_API_TOKEN=<token> -- npx -y @makenotion/notion-mcp-server` |
| **Linear**     | Issue 跟踪和项目管理集成                | `claude mcp add --transport sse linear https://mcp.linear.app/sse`                    |
| **Atlassian**  | Jira 和 Confluence 集成                 | `claude mcp add --transport sse atlassian https://mcp.atlassian.com/v1/sse`           |
| **Airtable**   | 数据库 CRUD 操作和 Schema 检查          | `claude mcp add airtable -e AIRTABLE_API_KEY=<token> -- npx -y @domdomegg/airtable-mcp-server` |
| **Figma**      | 设计稿转代码工作流                      | `claude mcp add --transport sse figma http://127.0.0.1:3845/sse`                      |
| **Zapier**     | 跨应用工作流自动化                      | `npx @composio/mcp@latest setup zapier --client claude`                              |

**使用场景**：
- 从代码讨论中直接创建 Jira 工单
- 将 Figma 设计稿转换为 React 组件
- 搜索 Notion 文档并生成会议摘要
- 自动化 Slack 通知和邮件发送

---

### 监控与调试

| 服务器名称   | 功能描述                       | 安装命令                                                    |
| ------------ | ------------------------------ | ----------------------------------------------------------- |
| **Sentry**   | 错误追踪和性能监控分析         | `claude mcp add --transport sse sentry https://mcp.sentry.dev/mcp` |
| **PostHog**  | 产品分析和功能标志管理         | `claude mcp add --transport sse posthog https://mcp.posthog.com/sse` |

**使用示例**：
```bash
# 安装 Sentry 后
> "过去 24 小时最常见的错误是什么？"
> "显示错误 ID abc123 的堆栈跟踪"

# 安装 PostHog 后
> "分析新功能的用户漏斗转化率"
> "A/B 测试的结果如何？"
```

---

### MCP 资源汇总网站

| 资源名称                 | 网址                                             | 说明                          |
| ------------------------ | ------------------------------------------------ | ----------------------------- |
| **官方 MCP Registry**    | https://registry.modelcontextprotocol.io/        | 官方 MCP 服务器注册中心       |
| **MCP 官方仓库**         | https://github.com/modelcontextprotocol/servers  | 官方参考实现                  |
| **MCPcat**               | https://mcpcat.io/                               | MCP 服务器目录和指南          |
| **awesome-mcp-servers**  | https://github.com/wong2/awesome-mcp-servers     | 社区精选 MCP 服务器列表       |
| **punkpeye/awesome-mcp** | https://github.com/punkpeye/awesome-mcp-servers  | 另一个优质 MCP 服务器合集     |
| **Microsoft MCP**        | https://github.com/microsoft/mcp                 | 微软官方 MCP 服务器实现       |

---

## Plugins 插件资源

Plugins 是 Claude Code 的功能扩展包，可以包含斜杠命令、子代理、技能、钩子和 MCP 服务器。

### 官方插件市场

**地址**: https://github.com/anthropics/claude-code/tree/main/plugins

**安装方式**:
```bash
# 添加官方市场
/plugin marketplace add anthropics/claude-code

# 安装官方插件
/plugin install feature-dev
/plugin install code-review
/plugin install security-scan
```

#### 官方核心插件

| 插件名称            | 功能描述                   |
| ------------------- | -------------------------- |
| **agent-sdk-dev**   | Agent SDK 开发辅助         |
| **pr-review-toolkit** | PR 审核工具集            |
| **commit-commands** | Git 提交相关命令           |
| **feature-dev**     | 完整的功能开发工作流       |
| **security-guidance** | 安全指导和最佳实践       |

---

### 社区插件市场

| 市场名称                           | GitHub 地址                                                | 说明                                    |
| ---------------------------------- | ---------------------------------------------------------- | --------------------------------------- |
| **ccplugins/awesome-claude-code-plugins** | https://github.com/ccplugins/awesome-claude-code-plugins | 200+ 插件，13 个分类                    |
| **GiladShoham/awesome-claude-plugins**    | https://github.com/GiladShoham/awesome-claude-plugins    | 遵循官方规范的精选插件市场              |
| **hekmon8/awesome-claude-code-plugins**   | https://github.com/hekmon8/awesome-claude-code-plugins   | Vibe Coding 最佳插件合集                |
| **quemsah/awesome-claude-plugins**        | https://github.com/quemsah/awesome-claude-plugins        | 243+ 插件，含 Skills v1.2.0 支持        |
| **jmanhype/awesome-claude-code**          | https://github.com/jmanhype/awesome-claude-code          | 多代理智能市场，68+ 专业代理            |
| **gmickel/gmickel-claude-marketplace**    | https://github.com/gmickel/gmickel-claude-marketplace    | Flow-Next 计划优先工作流                |

**安装社区市场**:
```bash
# 添加社区市场
/plugin marketplace add ccplugins/awesome-claude-code-plugins

# 浏览可用插件
/plugin

# 安装特定插件
/plugin install code-review@awesome-claude-code-plugins
```

---

### 开发工具类插件

| 插件名称              | 功能描述                      | 安装命令                                |
| --------------------- | ----------------------------- | --------------------------------------- |
| **ai-engineer**       | AI 工程开发辅助               | `/plugin install ai-engineer`           |
| **backend-architect** | 后端架构设计                  | `/plugin install backend-architect`     |
| **frontend-developer**| 前端开发工具                  | `/plugin install frontend-developer`    |
| **python-expert**     | Python 专家级辅助             | `/plugin install python-expert`         |
| **web-dev**           | Web 开发全栈支持              | `/plugin install web-dev`               |
| **mobile-app-builder**| 移动应用开发                  | `/plugin install mobile-app-builder`    |
| **refactor-assistant**| 安全重构配合自动测试          | `/plugin install refactor-assistant`    |

---

### DevOps 与 CI/CD 插件

| 插件名称                | 功能描述                     | 安装命令                                  |
| ----------------------- | ---------------------------- | ----------------------------------------- |
| **deploy-automation**   | 部署流水线操作               | `/plugin install deploy-automation`       |
| **docker-helper**       | Docker 容器工作流            | `/plugin install docker-helper`           |
| **k8s-deploy**          | Kubernetes 部署辅助          | `/plugin install k8s-deploy`              |
| **deployment-engineer** | 部署工程自动化               | `/plugin install deployment-engineer`     |
| **devops-automator**    | DevOps 自动化工具            | `/plugin install devops-automator`        |
| **infrastructure-maintainer** | 基础设施维护           | `/plugin install infrastructure-maintainer` |

---

### 测试与质量类插件

| 插件名称               | 功能描述                    | 安装命令                                 |
| ---------------------- | --------------------------- | ---------------------------------------- |
| **test-generator**     | 自动化测试用例生成          | `/plugin install test-generator`         |
| **unit-test-generator**| 单元测试生成器              | `/plugin install unit-test-generator`    |
| **coverage-analyzer**  | 测试覆盖率分析              | `/plugin install coverage-analyzer`      |
| **qa-automation**      | 端到端质量保证工作流        | `/plugin install qa-automation`          |
| **bug-detective**      | Bug 检测和追踪              | `/plugin install bug-detective`          |
| **api-tester**         | API 测试工具                | `/plugin install api-tester`             |
| **debugger**           | 调试辅助工具                | `/plugin install debugger`               |
| **optimize**           | 代码优化建议                | `/plugin install optimize`               |

---

### 文档与项目管理插件

| 插件名称                  | 功能描述                      | 安装命令                                    |
| ------------------------- | ----------------------------- | ------------------------------------------- |
| **doc-generator**         | 自动生成文档                  | `/plugin install doc-generator`             |
| **api-docs**              | OpenAPI/Swagger API 文档      | `/plugin install api-docs`                  |
| **readme-builder**        | 专业 README 文件生成          | `/plugin install readme-builder`            |
| **changelog-generator**   | 变更日志自动生成              | `/plugin install changelog-generator`       |
| **analyze-codebase**      | 代码库结构分析                | `/plugin install analyze-codebase`          |
| **documentation-generator** | 综合文档生成工具            | `/plugin install documentation-generator`   |
| **task-tracker**          | Issue 和任务管理集成          | `/plugin install task-tracker`              |
| **sprint-planner**        | 敏捷 Sprint 规划辅助          | `/plugin install sprint-planner`            |
| **standup-helper**        | 每日站会报告生成              | `/plugin install standup-helper`            |

---

### 安全类插件

| 插件名称              | 功能描述                   | 安装命令                                |
| --------------------- | -------------------------- | --------------------------------------- |
| **security-scan**     | 安全漏洞分析（官方）       | `/plugin install security-scan`         |
| **dependency-audit**  | 依赖安全检查               | `/plugin install dependency-audit`      |
| **secrets-detector**  | 代码中暴露的密钥检测       | `/plugin install secrets-detector`      |

---

### Git 工作流插件

| 插件名称            | 功能描述                    | 安装命令                              |
| ------------------- | --------------------------- | ------------------------------------- |
| **commit**          | 智能提交消息生成            | `/plugin install commit`              |
| **create-pr**       | PR 创建辅助                 | `/plugin install create-pr`           |
| **pr-review**       | PR 审核工具                 | `/plugin install pr-review`           |
| **fix-github-issue**| GitHub Issue 修复辅助       | `/plugin install fix-github-issue`    |
| **analyze-issue**   | Issue 分析工具              | `/plugin install analyze-issue`       |

---

## Skills 技能资源

Skills 是 Claude 的专业知识模块，教会 Claude 特定的工作流程和专业知识，是最轻量级的扩展方式。

### 官方技能库

**地址**: https://github.com/anthropics/skills

**安装方式**:
```bash
# 注册官方技能市场
/plugin marketplace add anthropics/skills

# 安装文档处理技能
/plugin install document-skills@anthropic-agent-skills

# 安装示例技能
/plugin install example-skills@anthropic-agent-skills
```

---

### 社区技能合集

| 合集名称                        | GitHub 地址                                                  | 说明                              |
| ------------------------------- | ------------------------------------------------------------ | --------------------------------- |
| **obra/superpowers**            | https://github.com/obra/superpowers                          | 20+ 实战技能（TDD、调试、协作）   |
| **travisvn/awesome-claude-skills** | https://github.com/travisvn/awesome-claude-skills         | Claude Skills 精选合集            |
| **VoltAgent/awesome-claude-skills** | https://github.com/VoltAgent/awesome-claude-skills       | 技能和资源合集                    |
| **ComposioHQ/awesome-claude-skills** | https://github.com/ComposioHQ/awesome-claude-skills     | AI 工作流定制技能                 |
| **alirezarezvani/claude-skills** | https://github.com/alirezarezvani/claude-skills            | 实际使用的技能集合                |
| **alirezarezvani/claude-code-skill-factory** | https://github.com/alirezarezvani/claude-code-skill-factory | 技能开发工具包          |
| **ChrisWiles/claude-code-showcase** | https://github.com/ChrisWiles/claude-code-showcase       | 完整项目配置示例                  |

**安装 Superpowers**:
```bash
# 添加 Superpowers 市场
/plugin marketplace add obra/superpowers-marketplace

# 使用内置命令
/brainstorm   # 头脑风暴
/write-plan   # 编写计划
/execute-plan # 执行计划
```

---

### 开发与技术类技能

| 技能名称                | 功能描述                              | 安装来源                    |
| ----------------------- | ------------------------------------- | --------------------------- |
| **test-driven-development** | TDD 开发工作流，先写测试再实现    | obra/superpowers            |
| **mcp-builder**         | MCP 服务器创建指南                    | anthropics/skills           |
| **webapp-testing**      | 使用 Playwright 测试本地 Web 应用     | anthropics/skills           |
| **frontend-design**     | React & Tailwind 项目设计决策         | anthropics/skills           |
| **artifacts-builder**   | 构建复杂的 claude.ai HTML artifacts   | anthropics/skills           |
| **playwright-skill**    | 通用浏览器自动化                      | 社区                        |
| **ios-simulator-skill** | iOS 应用构建、导航和测试自动化        | 社区                        |

---

### 文档处理类技能

| 技能名称   | 功能描述                                        | 安装来源         |
| ---------- | ----------------------------------------------- | ---------------- |
| **docx**   | Word 文档创建、编辑、分析，支持修订和批注       | anthropics/skills |
| **pdf**    | PDF 操作工具包：提取文本表格、创建新 PDF        | anthropics/skills |
| **pptx**   | PowerPoint 演示文稿创建编辑，支持布局模板图表   | anthropics/skills |
| **xlsx**   | Excel 电子表格操作，支持公式和格式化            | anthropics/skills |

**使用示例**:
```
> "使用 PDF skill 提取 path/to/file.pdf 中的表单字段"
> "创建一个包含本月销售数据的 Excel 报表"
```

---

### 创意与设计类技能

| 技能名称             | 功能描述                              | 安装来源          |
| -------------------- | ------------------------------------- | ----------------- |
| **algorithmic-art**  | 使用 p5.js 创建生成艺术               | anthropics/skills |
| **canvas-design**    | 设计 PNG/PDF 格式的视觉艺术           | anthropics/skills |
| **slack-gif-creator**| 创建优化的 Slack GIF 动画             | anthropics/skills |
| **claude-d3js-skill**| D3.js 数据可视化                      | 社区              |
| **web-asset-generator** | 生成 favicon、图标、社交媒体图片   | 社区              |

---

### 安全与研究类技能

| 技能名称                     | 功能描述                           | 安装来源    |
| ---------------------------- | ---------------------------------- | ----------- |
| **Trail of Bits Security**   | CodeQL/Semgrep 静态分析            | 社区        |
| **ffuf-web-fuzzing**         | Web 模糊测试渗透指南               | 社区        |
| **claude-scientific-skills** | 科学库和数据库访问                 | 社区        |

---

### 企业与沟通类技能

| 技能名称           | 功能描述                          | 安装来源          |
| ------------------ | --------------------------------- | ----------------- |
| **brand-guidelines** | 应用 Anthropic 官方品牌色彩和字体 | anthropics/skills |
| **internal-comms** | 撰写状态报告、新闻通讯等内部沟通  | anthropics/skills |
| **skill-creator**  | 交互式技能创建工具                | anthropics/skills |

---

### 技能创建基础模板

如果你想创建自己的技能，可以参考以下模板：

```markdown
---
name: my-custom-skill
description: 技能功能描述，以及何时使用它。用于描述X任务，当用户提到Y时触发。
---

# 我的自定义技能

## 使用说明
当执行此任务时，请遵循以下步骤：
1. 第一步说明
2. 第二步说明
3. 第三步说明

## 示例
- 示例用法 1
- 示例用法 2

## 注意事项
- 重要提示 1
- 重要提示 2
```

**存放位置**:
- 用户级：`~/.claude/skills/my-skill/SKILL.md`
- 项目级：`.claude/skills/my-skill/SKILL.md`

---

## 资源发现平台

| 平台名称                | 网址                                        | 说明                                |
| ----------------------- | ------------------------------------------- | ----------------------------------- |
| **claude-plugins.dev**  | https://claude-plugins.dev/skills           | 自动索引 GitHub 上所有公开 Skills   |
| **awesomeclaude.ai**    | https://awesomeclaude.ai/awesome-claude-code | Awesome Claude Code 可视化目录     |
| **scriptbyai.com**      | https://www.scriptbyai.com/claude-code-resource-list/ | Claude Code 资源列表 2026 版 |
| **agentskills.io**      | https://agentskills.io                      | Agent Skills 开放标准               |

---

## 参考链接

### 官方文档

| 文档                    | 链接                                                |
| ----------------------- | --------------------------------------------------- |
| **Agent Skills**        | https://code.claude.com/docs/en/skills              |
| **MCP**                 | https://code.claude.com/docs/en/mcp                 |
| **Plugins**             | https://code.claude.com/docs/en/plugins             |
| **Discover Plugins**    | https://code.claude.com/docs/en/discover-plugins    |
| **Plugin Marketplaces** | https://code.claude.com/docs/en/plugin-marketplaces |
| **MCP 官方网站**        | https://modelcontextprotocol.io                     |

### 社区资源

| 资源                                 | 链接                                                        |
| ------------------------------------ | ----------------------------------------------------------- |
| **MCP 官方 GitHub**                  | https://github.com/modelcontextprotocol/servers             |
| **官方 Skills GitHub**               | https://github.com/anthropics/skills                        |
| **awesome-claude-code-plugins**      | https://github.com/ccplugins/awesome-claude-code-plugins    |
| **awesome-claude-skills**            | https://github.com/travisvn/awesome-claude-skills           |
| **awesome-mcp-servers**              | https://github.com/wong2/awesome-mcp-servers                |
| **obra/superpowers**                 | https://github.com/obra/superpowers                         |
| **hesreallyhim/awesome-claude-code** | https://github.com/hesreallyhim/awesome-claude-code         |

---

## 快速入门指南

### 1. 安装你的第一个 MCP 服务器

```bash
# 推荐先安装 Sequential Thinking（结构化思考）
claude mcp add sequential-thinking npx -- -y @modelcontextprotocol/server-sequential-thinking

# 验证安装
/mcp
```

### 2. 添加插件市场并安装插件

```bash
# 添加社区市场
/plugin marketplace add ccplugins/awesome-claude-code-plugins

# 浏览并安装插件
/plugin
/plugin install code-review@awesome-claude-code-plugins
```

### 3. 安装官方 Skills

```bash
# 添加官方技能市场
/plugin marketplace add anthropics/skills

# 安装文档处理技能
/plugin install document-skills@anthropic-agent-skills
```

### 4. 验证所有扩展

```bash
# 查看 MCP 服务器状态
/mcp

# 查看已安装插件
/plugin list

# 询问可用技能
> "What Skills are available?"
```

---

> **提示**: 以上资源会随时间更新，建议定期访问官方仓库和社区资源获取最新版本。
>
> **免责声明**: 社区资源由第三方维护，使用前请仔细阅读相关文档和许可证。

---

> **最后更新**: 2026-01-17
> **数据来源**: 官方文档、GitHub、社区资源
