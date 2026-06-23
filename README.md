# 2026暑期AI集训共享仓库

## 仓库规则

1. 两大专题独立文件夹，每位成员拥有专属子目录，所有人仅在**个人专属文件夹**内上传文件，禁止修改、删除他人文件夹及仓库根目录公共文件
2. 每次上传前**先拉取最新内容**，再进行上传

## 全体成员统一上传流程（GitHub Desktop 版）

### 1. 克隆仓库到本地

1. 打开 GitHub Desktop，登录个人 GitHub 账号
2. 点击左上角 `File` → `Clone Repository`
3. 选中当前集体仓库，选择本地保存路径，完成克隆

### 2. 同步最新仓库内容（每次上传前必做）

1. 点击顶部 `Fetch origin` 获取远程更新
2. 点击 `Pull origin` 拉取所有人最新文件，避免上传冲突

### 3. 在个人文件夹内提交或修改文件

1. 在本地仓库**自己姓名命名**文件夹内，提交或修改文件

### 4. 本地提交修改

1. 返回 GitHub Desktop，左侧自动识别新增文件
2. 填写提交说明：`姓名+上传内容`（例：张三 上传第一周学习笔记）
3. 点击 `Commit to main` 完成本地提交

### 5. 上传同步到云端仓库

1. 点击顶部 `Push origin`，等待上传完成即操作结束

## 当前协作模式（默认） 

- 已加入的成员拥有 **Write 权限**，可在自己文件夹内直接上传/修改/删除文件 
- **所有修改直接生效，无需管理员审核**，提交后立即更新到远程仓库 main 分支 
- 规则：**仅允许操作自己文件夹，严禁修改他人文件或根目录公共文件**

# 集训安排

## 一、主讲人员与负责人安排

### 1. 深度学习组 （王妍负责组织）
赵雪杉、舒加怡、陈奕良、胡启帆、费子珊

### 2. 大模型与智能体组 （蒋佳璇负责组织）
吴浩然、梁冉芸、高瑞雪、张越涵、李世通

---

## 二、参与人员

廖老师、刘老师，智能决策与机器学习 Group 其他成员

---

## 三、时间安排

* 7月7日（星期二）开始第一次集训，7月10日（周五）第二次
* 每周两次，暂定每周二和周五晚上19:30开始

---

## 四、主讲内容

### 第一部分：深度学习基础

**学习目标：**
系统掌握深度学习的数学基础与核心模型，能够使用PyTorch实现线性网络、卷积神经网络、循环神经网络及Transformer等主流架构，具备独立完成深度学习实验的能力。

**参考资料：**

- 主要教材：《动手学深度学习》第二版
- 在线电子书：https://zh.d2l.ai/index.html
- B站视频课程：https://www.bilibili.com/list/1567748478/?sid=358497

**内容安排：**

| 讲次 | 主题 | 对应章节 |
|------|------|---------|
| 第1讲 | 预备知识（线性代数、微积分、概率、数据处理） | 第2章 |
| 第2讲 | 线性神经网络（线性回归、Softmax回归） | 第3章 |
| 第3讲 | 多层感知机（MLP、过拟合与正则化、Dropout） | 第4章 |
| 第4讲 | 深度学习计算（层、块、参数、GPU） | 第5章 |
| 第5讲 | 卷积神经网络（卷积层、池化层、LeNet） | 第6章 |
| 第6讲 | 现代卷积神经网络（AlexNet、VGG、ResNet等） | 第7章 |
| 第7讲 | 循环神经网络（序列模型、文本预处理、RNN） | 第8章 |
| 第8讲 | 现代循环神经网络（LSTM、GRU、Seq2Seq） | 第9章 |
| 第9讲 | 注意力机制（注意力汇聚、多头注意力、自注意力） | 第10章（上） |
| 第10讲 | Transformer架构 | 第10章（下） |
| 第11讲 | 优化算法（SGD、Adam、学习率调度） | 第11章 |
| 第12、13讲 | 计算性能与自然语言处理应用（BERT预训练） | 第12、14、15章 |

---

### 第二部分：大模型与智能体核心技术

**学习目标：**
系统掌握大模型智能体的核心原理与工程实现，覆盖大模型基础、智能体架构、模型微调与前沿应用四大模块，能够独立设计、实现、调试并评估大模型智能体。

**内容安排：**

| 讲次 | 主题 | 对应章节/资料 | 核心知识点 | 实践安排 |
| :---: | :--- | :--- | :--- | :--- |
| 第1周 | 大模型如何工作 | **必学**：条目1.1-1.5<br>**选学**：《Attention Is All You Need》原文；MoE架构直觉 | 1.1 Transformer架构直觉：注意力机制的功能定位（不要求数学推导）<br>1.2 Token化与上下文窗口：模型的输入表示与视野边界<br>1.3 预训练：下一词预测目标、训练数据构成与Scaling Laws<br>1.4 后训练：SFT / RLHF / RLVR，模型行为风格的形成机制<br>1.5 推理模型与测试时计算：OpenAI o系列、DeepSeek-R1的范式意义 | **实践任务**：围绕自身研究方向的一个具体问题（如“平台定价策略综述”），选取2-3个不同模型（如DeepSeek、豆包、Claude/ChatGPT），对比“普通模式”与“深度思考模式”下的输出差异。整理对比笔记，涵盖：模型间输出差异、推理模式带来的性能提升、观察到的幻觉现象。<br><br>**评估标准**：能够向非专业同学阐释大模型训练全流程；能够解释推理模型在数学/代码任务上的优势来源；对照实验具备可复现的方法论与记录。 |
| 第2周 | 模型生态、提示工程与API入门 | **必学**：条目1.6-1.8、2.1-2.2<br>**选学**：DeepSeek-V3 / R1技术报告细读 | 1.6 国内外大模型生态图谱：厂商定位、开源与闭源格局、科研场景下的选型策略<br>1.7 提示工程系统方法：角色设定、少样本示例、思维链、输出格式约束<br>1.8 大模型的局限：幻觉、知识截止、谄媚性，及科研使用中的红线<br>2.1 API基础：messages格式（system/user/assistant）、温度等采样参数、流式输出<br>2.2 结构化输出（JSON mode）与Function Calling / Tool Use：模型调用外部工具的能力机制 | **实践任务**：基于DeepSeek（或智谱）API实现一个具备工具调用能力的命令行问答程序。至少实现2个工具（如“本地CSV数据查询”和“数学计算”），由模型自主决策工具调用时机，系统提示需运用本周提示工程方法进行系统设计。演示运行效果并走读“模型返回工具调用 → 程序执行 → 结果回传”的完整循环。<br><br>**评估标准**：能够手绘function calling的完整消息流转图；工具调用循环为自主实现（非框架黑盒），能够回答实现细节；能演示一个自主发现的幻觉案例及其核验方法。 |
| 第3周 | 上下文工程、RAG与评测 | **必学**：条目2.3-2.7<br>**选学**：长上下文 vs RAG的取舍；rerank模型；Anthropic《Effective context engineering for AI agents》原文精读 | 2.3 上下文工程：从“提示工程”到“上下文工程”的演进逻辑——在正确时机将正确信息以正确形式置入上下文<br>2.4 Embedding与向量检索的原理<br>2.5 RAG（检索增强生成）：分块、检索、重排、生成的完整链路，适用场景与局限性<br>2.6 评测基础：大模型输出质量的科学评价方法，LLM-as-Judge的用法与潜在陷阱<br>2.7 Token经济学：API计费逻辑、prompt caching、成本控制策略 | **实践任务**：构建一个最小RAG问答demo。导入3-5篇本研究方向的文献，实现“分块 → 向量化 → 检索 → 带引用回答”的完整链路，并用5-10个自拟问题开展小规模评测（人工评判或LLM-as-Judge）。展示问答效果，并分析至少一个检索错误案例。<br><br>**评估标准**：能够阐明RAG各环节的潜在错误模式并结合demo举例；能够说明评测设计逻辑与结论可信度；能够估算单次问答的token成本。 |
| 第4周 | AI编程工具与Claude Code | **必学**：条目3.1-3.5<br>**选学**：Anthropic《Claude Code: Best practices for agentic coding》；Claude Agent SDK文档 | 3.1 AI编程范式演进：代码补全 → 对话式 → 智能体编程（Agentic Coding），各阶段的本质差异<br>3.2 Claude Code核心工作流：安装配置、需求表达方法、CLAUDE.md（项目记忆）规范<br>3.3 Claude Code进阶（上）：Plan Mode（先规划后执行）、Slash Commands、Skills（技能包）<br>3.4 Claude Code进阶（下）：Hooks（钩子）、Subagents（子智能体）、MCP服务接入<br>3.5 同类工具对比：Cursor、GitHub Copilot、Trae / 通义灵码 | **实践任务**：借助Claude Code（或替代工具）从零完成一个与研究相关的小工具并提交GitHub仓库，例如：问卷数据清洗与统计脚本、文献BibTeX整理器、实验数据可视化面板。全程记录与AI的协作过程：哪些由AI生成、哪些经人工审查修正、CLAUDE.md的编写思路。展示聚焦于“协作复盘”而非功能演示。<br><br>**评估标准**：工具可现场运行；能够阐明Plan Mode / CLAUDE.md / Subagent各自解决的问题；复盘中需包含至少一个“AI生成错误—人工发现并修正”的具体案例。 |
| 第5周 | 智能体与Harness Engineering + 项目开题 | **必学**：条目3.6-3.7、4.1-4.3<br>**选学**：Anthropic《Building Effective Agents》《Effective harnesses for long-running agents》原文精读 | 3.6 “Vibe Coding”的能力边界：哪些任务可交由AI自主完成，哪些需人工介入<br>3.7 AI辅助科研编程的规范：可复现性、代码审查、学术诚信边界<br>4.1 智能体的本质与Agent循环（感知 → 思考 → 工具调用 → 观察 → 迭代）、ReAct模式<br>4.2 Harness Engineering（上）：**Agent = Model + Harness**——模型负责推理，线束负责行动；harness的组件构成；产品护城河在线束层的业界共识<br>4.3 Harness Engineering（下）：上下文管理（压缩/记忆/子智能体隔离）、结果验证与长时运行智能体设计；Anthropic 2026年6月发布的动态工作流“A harness for every task” | **项目实战（可分多讲，每人都讲）**： 每人独立完成一个面向管理科学与工程研究场景的项目，可从下述项目清单中选择也可自行寻找研究场景搭建。<br><br>**评估标准**：能够不看资料阐述harness的组件构成及各组件功能。|
| 第6周 | MCP、智能体框架与多智能体 + 结题 | **必学**：条目4.4-4.9<br>**选学**：A2A协议规范；与项目相关的近两年文献2-3篇 | 4.4 **MCP（Model Context Protocol）**：智能体连接工具与数据的标准化协议，最小MCP server的实现与接入方式<br>4.5 Claude Agent SDK：基于Claude Code同款harness构建自定义智能体<br>4.6 智能体框架对比：Claude Agent SDK、LangGraph、CrewAI的设计哲学差异<br>4.7 多智能体系统：编排模式（编排者-工作者、辩论、流水线）、A2A协议、典型失效模式（错误级联、上下文撕裂）<br>4.8 智能体方法在管理科学研究中的应用前沿：基于LLM多智能体仿真的社会/管理实验（Generative Agents研究脉络）<br>4.9 个人AI智能体现象：**OpenClaw**与**Hermes Agent**——2026年开源自托管个人智能体（消息平台接入、持久记忆、自动技能创建），以harness视角与Claude Code对比（**了解级，不要求部署**） | **项目实战（可分多讲，每人都讲）**： 每人独立完成一个面向管理科学与工程研究场景的项目，可从下述项目清单中选择也可自行寻找研究场景搭建。<br><br>**评估标准**：能够列举至少两种多智能体失效模式及其工程对策。|


**实战项目选项**（每人选一，也可自拟报导师批准；★=入门档，★★=进阶档）

| # | 项目 | 难度 | 说明 |
|---|---|---|---|
| 1 | LLM 多智能体管理实验仿真 | ★★ | 用多个 LLM agent 模拟拍卖、谈判或供应链博弈场景，设计实验、记录 agent 行为、分析结果。可直接衔接行为运营/实验经济学方向的科研 |
| 2 | 智能文献综述助手 | ★ | RAG + 智能体：导入本方向 30-100 篇文献，实现"按主题问答、自动归纳研究脉络、生成带引用的综述初稿" |
| 3 | 企业数据分析智能体 | ★ | 自然语言 → 自动生成查询/分析代码 → 图表 → 管理建议，基于一份公开企业数据集（如电商订单、生产数据） |
| 4 | AI 辅助运筹优化智能体 | ★★ | LLM 理解自然语言描述的排产/选址/路径问题 → 自动建模并调用求解器（OR-Tools/Gurobi）→ 解释求解结果 |
| 5 | 学术写作与评审智能体工作流 | ★ | 多步智能体流水线：初稿分析 → 按期刊规范逐项审查 → 生成修改建议，体现 harness 设计思想 |

---

**参考资料：**

**入门与原理（第 1-2 周）**
- Andrej Karpathy《Intro to Large Language Models》 https://www.youtube.com/watch?v=zjkBMFhNj_g 、《Deep Dive into LLMs like ChatGPT》 https://www.youtube.com/watch?v=7xTGNNLPyMI ——B 站搜索"Karpathy 大语言模型"看搬运
- 3Blue1Brown 神经网络/Transformer 可视化系列——B 站官方账号 https://space.bilibili.com/88461692 ，原版合集 https://www.3blue1brown.com/topics/neural-networks 
- DeepSeek-V3 技术报告 https://arxiv.org/abs/2412.19437 、DeepSeek-R1 技术报告 https://arxiv.org/abs/2501.12948 
- 《Attention Is All You Need》（选学） https://arxiv.org/abs/1706.03762 

**API 与上下文工程（第 2-3 周）**
- DeepSeek 开放平台文档 https://api-docs.deepseek.com/zh-cn （含 API 入门、Function Calling、提示库、定价）
- 智谱开放平台文档 https://docs.bigmodel.cn ；阿里云百炼（通义）文档 https://help.aliyun.com/zh/model-studio/ 
- Anthropic 文档：Prompt Engineering 章节 https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/overview 、Tool Use 章节 https://docs.claude.com/en/docs/agents-and-tools/tool-use/overview 
- Anthropic 工程博客《Effective context engineering for AI agents》 https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents 
- LangChain 官方 RAG 教程 https://python.langchain.com/docs/tutorials/rag/ 

**AI 编程工具（第 4 周）**
- Claude Code 官方文档 https://code.claude.com/docs 
- Anthropic 工程博客《Claude Code: Best practices for agentic coding》 https://www.anthropic.com/engineering/claude-code-best-practices 
- Trae https://www.trae.cn 、通义灵码 https://lingma.aliyun.com （无法使用 Claude Code 者的替代实践载体）

**智能体与 Harness Engineering（第 5-6 周）**
- Anthropic《Building Effective Agents》 https://www.anthropic.com/engineering/building-effective-agents ——智能体设计模式的奠基性文章
- Anthropic《Effective harnesses for long-running agents》 https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents 、《Harness design for long-running application development》 https://www.anthropic.com/engineering/harness-design-long-running-apps 及 2026 年 6 月动态工作流文章《A harness for every task》（见工程博客索引 https://www.anthropic.com/engineering ）〔均需条件〕
- MCP 官方文档 https://modelcontextprotocol.io ；协议规范仓库 https://github.com/modelcontextprotocol/modelcontextprotocol ；Python SDK https://github.com/modelcontextprotocol/python-sdk 
- Claude Agent SDK 文档 https://docs.claude.com/en/api/agent-sdk/overview ；LangGraph 文档 https://langchain-ai.github.io/langgraph/ ；CrewAI 文档 https://docs.crewai.com 
- 论文：《ReAct: Synergizing Reasoning and Acting in Language Models》 https://arxiv.org/abs/2210.03629 
- 论文：《Generative Agents: Interactive Simulacra of Human Behavior》 https://arxiv.org/abs/2304.03442 ——LLM 多智能体仿真研究的代表作，管理科学方向重点读
- OpenClaw（开源个人 AI 助理，2026 年现象级项目，了解级）：GitHub https://github.com/openclaw/openclaw 、官网 https://openclaw.ai 
- Hermes Agent（Nous Research 开源自我改进智能体，了解级）：GitHub https://github.com/nousresearch/hermes-agent 、官方文档 https://hermes-agent.nousresearch.com/docs/ 

**学科交叉参考（第 5-6 周项目）**
- 检索关键词建议：LLM-based agents for social simulation / agent-based modeling with LLM / LLM for operations research——在 arXiv https://arxiv.org 、Google Scholar https://scholar.google.com 或 Semantic Scholar https://www.semanticscholar.org 检索近两年文献
- OR-Tools 文档 https://developers.google.com/optimization （项目 4 用）
- 
**教材参考**
- 教材：《动手学大模型智能体》（温睦宁、林江浩、张伟楠、俞勇著）
- 课件链接：https://haa.boyuai.com/slides/
- 视频课程：https://www.boyuai.com/course/course/6a3G_fJ50ejab_a
- GitHub：https://github.com/boyu-ai/Hands-on-AA
- 配套资源：课件 + 理论解读视频 + 源代码 + 课后习题 + 样例数据 + 提示词模板，以及相关章节引用的论文
- 《百面大模型》（包梦蛟、刘如日、朱俊达著，人民邮电出版社，2025版），覆盖从预训练底座到落地应用的大模型全链条搭建，理论介绍更为详尽；预训练、对齐与垂类微调（后训练）等章节可作为延伸阅读
---



## 五、集训形式及要求

* 每次集训都需学习 深度学习 与 大模型 两个专题
* 汇报人当晚现场抽签决定，请所有成员做好准备
* 汇报形式不做统一要求：支持 Markdown 文档、PPT 演示文稿或 GitHub 项目等
* 建立一个集训 GitHub（或 Gitee）组织仓库。每个专题一个目录，每人一个子目录，提交学习笔记与代码，供师门内部学习。每个项目需包含但不限于：README 文档（含运行说明）、可复现代码及必要的实验记录。**吴浩然负责创建与更新维护**
* 坚持“连接研究方向，赋能科研工作”的学习原则。在学习新技术时，须深入思考并明确其适用边界，即“该技术能否为我所用？适用于解决何种具体问题？”
