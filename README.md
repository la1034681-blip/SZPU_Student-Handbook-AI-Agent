# 深圳职业技术大学《学生手册》智能问答

用《学生手册（2025 年修订）》原文作答的问答系统 —— 答案带条款出处，手册里查不到就明说
「手册中没有相关规定」，不臆测、不编造。

## 效果

60 道真题实测（裁判模型统一用 DeepSeek，避免自评自判）：

    答案正确率   0.981
    出处命中率   1.000
    误拒率       0
    拒答准确率   100%
    平均响应     7.0 秒

（同一配置另一次跑出 0.943，差异来自免费模型本身的波动。）

## 它是怎么工作的

    学生提问
      ↓
    规划：判断问题要不要拆成几个小问题       ← 大模型
      ↓
    检索：语义检索（bge-m3）+ 关键词检索（BM25）两路并行
          ↓ 融合排序（RRF）
          ↓ 重排精选（bge-reranker-v2-m3）
      ↓
    生成：拿着原文条款写答案，每句都打 [E1] 角标   ← 大模型
      ↓
    自检：证据不够就再检索一轮
      ↓
    引用校验：角标对不上原文 → 降级或拒答（本项目自研，不靠模型自觉）
      ↓
    答案（角标可点击跳到原条款，附「内部过程」面板）

## 快速开始

### 本机运行（用本机显卡）

    pip install -r requirements-local.txt
    # 在项目根建 .env（可参考 .env.example），填上密钥
    双击「启动网页问答.bat」

需要在 `.env` 里配：

    ACCESS_CODE=          # 进网页的口令
    ADMIN_CODE=           # 查看使用统计的密钥
    FREE_LLM_API_KEY=     # 智谱 GLM-4-Flash（免费）
    EMBED_API_KEY=        # 硅基流动（免费）

### 部署到 Streamlit Cloud

1. 在 Streamlit Cloud 新建 App，指向本仓库，入口文件 `app.py`
2. Python 版本选 3.11
3. 在 Settings → Secrets 里填上面那几项，**并加一行**：

       EMBED_BACKEND = "api"

   线上没有显卡、内存上限约 2.7 GB，装不下本机那两个模型（共 4.4 GB），
   所以「问题编码 + 重排」这两步改调在线接口。编码模型仍是同一个
   `BAAI/bge-m3`，向量库照旧复用，检索结果不受影响
   （上线前已用 `src/dev/diag/check_embed_api.py` 比对验证：向量余弦中位
   1.00003、重排名次相关 0.9970、端到端条款重合 97.0%）。

密钥不会进仓库：`.env` 和 `.streamlit/secrets.toml` 都已被 `.gitignore` 排除。

## 目录结构

    app.py                     网页入口（Streamlit）
    requirements.txt           线上依赖（轻量，给 Streamlit Cloud 读）
    requirements-local.txt     本机依赖（含 torch 等，仅供本机环境复现）

    src/                       运行时模块，只放网站真正调用的
        qa_chain.py            问答链路（LangChain LCEL）
        agent_qa.py            智能体链路（规划 → 多路检索 → 生成 → 自检）
        retrievers.py          统一检索接口
        bm25.py                关键词检索（自研，字符 bigram）
        embed_api.py           在线编码与重排（部署到无显卡环境时用）
        embed_local.py         本机编码（用本机显卡）
        planner.py             问题拆解
        reflect.py             自检回环
        verify_citations.py    引用校验与拒答
        procedural_qa.py       程序化抽取式通道（不依赖大模型，备选）
        oplog.py               访问日志（异步写本地 + 云库）
        cloudstore.py          云库读写（建议墙、使用统计）
        store.py               本地存储层
        ask.py                 命令行问答入口

    src/tests/                 回归测试，改完代码跑一遍
    src/dev/build/             建库脚本：PDF → 条款 → 检索块 → 向量库
    src/dev/eval/              评测脚本：出题、跑分、阅卷、对比
    src/dev/diag/              诊断脚本：调参、消融、验证（多数是一次性的）

    data/                      向量库与结构化数据（约 9 MB，随仓库分发）
    assets/                    校徽等静态资源

## 数据是怎么来的

    PDF（正文含表格、分式、跨页条款）
      → 解析成 974 个条款
      → 切成 1510 个检索块（每块记住它属于哪条、原 PDF 第几页）
      → 用 bge-m3 离线算好向量（6 MB，一次算完永久复用）

这套流程的脚本都在 `src/dev/build/`。本仓库不收模型权重（GitHub 单文件
上限 100 MB，且线上不需要），模型请自行下载或使用在线接口。

## 隐私

- 访问日志分两张表：`qa_log` 存完整问答（只授写入），`qa_usage` 只存
  匿名编号与时间（公开读，供统计看板）。看板因此读不到提问原文。
- 匿名编号是一串随机十六进制，写在网址 `?u=` 里，不含任何身份信息。

## 已知限制

- 免费大模型（GLM-4-Flash）有波动，同一问题可能两次表述不同
- 提问目前没有频率限制；多人同时使用可能触及上游免费档的并发上限
- 使用统计看板没有明细下钻与导出功能
