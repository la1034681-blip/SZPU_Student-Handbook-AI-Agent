# src/dev 目录说明

这里全是**开发期脚本**，不参与线上运行（线上只跑 `app.py` + `src/{core,retrieval,infra}`）。
分四个目录，按「什么时候会用到」分：

| 目录 | 什么时候用 | 里面有什么 |
|---|---|---|
| `build/` | 手册换版本时**一次性**跑 | PDF → 条款 → 索引 → 向量，加上解析质量审计 |
| `eval/` | 想量「有没有变差」时 | 60 题评测、检索评测、裁判、结果汇总 |
| `diag/` | 改完代码想**确认没改坏**时 | 守门脚本、探针、截图、保活 |
| `tests/`（在 `src/tests`） | 每次改动都跑 | 24 个回归测试 |

## diag/ 里的命名约定（看前缀就知道该不该跑）

| 前缀 | 含义 | 跑起来什么样 |
|---|---|---|
| `check_*` | **守门脚本**：验证某条不变量还成立（该拦的拦、该放的不放、手册题不碰官网…） | 打印「通过 / 失败」若干行，退出码非 0 即失败。改代码后必跑 |
| `probe_*` | **探针**：拿一批真问题去问，看效果与耗时 | 打印每题走了哪条路、有没有降级、官网有没有命中 |
| `diag_*` | **单点诊断**：只查一件事（某题分数分布、延迟、精排效果） | 打印一堆中间分数，人来判断 |
| `_*.py` | 公共模块（浏览器封装、密钥读取），不被单独执行 | — |

## 常跑的几条

```bash
# 回归（24 个脚本逐个跑；不要用 pytest src/tests，脚本是 main()+sys.exit 形态）
python src/tests/run_all.py

# 三道路径守卫
python src/dev/diag/check_subject_guard.py      # 主体闸：7 拦 5 放 + 60 题零误拦
python src/dev/diag/check_followup_guard.py     # 换话题不被上一题带偏
python src/dev/diag/check_web_link.py           # 手册题不碰官网、官网题才走官网

# 官网兜底本体
python src/dev/diag/probe_web.py                # 20 题
python src/dev/diag/probe_hard_cases.py         # 16 组刁钻问法

# 线上体检
python src/dev/diag/check_cf_ai.py <令牌> <账户ID>   # 编码/重排接口五项体检
python src/dev/diag/probe_online_cases.py <地址>     # 线上连问几道
python src/dev/diag/keepalive.py                     # 防 Cloud 休眠（真浏览器）

# 60 题评测（裁判必须显式指定，别用被测模型自己判自己）
python src/dev/eval/evaluate_generation.py data --agent --judge-backend deepseek --out result.json
```

## 两条纪律（都是踩过坑换来的）

1. **诊断脚本开跑先自报运行条件**（编码后端 / 重排器 / 有没有降级）。
   否则一旦静默降级成纯关键词，量出来的是 BM25 分，结论却写成「两类可分」。
2. **超时不许当成功**。判据要「正文必须变化」，不能读完最后一条消息就当拿到新答复。
