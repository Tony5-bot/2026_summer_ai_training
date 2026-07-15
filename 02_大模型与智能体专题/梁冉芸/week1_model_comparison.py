#!/usr/bin/env python3
"""
第 1 周实践：多模型对比实验
研究问题：多模态虚假新闻检测的核心挑战与解决思路综述
对比维度：
  1. 模型间输出差异（DeepSeek-V4-Flash vs DeepSeek-V4-Pro）
  2. 普通模式 vs 深度思考模式的推理质量
  3. 幻觉现象记录与核验

运行：
  pip install openai python-dotenv
  export DEEPSEEK_API_KEY=sk-xxxx
  python model_comparison.py
"""

import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


# ─── 客户端配置 ──────────────────────────────────────────────────────────────

def get_client() -> OpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("请设置 DEEPSEEK_API_KEY 环境变量")
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


# ─── 对比问题集（针对多模态虚假新闻检测方向）────────────────────────────────

COMPARISON_TASKS = [
    {
        "id": "T1",
        "category": "综述类（知识广度）",
        "question": (
            "请综述多模态虚假新闻检测领域的核心技术挑战，"
            "包括图文不一致检测、跨模态推理和证据获取三个维度，"
            "并对每个挑战的代表性解决方案做简要说明。"
            "在回答中，请明确标注任何引用或数字的来源，对于不确定的内容请标注[不确定]。"
        ),
        "note": "此问题用于检测幻觉：模型可能编造论文或数据",
    },
    {
        "id": "T2",
        "category": "方法对比（需要推理）",
        "question": (
            "请比较基于变分自编码器（VAE）的多模态虚假新闻检测方法"
            "与基于大语言模型多智能体辩论的方法的优缺点，"
            "从计算复杂度、可解释性、对新型虚假新闻的泛化能力三个维度分析。"
            "请给出具体的技术细节，对于不确定的内容请标注[不确定]。"
        ),
        "note": "此问题需要多步推理，适合观察推理模式的优势",
    },
    {
        "id": "T3",
        "category": "研究设计（应用性）",
        "question": (
            "假设你正在设计一个实验，评估大语言模型在多模态虚假新闻检测中"
            "作为零样本检测器（zero-shot detector）的性能。"
            "请设计：(1)实验的评估指标；(2)基线模型选择逻辑；"
            "(3)至少两个可能产生假阳性的典型样本类型。"
            "请给出具体可行的方案，对于不确定的内容请标注[不确定]。"
        ),
        "note": "此问题考查应用型推理，观察模型能否给出合理研究设计",
    },
]

# 模型配置 - 使用 extra_body 明确控制 thinking 模式
MODEL_CONFIGS = [
    {
        "label": "DeepSeek-V4-Pro（普通模式）",
        "model": "deepseek-v4-pro",
        "temperature": 0.7,
        "extra_body": {"thinking": {"type": "disabled"}},  # 显式关闭思考模式
    },
    {
        "label": "DeepSeek-V4-Pro（深度思考模式）",
        "model": "deepseek-v4-pro",
        "temperature": 0.7,
        "extra_body": {"thinking": {"type": "enabled"}},  # 显式开启思考模式
        "reasoning_effort": "high",  # 思考强度
    },
]

SYSTEM_PROMPT = (
    "你是一名计算机科学领域的研究助手，专长是多模态学习和虚假信息检测。"
    "请给出准确、有深度的学术性回答。"
    "对于不确定的内容，请明确标注[不确定]，不要编造。"
    "如果引用具体论文或数据，请提供完整的引用信息以便核实。"
)


# ─── 核心对比函数 ─────────────────────────────────────────────────────────────

def query_model(client: OpenAI, config: dict, question: str,
                system_prompt: str = "") -> dict:
    """发送请求并记录完整信息"""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})

    start_time = time.time()
    try:
        # 构建请求参数
        request_params = {
            "model": config["model"],
            "messages": messages,
            "temperature": config["temperature"],
        }

        # 添加 extra_body（如果存在）
        if "extra_body" in config:
            request_params["extra_body"] = config["extra_body"]

        # 添加 reasoning_effort（如果存在）
        if "reasoning_effort" in config:
            request_params["reasoning_effort"] = config["reasoning_effort"]

        resp = client.chat.completions.create(**request_params)

        elapsed = time.time() - start_time
        msg = resp.choices[0].message
        usage = resp.usage

        result = {
            "model_label": config["label"],
            "model": config["model"],
            "content": msg.content or "",
            "thinking": getattr(msg, "reasoning_content", None),  # 思考过程
            "elapsed_seconds": round(elapsed, 2),
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "finish_reason": resp.choices[0].finish_reason,
            "error": None,
        }
    except Exception as e:
        elapsed = time.time() - start_time
        result = {
            "model_label": config["label"],
            "model": config["model"],
            "content": "",
            "thinking": None,
            "elapsed_seconds": round(elapsed, 2),
            "input_tokens": 0,
            "output_tokens": 0,
            "finish_reason": "error",
            "error": str(e),
        }
    return result


def analyze_hallucination(response_content: str) -> dict:
    """
    幻觉风险分析（基于规则）
    检测模型是否给出了具体的论文名/作者/数字等高风险信息
    """
    import re
    risk_signals = []

    # 检测具体数字（准确率等）
    numbers = re.findall(r'\b\d+\.?\d*\s*%', response_content)
    if numbers:
        risk_signals.append(f"包含具体数字：{numbers[:3]}（需人工核实）")

    # 检测论文引用格式（高风险！）
    citations = re.findall(r'[A-Z][a-z]+\s+et\s+al\.?\s*[\(\（]\d{4}[\)\）]', response_content)
    if citations:
        risk_signals.append(f"包含引用格式：{citations[:3]}（极高风险，必须核实是否存在）")

    # 检测带有年份的论文引用（更宽泛的匹配）
    paper_refs = re.findall(r'["“」]?[A-Za-z\s]+?["”」]?\s*[\(\（]\d{4}[\)\）]', response_content)
    if paper_refs and not citations:  # 如果没有匹配到et al.格式，但匹配到"标题(年份)"格式
        risk_signals.append(f"包含论文引用格式：{paper_refs[:3]}（需核实论文是否存在）")

    # 检测不确定标注
    uncertainty_markers = ["不确定", "可能", "据我了解", "大约", "[不确定]", "推测", "大致"]
    has_uncertainty = any(m in response_content for m in uncertainty_markers)

    # 检测是否包含具体方法名称或技术细节（可能是编造的）
    tech_terms = re.findall(r'\b[A-Z][a-z]*(?:[A-Z][a-z]*)*\s+(?:模型|算法|方法|框架)\b', response_content)
    if len(tech_terms) > 5:  # 如果提到太多具体方法名，可能是幻觉
        risk_signals.append(f"包含大量具体技术名称：{tech_terms[:3]}...（需核实真实性）")

    risk_level = "HIGH" if citations else ("MEDIUM" if numbers or paper_refs else "LOW")

    return {
        "high_risk_elements": risk_signals,
        "has_uncertainty_markers": has_uncertainty,
        "risk_level": risk_level,
    }


# ─── 主程序 ──────────────────────────────────────────────────────────────────

def run_comparison():
    client = get_client()
    report = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "research_topic": "多模态虚假新闻检测",
            "tasks_count": len(COMPARISON_TASKS),
            "models_count": len(MODEL_CONFIGS),
        },
        "results": [],
        "summary": {},
    }

    print("=" * 65)
    print("第 1 周实践：多模型对比实验")
    print("研究方向：多模态虚假新闻检测")
    print("=" * 65)

    for task in COMPARISON_TASKS:
        print(f"\n{'─' * 65}")
        print(f"[{task['id']}] {task['category']}")
        print(f"问题：{task['question'][:80]}...")
        print(f"{'─' * 65}")

        task_result = {
            "task_id": task["id"],
            "category": task["category"],
            "question": task["question"],
            "responses": [],
        }

        for config in MODEL_CONFIGS:
            print(f"\n▶ 正在查询：{config['label']}...")

            # 显示思考模式状态
            if "thinking" in config.get("extra_body", {}):
                thinking_status = config["extra_body"]["thinking"]["type"]
                if thinking_status == "enabled":
                    print(f"  🧠 模式：深度思考 (reasoning_effort={config.get('reasoning_effort', 'N/A')})")
                else:
                    print(f"  ⚡ 模式：普通模式（思考已禁用）")
            else:
                print(f"  ⚡ 模式：普通模式（默认）")

            # 发送主问题（包含幻觉检查要求）
            result = query_model(client, config, task["question"], SYSTEM_PROMPT)

            # 直接对回答进行幻觉分析
            result["hallucination_analysis"] = analyze_hallucination(result["content"])

            task_result["responses"].append(result)

            # 打印摘要
            print(f"  ✓ 耗时 {result['elapsed_seconds']}s | "
                  f"tokens: {result['input_tokens']}↑ {result['output_tokens']}↓ | "
                  f"幻觉风险: {result['hallucination_analysis']['risk_level']}")

            # 检查是否真的有思考过程输出（用于验证配置是否生效）
            if result["thinking"]:
                think_preview = result["thinking"][:100].replace("\n", " ")
                print(f"  💭 思考过程（前100字）：{think_preview}...")
            else:
                print(f"  ❌ 无思考过程输出（符合当前模式设置）")

            answer_preview = result["content"][:150].replace("\n", " ")
            print(f"  📝 回答预览：{answer_preview}...")

            # 打印幻觉检测结果
            if result["hallucination_analysis"]["high_risk_elements"]:
                print(f"  ⚠️ 检测到潜在幻觉：{result['hallucination_analysis']['high_risk_elements'][0]}")
                if len(result["hallucination_analysis"]["high_risk_elements"]) > 1:
                    print(f"     及其他 {len(result['hallucination_analysis']['high_risk_elements']) - 1} 个风险项")

        report["results"].append(task_result)

    # ── 生成汇总分析 ────────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print("汇总分析")
    print(f"{'=' * 65}")

    # 根据模型名称中的关键字分类（flash / pro）
    flash_times, pro_times = [], []
    flash_tokens, pro_tokens = [], []
    flash_risks, pro_risks = [], []
    flash_has_thinking, pro_has_thinking = [], []

    for task_result in report["results"]:
        for resp in task_result["responses"]:
            if "flash" in resp["model"].lower():
                flash_times.append(resp["elapsed_seconds"])
                flash_tokens.append(resp["output_tokens"])
                flash_risks.append(resp["hallucination_analysis"]["risk_level"])
                flash_has_thinking.append(1 if resp["thinking"] else 0)
            elif "pro" in resp["model"].lower():
                pro_times.append(resp["elapsed_seconds"])
                pro_tokens.append(resp["output_tokens"])
                pro_risks.append(resp["hallucination_analysis"]["risk_level"])
                pro_has_thinking.append(1 if resp["thinking"] else 0)

    def avg(lst):
        return round(sum(lst) / len(lst), 2) if lst else 0

    def count_risk(risks, level):
        return sum(1 for r in risks if r == level)

    summary = {
        "flash_avg_time": avg(flash_times),
        "pro_avg_time": avg(pro_times),
        "flash_avg_output_tokens": avg(flash_tokens),
        "pro_avg_output_tokens": avg(pro_tokens),
        "time_overhead_pct": round(
            (avg(pro_times) - avg(flash_times)) / max(avg(flash_times), 0.01) * 100, 1
        ),
        "flash_risk_high": count_risk(flash_risks, "HIGH"),
        "flash_risk_medium": count_risk(flash_risks, "MEDIUM"),
        "flash_risk_low": count_risk(flash_risks, "LOW"),
        "pro_risk_high": count_risk(pro_risks, "HIGH"),
        "pro_risk_medium": count_risk(pro_risks, "MEDIUM"),
        "pro_risk_low": count_risk(pro_risks, "LOW"),
        "flash_thinking_count": sum(flash_has_thinking),
        "pro_thinking_count": sum(pro_has_thinking),
        "flash_total": len(flash_has_thinking),
        "pro_total": len(pro_has_thinking),
    }
    report["summary"] = summary

    print(f"DeepSeek-V4普通模式平均耗时：{summary['flash_avg_time']}s")
    print(f"DeepSeek-V4深度思考平均耗时：{summary['pro_avg_time']}s")
    print(f"推理模式时间开销：+{summary['time_overhead_pct']}%")
    print(f"Flash 平均输出 token：{summary['flash_avg_output_tokens']}")
    print(f"Pro 平均输出 token：{summary['pro_avg_output_tokens']}")
    print(f"\n思考模式验证：")
    print(f"  普通模式有思考过程输出：{summary['flash_thinking_count']}/{summary['flash_total']} (应为0)")
    print(f"  深度思考有思考过程输出：{summary['pro_thinking_count']}/{summary['pro_total']} (应为{summary['pro_total']})")
    print(f"\n幻觉风险统计：")
    print(
        f"  Flash: HIGH={summary['flash_risk_high']}, MEDIUM={summary['flash_risk_medium']}, LOW={summary['flash_risk_low']}")
    print(
        f"  Pro:   HIGH={summary['pro_risk_high']}, MEDIUM={summary['pro_risk_medium']}, LOW={summary['pro_risk_low']}")

    # ── 保存完整报告 ─────────────────────────────────────────────────────────
    output_path = "comparison_report.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 完整报告已保存至：{output_path}")

    # ── 生成 Markdown 对比笔记 ───────────────────────────────────────────────
    generate_markdown_notes(report)


def generate_markdown_notes(report: dict):
    """从 JSON 报告生成可汇报的 Markdown 对比笔记"""
    lines = [
        "# 第1周实践：多模型对比实验报告",
        "",
        f"> 实验时间：{report['metadata']['timestamp']}",
        f"> 研究方向：{report['metadata']['research_topic']}",
        "",
        "## 实验设计",
        "",
        "**对比目标**：针对多模态虚假新闻检测方向，对比 DeepSeek-V4-Flash（普通模式，思考已禁用）与 DeepSeek-V4-Pro（深度思考模式，思考已启用，reasoning_effort=high）在以下维度的差异：",
        "1. 综述类问题的知识广度与准确性",
        "2. 方法对比问题的推理深度",
        "3. 研究设计问题的应用性",
        "4. 幻觉现象的发生率与类型",
        "",
        "**思考模式控制**：",
        f"- Flash 模式：`extra_body={{\"thinking\": {{\"type\": \"disabled\"}}}}` （应无思考过程）",
        f"- Pro 模式：`extra_body={{\"thinking\": {{\"type\": \"enabled\"}}}}, reasoning_effort=\"high\"` （应有思考过程）",
        "",
        "**验证结果**：",
        f"- Flash 有思考过程输出：{report['summary']['flash_thinking_count']}/{report['summary']['flash_total']} (应为0) ✅" if
        report['summary'][
            'flash_thinking_count'] == 0 else f"- Flash 有思考过程输出：{report['summary']['flash_thinking_count']}/{report['summary']['flash_total']} ⚠️ (预期为0，请检查配置)",
        f"- Pro 有思考过程输出：{report['summary']['pro_thinking_count']}/{report['summary']['pro_total']} (应为{report['summary']['pro_total']}) ✅" if
        report['summary']['pro_thinking_count'] == report['summary'][
            'pro_total'] else f"- Pro 有思考过程输出：{report['summary']['pro_thinking_count']}/{report['summary']['pro_total']} ⚠️ (预期为{report['summary']['pro_total']}，请检查配置)",
        "",
        "**可复现方法**：",
        "- 固定 system prompt（见代码 SYSTEM_PROMPT）",
        "- 每个问题都包含幻觉检查提示",
        "- 记录每次请求的 token 数、耗时、finish_reason",
        "- 使用规则自动检测高风险内容（具体数字、论文引用等）",
        "",
        "## 逐题对比",
    ]

    for task_result in report["results"]:
        lines += [
            "",
            f"### [{task_result['task_id']}] {task_result['category']}",
            "",
            f"**问题**：{task_result['question']}",
            "",
        ]
        for resp in task_result["responses"]:
            # 判断是否为思考模式
            mode_icon = "🧠" if "deepseek-v4-pro" in resp["model"].lower() else "⚡"
            mode_text = "深度思考 (enabled)" if "deepseek-v4-pro" in resp["model"].lower() else "普通模式 (disabled)"
            has_thinking = "✅ 有" if resp["thinking"] else "❌ 无"

            lines += [
                f"#### {mode_icon} {resp['model_label']} ({mode_text})",
                "",
                f"- 耗时：{resp['elapsed_seconds']}s",
                f"- 输出 token：{resp['output_tokens']}",
                f"- 思考过程：{has_thinking}",
                f"- 幻觉风险评级：{resp['hallucination_analysis']['risk_level']}",
            ]
            if resp["hallucination_analysis"]["high_risk_elements"]:
                lines.append(f"- ⚠️ 高风险元素：")
                for elem in resp['hallucination_analysis']['high_risk_elements']:
                    lines.append(f"  - {elem}")
            if resp["hallucination_analysis"]["has_uncertainty_markers"]:
                lines.append("- ✅ 包含不确定性标注（符合要求）")
            if resp.get("thinking"):
                lines += [
                    "",
                    "<details><summary>💭 思考过程（点击展开）</summary>",
                    "",
                    "```",
                    resp["thinking"][:800] + ("..." if len(resp["thinking"]) > 800 else ""),
                    "```",
                    "</details>",
                ]
            lines += [
                "",
                "**回答**：",
                "",
                resp["content"],  # 完整内容
                "",
            ]

    # ── 汇总结论表格 ──────────────────────────────────────────────────────
    lines += [
        "## 汇总结论",
        "",
        "| 指标 | DeepSeek-V4-Flash（普通） | DeepSeek-V4-Pro（深度思考） |",
        "|---|---|---|",
        f"| 平均耗时 | {report['summary']['flash_avg_time']}s | {report['summary']['pro_avg_time']}s |",
        f"| 平均输出token | {report['summary']['flash_avg_output_tokens']} | {report['summary']['pro_avg_output_tokens']} |",
        f"| 推理模式时间开销 | +{report['summary']['time_overhead_pct']}% | — |",
        f"| HIGH风险幻觉数 | {report['summary']['flash_risk_high']} | {report['summary']['pro_risk_high']} |",
        f"| MEDIUM风险幻觉数 | {report['summary']['flash_risk_medium']} | {report['summary']['pro_risk_medium']} |",
        f"| LOW风险幻觉数 | {report['summary']['flash_risk_low']} | {report['summary']['pro_risk_low']} |",
        "",
    ]

    # ── 自动收集幻觉案例 ──────────────────────────────────────────────────
    lines.append("## 观察到的幻觉案例及核验方法")
    lines.append("")
    hallucination_cases = []
    for task_result in report["results"]:
        for resp in task_result["responses"]:
            for elem in resp["hallucination_analysis"]["high_risk_elements"]:
                mode = "深度思考" if "pro" in resp["model"].lower() else "普通"
                hallucination_cases.append({
                    "task": task_result["task_id"],
                    "model": f"{resp['model_label']} ({mode})",
                    "detail": elem,
                })

    if hallucination_cases:
        lines.append("**自动检测到的潜在幻觉风险项**：")
        lines.append("")
        lines.append("| 任务 | 模型 | 风险描述 |")
        lines.append("|---|---|---|")
        for case in hallucination_cases:
            lines.append(f"| {case['task']} | {case['model']} | {case['detail']} |")
        lines.append("")
        lines.append("> ⚠️ 以上内容均需人工核实，尤其注意论文引用和具体数字。")
    else:
        lines.append("本次实验未检测到明显的幻觉风险信号（但建议仍对关键引用进行人工验证）。")

    lines += [
        "",
        "**核验方法**：",
        "1. 对模型给出的论文引用，使用 Google Scholar 搜索验证是否存在",
        "2. 对具体数字（准确率等），回到原论文确认",
        "3. 使用 `analyze_hallucination()` 函数自动标记高风险内容",
        "4. 检查[不确定]标注是否合理使用",
        "",
        "## 推理模式在此任务上的增益分析",
        "",
        "**深度思考模式 (thinking=enabled, reasoning_effort=high) 的观察**：",
        "",
        "**有明显增益的场景**：",
        "- [ ] 多步推理的方法对比分析",
        "- [ ] 复杂研究设计类问题",
        "- [ ] 需要详细论证的综述类问题",
        "",
        "**无明显增益或可能带来问题的场景**：",
        "- [ ] 简单事实检索",
        "- [ ] 思考过深导致过度复杂化简单问题",
        "- [ ] 可能增加幻觉风险（需要实验验证）",
        "",
        "**幻觉风险对比**：",
        f"- 普通模式：HIGH={report['summary']['flash_risk_high']}, MEDIUM={report['summary']['flash_risk_medium']}, LOW={report['summary']['flash_risk_low']}",
        f"- 深度思考：HIGH={report['summary']['pro_risk_high']}, MEDIUM={report['summary']['pro_risk_medium']}, LOW={report['summary']['pro_risk_low']}",
        "",
        "**思考模式控制验证**：",
        f"- Flash (disabled)：有思考过程 {report['summary']['flash_thinking_count']}/{report['summary']['flash_total']} 次 ✅" if
        report['summary'][
            'flash_thinking_count'] == 0 else f"- Flash (disabled)：有思考过程 {report['summary']['flash_thinking_count']}/{report['summary']['flash_total']} 次 ⚠️ 配置可能未生效",
        f"- Pro (enabled)：有思考过程 {report['summary']['pro_thinking_count']}/{report['summary']['pro_total']} 次 ✅" if
        report['summary']['pro_thinking_count'] == report['summary'][
            'pro_total'] else f"- Pro (enabled)：有思考过程 {report['summary']['pro_thinking_count']}/{report['summary']['pro_total']} 次 ⚠️ 配置可能未生效",
    ]

    md_path = "模型对比实验笔记.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"✅ Markdown 笔记已保存至：{md_path}")


if __name__ == "__main__":
    run_comparison()