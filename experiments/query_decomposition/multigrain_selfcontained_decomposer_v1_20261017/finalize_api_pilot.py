"""Serialize literal Codex judgments and reconcile this completed API pilot.

The authored judgments below come from reading every original question and both
actual outputs. This code does not infer semantic correctness from model output.
P=all core checks pass; a string of FSCU gives explicitly failed checks;
? followed by checks gives uncertainty; X is an unavailable API response.
"""

from collections import Counter
import json

from .run_api_pilot import HERE, OUT, rows, sha, summarize
from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import utc, write_json


# qid, v1 authored code/reason, v2 authored code/reason. Original-only terminal
# repetitions are logged separately from the shared semantic-content assessment.
AUTHORED = [
    ("5ae1f079554299234fd04343", "P", "原题是已明确公路到县的局部关系；输出忠实改述原题。按共同内容口径可用，但违反v1排除原题的指令。", "U", "新增Which county is located in the commonwealth of Kentucky?没有限定到目标公路，也未明确询问完整候选集合；它没有独立的必要支持用途。"),
    ("c116cee408ac11ebbd83ac1f6bf848b6", "P", "分别取得两部电影的导演身份和出生时间，保留电影与属性对应。", "P", "分别取得两部电影的导演身份和出生时间，保留电影与属性对应。"),
    ("2hop__124412_402641", "P", "询问Chiang Hsiao-wen的父亲，提供后续父亲之子问题的对象。", "P", "保留父亲身份问题，并自然输出原问题；原题重复由程序追加规则处理。"),
    ("5ae33c4d5542992f92d82262", "FCU", "把曾任阿肯色州州长的人改为任职同时担任州长的人，且插入输入没有的Bill Clinton及其前任问题。", "?C", "只有原题的语法改述，没有较简短的任职对象或职位支持问题。原题可被视作带限定的局部人物关系，也可展开任职对象与曾任州长的条件，覆盖判为不确定。"),
    ("2hop__156085_436709", "P", "按原句表面结构询问Mickey Mouse配偶的身份和该配偶的创作者，没有改成创作者的配偶。第二题属于原题等价表达，另记v1指令偏离。", "P", "按原句表面结构询问配偶身份及该配偶的创作者，保留关系方向；末题等价原题。"),
    ("73dcc04808de11ebbd9eac1f6bf848b6", "P", "两个独立问题取得两位电影导演的去世日期；允许以中等粒度直接请求属性，不强制再列姓名。", "P", "两个独立问题取得两位电影导演的去世日期；允许中等粒度，不强制姓名节点。"),
    ("5ae0c7b055429945ae95944b", "P", "输出保留团体、1590年代和剧场类型。仅改述原题另记指令偏离；不因没有强制增加场馆身份而判语义失败。", "P", "剧场身份与剧场类型形成有用途的两个描述层次，保留团体和1590年代。"),
    ("ab622778089911ebbd77ac1f6bf848b6", "P", "两部电影分别取得导演及出生日期，足以支持年龄先后比较。", "P", "两部电影分别取得导演及出生日期或年龄；birth date or age表述有时间口径风险，但含有正确所需属性，未按确定语义错误处罚。"),
    ("2hop__503085_110222", "P", "画家身份与画家的出生日期均独立完整；末题等价原题，另记v1指令偏离。", "P", "画家身份与画家的出生日期均独立完整，允许自然产生原题。"),
    ("5abf0bf95542997719eab63a", "P", "两题分别识别研究对象和其额外职业，第二题保留José Rubia Barcia及诗人/剧作家描述。", "S", "第二题只写that Spanish figure，没有姓名或与José Rubia Barcia的关系，无法独立识别对象。"),
    ("c51148bc0bdb11eba7f7acde48001122", "U", "原题括号Daughter Of Selim Ii已给父亲身份，再问父亲姓名冗余；Selim II本身来自输入，不是模型凭空填入事实。", "U", "原题括号已给父亲身份，仍重复询问父亲。两题还省略原始标题消歧，记录为风险，不把Selim II当作外部事实。"),
    ("2hop__635187_861533", "P", "地区所在省和该省的邻省问题均可独立理解；后题等价原题，另记v1指令偏离。", "P", "地区所在省和该省的邻省问题均可独立理解，允许自然产生原题。"),
    ("5ab8cd3d5542991b5579f009", "U", "询问整个乐队成员名单没有明确支持乐队命名来源的必要用途；最后命名问题还缺少原始时间限定。", "P", "取得指定时期的乐队名称，并保留带时期限定的命名来源问题。active from的措辞有成员任期/乐队活动期混淆风险，保留在原因中，不据此认定已发生事实错误。"),
    ("b0fe569c0bdd11eba7f7acde48001122", "P", "歌曲表演者及其来源国均用完整描述表达；后题等价原题，另记v1指令偏离。", "P", "歌曲表演者及其来源国均用完整描述表达。"),
    ("3hop1__540585_568433_51423", "P", "制作公司、总部所在地及该城市中的城堡均保留完整描述。", "S", "最后一题用that city指代前题所得城市，没有在该题内给出完整定位描述。"),
    ("5abad3445542996cc5e49ea0", "F", "把traded to改成join，保留了赛季但弱化原题要求的转会方式。此项按本次保留关系的严格口径判不合格，不声称已经造成下游答错。", "P", "兄弟身份和赛季后的交易去向均保留完整对象描述及traded关系。"),
    ("f77e7afa0bda11eba7f7acde48001122", "P", "大学创办者身份及其配偶关系均独立完整；后题等价原题，另记v1指令偏离。", "P", "大学创办者身份及其配偶关系均独立完整。"),
    ("3hop1__573244_339990_15538", "S", "第二题只写that state，不能离开第一题独立识别州。", "P", "地点问题和恢复死刑的州问题保留Kashinka Hollow定位描述。Where可能返回不同地点层级，记录为检索精度风险。"),
    ("5a8333115542993344745feb", "S", "第二题用that county依赖前题。", "U", "新增询问包含Helena的县之县治，方向与所需县身份不同，且原题已给Helena是县治；后两题已有所需信息，第一题冗余。"),
    ("1a25fd560bb011ebab90acde48001122", "F", "第二题省略1st Marquis Of La Laguna这一输入提供的身份消歧；按要求保留完整身份限定的严格口径判不合格，不声称事实指向一定错误。", "P", "父亲及父亲的父亲两题均保留完整姓名/爵位消歧，且没有猜测实际人名。"),
    ("3hop1__483634_720378_15538", "S", "最后一题用that state依赖前题；其余县到州描述有用途。", "S", "最后一题仍用that state依赖前题，运行示例没有消除此问题。"),
    ("5a7912f655429974737f796c", "P", "演员身份与出生日期均保留电影和角色限定；后题等价原题，另记v1指令偏离。", "P", "演员身份与出生日期均保留电影和角色限定。"),
    ("20bf77b40bb011ebab90acde48001122", "P", "原题是局部配偶关系，输出只是同义改述；共同内容可用，但违反v1要求局部问题输出空数组。", "P", "局部配偶关系返回空数组，由程序追加原题。"),
    ("3hop1__51933_40769_64047", "?CU", "没有明确询问豪华部门身份；直接询问RX 350改款日期也可能足以支持原题，不能仅按缺失原子节点判错。中间公司问题是否提供新增用途不确定。", "?CU", "豪华部门身份仍未明确展开，新增车身样式问题是否帮助确定改款日期不确定；RX 350已在原题给出，不能要求所有桥接必须逐边展开。"),
    ("5a89a02955429946c8d6e975", "P", "两部电影各自发行日期，未猜测电影版本。原始片名歧义保留。", "P", "两部电影各自发行日期，未猜测电影版本。原始片名歧义保留。"),
    ("6a8a71280bb011ebab90acde48001122", "F", "把未限定父系或母系的uncle限制为父亲的兄弟姐妹，加入原题未给的父系路径。", "U", "Who is John Kennedy...是对已指定人物的泛化身份问题，没有独立必要用途；后题只是原题。"),
    ("4hop2__567956_14670_8987_8529", "SC", "Country B在各子问题中没有原题提供的关系描述，且缺少定位A/B的桥接。If so又依赖前文判断。", "SC", "Which country is Country A/B?没有任何关系限定，其他问题也只用未绑定的Country B；缺少可独立执行的国家定位问题。"),
    ("5ae14def55429901ffe4ae4a", "P", "分别询问两人是否为导演，供原题完成合取。", "P", "分别询问两人是否为导演，供原题完成合取。"),
    ("ed0838b508dc11ebbd9dac1f6bf848b6", "P", "分别取得两部电影的来源国；produced与originate之间有措辞风险，但这里作为来源国查询初审可用，没有用外部事实核对。", "P", "分别询问两部电影的来源地，可供国家比较；Where比country宽，作为地点层级风险另记。"),
    ("4hop2__71753_648517_70784_79935", "F", "把north of the region where Israel is located缩成north of Israel，改变空间关系对象。原题后半句本身有歧义，不据此强行补足事实。", "S", "使用identified in answer to the first/second question，明确依赖其他问题；原题后半句歧义保留，未靠背景事实解决。"),
    ("5a77685e5542993569682d83", "X", "APIConnectionError，未收到完整回复；不重发、不编造语义判断，保留36题分母。", "P", "分别取得两场战役的发生时间供先后比较，保留原始战役名歧义。"),
    ("ee5cf4a008a711ebbd7fac1f6bf848b6", "P", "分别取得两个博物馆的成立时间，保留完整名称。", "P", "分别取得两个博物馆的成立时间，保留完整名称。"),
    ("4hop1__408432_32392_823060_610794", "S", "最后一题用that city指代前面的边界城市。", "P", "电台所在地、州首府、相邻城市和县四层均保留完整描述，最后一题不依赖其他题的代词。"),
    ("5a7ca95b55429935c91b5287", "P", "分别取得两位作家的出生与去世日期，保留人名、国籍和职业限定。", "P", "分别取得两位作家的出生与去世日期，保留人名、国籍和职业限定。"),
    ("fa7ad32b093411ebbdaeac1f6bf848b6", "P", "分别询问两部电影的发行时间，供原题比较。", "P", "分别询问两部电影的发行时间，供原题比较。"),
    ("4hop1__151650_5274_458768_33637", "FSC", "将only group替换为另一家record label，遗漏先确定Desde El Principio唱片公司的层次，并多次用that record label/headquarters/city跨题指代。", "FS", "that record label、that larger group、that city均需前题；only这一限定也在较大集团问题中丢失。"),
]


def main():
    cases = rows(OUT / "cases.frozen.jsonl")
    authored = {r[0]: r[1:] for r in AUTHORED}
    assert len(authored) == len(cases) == 36 and set(authored) == {r["qid"] for r in cases}
    reviewed = []
    names = {"F": "faithfulness_and_unknowns", "S": "self_containment", "C": "supporting_information_coverage", "U": "distinct_usefulness"}
    for case in cases:
        pair = {k: case[k] for k in ("qid", "dataset", "question", "type")}
        for i, arm in enumerate(("v1", "v2")):
            code, reason = authored[case["qid"]][i*2:i*2+2]
            path = OUT / "cases" / arm / case["dataset"] / (case["qid"] + ".json")
            raw = json.loads(path.read_text(encoding="utf-8"))
            assert raw["question"] == case["question"] and raw["qid"] == case["qid"]
            if code == "X":
                assert raw["status"] != "success"
                verdict, checks = "unavailable", {name: "not_scored" for name in names.values()}
            else:
                assert raw["status"] == "success" and raw["finish_reason"] == "stop" and raw["mechanical"]["array_of_nonempty_strings"]
                verdict = "pass" if code == "P" else "uncertain" if code.startswith("?") else "fail"
                checks = {name: "pass" for name in names.values()}
                if code != "P":
                    for letter in code.lstrip("?"):
                        checks[names[letter]] = "uncertain" if code.startswith("?") else "fail"
            pair[arm] = {"verdict": verdict, "checks": checks, "reason": reason, "literal_answer": raw.get("answer"),
                         "request_id": raw["request_id"], "raw_result_sha256": sha(path), "raw_file": str(path.relative_to(OUT))}
        reviewed.append(pair)
    review_path = OUT / "AUTHORED_SEMANTIC_REVIEW.jsonl"
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in reviewed)
    if review_path.exists():
        assert review_path.read_text(encoding="utf-8") == text
    else:
        review_path.write_text(text, encoding="utf-8")
    mechanical = summarize()
    groups = {}
    for dataset in ("overall", "hotpotqa", "2wiki", "musique"):
        subset = reviewed if dataset == "overall" else [r for r in reviewed if r["dataset"] == dataset]
        group = {arm: {"planned": len(subset), **dict(Counter(r[arm]["verdict"] for r in subset)),
                       "pass_rate_full_denominator": sum(r[arm]["verdict"] == "pass" for r in subset)/len(subset),
                       "failed_checks": {name: sum(r[arm]["checks"][name] == "fail" for r in subset) for name in names.values()}}
                 for arm in ("v1", "v2")}
        common = [r for r in subset if all(r[a]["verdict"] != "unavailable" for a in ("v1", "v2"))]
        group["common"] = {"n": len(common), "v1_pass": sum(r["v1"]["verdict"] == "pass" for r in common),
                           "v2_pass": sum(r["v2"]["verdict"] == "pass" for r in common),
                           "gained_pass": [r["qid"] for r in common if r["v1"]["verdict"] != "pass" and r["v2"]["verdict"] == "pass"],
                           "lost_pass": [r["qid"] for r in common if r["v1"]["verdict"] == "pass" and r["v2"]["verdict"] != "pass"]}
        groups[dataset] = group
    result = {"utc": utc(), "review_context": "Current Codex, one shared nonblind context; literal query/output review, not independent human gold.",
              "rubric": "PILOT_REVIEW_RULES.txt; shared content assessment does not reward v2 merely for relaxing original-question exclusion.",
              "uncertain_cases": "RX 350 bridge utility in both arms; Robert Suettinger local-vs-decomposable reading in v2.",
              "strict_case_notes": "v1 trade/join and dropped 1st Marquis qualifier are treated as strict-contract failures; this is not proof of downstream factual errors.",
              "groups": groups, "mechanical": mechanical,
              "claims": "Descriptive prompt diagnostic only; no 4B, downstream F1, independent-test or significant-superiority claim."}
    write_json(OUT / "RESULTS.json", result)
    lines = ["自包含多粒度分解prompt：36题配对API小测", "",
             "三个数据集各12题，共36题×2版。复用已观察的诊断问题，不是独立测试集。",
             "输入仅为实际system prompt和原query；无标准答案、参考分解、文档或检索结果。",
             "模型 qwen-plus-2025-12-01，temperature=0，thinking关闭，max_tokens=1024。",
             "v1为旧prompt；v2删除RAG背景、采用贯穿例子，并条件允许自然产生原query。属于合并改动，不能归因于某一句。", "",
             "范围\t方案\t完整回应\t内容初审通过\t失败\t不确定\t无可评回复"]
    for ds, group in groups.items():
        for arm in ("v1", "v2"):
            r = group[arm]
            lines.append(f"{ds}\t{arm}\t{r['planned']-r.get('unavailable',0)}/{r['planned']}\t{r.get('pass',0)}/{r['planned']} ({r['pass_rate_full_denominator']:.2%})\t{r.get('fail',0)}\t{r.get('uncertain',0)}\t{r.get('unavailable',0)}")
    c = groups["overall"]["common"]
    lines += ["", f"共同有效{c['n']}题：v1通过{c['v1_pass']}，v2通过{c['v2_pass']}；转为通过{len(c['gained_pass'])}题，失去通过{len(c['lost_pass'])}题。",
              "两版均有6题出现不能独立理解的指代。没有#N不能证明没有跨题引用。",
              "v2有4题精确重现原题，按新规则允许；程序按规范化完全相同文本去重后追加原题，未做语义同义识别。",
              "结果为当前Codex同一上下文逐题初审；非盲审、非人工金标。局部/桥接边界、关系弱化、消歧省略的严格判法在逐题理由中明示。",
              "1条v1请求APIConnectionError后状态不明，未重发。仅续发其余9条尚未提交请求；72条计划均保留。",
              f"本次已返回用量按既有单价计{mechanical['pilot_usage_list_price_cny']:.7f}元；未核销预留{mechanical['unknown_reserved_cny']:.7f}元。实际账单未核对。",
              "初期Windows路径分隔符导致打印的本次小计为0，现已按全局账本重算；逐请求计费和全局预算未丢失。",
              "尚不能冻结为90%正确的标注器，也不能据此推断4B重训或下游RAG提升。", "",
              "文件：PROTOCOL_FROZEN.json；cases.frozen.jsonl；api/下请求与原始响应；AUTHORED_SEMANTIC_REVIEW.jsonl；RESULTS.json。"]
    (OUT / "REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"groups": groups, "cost": mechanical["pilot_usage_list_price_cny"], "unresolved": mechanical["unknown_reserved_cny"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
