# Copyright 2026 the macaq-mlx authors. Licensed under the Apache License, Version 2.0.
"""Write lm-eval task files for CMMLU read from its official CSV release (task group `cmmlu_local`).

Why: the CMMLU copy on the Hugging Face hub ships a loading script, which current `datasets` releases no longer run.
The prompt (description, question layout, "答案：") and the metrics are the same as lm-eval's own cmmlu tasks.

1. Download cmmlu_v1_0_1.zip from the dataset repo `lmlmcat/cmmlu` on Hugging Face (CMMLU is CC BY-NC 4.0; it is not
   redistributed here) and unzip it; the folder must contain test/<subject>.csv and dev/<subject>.csv.
2. python bench/lmeval/make_cmmlu_tasks.py /abs/path/to/cmmlu-unzipped
   writes bench/lmeval/tasks/cmmlu_local/*.yaml (the CSV paths are written as absolute paths).
"""
import os
import sys

SUBJECTS = {
    "agronomy": "农学", "anatomy": "解剖学", "ancient_chinese": "古汉语", "arts": "艺术学", "astronomy": "天文学",
    "business_ethics": "商业伦理", "chinese_civil_service_exam": "中国公务员考试", "chinese_driving_rule": "中国驾驶规则",
    "chinese_food_culture": "中国饮食文化", "chinese_foreign_policy": "中国外交政策", "chinese_history": "中国历史",
    "chinese_literature": "中国文学", "chinese_teacher_qualification": "中国教师资格", "clinical_knowledge": "临床知识",
    "college_actuarial_science": "大学精算学", "college_education": "大学教育学",
    "college_engineering_hydrology": "大学工程水文学", "college_law": "大学法律", "college_mathematics": "大学数学",
    "college_medical_statistics": "大学医学统计", "college_medicine": "大学医学", "computer_science": "计算机科学",
    "computer_security": "计算机安全", "conceptual_physics": "概念物理学",
    "construction_project_management": "建设工程管理", "economics": "经济学", "education": "教育学",
    "electrical_engineering": "电气工程", "elementary_chinese": "小学语文", "elementary_commonsense": "小学常识",
    "elementary_information_and_technology": "小学信息技术", "elementary_mathematics": "初等数学", "ethnology": "民族学",
    "food_science": "食品科学", "genetics": "遗传学", "global_facts": "全球事实", "high_school_biology": "高中生物",
    "high_school_chemistry": "高中化学", "high_school_geography": "高中地理", "high_school_mathematics": "高中数学",
    "high_school_physics": "高中物理学", "high_school_politics": "高中政治", "human_sexuality": "人类性行为",
    "international_law": "国际法学", "journalism": "新闻学", "jurisprudence": "法理学",
    "legal_and_moral_basis": "法律与道德基础", "logical": "逻辑学", "machine_learning": "机器学习",
    "management": "管理学", "marketing": "市场营销", "marxist_theory": "马克思主义理论", "modern_chinese": "现代汉语",
    "nutrition": "营养学", "philosophy": "哲学", "professional_accounting": "专业会计", "professional_law": "专业法学",
    "professional_medicine": "专业医学", "professional_psychology": "专业心理学", "public_relations": "公共关系",
    "security_study": "安全研究", "sociology": "社会学", "sports_science": "体育学",
    "traditional_chinese_medicine": "中医中药", "virology": "病毒学", "world_history": "世界历史",
    "world_religions": "世界宗教",
}

TASK = """task: cmmlu_local_{subj}
dataset_path: csv
dataset_kwargs:
  data_files:
    test: {root}/test/{subj}.csv
    dev: {root}/dev/{subj}.csv
description: '以下是关于{name}的单项选择题，请直接给出正确答案的选项。


  '
test_split: test
fewshot_split: dev
fewshot_config:
  sampler: first_n
output_type: multiple_choice
doc_to_text: '{{{{Question.strip()}}}}

  A. {{{{A}}}}

  B. {{{{B}}}}

  C. {{{{C}}}}

  D. {{{{D}}}}

  答案：'
doc_to_choice:
- A
- B
- C
- D
doc_to_target: '{{{{[''A'', ''B'', ''C'', ''D''].index(Answer)}}}}'
metric_list:
- metric: acc
  aggregation: mean
  higher_is_better: true
- metric: acc_norm
  aggregation: mean
  higher_is_better: true
metadata:
  version: 1.0
"""

GROUP_HEAD = "group: cmmlu_local\ntask:\n"
GROUP_TAIL = """aggregate_metric_list:
- aggregation: mean
  metric: acc
  weight_by_size: true
- aggregation: mean
  metric: acc_norm
  weight_by_size: true
metadata:
  version: 1.0
"""


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    root = os.path.abspath(sys.argv[1])
    missing = [s for s in SUBJECTS for split in ("test", "dev")
               if not os.path.exists(os.path.join(root, split, f"{s}.csv"))]
    if missing:
        raise SystemExit(f"{root} lacks {len(missing)} CSV files, e.g. {missing[:3]}")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tasks", "cmmlu_local")
    os.makedirs(out, exist_ok=True)
    for subj, name in SUBJECTS.items():
        with open(os.path.join(out, f"cmmlu_local_{subj}.yaml"), "w", encoding="utf-8") as f:
            f.write(TASK.format(subj=subj, name=name, root=root))
    with open(os.path.join(out, "_cmmlu_local.yaml"), "w", encoding="utf-8") as f:
        f.write(GROUP_HEAD + "".join(f"- cmmlu_local_{s}\n" for s in sorted(SUBJECTS)) + GROUP_TAIL)
    print(f"wrote {len(SUBJECTS)} subject tasks and the group to {out}")


if __name__ == "__main__":
    main()
