"""Write a patient-holdout report, separating ratio, vote and probability."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def method_metrics(full: dict, three: dict) -> list[tuple[str, dict]]:
    """Keep the preselected seed72 reference rather than choosing a best seed."""
    first = next(m for m in full['per_member'].values() if m['seed'] == 72)
    return [('單模型 seed72', first['fixed70']),
            ('3模型平均比值', three['fixed70']['mean_ratio']),
            ('3模型多數決', three['fixed70']['majority_vote']),
            ('5模型平均比值', full['fixed70']['mean_ratio']),
            ('5模型多數決', full['fixed70']['majority_vote'])]


def main() -> None:
    root = Path(sys.argv[1])
    full = json.loads((root / 'ensemble5.json').read_text())
    three = json.loads((root / 'ensemble3.json').read_text())
    audit = json.loads((root / 'cohort_audit.json').read_text())
    timing = json.loads((root / 'complete.json').read_text())
    reviewed_n = full['fixed70']['mean_ratio']['n']
    reserved_n = audit.get('test_trained_against', audit.get('test', 200))
    assert reviewed_n == audit.get('test_scored', reserved_n)
    cohorts = []
    if (root / 'ensemble5_official200.json').exists():
        original = json.loads((root / 'ensemble5_official200.json').read_text())
        original3 = json.loads((root / 'ensemble3_official200.json').read_text())
        assert original['fixed70']['mean_ratio']['n'] == reserved_n
        cohorts.append(('原固定測試集', original, original3))
    cohorts.append(('依審查紀錄排除後的測試集', full, three))
    lines = ['# CT＋LAA 二階段回歸五模型結果', '',
        f"{audit['n']} 人；{audit['train']} 人訓練、{reserved_n} 人保留未參與訓練。五個種子 72–76，各80 epochs。", '',
        f"專案醫師審查紀錄另列排除 {reserved_n-reviewed_n} 人；審查後實際評估 {reviewed_n} 人。排除者沒有移入訓練集。兩種評估分開報告，不能將188人的分數標成200人。", '',
        '每個模型預測 FEV1/FVC 百分比值；小於70判為異常。平均比值與多數決是不同決策規則。']
    keys = ['auc','accuracy','balanced_accuracy','sensitivity','specificity']
    for cohort_name, five, three_result in cohorts:
        n = five['fixed70']['mean_ratio']['n']
        methods = method_metrics(five, three_result)
        assert all(m['n'] == n for _, m in methods)
        assert len(five['patients']) == len(three_result['patients']) == n
        assert set(five['patients']) == set(three_result['patients'])
        lines += ['', f'## {cohort_name}（{n} 人）', '',
                  '| 方法 | AUC | Accuracy | Balanced accuracy | Sensitivity | Specificity |',
                  '| --- | --- | --- | --- | --- | --- |']
        for name, metrics in methods:
            lines.append('| '+name+' | '+' | '.join(f'{metrics[k]:.4f}' for k in keys)+' |')
        lines += ['', f"五模型平均比值 MAE：{five['fixed70']['ratio_mae']:.4f} 個百分點。",
                  f"平均比值 AUC 95% bootstrap CI：{five['fixed70']['mean_ratio']['auc_95ci']}。",
                  f"兩種五模型判定不同：{five['fixed70']['rules_disagree_on']} 人。"]
    lines += ['',
              'AUC 的分數來源：平均法使用負的預測比值；投票法使用0–5票數。',
              '3/5票=三個模型判為異常，不代表60%患病機率；例如FEV1/FVC 68%也是肺功能比值，不是患病機率。', '',
              '原200人測試集保留，但此測試集曾反覆用於歷史方法比較，不能當成全新外部驗證。',
              '188人版本的排除名單來自專案 doctor_review_exclusions.json；屬審查後子集，不可直接和歷史200人結果比較。此報告沒有重新判定排除理由或更動標籤。',
              '新增病例標籤沿用經核實影像與肺功能配對；包含無post-BD時使用pre-BD情況。',
              '新增肺遮罩採TotalSegmentator fast；舊密度通道沿用既有快取。',
              '不在200人上挑epoch或調分類門檻；本次結果不是臨床部署認證。', '',
              f"前處理至首輪評估完成的經過時間（含中斷重跑）：{timing['elapsed_seconds']/3600:.2f} 小時；不含其後補算原200人評估。", '',
              '![五模型比較](comparison.png)']
    (root / 'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    fig, axes = plt.subplots(1, len(cohorts), figsize=(9*len(cohorts),4), squeeze=False)
    labels = ['Seed72', 'Mean3', 'Vote3', 'Mean5', 'Vote5']
    for ax, (_, five, three_result) in zip(axes[0], cohorts):
        n = five['fixed70']['mean_ratio']['n']
        methods = method_metrics(five, three_result)
        bars = ax.bar(labels, [m['auc'] for _,m in methods])
        ax.bar_label(bars, fmt='%.4f')
        label = 'Original holdout' if n == reserved_n else 'Reviewed subset'
        ax.set(ylim=(0,1), ylabel='AUC', title=f'CT + LAA: {label}, n={n}')
    fig.tight_layout()
    fig.savefig(root / 'comparison.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
