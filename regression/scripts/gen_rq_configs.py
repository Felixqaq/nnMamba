"""Generate the 10 aligned RQ1/2/3 configs from one shared protocol template.

Guarantees the training protocol is byte-identical across all configs; the only
differences are model.name / tapct_features / manifest path and the task-inherent
data fields (target_mode, num_classes, loss, target_normalization, class_indices).
"""
from __future__ import annotations
from collections import OrderedDict
from pathlib import Path

import yaml

REG = Path(__file__).resolve().parents[1]
TAPCT = "./embeddings/tapct_s_3d/features.npz"

MODEL_COMMON = OrderedDict(
    in_channels=1, hidden_dim=256, dropout=0.3, base_channels=32, blocks=3,
    attn_heads=8, attn_layers=1, attn_mlp_ratio=2.0, attn_dropout=0.1,
    tapct_embedding_dim=1152, fusion_projection_dim=128, fusion_dropout=0.1,
    feature_size=24, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24],
    window_size=4, patch_size=2, use_checkpoint=False, use_v2=True,
)

AUG_CLS = OrderedDict(
    enabled=True, balance_then_augment=True, views_per_sample=5, probability=1.0,
    rotation_degrees=7.0, translation_fraction=0.05, scale_range=[0.95, 1.05],
    intensity_scale_range=[0.95, 1.05], intensity_shift_range=[-0.1, 0.1],
    noise_std=0.03,
)

PFT = "../pft.json"
ANGLE_SPLIT = "../by_angle_all/reclassification_manifest.json"

TASKS = [
    dict(key="rq1.normal_v_abnormal", target_mode="normal_v_abnormal", num_classes=2,
         loss="cross_entropy", tgt_norm="none", class_indices=[0, 1],
         task="RQ1_normal_v_abnormal",
         source_dir="../classification/datasets/normal_v_abnormal_66",
         extra=OrderedDict(), aug="cls", stem="rq1_nva66"),
    dict(key="rq2a.angle_3class", target_mode="angle_3class", num_classes=3,
         loss="cross_entropy", tgt_norm="none", class_indices=[0, 1, 2],
         task="RQ2a_angle_3class", source_dir="../by_angle_all",
         extra=OrderedDict(pft_json=PFT, angle_split_manifest=ANGLE_SPLIT),
         aug="cls", stem="rq2a_angle3class"),
    dict(key="rq2b.angle_binary", target_mode="angle_binary_extreme", num_classes=2,
         loss="cross_entropy", tgt_norm="none", class_indices=[0, 1],
         task="RQ2b_angle_binary_extreme", source_dir="../by_angle_all",
         extra=OrderedDict(pft_json=PFT, angle_split_manifest=ANGLE_SPLIT),
         aug="cls", stem="rq2b_anglebin"),
    dict(key="rq2c.angle_reg", target_mode="angle", num_classes=1,
         loss="auto", tgt_norm="zscore", class_indices=None,
         task="RQ2c_angle_regression", source_dir="../by_angle_all",
         extra=OrderedDict(pft_json=PFT, angle_split_manifest=ANGLE_SPLIT),
         aug="reg", stem="rq2c_anglereg"),
    dict(key="rq3.oi_emphysema", target_mode="oi_emphysema", num_classes=2,
         loss="cross_entropy", tgt_norm="none", class_indices=[0, 1],
         task="RQ3_oi_emphysema", source_dir="../by_angle_all",
         extra=OrderedDict(oi_threshold=3.0, pft_json=PFT,
                           oi_json="./oi_processed.json",
                           angle_split_manifest=ANGLE_SPLIT),
         aug="cls", stem="rq3_oiemph"),
]

ARMS = [
    dict(arm="image", name="hybrid_mamba_attention", tapct=None),
    dict(arm="fusion", name="hybrid_mamba_tapct_fusion", tapct=TAPCT),
]


def build(t: dict, a: dict) -> OrderedDict:
    is_reg = t["aug"] == "reg"
    model = OrderedDict(name=a["name"])
    model.update(num_classes=t["num_classes"])
    model.update(MODEL_COMMON)

    training = OrderedDict(
        epochs=100, batch_size=12, eval_batch_size=12, swin_batch_size=5,
        swin_eval_batch_size=6, learning_rate=0.0001, weight_decay=0.001,
        k_folds=5, eval_interval=5, save_interval=10, seed=42, loss=t["loss"],
        clip_grad_norm=1.0, amp=False, track_train_metrics=False,
        class_weight_mode="none",
    )
    early = OrderedDict(enabled=False, patience=6, min_delta=0.005)
    gradcam = OrderedDict(enabled=False, max_samples=8,
                          target_layer="image_encoder.attention_layers",
                          target_class=0)

    if is_reg:
        aug = OrderedDict(enabled=False)
    else:
        aug = OrderedDict(AUG_CLS)
        aug["class_indices"] = t["class_indices"]
        # keep field order: class_indices right after probability
        aug = OrderedDict(
            enabled=True, balance_then_augment=True, views_per_sample=5,
            probability=1.0, class_indices=t["class_indices"],
            rotation_degrees=7.0, translation_fraction=0.05,
            scale_range=[0.95, 1.05], intensity_scale_range=[0.95, 1.05],
            intensity_shift_range=[-0.1, 0.1], noise_std=0.03,
        )

    data = OrderedDict(target_mode=t["target_mode"], source_dir=t["source_dir"],
                       labels_json="../patient_angle_classification_by_group.json")
    data.update(t["extra"])
    data.update(
        manifest=f"./datasets/generated/{t['stem']}_manifest.{a['arm']}.json",
        tapct_features=a["tapct"],
        image_size=[112, 136, 112], intensity_window=[-1000.0, 400.0],
        input_normalization="zscore", target_normalization=t["tgt_norm"],
        cache_data=True, num_workers=4, pin_memory=True, prefetch_factor=4,
        angle_bin_count=5, balanced_sampling=(not is_reg), augmentation=aug,
    )

    return OrderedDict(
        experiment=OrderedDict(
            name=f"{t['task']} {a['arm']}",
            description=(f"{t['target_mode']} | {a['name']} | 66-patient aligned "
                         f"protocol (100ep, no early stop, seed 42, 5-fold)."),
        ),
        model=model, training=training, early_stopping=early, gradcam=gradcam,
        data=data, task=t["task"],
        resume=OrderedDict(enabled=False, uuid=None, start_fold=0),
        gpu=OrderedDict(device_id="0"),
    )


def represent_odict(dumper, data):
    return dumper.represent_mapping("tag:yaml.org,2002:map", data.items())


def main() -> None:
    yaml.add_representer(OrderedDict, represent_odict)
    for t in TASKS:
        for a in ARMS:
            cfg = build(t, a)
            path = REG / f"config.{t['key']}.{a['arm']}.yaml"
            with path.open("w", encoding="utf-8") as f:
                f.write(f"# {t['task']} — {a['arm']} arm (auto-generated by "
                        "scripts/gen_rq_configs.py)\n")
                yaml.dump(cfg, f, sort_keys=False, default_flow_style=False,
                          allow_unicode=True)
            print(f"wrote {path.name}")


if __name__ == "__main__":
    main()
