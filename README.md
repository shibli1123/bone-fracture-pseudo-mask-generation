# Weakly Supervised Bone Fracture Segmentation

This repository contains a reproducible research pipeline for image-level fracture classification, class activation map (CAM) localization, pseudo-mask creation, and optional U-Net training. It is a research prototype, not a clinical diagnostic tool.

## Current evidence and limits

The checked-in data currently contains one class-folder X-ray source. It does not include the FracAtlas expert masks, the other sources named in the research guide, patient identifiers, or validated segmentation labels. No deep-learning model has been run in this workspace: PyTorch and torchvision are unavailable here. The draft in `docs/` reports the verified dataset inventory and is explicitly not a completed empirical paper. Do not treat the audit figure as model performance.

Exact SHA-256 duplicates are grouped so they cannot cross splits, and duplicates are collapsed for model training. This is not patient-level separation and does not detect near duplicates. The data audit reports label conflicts, source/split counts, and candidate mask matches. Review the audit before training.

## Quick start

Run from `Code/` with Python 3.10 or newer. Install dependencies in a virtual environment, then prepare and inspect the manifest:

```powershell
python -m pip install -r requirements.txt
python prepare_data.py
Get-Content ..\outputs\data\audit.md
python shortcut_check.py
```

The default input is `../Datasets`; generated files go under `../outputs/` (excluded from Git). You can use `--data` and `--out` to select other locations.

Train one classifier configuration and generate CAMs:

```powershell
python train_classifier.py --model resnet50 --image-size 448 --seed 0
python cam_generate.py --checkpoint ..\outputs\classifiers\resnet50_448_s0\best.pt --method layercam --split all --tta-flip
```

Supported classifiers are ResNet-50 and DenseNet-121. ImageNet weights are downloaded by torchvision unless `--no-pretrained` is supplied. Classifier selection uses validation data; the test split is not evaluated during training. Repeat with seeds 0, 1, and 2 for a paper study.

When expert masks are available, tune a global CAM threshold on validation only and use the selected value once on the test split:

```powershell
python evaluate.py --index ..\outputs\cams\resnet50_448_s0_layercam_tta\index.csv --split val --sweep
python evaluate.py --index ..\outputs\cams\resnet50_448_s0_layercam_tta\index.csv --split test --threshold 0.40
```

Replace `0.40` with the validation-selected value. The evaluator reports Dice, IoU, precision, recall, pointing-game accuracy, and group-bootstrap confidence intervals. It refuses to sweep thresholds on the test split. Thresholded CAMs are coarse localization cues and must not be described as expert fracture boundaries.

Create pseudo-masks and optionally train a U-Net:

```powershell
python make_masks.py --index ..\outputs\cams\resnet50_448_s0_layercam_tta\index.csv --threshold 0.40
python train_segmentation.py --mask-index ..\outputs\pseudo_masks\index.csv
```

The U-Net trainer reads only train and validation rows and never evaluates the test split. Its validation score measures agreement with the supplied training labels; when those labels are CAM pseudo-masks, this is not an expert-mask segmentation score. Evaluate its checkpoint separately against held-out FracAtlas expert masks before making segmentation claims.

## Data layout

The data scanner accepts images under directories named `fractured` and `not fractured` (also recognizes `not_fractured`, `normal`, and common aliases). Existing `train`, `val`/`validation`, and `test` directory names are retained when exact duplicate groups agree. Otherwise, deterministic hash-grouped splits are assigned. Patient metadata should be used for grouping when available.

Mask pairing is currently filename-based for files stored in directories named `mask`, `masks`, `annotations`, or `ground-truth`. Confirm every candidate pair before using it. COCO JSON polygons are not converted automatically in this version.

## Code map

- `Code/prepare_data.py`: image inventory, exact-hash grouping, manifest, and audit.
- `Code/train_classifier.py`: transfer-learning classifier with head warm-up and fine-tuning.
- `Code/cam_generate.py`, `Code/wsss/cam.py`: Grad-CAM, Grad-CAM++, LayerCAM, and optional flip averaging.
- `Code/evaluate.py`, `Code/wsss/masks.py`: validation threshold sweep and mask metrics.
- `Code/make_masks.py`: thresholding and morphology for pseudo-mask export.
- `Code/shortcut_check.py`: low-level image-statistics diagnostic.
- `Code/baselines.py`: empty, central-ellipse, and random localization baselines.
- `Code/train_segmentation.py`: small U-Net baseline trained on paired masks.
- `Code/aggregate_results.py`: aggregate run summaries across seeds.
- `Code/export_audit_outputs.py`: export the repository inventory counts, audit-only plot, and explicit model-run status.
- `Code/build_paper_draft.py`: generate the Word draft from the supplied IEOM formatting template.
- `Code/ieom_paper_format.docx`: content-cleared copy of the author-supplied format, retaining its page design and header/footer.
- `Code/bone_fracture_complete_all_in_one_colab (1).ipynb`: original exploratory notebook; the scripts above are canonical.

## Current outputs

- `docs/Bone_Fracture_WSSS_IEOM_Draft.docx`: paper draft using the supplied IEOM manuscript layout, with Shibli Sanjid Faheem and Yousuf Abdullah as authors. It states that no classifier or segmentation model was run.
- `docs/outputs/dataset_split_counts.csv` and `run_and_data_summary.json`: reproducible inventory counts from the dataset manifest.
- `docs/outputs/dataset_distribution.png`: image-count plot by existing split and class; it is not a model result.
- `docs/outputs/model_run_status.csv`: records the classifier and U-Net run status and why no performance output exists.

Regenerate the audit outputs and paper after preparing the manifest with:

```powershell
python Code\prepare_data.py
python Code\export_audit_outputs.py
python Code\build_paper_draft.py
```

## Reproducibility checklist

Before calling this study submission-ready, add and verify all four intended data sources; provide source licenses and provenance; provide patient/study grouping where permitted; convert and audit FracAtlas expert masks; run all model/CAM ablations for three seeds; select thresholds on validation only; evaluate the test set once; report per-source classifier AUC and segmentation metrics with uncertainty; run shortcut and trivial-baseline checks; select representative success/failure overlays; and complete the result tables in `docs/`.

The research guide also mentions MedSAM refinement, GRAZPEDWRI box constraints, and a ResNet-34 segmentation encoder. Those optional components are not implemented in this baseline code. Do not report them as completed experiments.

## Publication target

The guide targets the 9th IEOM Bangladesh International Conference. Its official author page specifies an 8–12 page paper and an IEOM Word template. The current manuscript is based on the official template but remains incomplete until data, experiments, citations, author affiliations, and results are verified. The conference page currently gives inconsistent submission-deadline information; check directly with IEOM before relying on a date.

## Authors

Shibli Sanjid Faheem and Yousuf Abdullah
