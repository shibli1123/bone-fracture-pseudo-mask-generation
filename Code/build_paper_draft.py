from __future__ import annotations

from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = Path(__file__).with_name("ieom_paper_format.docx")
OUTPUT = ROOT / "docs" / "Bone_Fracture_WSSS_IEOM_Draft.docx"
AUDIT = ROOT / "outputs" / "data" / "manifest.csv"
FIGURE = ROOT / "docs" / "outputs" / "dataset_distribution.png"


def remove_body_content(document):
    body = document._element.body
    for child in list(body):
        if child.tag != qn("w:sectPr"):
            body.remove(child)


def set_cell_fill(cell, fill: str):
    tc_pr = cell._tc.get_or_add_tcPr()
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), fill)
    tc_pr.append(shading)


def set_table_borders(table):
    tbl_pr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        element = OxmlElement(f"w:{edge}")
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), "4")
        element.set(qn("w:space"), "0")
        element.set(qn("w:color"), "D9D9D9")
        borders.append(element)
    tbl_pr.append(borders)


def format_run(run, size=10, bold=False, italic=False, color="000000"):
    run.font.name = "Times New Roman"
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "Times New Roman")
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    run.font.color.rgb = RGBColor.from_string(color)


def add_paragraph(document, text="", *, size=10, bold=False, italic=False,
                  alignment=WD_ALIGN_PARAGRAPH.JUSTIFY, after=5, before=0,
                  keep=False, hanging=0):
    paragraph = document.add_paragraph()
    paragraph.alignment = alignment
    fmt = paragraph.paragraph_format
    fmt.space_before, fmt.space_after = Pt(before), Pt(after)
    fmt.line_spacing = 1.0
    fmt.keep_together = keep
    if hanging:
        fmt.left_indent = Inches(hanging)
        fmt.first_line_indent = Inches(-hanging)
    if text:
        format_run(paragraph.add_run(text), size=size, bold=bold, italic=italic)
    return paragraph


def add_heading(document, text, level=1):
    size = 12 if level == 1 else 11
    paragraph = add_paragraph(document, text, size=size, bold=True,
                              alignment=WD_ALIGN_PARAGRAPH.LEFT, before=5, after=2, keep=True)
    paragraph.paragraph_format.keep_with_next = True
    return paragraph


def add_table(document, headers, rows, widths):
    table = document.add_table(rows=1, cols=len(headers))
    table.autofit = False
    for i, width in enumerate(widths):
        table.columns[i].width = Inches(width)
    for i, label in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.width = Inches(widths[i])
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        set_cell_fill(cell, "1F4E78")
        paragraph = cell.paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.space_after = Pt(3)
        format_run(paragraph.add_run(label), size=9, bold=True, color="FFFFFF")
    for row_index, values in enumerate(rows):
        cells = table.add_row().cells
        for i, value in enumerate(values):
            cells[i].width = Inches(widths[i])
            cells[i].vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            if row_index % 2 == 1:
                set_cell_fill(cells[i], "F2F6FA")
            paragraph = cells[i].paragraphs[0]
            paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
            paragraph.paragraph_format.space_after = Pt(3)
            format_run(paragraph.add_run(str(value)), size=9)
    set_table_borders(table)
    add_paragraph(document, "", after=2)
    return table


def inventory_rows():
    if not AUDIT.exists():
        return [("Repository image folders", "1,927", "1,897", "0", "No expert masks")]
    frame = pd.read_csv(AUDIT)
    total = len(frame)
    unique = frame.sha256.nunique()
    masks = int(frame.mask_path.fillna("").astype(str).str.len().gt(0).sum())
    return [("BONE_FRACTURE_DATASET/cropped_dataset", f"{total:,}", f"{unique:,}",
             f"{masks:,}", "Class labels; provenance not supplied")]


def build():
    if not TEMPLATE.exists():
        raise FileNotFoundError(f"Official IEOM Bangladesh 2026 template is missing: {TEMPLATE}")
    document = Document(str(TEMPLATE))
    remove_body_content(document)
    for section in document.sections:
        section.page_width = Inches(8.5)
        section.page_height = Inches(11)
        section.top_margin = section.bottom_margin = Inches(1)
        section.left_margin = section.right_margin = Inches(1)

    # Title and author block follow the official IEOM Bangladesh 2026 template.
    add_paragraph(document,
        "Weakly Supervised Bone Fracture Segmentation and Pseudo Mask Generation from X-Ray Images Using Deep Learning",
        size=18, bold=True, alignment=WD_ALIGN_PARAGRAPH.CENTER, after=9, keep=True)
    add_paragraph(document, "Shibli Sanjid Faheem and Yousuf Abdullah", size=12, bold=True,
                  alignment=WD_ALIGN_PARAGRAPH.CENTER, after=2, keep=True)
    add_paragraph(document, "Department of Computer Science and Engineering", size=12,
                  alignment=WD_ALIGN_PARAGRAPH.CENTER, after=0)
    add_paragraph(document, "Independent University, Bangladesh", size=12,
                  alignment=WD_ALIGN_PARAGRAPH.CENTER, after=0)
    add_paragraph(document, "shiblisanjidfaheem@gmail.com", size=12,
                  alignment=WD_ALIGN_PARAGRAPH.CENTER, after=8)

    add_paragraph(document, "Abstract", size=12, bold=True,
                  alignment=WD_ALIGN_PARAGRAPH.CENTER, after=3, keep=True)
    abstract = (
        "Weakly supervised segmentation may reduce pixel-level annotation demands in bone-fracture X-ray analysis, "
        "but class activation maps (CAMs) provide coarse evidence rather than fracture boundaries. This paper defines "
        "a reproducible pipeline for transfer-learning classification, Grad-CAM, Grad-CAM++, and LayerCAM localization, "
        "validation-only threshold selection, and optional U-Net training on pseudo-masks. The current repository audit "
        "identified 1,927 labeled images from one class-folder source, representing 1,897 exact image hashes. No expert "
        "pixel masks, patient identifiers, or trained model results are currently available in the checkout. Accordingly, "
        "this manuscript reports the protocol and data status only; it makes no performance claims. A held-out FracAtlas "
        "evaluation, multiple-seed experiments, confidence intervals, and baseline comparisons remain necessary before "
        "submission. The implementation groups exact duplicates, keeps test-threshold selection separate from validation, "
        "and records source-level audit information. Near-duplicate and patient-level leakage remain unresolved without "
        "additional metadata."
    )
    add_paragraph(document, abstract, size=10, after=5)
    add_paragraph(document, "Keywords", size=12, bold=True, alignment=WD_ALIGN_PARAGRAPH.LEFT, after=1, keep=True)
    add_paragraph(document, "Weak supervision, bone fracture, X-ray imaging, class activation maps, pseudo-masks",
                  size=10, after=8)
    add_paragraph(document, "Protocol and status draft. Not ready for submission until the marked experiments and author details are completed.",
                  size=9, italic=True, alignment=WD_ALIGN_PARAGRAPH.CENTER, after=8)

    add_heading(document, "1. Introduction")
    add_paragraph(document,
        "Pixel-level fracture annotation requires expert review and is less abundant than image-level fracture labels. "
        "Weakly supervised learning therefore offers a practical research question: whether image-level supervision can "
        "produce useful localization maps and pseudo-masks for downstream segmentation. Public musculoskeletal radiograph "
        "collections such as FracAtlas provide image labels, boxes, and expert masks, while GRAZPEDWRI-DX provides a large "
        "annotated pediatric wrist dataset (Abedeen et al. 2023; Nagy et al. 2022). Their different populations and label "
        "structures make source-aware evaluation essential.")
    add_paragraph(document,
        "CAM methods indicate image regions that contribute to a classifier's output, but their spatial resolution and "
        "discriminative nature can omit parts of a target (Zhou et al. 2016; Selvaraju et al. 2017). Grad-CAM++, LayerCAM, "
        "and more recent CAM refinement methods address aspects of localization quality, but do not turn image-level labels "
        "into expert segmentation annotations (Chattopadhyay et al. 2018; Jiang et al. 2021; Lei et al. 2023; Fu et al. 2025). "
        "This distinction is especially important for thin fracture lines, where apparent overlap can depend strongly on "
        "mask definition and threshold selection.")
    add_paragraph(document,
        "The repository at the time of this audit contains one labeled image-folder source and no expert masks or patient "
        "identifiers. The experiment described here is consequently a protocol and implementation report. Numerical results "
        "are intentionally withheld until the required datasets and evaluations are complete.")

    add_heading(document, "1.1 Objectives")
    add_paragraph(document,
        "The study will (1) compare ResNet-50 and DenseNet-121 fracture classifiers; (2) compare Grad-CAM, Grad-CAM++, and "
        "LayerCAM pseudo-masks using validation-selected thresholds; (3) measure localization against expert masks on a "
        "held-out FracAtlas split; (4) compare pseudo-mask-trained U-Net results with real-mask and trivial baselines; and "
        "(5) assess data leakage and image-statistic shortcuts. These objectives remain pending empirical completion.")

    add_heading(document, "2. Literature Review")
    add_paragraph(document,
        "Class activation mapping uses feature-map evidence to localize discriminative image regions from image-level "
        "classification labels (Zhou et al. 2016). Grad-CAM generalizes this idea using gradients at a convolutional layer "
        "(Selvaraju et al. 2017), and Grad-CAM++ changes gradient weighting to improve explanations for multiple instances "
        "(Chattopadhyay et al. 2018). LayerCAM uses positive gradients at convolutional layers to provide hierarchical "
        "localization maps (Jiang et al. 2021). These maps can be thresholded into seeds, but CAM-only weak supervision is "
        "known to produce incomplete regions. AffinityNet propagates local responses using learned pixel affinities (Ahn and "
        "Kwak 2018); DCAM and UM-CAM propose later CAM refinement strategies (Lei et al. 2023; Fu et al. 2025). The current "
        "implementation evaluates the three gradient-based CAM methods and simple thresholding; it does not implement these "
        "learned refinement approaches.")
    add_paragraph(document,
        "ResNet and DenseNet are established image-classification backbones (He et al. 2016; Huang et al. 2017), while U-Net "
        "is a common encoder-decoder architecture for biomedical segmentation (Ronneberger et al. 2015). FracAtlas supplies "
        "a relevant fracture dataset with segmentation annotations (Abedeen et al. 2023); GRAZPEDWRI-DX supplies pediatric "
        "wrist images and expert region labels (Nagy et al. 2022). These sources should not be pooled without reporting "
        "population, anatomy, annotation, and split differences.")
    add_paragraph(document,
        "Shortcut learning describes models that exploit predictive but non-causal cues and fail under changed conditions "
        "(Geirhos et al. 2020). For radiographs, source, acquisition, and image processing artifacts can correlate with labels. "
        "The planned low-level image-statistics baseline is a warning screen, not proof that a model is unbiased. Exact duplicate "
        "hashes reduce one form of leakage, but they cannot replace patient or study grouping. Threshold selection follows "
        "validation-only practice; Otsu thresholding is included as an unsupervised comparator (Otsu 1979). Confidence "
        "intervals will be estimated by bootstrapping independent hash groups rather than individual duplicate files (Efron 1979).")

    add_heading(document, "3. Methods")
    add_heading(document, "3.1 Data inventory and provenance", level=2)
    add_paragraph(document,
        "The current data root is Datasets/BONE_FRACTURE_DATASET/cropped_dataset, organized into train, val, and test folders "
        "with fractured and not-fractured labels. The generated manifest contains 1,927 image files and 1,897 unique SHA-256 "
        "hashes. It records 936 train images, 600 validation images, and 391 test images. No candidate pixel masks were found. "
        "The source citation, collection details, license, and patient grouping metadata were not included in the checkout and "
        "must be added before publication.")
    add_table(document,
        ["Repository source", "Images", "Unique hashes", "Masks", "Status"],
        inventory_rows(), [2.25, 0.8, 0.85, 0.65, 1.95])
    add_paragraph(document,
        "The guide additionally proposes FracAtlas, two Kaggle fracture datasets, and up to 4,000 GRAZPEDWRI-DX wrist images. "
        "Those sources are not present in the checked-in dataset directory and are not counted in the inventory above. "
        "FracAtlas should supply the expert-mask evaluation subset after its official annotations are converted and audited. "
        "Mask matching in the current preparation script is filename-based; no segmentation metrics should be reported until "
        "every image-mask pair is verified.")

    add_heading(document, "3.2 Leakage-aware preparation", level=2)
    add_paragraph(document,
        "The preparation script inventories image files in class-named directories, computes SHA-256 hashes, preserves an "
        "existing train/validation/test assignment when every exact-duplicate group agrees, and assigns a deterministic "
        "grouped split if it does not. Exact duplicates are kept together and collapsed to one sample for model training. "
        "The current audit found zero exact-hash groups crossing splits and zero duplicate groups with conflicting labels. "
        "Patient or study identifiers are absent, so exact-hash grouping does not rule out near-duplicate or patient-level "
        "leakage. All counts and exclusion rules should be rerun after adding sources.")

    add_heading(document, "3.3 Classifier training")
    add_paragraph(document,
        "The implemented classifier candidates are ResNet-50 and DenseNet-121 initialized from ImageNet weights, with a new "
        "two-class output head. Grayscale radiographs are replicated to three channels, resized to 448 by 448 pixels, and "
        "normalized with ImageNet channel statistics. Training uses class-balanced sampling and AdamW. The classifier head "
        "is warmed up for two epochs before fine-tuning the full model. The default run is 20 epochs, with the checkpoint "
        "selected by validation AUC when both validation classes are present, otherwise by validation F1. The test split is "
        "not used for checkpoint selection. Three independent seeds are required for reported comparisons.")

    add_heading(document, "3.4 CAM generation and pseudo-mask construction")
    add_paragraph(document,
        "The final convolutional block is used to compute Grad-CAM, Grad-CAM++, or LayerCAM for the fracture logit. Optional "
        "horizontal-flip test-time averaging combines the original map with the spatially reversed map. CAMs are normalized "
        "per image and saved with a manifest index. Global thresholds are swept on validation masks only; Otsu and percentile "
        "thresholds are available as alternatives. Binary masks receive 3 by 3 morphological opening and closing. CAMs are "
        "classifier evidence maps, so their thresholded outputs are treated as noisy pseudo-labels rather than anatomical "
        "ground truth.")

    add_heading(document, "3.5 Metrics and uncertainty")
    add_paragraph(document,
        "Classification reporting will include fracture-class AUC, accuracy, precision, recall, and F1 by source. Localization "
        "reporting will include Dice, IoU, precision, recall, and pointing-game accuracy on expert masks. Dice and F1 are "
        "algebraically equivalent for binary pixel labels; only Dice will be used for mask overlap. Empty-mask conventions "
        "are fixed in the implementation and must be reported with results. Confidence intervals will use 2,000 bootstrap "
        "resamples over exact-hash groups. Because patient IDs are unavailable, intervals will not represent patient-level "
        "sampling uncertainty. Thresholds will be selected on validation only and applied once to the held-out test split.")

    add_heading(document, "3.6 Segmentation baselines")
    add_paragraph(document,
        "A compact U-Net is implemented for paired binary masks. It can be trained on pseudo-masks or supplied expert masks, "
        "but its validation score is agreement with the supplied labels and is not an expert-mask score when pseudo-labels "
        "are used. Planned comparisons are a pseudo-mask-trained U-Net, a real-mask-trained upper-bound model, an empty-mask "
        "baseline, a central-ellipse baseline, and a deterministic random-mask baseline. Dice loss follows the common "
        "soft-overlap formulation used in medical image segmentation (Milletari et al., 2016). These comparisons cannot be run on "
        "the current checkout because it contains no expert masks.")

    add_heading(document, "4. Results and Discussion")
    add_paragraph(document,
        "No model training or expert-mask evaluation has been run in this workspace. The available Python runtime does not "
        "contain PyTorch, torchvision, scikit-learn, OpenCV, or matplotlib, and the repository contains no pixel masks. "
        "Table 2 is a completion ledger, not a results table. Values must be populated from saved outputs after the required "
        "experiments; estimated or illustrative values must not be substituted.")
    add_table(document,
        ["Analysis", "Current status", "Required evidence"],
        [
            ("Classifier AUC by source and seed", "Not run", "Trained checkpoints and held-out predictions"),
            ("CAM Dice/IoU and pointing game", "Not available", "Verified expert masks; validation and test outputs"),
            ("Threshold and backbone ablations", "Not run", "Three seeds; validation-selected threshold"),
            ("Pseudo-mask vs real-mask U-Net", "Not available", "Expert-mask train/validation/test partitions"),
            ("Shortcut and trivial baselines", "Not run", "Audit CSV and expert-mask baseline CSV"),
        ], [2.0, 1.25, 3.25])
    if FIGURE.exists():
        image_paragraph = document.add_paragraph()
        image_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        image_paragraph.add_run().add_picture(str(FIGURE), width=Inches(5.9))
        add_paragraph(document,
            "Figure 1: Image counts by existing split and class in the repository dataset; these are inventory counts, not model results.",
            size=9, alignment=WD_ALIGN_PARAGRAPH.CENTER, after=5)
    add_paragraph(document,
        "No classifier or segmentation model was run for this paper. ResNet-50 and DenseNet-121 are implemented as candidate "
        "classifiers, and a compact U-Net is implemented as a segmentation baseline, but the required PyTorch runtime and "
        "expert masks were unavailable. The plotted values describe only the indexed image files.")

    add_heading(document, "5. Discussion")
    add_paragraph(document,
        "The present repository supports a leakage-aware classification and localization workflow, but it does not yet support "
        "an empirical claim about fracture segmentation quality. The main contribution at this stage is a reproducible "
        "protocol that separates validation threshold selection from final test evaluation and makes the data limitations "
        "visible. A completed study should compare CAM methods within the same source and split, report source-specific "
        "classification behavior, and show successful and failed localization overlays. It should also compare the learned "
        "maps with simple baselines and investigate whether image-level acquisition cues predict labels.")
    add_paragraph(document,
        "The low-resolution nature of CAMs may be poorly matched to narrow fracture structures. Mask overlap metrics should "
        "therefore be interpreted alongside pointing-game accuracy and representative overlays. Any use of the pseudo-mask "
        "trained U-Net must be evaluated against expert annotations on held-out images; agreement with its own pseudo-labels "
        "cannot establish clinical boundary accuracy. Results from different anatomical regions or sources should not be "
        "pooled without stratification.")

    add_heading(document, "6. Limitations")
    add_paragraph(document,
        "The present dataset inventory contains a single source with undocumented provenance and no expert pixel masks. The "
        "source labels, image quality, class definitions, licensing, and collection protocol require verification. Exact hashes "
        "do not detect transformed near duplicates or repeated patients. The current code does not parse COCO polygons, use "
        "GRAZPEDWRI-DX bounding boxes, perform MedSAM refinement, train a pretrained ResNet-34 U-Net, or provide external "
        "validation. The runtime used for this work lacks the deep-learning dependencies needed to train the models. Therefore, "
        "no accuracy, localization, segmentation, robustness, or clinical utility claim is supported at this time.")

    add_heading(document, "7. Conclusion")
    add_paragraph(document,
        "This manuscript defines a weakly supervised fracture localization pipeline and reports the exact data status of the "
        "current repository. The audit found 1,927 class-labeled images, 1,897 unique exact-image hashes, and no paired expert "
        "masks. The empirical study remains incomplete. Publication claims should be added only after the intended sources are "
        "documented, expert-mask pairs are validated, multi-seed experiments and baselines are run, and the held-out test set "
        "is evaluated once using thresholds selected on validation data.")

    add_heading(document, "Acknowledgements")
    add_paragraph(document, "[Add only verified funding, facilities, or contributor acknowledgements.]", alignment=WD_ALIGN_PARAGRAPH.LEFT)

    add_heading(document, "References")
    references = [
        "Abedeen, I., Rahman, M. A., Prottyasha, F. Z., Ahmed, T., Chowdhury, T. M., and Shatabda, S., FracAtlas: A Dataset for Fracture Classification, Localization and Segmentation of Musculoskeletal Radiographs, Scientific Data, vol. 10, 521, 2023. https://doi.org/10.1038/s41597-023-02432-4.",
        "Ahn, J., and Kwak, S., Learning Pixel-Level Semantic Affinity With Image-Level Supervision for Weakly Supervised Semantic Segmentation, Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 4981-4990, 2018. https://doi.org/10.1109/CVPR.2018.00523.",
        "Chattopadhyay, A., Sarkar, A., Howlader, P., and Balasubramanian, V. N., Grad-CAM++: Generalized Gradient-Based Visual Explanations for Deep Convolutional Networks, Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 839-847, 2018. https://doi.org/10.1109/WACV.2018.00097.",
        "Efron, B., Bootstrap Methods: Another Look at the Jackknife, The Annals of Statistics, vol. 7, no. 1, pp. 1-26, 1979. https://doi.org/10.1214/aos/1176344552.",
        "Fu, J., Wang, G., Lu, T., Yue, Q., Vercauteren, T., Ourselin, S., and Zhang, S., UM-CAM: Uncertainty-Weighted Multi-Resolution Class Activation Maps for Weakly-Supervised Segmentation, Pattern Recognition, vol. 160, 111204, 2025. https://doi.org/10.1016/j.patcog.2024.111204.",
        "Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., and Wichmann, F. A., Shortcut Learning in Deep Neural Networks, Nature Machine Intelligence, vol. 2, pp. 665-673, 2020. https://doi.org/10.1038/s42256-020-00257-z.",
        "He, K., Zhang, X., Ren, S., and Sun, J., Deep Residual Learning for Image Recognition, Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 770-778, 2016. https://doi.org/10.1109/CVPR.2016.90.",
        "Huang, G., Liu, Z., van der Maaten, L., and Weinberger, K. Q., Densely Connected Convolutional Networks, Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 4700-4708, 2017.",
        "Jiang, P.-T., Zhang, C.-B., Hou, Q., Cheng, M.-M., and Wei, Y., LayerCAM: Exploring Hierarchical Class Activation Maps for Localization, IEEE Transactions on Image Processing, vol. 30, pp. 5875-5888, 2021. https://doi.org/10.1109/TIP.2021.3089943.",
        "Lei, J., Yang, G., Wang, S., Feng, Z., and Liang, R., DCAM: Disturbed Class Activation Maps for Weakly Supervised Semantic Segmentation, Journal of Visual Communication and Image Representation, vol. 94, 103852, 2023. https://doi.org/10.1016/j.jvcir.2023.103852.",
        "Milletari, F., Navab, N., and Ahmadi, S.-A., V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation, Proceedings of the Fourth International Conference on 3D Vision, pp. 565-571, 2016. https://doi.org/10.1109/3DV.2016.79.",
        "Nagy, E., Janisch, M., Hržić, F., Sorantin, E., and Tschauner, S., A Pediatric Wrist Trauma X-Ray Dataset (GRAZPEDWRI-DX) for Machine Learning, Scientific Data, vol. 9, 222, 2022. https://doi.org/10.1038/s41597-022-01328-z.",
        "Otsu, N., A Threshold Selection Method from Gray-Level Histograms, IEEE Transactions on Systems, Man, and Cybernetics, vol. 9, no. 1, pp. 62-66, 1979. https://doi.org/10.1109/TSMC.1979.4310076.",
        "Ronneberger, O., Fischer, P., and Brox, T., U-Net: Convolutional Networks for Biomedical Image Segmentation, Medical Image Computing and Computer-Assisted Intervention, Lecture Notes in Computer Science, vol. 9351, pp. 234-241, 2015. https://doi.org/10.1007/978-3-319-24574-4_28.",
        "Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., and Batra, D., Grad-CAM: Visual Explanations From Deep Networks via Gradient-Based Localization, Proceedings of the IEEE International Conference on Computer Vision, pp. 618-626, 2017. https://doi.org/10.1109/ICCV.2017.74.",
        "Zhou, B., Khosla, A., Lapedriza, A., Oliva, A., and Torralba, A., Learning Deep Features for Discriminative Localization, Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 2921-2929, 2016.",
    ]
    for reference in references:
        add_paragraph(document, reference, size=9, alignment=WD_ALIGN_PARAGRAPH.LEFT, after=4, hanging=0.25)

    add_heading(document, "Biographies")
    add_paragraph(document,
        "Shibli Sanjid Faheem is an undergraduate student in Computer Science and Engineering with a minor in Management Information Systems at Independent University, Bangladesh.",
        alignment=WD_ALIGN_PARAGRAPH.LEFT)
    add_paragraph(document,
        "Yousuf Abdullah is a co-author of this study and is affiliated with the Department of Computer Science and Engineering, Independent University, Bangladesh.",
        alignment=WD_ALIGN_PARAGRAPH.LEFT)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document.save(str(OUTPUT))
    print(f"Created {OUTPUT}")


if __name__ == "__main__":
    build()

