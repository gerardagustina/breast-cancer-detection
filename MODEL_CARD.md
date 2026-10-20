# Model Card: Breast Ultrasound Malignancy Classifier (v1.0)

## Model details
- **Developer:** Gerard Agustina (portfolio project)
- **Version:** v1.0 (October 2026). Weights available in the [v1.0 release](https://github.com/gerardagustina/breast-cancer-detection/releases/tag/v1.0)
- **Architecture:** DenseNet121 (torchvision), initialised with ImageNet weights; classification head replaced by a single output (logit)
- **Input:** one grayscale breast ultrasound image, 224×224, intensities scaled to [0, 1]
- **Output:** probability of malignancy (sigmoid of the logit)
- **Training:** Adam (learning rate 1e-4), batch size 32, binary cross-entropy with class weight 2.71 for malignant cases, data augmentation (horizontal flip, ±15° rotation, ±10% zoom), early stopping on validation AUC (best epoch: 8). Seed 42; results verified to be reproducible.

## Intended use
- **Educational and research demonstration** of a medical image classification workflow.
- **Not intended for clinical use.** It has not been clinically validated or approved by any regulatory body.

## Out-of-scope uses
- Diagnosis, screening or any clinical decision-making.
- Other imaging modalities (e.g., mammography, MRI).
- Images from devices, hospitals or populations different from the training data without prior validation.

## Training and evaluation data
- **BreastMNIST** (MedMNIST v2), derived from the **BUSI** dataset: 780 breast ultrasound images from a single hospital.
- Splits: 546 train / 78 validation / 156 test, stratified, 27% malignant in each split.
- Labels: malignant vs. normal/benign (normal and benign are merged into one class).
- No patient identifiers are available, so a patient-level split could not be verified.

## Evaluation results (test set, n = 156, 42 malignant)

| Metric | Threshold 0.5 | Clinical threshold 0.12 |
|---|---|---|
| AUC | 0.937 | 0.937 |
| Sensitivity | 78.6% (33/42) | 92.9% (39/42), 95% CI ≈ 81–98% |
| Specificity | 92.1% (105/114) | 81.6% (93/114) |
| PPV | 78.6% | 65.0% |
| NPV | 92.1% | 96.9% |

The clinical threshold was selected on the validation set to reach 100% validation sensitivity, and then applied once to the test set. PPV and NPV depend on prevalence (27% here, much lower in real screening populations).

## Explainability
Grad-CAM maps highlight the lesion in correctly detected cancers. The 3 missed cancers involved: no clearly visible lesion, a malignant lesion with benign-looking features, and a lesion at the image border.

## Risks and limitations
- **False negatives:** missed cancers may falsely reassure patients and delay diagnosis. At the clinical threshold, 3 of 42 test cancers were missed, with high confidence.
- **Small dataset and wide uncertainty:** the clinical threshold was unstable across random seeds (0.12–0.40), and test sensitivity has a wide confidence interval.
- **Single-centre data, no external validation:** performance on other populations or ultrasound devices is unknown.
- **No subgroup analysis:** performance by age, breast density or lesion type could not be assessed.
- **Uncalibrated probabilities:** outputs should not be interpreted as true risk estimates.
- **Possible centre bias:** lesions at the image border may be under-detected.

## Recommendations
Before any real-world use: external validation on multi-centre data, probability calibration, cross-validated threshold selection and evaluation by clinical experts.