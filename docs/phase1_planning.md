# Phase 1: Research and Planning

Status: **complete. Task definition (Section 5) confirmed by the user.**
Nothing in this document is an experimental result. All performance statements
are proposed work until Phase 6 produces measured numbers.

---

## 1. Problem statement

Labelling chest X-rays takes radiologist time, so labelled sets are small while
unlabelled images are plentiful. This project asks whether a semi-supervised
method based on Deep Embedded Clustering (SSLDEC) can use a large pool of
unlabelled chest X-rays to classify a radiographic finding better than a model
trained only on the small labelled subset, under a patient-level evaluation
protocol.

## 2. Objectives

1. Reproduce the core SSLDEC mechanism (Student's t soft assignment, sharpened
   target distribution, KL loss, labelled anchoring of clusters).
2. Adapt it to an ImageNet-pretrained CNN on chest X-rays, documenting every
   change from the original paper.
3. Compare three methods on one fixed, patient-level test set:
   supervised-only, basic pseudo-labelling, SSLDEC.
4. Measure how the gap changes as the labelled fraction varies.
5. Serve the trained model through a FastAPI + React research prototype that
   clearly labels outputs as non-diagnostic.
6. Optional: semi-supervised lung-field segmentation on a dataset with
   pixel-level masks.

### Out of scope

- Cancer diagnosis. No dataset used here has pathology-confirmed cancer labels.
- Clinical deployment, treatment advice, or regulatory claims.
- Lesion-level segmentation (no lesion masks available; see Section 6).

## 3. The paper: what it actually does

Enguehard, O'Halloran, Gholipour. "Semi-Supervised Learning With Deep Embedded
Clustering for Image Classification and Segmentation." *IEEE Access* 7
(2019): 11093-11104. doi:10.1109/ACCESS.2019.2891970. PMC6777718.

### 3.1 Background: unsupervised DEC (Xie et al., 2016)

An encoder maps an image x_i to an embedding z_i. There are K learnable cluster
centres mu_1..mu_K in embedding space.

**Soft assignment** (Student's t kernel, alpha = degrees of freedom; alpha = 1
in the paper):

    q_ij = (1 + ||z_i - mu_j||^2 / alpha)^(-(alpha+1)/2)
           / sum_j' (1 + ||z_i - mu_j'||^2 / alpha)^(-(alpha+1)/2)

q_ij is the probability that sample i belongs to cluster j. Heavy tails make
the assignment tolerant of points far from every centre.

**Target distribution** (sharpened, frequency-normalised):

    f_j  = sum_i q_ij                         (soft size of cluster j)
    p_ij = (q_ij^2 / f_j) / sum_j' (q_ij'^2 / f_j')

Squaring q pushes confident assignments closer to 1. Dividing by f_j stops
large clusters from absorbing everything.

**Loss:**

    L_KL = KL(P || Q) = sum_i sum_j p_ij log(p_ij / q_ij)

P is treated as a fixed target (no gradient). Minimising L_KL moves both the
embeddings z_i and the centres mu_j so that confident assignments become more
confident: self-training in embedding space.

Unsupervised DEC has no notion of classes. Cluster 3 is just "cluster 3", and
it may group images by patient position, device, or scanner instead of by
disease.

### 3.2 SSLDEC, the paper's semi-supervised version

- **One cluster per class (K = number of classes).** Labelled samples are
  trained so that q_i matches their known class, which ties cluster j to
  class j. This is what makes cluster IDs meaningful. The labels are anchors,
  not an after-the-fact mapping.
- **Labelled loss:** the paper says it is a KL divergence between q and the
  empirical (one-hot) label distribution, but it gives no equation. For a
  one-hot target this equals cross-entropy on q: `L_sup = -log q_{i, y_i}`.
- **Algorithm (from the text):**
  1. Pretrain on labelled data only for 100 epochs.
  2. Repeat: predict q for all data, compute P, select a class-balanced
     subset, and train on it with P as the target.
  3. After each unlabelled epoch, retrain on labelled data until training
     accuracy reaches 100%.
- **Settings:** alpha = 1, Adam, batch 128, learning rate 1e-3 to 1e-4,
  5-model ensemble, trained from scratch.
- **Experiments:** half-moons (toy), MNIST (100/1000 labels), SVHN (500/1000
  labels), and iSeg2017 infant brain MRI tissue segmentation (CSF/GM/WM) with a
  3D dense FCN applying the clustering layer per voxel. The paper reports the
  numbers it reports; we do not reuse them as expectations for chest X-rays.

### 3.3 Gaps in the paper that we must resolve ourselves

| Gap in the paper | Our decision |
|---|---|
| Centre initialisation is inconsistent (random vs. "pretraining initialises clusters") | After supervised warm-up, set mu_j to the mean embedding of labelled class-j samples |
| Labelled loss has no equation | `L_sup = CE(q_i, y_i)`, i.e. KL(one-hot ‖ q) |
| Stop when labelled accuracy hits 100% | Stop on validation metric. A pretrained DenseNet reaches 100% train accuracy quickly and would overfit; the paper itself warns about this. |
| Class-balanced subset selection mechanism not specified | Confidence threshold on p_ij, then cap pseudo-labelled samples per class |

## 4. Mismatch: paper experiments vs. chest X-ray "cancer" classification

1. **No cancer ground truth.** ChestX-ray14 labels such as "Mass" and
   "Nodule" describe radiographic findings mined from report text. They are
   not biopsy-confirmed malignancy. Most nodules are benign. The project
   therefore classifies a *suspicious focal opacity*, never "cancer".
2. **Label noise.** The labels come from NLP on reports (the dataset card
   estimates >90% accuracy, which still means many wrong labels). MNIST and
   SVHN labels are essentially clean. Noisy anchors can pull clusters the
   wrong way.
3. **Class imbalance.** MNIST/SVHN are roughly balanced. Nodule/mass images
   are a minority. DEC's f_j normalisation actively pushes clusters towards
   equal size, which on an imbalanced pool biases predictions towards the
   minority class (more false positives). We will test a prior-corrected
   target that uses class priors estimated from the labelled set.
4. **Weak cluster structure.** Digits form tight clusters. A small nodule is a
   few pixels of a 1024x1024 image. Embeddings are dominated by anatomy,
   positioning, devices, and hospital, so the cluster assumption may hold
   poorly. This is a real risk that the experiments may confirm.
5. **Pretrained vs. from scratch.** The paper trains small CNNs from scratch.
   We use ImageNet-pretrained DenseNet121 with a projection head, which changes
   the training dynamics (strong supervised baseline, fast overfitting).
6. **Multi-label source data.** NIH images can carry several findings. The
   paper's method is single-label (softmax over clusters). We define a
   single-label binary subset (Section 5) and exclude ambiguous images.
7. **Segmentation.** The paper had voxel-wise labels. ChestX-ray14 provides
   only bounding boxes for a small subset (about 1k images), not masks, so
   lesion segmentation cannot be trained or evaluated honestly (Section 6).
8. **Compute.** A 5-model ensemble at 1024 px is out of reach on a 6 GB GPU.
   We train single models at 224 px (ensembling optional).

## 5. Dataset and task (proposed; confirm before Phase 3)

**Dataset:** NIH ChestX-ray14. 112,120 frontal chest X-rays (PNG, 1024x1024)
of 30,805 patients, 14 report-mined finding labels plus "No Finding",
metadata CSV with patient ID, age, sex, and view position. The dataset ships
an official patient-disjoint train/val vs. test list.

- Official access: NIH Clinical Center Box folder
  `https://nihcc.app.box.com/v/ChestXray-NIHCC`, which includes the README,
  `Data_Entry_2017.csv`, `BBox_List_2017.csv`, `train_val_list.txt`,
  `test_list.txt`, and 12 image archives (about 45 GB in total).
- **To verify in Phase 3 (do not skip):** read the official README in that
  folder for the usage terms and required citation. Secondary sources
  describe it as having no use restrictions / CC0, but we cite the README
  itself.
- Citation: Wang X., Peng Y., Lu L., Lu Z., Bagheri M., Summers R.M.
  "ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on
  Weakly-Supervised Classification and Localization of Common Thorax
  Diseases." CVPR 2017.

**Proposed task: binary, single-label.**

| Class | Definition |
|---|---|
| 0 `no_finding` | label string is exactly "No Finding" |
| 1 `nodule_or_mass` | label contains "Nodule" or "Mass" (other findings allowed alongside) |
| excluded | any other finding without Nodule/Mass |

Reasoning: it is the dataset label closest to the cancer-related motivation
while remaining honest (a finding, not a diagnosis), it fits the paper's
single-label clustering formulation, and it is a hard, realistic small-lesion
problem.

Alternative if results are too weak to compare methods meaningfully:
`no_finding` vs `any_finding` (larger positive class, easier signal).

**Semi-supervised simulation.** We do have labels for all images, but we hide
them for the unlabelled pool. This is the standard protocol: it lets us measure
pseudo-label accuracy, which a real unlabelled set would not allow. Starting
point: 5% of training patients labelled and the rest unlabelled. Later
experiments sweep 1/5/10/25%, all on the same fixed test set.

**Leakage control.** All splits are made by patient ID, and a patient's
images never cross splits. The test set is the official `test_list.txt` (or
a patient-disjoint subset of it) and is never used for tuning.

## 6. Segmentation plan (optional extension)

ChestX-ray14 has no pixel masks. The honest options are:

- **Lung-field segmentation** on public datasets with radiologist-supervised
  lung masks: Montgomery County (138 images; masks by Candemir et al. 2014)
  and Shenzhen (lung masks available through derived collections). U-Net,
  with a semi-supervised extension (labelled Dice + CE loss, unlabelled
  consistency or per-pixel DEC loss as in the paper). Metrics: Dice and IoU.
- **Lesion segmentation:** future work. It needs pixel-annotated lesion data
  (e.g., a nodule dataset with masks), not image-level labels or boxes.

Official access and mask licensing for these datasets get verified when the
extension starts.

## 7. Proposed architecture

```
                           ┌───────────────────────────────┐
 image (1x224x224 → 3ch) → │ DenseNet121 (ImageNet init)   │ → 1024-d pooled feature
                           └───────────────────────────────┘
                                          │
                              projection head (MLP) → z ∈ R^64
                                          │
                       clustering layer: centres mu_1..mu_K (K = 2)
                                          │
                               q_i (Student's t, alpha=1)
                    ┌─────────────────────┴─────────────────────┐
           labelled: L_sup = CE(q_i, y_i)       unlabelled: L_DEC = KL(P||Q)
                                                (P refreshed periodically,
                                                 confidence-masked, class-capped)

    L = L_sup + lambda(t) * L_DEC     lambda ramps up from 0 after warm-up

 Baselines: (1) same backbone + linear head, labelled only
            (2) same backbone + pseudo-labelling (threshold on softmax)
```

System view:

```
React (Vite) ──HTTP multipart──▶ FastAPI ──▶ predictor service ──▶ PyTorch model (artifacts/)
      ▲                              │  validation, same preprocessing as training
      └──────── JSON result ◀────────┘  (class, model score, disclaimer, model version)
```

### UI quality requirements (Phase 7)

The user has asked for the best UI they can get, so the frontend is a
first-class deliverable, not an afterthought:

- Clean clinical-research look: a neutral palette, one accent colour, strong
  typography, light and dark themes.
- Drag-and-drop upload with an instant preview, file-type/size feedback, and
  a loading state during inference.
- Results card: predicted category, the model score shown with a visual bar
  and an explicit "not a calibrated probability / not a diagnosis" note, the
  model version, and the inference time.
- Optional heatmap overlay (Grad-CAM) toggle on the image, labelled as a
  model-attention visualisation, not a lesion location.
- A model card page: dataset, task, metrics from real Phase 6 runs, and
  limitations.
- Responsive from phone to desktop; accessible (keyboard, contrast, alt text).
- Clear, human error messages for invalid, corrupt, or too-large files and
  for backend-unavailable states.

## 8. Roadmap

| Phase | Output | Verification |
|---|---|---|
| 1 Planning | this document, task decision | task confirmed |
| 2 Environment | venv, deps, `scripts/check_env.py` passes | GPU forward pass succeeds |
| 3 Data | download, validation, resized copies, patient-level splits, labelled/unlabelled CSVs | split leakage test passes, class counts reported |
| 4 Baseline | supervised DenseNet121, checkpoint, test metrics | metrics computed once on test |
| 5 SSLDEC | DEC layer, losses, training loop, logs | unit tests on q/P/KL; training curves |
| 6 Comparison | pseudo-labelling baseline, label-fraction sweep, tables/plots from logs | all numbers traceable to run files |
| 7 Web app | FastAPI + React, API contract | API tests, manual upload test |
| 8 Test/deploy | pytest suite, Docker, optional EC2 | `docker compose up` works locally |
| Seg (opt.) | U-Net + SSL on lung masks | Dice/IoU on held-out patients |
| Report | academic write-up, slides, viva Q&A | experiments labelled real vs. proposed |
