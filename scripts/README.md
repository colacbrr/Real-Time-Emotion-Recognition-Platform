# Scripts Guide

For the new educational comparison use `evaluation.py`: shared manifest, exact-duplicate/identity checks, majority baseline, HOG-SVM, frozen ResNet18 transfer learning, common predictions and metrics. See [the evaluation protocol](../docs/ACCURACY.md). The scripts listed below are historical workflows; their different splits and metadata comparisons do not establish a common-test comparison.

The `scripts/` directory contains the model-training and evaluation workflows shared with the core emotion-prediction repository.

## Classical Models

- `train_model.py`: train one registry-defined feature model such as `hog_svm` or `lbp_svm`
- `validate_model.py`: validate a saved classical model on the labeled validation split
- `compare_models.py`: train and compare all registry models on the validation split
- `hog_model_search.py`: search HOG parameter combinations

## CNN Models

- `cnn_train.py`: train a CNN checkpoint such as `resnet18`, `resnet50`, `efficientnet_b0`, or `small_cnn`
- `cnn_validate.py`: evaluate a saved CNN checkpoint and write a validation report
- `cnn_compare_models.py`: compare stored CNN checkpoint metadata
- `cnn_make_submission.py`: generate `submission.csv` from one or more CNN checkpoints

## Data Preparation

- `split_dataset.py`: create dataset splits
- `rebalance_split.py`: rebalance split distributions when needed

## Submission and Prediction

- `make_submission.py`: generate a submission CSV from the best classical model
- `predict_submission.py`: run prediction-oriented workflows against the unlabeled split

## Model Metadata

- `model_registry.py`: load and query the `models/registry.json` configuration

## Recovered academic assets

- `audit_recovery.py`: read-only ZIP/image/label audit, historical manifest and similarity candidates.
- `evaluate_recovered.py`: recovered HOG plus majority baseline on historical validation; explicitly exploratory, no fitting.
- `split_protocol.py`: optional unofficial nested stratified group split; preserves exact duplicate/person groups and refuses existing assignments. Use only after source/protocol review.
- `evaluation.py evaluate-saved`: evaluate the selected validation artifact without refitting; verifies manifest, implementation and artifact hashes.

## Duplicate review

- `review_candidates.py`: verifies recovered images, renders class-label-free contact sheets, computes SSIM/RMSE and limited-alignment NCC; does not assign review decisions or infer identities.
- `duplicate_policy.py`: checks complete annotations against evidence, builds conservative image groups and proposes quarantine/exclusions while preserving all original labels and splits.
- `evaluation.py` and `split_protocol.py` preserve optional `duplicate_group` annotations and prevent cross-split leakage. See [review decisions and limitations](../docs/publicare/duplicate_review.md).

- `verify_duplicate_review.py`: verifies recorded hashes, unchanged recovered source files and byte-identical regeneration of the proposal; checks integrity rather than performing a second visual review.

## Derived internal protocol

- `build_internal_inventory.py`: verifies the recorded recovery/review, applies reviewed selection to CSVs only, preserves all IDs and original labels, and refuses eligible/held-out hash or image-group overlap.
- `recover_resnet18_weights.py`: explicit recovery of the official Torchvision IMAGENET1K_V1 artifact, with publisher filename hash-prefix and strict architecture checks; training scripts still do not download.
- The frozen retrospective protocol and cost pilot are described in [the derived inventory](../docs/publicare/inventar_derivat.md).

- `export_internal_results.py`: verifies frozen configuration, saved-artifact provenance and metrics recalculated from all three test exports; writes the common JSON/Markdown report, per-class recall and CSV confusion matrices.

- `verify_internal_study.py`: checks the entire recorded v1 chain without fitting or network access; byte-identical regeneration of derived inventories and numeric reports, source/ID conservation and frozen-protocol checks.

## FER2013 source investigation

- `fetch_fer2013_reference.py`: explicit public Kaggle v1 reference/metadata download, without authentication bypass; saved checksums and provenance.
- `match_fer2013_reference.py`: offline byte/pixel hash matching of recovered images against the reference archive; retains the distinction between uploader license declaration and independently verified image rights, and withholds official predict reference classes/paths.
- [Evidence and license sources](../docs/publicare/sursa_fer2013/raport.md). The frozen study records are preserved as historical snapshots; source investigation is a separate addendum.

## Extensie pentru articolul de reproducibilitate

`scientific_study.py` implementează protocolul v2 separat de codul v1 înghețat: `plan`, `pilot`, `run`, `verify`. `verify_scientific_artifacts.py` verifică foldurile și reproduce predicțiile finale din artefacte fără fit. `export_scientific_study.py` generează tabele și figuri din rezultate existente, cu Python/Matplotlib. Vezi [comenzile complete](../docs/publicare/studiu_v2/README.md) și [manuscrisul](../docs/publicare/articol/README.md).

`verify_published_results.py` este verificarea portabilă a dovezilor numerice, fără imagini sau weights: hash-uri, subseturi, predicții, metrici și bootstrap. Dependințele sunt în `requirements/verify.txt`.

`stop_local_app.py` oprește numai serverele de dezvoltare din directoarele acestui repository, pe Linux. Opțiunea `--dry-run` listează PID-urile fără a trimite semnale.
