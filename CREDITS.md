# ASTRA Dataset Credits

ASTRA combines original examples written for this project with selected examples from public datasets. Dataset provenance is tracked here so that the source and license remain visible.

## Original ASTRA examples

- **Creator:** Roger / How2FixBug
- **Source:** `Main/Models/Astra 0.1/dataset.txt`
- **License:** Project-authored material; reuse only under the terms chosen by the project owner.
- The additional custom examples in this dataset were written specifically for ASTRA and are not copied from the external datasets below.

## OpenAssistant Conversations Dataset (OASST1)

- **Dataset:** [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1)
- **Creators:** OpenAssistant community and contributors
- **License:** Apache License 2.0
- **Included subset:** English, ready-for-export message pairs; the importer filters deleted, negative-review, flagged, synthetic, and non-top-ranked responses where metadata is available.
- **Attribution:** OpenAssistant, “OpenAssistant Conversations Dataset (OASST1),” 2023. See the dataset card and included LICENSE file.
- **Paper:** [The OpenAssistant Conversations Dataset](https://arxiv.org/abs/2304.07327)

## Aya Dataset

- **Dataset:** [CohereLabs/aya_dataset](https://huggingface.co/datasets/CohereLabs/aya_dataset)
- **Creators:** Cohere Labs Aya Open Science Initiative and community contributors
- **License:** Apache License 2.0 (verify the dataset's current card and license file before each new redistribution).
- **Included subset:** English prompt/completion rows only, when the dataset viewer exposes the expected fields.
- **Attribution:** Cohere Labs, “Aya Dataset,” 2024. See the dataset card.
- **Paper:** [Aya: An Open Dataset for Multilingual Instruction Tuning](https://arxiv.org/abs/2402.06619)

## Not included

Databricks Dolly 15K was reviewed but is **not** included by the importer. Its CC BY-SA 3.0 license has a ShareAlike condition, not just a simple credit requirement. ASTRA will not silently mix it into the main dataset.

## Import and provenance notes

- Run `python3 import_datasets.py` from this directory to import up to 5,000 examples per supported source.
- The importer uses Python's standard library and downloads OASST1 plus English Aya rows. It does not install packages.
- Imported examples have an empty `<think>` section. The source datasets do not provide ASTRA reasoning summaries, and the importer does not invent or copy hidden reasoning.
- The importer filters exact duplicate prompt/answer pairs. Dataset sources and licenses remain listed here even when the local importer adds zero new examples.
- External dataset licenses may impose additional notice, attribution, patent, or redistribution obligations. Keep the original license files and notices when redistributing source material; this credits page is not a substitute for reading the full licenses.
- The source dataset owners retain their rights. This page records provenance; it does not claim ownership of third-party examples.
