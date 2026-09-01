# Third-party notices and data attribution

The repository's original software is licensed under the [MIT License](LICENSE).
That license does not relicense third-party datasets, sample images, pretrained
weights, papers, names, or marks. Those materials remain governed by their
respective terms.

## SID_Set / SIDA

CamTrace-6M was trained and evaluated on a subset of **SID_Set (Social media
Image Detection dataSet)** by Zhenglin Huang, Jinwei Hu, Xiangtai Li, Yiwei He,
Xingyu Zhao, Bei Peng, Baoyuan Wu, Xiaowei Huang, and Guangliang Cheng.

- Dataset: https://huggingface.co/datasets/saberzl/SID_Set
- Paper: *SIDA: Social Media Image Deepfake Detection, Localization and
  Explanation with Large Multimodal Model* (CVPR 2025)
- License stated by the dataset publisher: [Creative Commons Attribution 4.0
  International](https://creativecommons.org/licenses/by/4.0/)

This project decodes, crops, JPEG re-encodes, adds controlled transformations
to, and computes model outputs from SID_Set images. The two images under
`demo/evaluation_inputs/` and the transformed excerpts in
`reports/error_contact_sheet.png` are included as attributed evaluation and
error-analysis evidence under CC BY 4.0. They are not covered by this
repository's MIT software license. No endorsement by the SID_Set authors or
the creators of its source images is implied.

Demo filename mapping to the unmodified SID_Set identifiers:

- `demo/evaluation_inputs/sid_real_ccby.jpg` ← `37eb6c20f4b49b58.jpg`
- `demo/evaluation_inputs/sid_synthetic_ccby.png` ←
  `full_synthetic_008406.png`

SID_Set states that some real-image material originates from Open Images V7,
COCO, and Flickr30k and is provided under the attribution terms described on
the SID_Set dataset card. The original image identifiers are retained in the
project's machine-readable error report and the mapping above.

Suggested citation:

```bibtex
@inproceedings{huang2025sida,
  title     = {SIDA: Social Media Image Deepfake Detection, Localization and Explanation with Large Multimodal Model},
  author    = {Huang, Zhenglin and Hu, Jinwei and Li, Xiangtai and He, Yiwei and Zhao, Xingyu and Peng, Bei and Wu, Baoyuan and Huang, Xiaowei and Cheng, Guangliang},
  booktitle = {Conference on Computer Vision and Pattern Recognition},
  year      = {2025}
}
```

## COCO val2017 and WildFake / DALL·E Advanced

COCO val2017 photographs and the DALL·E Advanced subset of WildFake were
used only for the organizer-provided held-out demonstration and content-blind
confound audit. They were never used to fit CamTrace-6M and are not
redistributed in this repository or release archive.

- COCO: https://cocodataset.org/#home
- WildFake dataset page supplied in the challenge brief:
  https://modelscope.cn/datasets/hy2628982280/WildFake/summary
- WildFake paper: Yan Hong et al., *WildFake: A Large-Scale and Hierarchical
  Dataset for AI-Generated Images Detection*, AAAI 2025.

Users who reproduce the audit must obtain these datasets from their official
sources and comply with the licenses and terms applying to each image.

## CLIP and OpenCLIP

CamTrace-6M uses the OpenAI CLIP ViT-L/14 QuickGELU vision backbone through
OpenCLIP. The backbone remains frozen and its weights are not included in the
CamTrace-6M checkpoint or submission archive; OpenCLIP downloads them from the
upstream source on first use.

- OpenAI CLIP: https://github.com/openai/CLIP — MIT License
- OpenCLIP: https://github.com/mlfoundations/open_clip — MIT License

The names OpenAI, CLIP, PyTorch, Hugging Face, NVIDIA, CUDA, and other
third-party names are used only to identify compatible technologies. No
endorsement or affiliation is implied.

## Python dependencies

Runtime and development dependencies are resolved from their official package
indexes by `uv.lock`; their source code is not vendored here. Each dependency
remains under its upstream license. See `pyproject.toml` and `uv.lock` for the
complete, reproducible dependency set.
