from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ModelFileSpec:
    filename: str
    expected_sha256: str  # Empty string for small JSON/template configs
    size_bytes: int
    repo_id: Optional[str] = None
    revision: Optional[str] = None
    remote_filename: Optional[str] = None


@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    repo_id: str
    revision: str
    source_url: str
    license: str
    architecture: str
    training_dataset: str
    intended_task: str
    preprocessing: str
    limitations: str
    primary_weight_file: str
    files: List[ModelFileSpec]


SMAUG_REVISION = "c7c95951e04a87b884a4ad479401b1128c3903bd"

MODEL_CATALOG: Dict[str, ModelSpec] = {
    "ai_generated_detector": ModelSpec(
        key="ai_generated_detector",
        name="Smaug AI Image Detector (Primary)",
        repo_id="abacusai/Smaug-AI-Image-Detector",
        revision=SMAUG_REVISION,
        source_url="https://huggingface.co/abacusai/Smaug-AI-Image-Detector",
        license="Apache-2.0",
        architecture="EmbeddingGemma-2 vision-only encoder (439M params) + SentenceTransformer Pooling/Normalize/Dense classifier (bfloat16 on CUDA, float32 on CPU)",
        training_dataset="Fine-tuned by Abacus.AI on Google EmbeddingGemma-2 for modern synthetic generators (GPT-Image, FLUX, Midjourney, DALL-E, Sora/Veo frames); benchmarked on OpenFake & CommunityForensics.",
        intended_task="Primary full-image AI-generated vs. real image detector scored via three 528x528 crops at native resolution.",
        preprocessing="Official SmaugImageDetect views(img, crops=3): EXIF-transpose RGB -> shortest edge to 576px (upscale to 528 if smaller) -> 3 native-res 528x528 crops (start, center, end) -> 140 vision tokens -> sigmoid(mean crop logits).",
        limitations="Whole-image synthetic probability only; uncalibrated raw sigmoid score; unstable in fp16 (requires bf16 on CUDA or fp32 on CPU). Does not localize partial edits on real photos and accuracy can vary under heavy recompression or unseen generators.",
        primary_weight_file="model.safetensors",
        files=[
            ModelFileSpec("config.json", "", 3077),
            ModelFileSpec("config_sentence_transformers.json", "", 1326),
            ModelFileSpec("modules.json", "", 537),
            ModelFileSpec("processor_config.json", "", 1788),
            ModelFileSpec("sentence_bert_config.json", "", 747),
            ModelFileSpec("tokenizer_config.json", "", 1600),
            ModelFileSpec("chat_template.jinja", "", 1016),
            ModelFileSpec(
                "smaug_image_detect.py",
                "ad8a17d78e610127d6e23fc7fb4af27a3e7a39da5734ea643dfaafdc667f282a",
                3015,
            ),
            ModelFileSpec(
                "model.safetensors",
                "c5ae3fac7470126f25d2dca2386d8728f7c828fae53e8ac9eb006937821ea006",
                877599816,
            ),
            ModelFileSpec(
                "tokenizer.json",
                "4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4",
                32170510,
            ),
            ModelFileSpec("1_Pooling/config.json", "", 90),
            ModelFileSpec("2_Normalize/config.json", "", 97),
            ModelFileSpec("3_Dense/config.json", "", 225),
            ModelFileSpec(
                "3_Dense/model.safetensors",
                "b09918c3838ef3b9dbd7057b0a8d382eced8fa06363ed62a5f60abe65cc65809",
                1690,
            ),
        ],
    ),
    "univfd_detector": ModelSpec(
        key="univfd_detector",
        name="UniversalFakeDetect (UnivFD · Secondary Comparison)",
        repo_id="openai/clip-vit-large-patch14 + siddharthksah/deepsafe-weights",
        revision="32bd64288804d66eefd0ccbe215aa642df71cc41",
        source_url="https://github.com/Arjunkalliyadath/ai-image-detector",
        license="MIT (UnivFD implementation, fc_weights head & OpenAI CLIP ViT-L/14)",
        architecture="CLIPVisionModelWithProjection (ViT-L/14, 768-d) + L2-norm + Linear(768->1) UnivFD head",
        training_dataset="ProGAN synthetic/real benchmark (Ojha et al., CVPR 2023 UnivFD) · Retained as secondary comparison baseline.",
        intended_task="Secondary comparison full-image synthetic/AI-generated vs. real detector (raw sigmoid P(ai)).",
        preprocessing="EXIF-transpose RGB -> Bicubic resize 224 -> CenterCrop 224x224 -> CLIP mean=[0.4815,0.4578,0.4082], std=[0.2686,0.2613,0.2758] -> L2-normalize",
        limitations="Secondary comparison detector only. Trained on older ProGAN distributions; raw sigmoid score is uncalibrated and less sensitive to modern generators than Smaug.",
        primary_weight_file="fc_weights.pth",
        files=[
            ModelFileSpec(
                "config.json",
                "",
                4519,
                repo_id="openai/clip-vit-large-patch14",
                revision="32bd64288804d66eefd0ccbe215aa642df71cc41",
                remote_filename="config.json",
            ),
            ModelFileSpec(
                "preprocessor_config.json",
                "",
                316,
                repo_id="openai/clip-vit-large-patch14",
                revision="32bd64288804d66eefd0ccbe215aa642df71cc41",
                remote_filename="preprocessor_config.json",
            ),
            ModelFileSpec(
                "model.safetensors",
                "a2bf730a0c7debf160f7a6b50b3aaf3703e7e88ac73de7a314903141db026dcb",
                1710540580,
                repo_id="openai/clip-vit-large-patch14",
                revision="32bd64288804d66eefd0ccbe215aa642df71cc41",
                remote_filename="model.safetensors",
            ),
            ModelFileSpec(
                "fc_weights.pth",
                "477100745713bcc957beb2b40859536859b6483fd6301b3b9293151b194c7847",
                4083,
                repo_id="siddharthksah/deepsafe-weights",
                revision="71706520a9e12494b3ebc24e6091f5b22b9efcaf",
                remote_filename="universalfakedetect/fc_weights.pth",
            ),
        ],
    ),
    "face_manipulation_detector": ModelSpec(
        key="face_manipulation_detector",
        name="EfficientNet-B0 FaceForensics++ C23 Detector",
        repo_id="Xicor9/efficientnet-b0-ffpp-c23",
        revision="21aebf2d1c8a7ba85eb61734884c50c877c38ea3",
        source_url="https://huggingface.co/Xicor9/efficientnet-b0-ffpp-c23",
        license="MIT (weights) / FaceForensics++ Research-Only terms",
        architecture="torchvision.models.efficientnet_b0 (classifier[1]=Linear(1280,2)) + OpenCV Haar face detector",
        training_dataset="FaceForensics++ (FF++) C23 (Real vs. DeepFake, FaceSwap, Face2Face, NeuralTextures)",
        intended_task="Face manipulation / face-swap classification on OpenCV-detected face crops (0=Real, 1=Fake)",
        preprocessing="OpenCV Haar face crop (+25% margin) -> Resize 224x224 RGB -> ToTensor() [0..1]",
        limitations="Research/academic checkpoint; raw score is uncalibrated; requires detectable frontal/near-frontal face; accuracy drops on heavy occlusion or unseen manipulation methods.",
        primary_weight_file="efficientnet_b0_ffpp_c23.pth",
        files=[
            ModelFileSpec(
                "efficientnet_b0_ffpp_c23.pth",
                "2a7963b787ece2594a79160f20e53d6bc621f6bfe34ee21d139efa81bcb70c84",
                16339854,
            ),
        ],
    ),
    "dinov2_extractor": ModelSpec(
        key="dinov2_extractor",
        name="DINOv2 ViT-S/14 Feature Extractor",
        repo_id="facebook/dinov2-small",
        revision="ed25f3a31f01632728cabb09d1542f84ab7b0056",
        source_url="https://huggingface.co/facebook/dinov2-small",
        license="Apache-2.0",
        architecture="Dinov2Model (ViT-S/14, 384-d embeddings, 12 layers, 6 attention heads)",
        training_dataset="LVD-142M (self-supervised visual representation learning)",
        intended_task="384-d image feature extraction & experimental CLS-to-patch attention visualization (not a standalone deepfake classifier)",
        preprocessing="Shortest edge 256 -> center crop 224x224 RGB -> ImageNet mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]",
        limitations="Self-supervised backbone only; never emits synthetic classification scores without a trained task head.",
        primary_weight_file="model.safetensors",
        files=[
            ModelFileSpec("config.json", "", 547),
            ModelFileSpec("preprocessor_config.json", "", 436),
            ModelFileSpec(
                "model.safetensors",
                "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1",
                88249960,
            ),
        ],
    ),
}
