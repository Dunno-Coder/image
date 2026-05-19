from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "datasets"
WEIGHTS_DIR = ROOT_DIR / "weights"
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_CSV = RESULTS_DIR / "results.csv"
IMPROVEMENT_REPORT_CSV = RESULTS_DIR / "improvement_report.csv"
BEST_BACKBONE_CONFIG_JSON = RESULTS_DIR / "best_backbone_config.json"
FEEDBACK_DIR = ROOT_DIR / "feedback"
FEEDBACK_CSV = FEEDBACK_DIR / "feedback.csv"

SUPPORTED_DATASETS = ("genimage", "cifake", "ai_vs_real")
SUPPORTED_MODELS = (
    "dinov3_linear",
    "siglip2_linear",
    "aimv2_linear",
    "proposed_mnff_edl",
)
BASELINE_MODELS = ("dinov3_linear", "siglip2_linear", "aimv2_linear")

CLASS_TO_IDX = {"real": 0, "ai": 1}
IDX_TO_CLASS = {0: "real", 1: "ai"}
NUM_CLASSES = 2

DEFAULT_IMAGE_SIZE = 224
DEFAULT_SEED = 42
DEFAULT_UNCERTAINTY_THRESHOLD = 0.50

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# The proposed model uses the strongest selected foundation backbone by default.
# Change this after experiments if another baseline backbone wins consistently.
DEFAULT_PROPOSED_BACKBONE = "dinov3_linear"
