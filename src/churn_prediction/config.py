"""Configuration manager for the Customer Churn Prediction pipeline."""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


def find_project_root() -> Path:
    """Recursively search upward from current file to find project root marked by configs/ or .git."""
    curr = Path(__file__).resolve().parent
    for parent in [curr] + list(curr.parents):
        if (parent / "configs" / "config.yaml").exists() or (parent / "requirements.txt").exists():
            return parent
    return Path.cwd()


PROJECT_ROOT = find_project_root()


def load_raw_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration dictionary from YAML file."""
    if config_path is None:
        target = PROJECT_ROOT / "configs" / "config.yaml"
    else:
        target = Path(config_path)
        if not target.is_absolute():
            target = PROJECT_ROOT / target

    if not target.exists():
        raise FileNotFoundError(f"Configuration file not found at: {target}")

    with open(target, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class AppConfig:
    """Structured configuration object with resolved paths."""

    raw: Dict[str, Any] = field(default_factory=dict)

    # Core properties
    project_name: str = "Customer Churn Prediction"
    random_seed: int = 42

    # Paths (all resolved to absolute Paths)
    project_root: Path = field(default_factory=find_project_root)
    raw_data_path: Path = field(init=False)
    processed_dir: Path = field(init=False)
    database_path: Path = field(init=False)
    models_dir: Path = field(init=False)
    best_model_path: Path = field(init=False)
    reports_dir: Path = field(init=False)
    figures_dir: Path = field(init=False)
    metrics_comparison_path: Path = field(init=False)
    sql_schema_path: Path = field(init=False)
    sql_queries_path: Path = field(init=False)

    # Data column specs
    id_column: str = "customerID"
    target_column: str = "Churn"
    positive_label: str = "Yes"
    negative_label: str = "No"
    test_size: float = 0.20
    cv_folds: int = 5
    stratify: bool = True
    numeric_features: List[str] = field(default_factory=list)
    categorical_features: List[str] = field(default_factory=list)

    # Model parameters
    logistic_regression_params: Dict[str, Any] = field(default_factory=dict)
    random_forest_params: Dict[str, Any] = field(default_factory=dict)
    gradient_boosting_params: Dict[str, Any] = field(default_factory=dict)

    # Evaluation
    default_threshold: float = 0.50

    def __post_init__(self):
        if not self.raw:
            self.raw = load_raw_config()

        self.project_name = self.raw.get("project", {}).get("name", self.project_name)
        self.random_seed = int(
            os.getenv("RANDOM_SEED", self.raw.get("environment", {}).get("random_seed", 42))
        )

        paths = self.raw.get("paths", {})
        self.raw_data_path = self._resolve(
            os.getenv("DATA_RAW_PATH", paths.get("raw_data", "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"))
        )
        self.processed_dir = self._resolve(paths.get("processed_dir", "data/processed"))
        self.database_path = self._resolve(
            os.getenv("DATABASE_PATH", paths.get("database", "data/churn_analysis.db"))
        )
        self.models_dir = self._resolve(paths.get("models_dir", "models"))
        self.best_model_path = self._resolve(
            os.getenv("MODEL_OUTPUT_PATH", paths.get("best_model_path", "models/best_model.joblib"))
        )
        self.reports_dir = self._resolve(paths.get("reports_dir", "reports"))
        self.figures_dir = self._resolve(paths.get("figures_dir", "reports/figures"))
        self.metrics_comparison_path = self._resolve(
            paths.get("metrics_comparison", "reports/model_comparison.csv")
        )
        self.sql_schema_path = self._resolve(paths.get("sql_schema", "sql/schema.sql"))
        self.sql_queries_path = self._resolve(paths.get("sql_queries", "sql/analysis_queries.sql"))

        data_sec = self.raw.get("data", {})
        self.id_column = data_sec.get("id_column", "customerID")
        self.target_column = data_sec.get("target_column", "Churn")
        self.positive_label = data_sec.get("positive_label", "Yes")
        self.negative_label = data_sec.get("negative_label", "No")
        self.test_size = float(data_sec.get("test_size", 0.20))
        self.cv_folds = int(data_sec.get("cv_folds", 5))
        self.stratify = bool(data_sec.get("stratify", True))

        self.numeric_features = data_sec.get(
            "numeric_features", ["tenure", "MonthlyCharges", "TotalCharges"]
        )
        self.categorical_features = data_sec.get(
            "categorical_features",
            [
                "gender",
                "SeniorCitizen",
                "Partner",
                "Dependents",
                "PhoneService",
                "MultipleLines",
                "InternetService",
                "OnlineSecurity",
                "OnlineBackup",
                "DeviceProtection",
                "TechSupport",
                "StreamingTV",
                "StreamingMovies",
                "Contract",
                "PaperlessBilling",
                "PaymentMethod",
            ],
        )

        models_sec = self.raw.get("models", {})
        self.logistic_regression_params = models_sec.get("logistic_regression", {})
        self.random_forest_params = models_sec.get("random_forest", {})
        self.gradient_boosting_params = models_sec.get("gradient_boosting", {})

        eval_sec = self.raw.get("evaluation", {})
        self.default_threshold = float(eval_sec.get("default_threshold", 0.50))

    def _resolve(self, path_str: str) -> Path:
        p = Path(path_str)
        if p.is_absolute():
            return p
        return (self.project_root / p).resolve()


def get_config(config_path: Optional[str] = None) -> AppConfig:
    """Factory function to get instantiated AppConfig."""
    raw = load_raw_config(config_path)
    return AppConfig(raw=raw)
