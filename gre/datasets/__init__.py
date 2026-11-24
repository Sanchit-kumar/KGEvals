"""Built-in datasets. Importing this module registers them."""
from gre.datasets import entity_matching, local, mine, redocred, text2kgbench, toy  # noqa: F401
from gre.datasets.base import Dataset, data_dir, load_samples, parse_dataset_spec  # noqa: F401

# Named suites: `suite: kg_sota` in YAML or `gre run --suite kg_sota`.
SUITES: dict[str, list[str]] = {
    # Extraction + ER track, strongest public benchmarks per stage
    "kg_sota": ["text2kgbench_webnlg", "text2kgbench_tekgen", "redocred", "mine",
                "wdc_products", "abt_buy", "amazon_google"],
    "extract": ["text2kgbench_webnlg", "text2kgbench_tekgen", "redocred"],
    "resolve": ["redocred", "wdc_products", "abt_buy", "amazon_google"],
    "graph": ["mine", "redocred"],
    "smoke": ["toy"],
}
