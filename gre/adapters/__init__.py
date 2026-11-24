"""Built-in reference adapters (baselines). Importing registers them."""
from gre.adapters.extract import llm_structured  # noqa: F401
from gre.adapters.graph import networkx_store  # noqa: F401
from gre.adapters.resolve import baselines  # noqa: F401
