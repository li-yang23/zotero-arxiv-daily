from collections.abc import Iterator
from typing import Any

from omegaconf import OmegaConf


def iter_generation_kwargs(llm_params: Any) -> Iterator[dict[str, Any]]:
    generation_kwargs = llm_params.get("generation_kwargs", {}) or {}
    if OmegaConf.is_config(generation_kwargs):
        generation_kwargs = OmegaConf.to_container(generation_kwargs, resolve=True)
    else:
        generation_kwargs = dict(generation_kwargs)

    models = generation_kwargs.pop("model", None)
    if models is None:
        yield generation_kwargs
        return

    if isinstance(models, str):
        yield {**generation_kwargs, "model": models}
        return

    for model in models:
        yield {**generation_kwargs, "model": str(model)}
