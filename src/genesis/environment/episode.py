"""Assemble a trusted environment episode from validated in-memory records."""

from collections.abc import Mapping, Sequence
from copy import deepcopy

from genesis.domain.models import EpisodeTruth
from genesis.evaluation.terminal import EpisodeEvaluator

from .auditor_env import Episode
from .tools import PublicTools


def build_episode(
    *,
    task: str,
    manifest: Mapping[str, object],
    policy: Mapping[str, object],
    tables: Mapping[str, Sequence[Mapping[str, object]]],
    truth: EpisodeTruth,
    tolerances_minor: Mapping[str, int] | None = None,
) -> Episode:
    """Use one public record catalogue for research and evidence validation."""
    episode_id = manifest["episode_id"]
    if type(episode_id) is not str or not episode_id or episode_id != truth["episode_id"]:
        raise ValueError("manifest and truth must identify the same episode")
    manifest = deepcopy(dict(manifest))
    policy = deepcopy(dict(policy))
    tools = PublicTools(episode_id=episode_id, policy=policy, tables=tables)
    return Episode(
        task=task,
        manifest=manifest,
        policy=policy,
        handle_action=tools,
        evaluate=EpisodeEvaluator(
            truth["issues"], tools.public_evidence_ids, tolerances_minor=tolerances_minor,
        ),
    )
