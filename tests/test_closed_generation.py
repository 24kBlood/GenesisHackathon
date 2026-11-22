"""Public identifiers and replay at the sealed generation boundary."""

from unittest import TestCase
from unittest.mock import patch

from genesis.evaluation.judge import closed_environment
from genesis.generation import generate_closed_episode, new_closed_case


class ClosedGenerationTests(TestCase):
    config = {"schema_version": "1.0.0", "error_counts": {"fx_conversion": 1}}

    def test_public_id_is_supplied_independently_of_the_case_seed(self):
        token = "0123456789abcdef" * 2
        with patch("genesis.generation.company.token_hex", return_value=token) as make_id:
            with patch("genesis.generation.company.randbits", side_effect=[1, 2]) as make_seed:
                first_seed, first_config = new_closed_case(self.config)
                second_seed, second_config = new_closed_case(self.config)
        self.assertEqual(make_id.call_args_list[0].args, (16,))
        self.assertEqual(make_seed.call_args_list[0].args, (128,))
        self.assertNotEqual(first_seed, second_seed)
        self.assertEqual(first_config, second_config)
        expected_id = "episode-" + token
        for seed in (first_seed, second_seed):
            case = generate_closed_episode(seed, first_config)
            self.assertEqual(case, generate_closed_episode(seed, first_config))
            self.assertEqual(case.public.manifest["episode_id"], expected_id)
            self.assertEqual(case.evaluator.truth["episode_id"], expected_id)
            observation, _ = closed_environment().reset(seed, first_config)
            self.assertEqual(observation["manifest"]["episode_id"], expected_id)
            self.assertNotIn("seed", observation)

    def test_closed_path_rejects_public_seeds_and_legacy_ids(self):
        seed, config = new_closed_case(self.config)
        for rejected_seed, rejected_config in (
            (20260304, config),
            (seed, self.config),
            (seed, {**self.config, "episode_id": "episode-" + "a" * 24}),
        ):
            with self.subTest(seed=rejected_seed, config=rejected_config):
                with self.assertRaises(ValueError):
                    closed_environment().reset(rejected_seed, rejected_config)
