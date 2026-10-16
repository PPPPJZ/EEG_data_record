from __future__ import annotations

import tempfile
import unittest
from collections import Counter
from pathlib import Path

from config import CLASS_LEFT, CLASS_RIGHT
from experiment import (
    build_trial_plans,
    build_trial_sequence,
    create_session_config,
)


def make_config(output_dir: Path, *, seed: int = 123):
    return create_session_config(
        subject_id="S001",
        experiment_date="2026-09-21",
        description="test",
        left_trials=3,
        right_trials=5,
        output_dir=str(output_dir),
        random_seed=seed,
    )


class SequenceTest(unittest.TestCase):
    def test_sequence_has_exact_requested_counts_and_is_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = make_config(Path(directory))

            first = build_trial_sequence(config)
            second = build_trial_sequence(config)

        self.assertEqual(first, second)
        self.assertEqual(
            Counter(first),
            Counter({CLASS_LEFT: 3, CLASS_RIGHT: 5}),
        )

    def test_trial_plan_has_required_boundaries(self) -> None:
        plans = build_trial_plans([CLASS_LEFT, CLASS_RIGHT])

        self.assertEqual(plans[0].trial_id, 1)
        self.assertEqual(plans[0].planned_rest_start_s, 0.0)
        self.assertEqual(plans[0].planned_cue_start_s, 2.5)
        self.assertEqual(plans[0].planned_cue_end_s, 4.0)
        self.assertEqual(plans[0].planned_trial_end_s, 8.0)
        self.assertEqual(plans[1].planned_rest_start_s, 8.0)
        self.assertEqual(plans[1].planned_trial_end_s, 16.0)

    def test_nondivisible_trial_count_assigns_blocks(self) -> None:
        plans = build_trial_plans(
            [CLASS_LEFT, CLASS_RIGHT, CLASS_LEFT, CLASS_RIGHT, CLASS_LEFT],
            break_every_trials=2,
        )

        self.assertEqual([plan.block_id for plan in plans], [1, 1, 2, 2, 3])

    def test_scheduled_break_count_excludes_final_block(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = create_session_config(
                subject_id="S001",
                experiment_date="2026-09-21",
                description="",
                left_trials=3,
                right_trials=2,
                break_every_trials=2,
                output_dir=directory,
            )

        self.assertEqual(config.scheduled_break_count, 2)

    def test_rejects_relative_output_path(self) -> None:
        with self.assertRaisesRegex(ValueError, "绝对路径"):
            create_session_config(
                subject_id="S001",
                experiment_date="2026-09-21",
                description="",
                left_trials=1,
                right_trials=1,
                output_dir="relative/path",
            )


if __name__ == "__main__":
    unittest.main()
