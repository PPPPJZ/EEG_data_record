"""临时 H5 验证数据对齐、无泄漏划分及 checkpoint 推理闭环。"""
from __future__ import annotations
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import h5py
import numpy as np
import torch
from adfcnn_lib.checkpoint import save_teacher
from adfcnn_lib.config import ModelConfig, PreprocessingConfig, RESULTS_SEEDS, TrainingConfig
from adfcnn_lib.data import load_session, make_folds, split_fold_indices
from adfcnn_lib.inference import Predictor
from adfcnn_lib.model import ADFCNN
from adfcnn_lib.preprocessing import preprocess_window
from adfcnn_lib.training import train_fold

# 测试参数配置区：短训、CPU、临时目录，不修改正式实验默认值。
TEST_SEED = 19
TEST_TRIALS = 10
TEST_TRAINING = TrainingConfig(epochs=3, batch_size=4, device="cpu")


def create_session(path: Path) -> None:
    config = PreprocessingConfig()
    raw = np.random.default_rng(TEST_SEED).normal(0, 10, (12000, 8)).astype(np.float32)
    dtype = [
        ("trial_id", "i4"), ("class_id", "i1"), ("completed", "?"),
        ("cue_start_sample", "i8"), ("trial_end_sample", "i8"),
    ]
    trials = np.zeros(TEST_TRIALS + 2, dtype=dtype)
    for index in range(len(trials)):
        trials[index] = (index + 1, index % 2, True, index * 1000, index * 1000 + 900)
    trials[-2]["completed"] = False
    trials[-1]["trial_end_sample"] = trials[-1]["cue_start_sample"] + 749
    protocol = {
        "sample_rate_hz": config.sampling_rate_hz, "channel_count": 8,
        "channel_labels": list(config.channel_names), "signal_unit": "uV",
        "class_codes": {"0": "left_hand", "1": "right_hand"},
    }
    with h5py.File(path, "w") as source:
        source.create_dataset("metadata/json", data=json.dumps({"status": "incomplete", "subject_id": "test"}))
        source.create_dataset("protocol/json", data=json.dumps(protocol))
        source.create_dataset("eeg/raw_uv", data=raw)
        source.create_dataset("trials/table", data=trials)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.original_threads)

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "motor_imagery.h5"
        create_session(self.path)
        self.session = load_session(self.path)

    def test_completed_trials_labels_and_shuffle_stay_aligned(self) -> None:
        session = self.session
        self.assertEqual(session.inputs.shape, (10, 1, 8, 750))
        self.assertEqual(session.summary["excluded_trials"], 2)
        self.assertEqual(session.summary["class_counts"], [5, 5])
        self.assertEqual(set(session.trial_ids.tolist()), set(range(1, 11)))
        self.assertFalse(np.array_equal(session.trial_ids, np.arange(1, 11)))
        np.testing.assert_array_equal(session.labels, (session.trial_ids - 1) % 2)
        repeated = load_session(self.path)
        np.testing.assert_array_equal(session.trial_ids, repeated.trial_ids)
        np.testing.assert_array_equal(session.inputs, repeated.inputs)

    def test_training_preprocessing_equals_feedback_window(self) -> None:
        with h5py.File(self.path, "r") as source:
            for i, trial_id in enumerate(self.session.trial_ids):
                start = int((trial_id - 1) * 1000)
                raw_uv = source["eeg/raw_uv"][start:start + 750]
                actual = preprocess_window(raw_uv, self.session.preprocessing)
                np.testing.assert_array_equal(actual, self.session.inputs[i])
        np.testing.assert_allclose(self.session.inputs.mean(axis=-1), 0, atol=1e-6)
        np.testing.assert_allclose(self.session.inputs.std(axis=-1), 1, atol=1e-6)

    def test_folds_are_stratified_disjoint_and_cover_once_per_seed(self) -> None:
        for seed in RESULTS_SEEDS:
            folds = make_folds(self.session.labels, seed)
            np.testing.assert_array_equal(folds, make_folds(self.session.labels, seed))
            all_test, all_validation = [], []
            for test_fold in range(1, 6):
                train, validation, test = split_fold_indices(folds, test_fold)
                self.assertEqual((len(train), len(validation), len(test)), (6, 2, 2))
                groups = [set(x) for x in (train, validation, test)]
                self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
                self.assertEqual(set.union(*groups), set(range(10)))
                np.testing.assert_array_equal(np.bincount(self.session.labels[test], minlength=2), [1, 1])
                all_test.extend(test)
                all_validation.extend(validation)
            self.assertEqual(sorted(all_test), list(range(10)))
            self.assertEqual(sorted(all_validation), list(range(10)))

    def test_invalid_window_or_nonfinite_values_fail(self) -> None:
        for values in (np.zeros((8, 750)), np.zeros((749, 8)), np.full((750, 8), np.nan), np.zeros((750, 8), dtype=complex)):
            with self.assertRaises(ValueError):
                preprocess_window(values, self.session.preprocessing)
        zero = preprocess_window(np.zeros((750, 8)), self.session.preprocessing)
        np.testing.assert_array_equal(zero, np.zeros((1, 8, 750), dtype=np.float32))

    def test_mismatched_channel_order_fails(self) -> None:
        with h5py.File(self.path, "r+") as source:
            protocol = json.loads(source["protocol/json"][()].decode())
            protocol["channel_labels"].reverse()
            del source["protocol/json"]
            source.create_dataset("protocol/json", data=json.dumps(protocol))
        with self.assertRaisesRegex(ValueError, "通道顺序"):
            load_session(self.path)

    def test_less_than_five_trials_per_class_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "每类至少"):
            make_folds(np.array([0] * 5 + [1] * 4), TEST_SEED)

    def test_train_selects_validation_model_and_evaluates_test_once(self) -> None:
        from adfcnn_lib import training
        folds = make_folds(self.session.labels, TEST_SEED)
        original = training.evaluate
        calls = []
        scripted_scores = [(0.3, 0.8), (0.2, 0.8), (0.1, 0.6), (0.4, 0.5)]
        def evaluate_spy(model, loader, device):
            original(model, loader, device)
            calls.append(loader.dataset.tensors[1].numpy().copy())
            return scripted_scores[len(calls) - 1]
        with patch.object(training, "evaluate", side_effect=evaluate_spy):
            result = train_fold(self.session, folds, 1, TEST_SEED, TEST_TRAINING)
        self.assertEqual(len(calls), TEST_TRAINING.epochs + 1)
        self.assertEqual(result.best_epoch, 2)
        self.assertEqual(result.validation_accuracy, 0.8)
        self.assertEqual(result.validation_loss, 0.2)
        self.assertEqual(result.test_accuracy, 0.5)

    def test_checkpoint_load_and_repeated_feedback_inference(self) -> None:
        folds = make_folds(self.session.labels, TEST_SEED)
        result = train_fold(self.session, folds, 1, TEST_SEED, TEST_TRAINING)
        checkpoint = Path(self.temporary.name) / "teacher.pt"
        save_teacher(checkpoint, self.session, result, folds, 1, TEST_SEED, TEST_TRAINING, ModelConfig())
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.assertEqual(payload["preprocessing_config"], asdict(self.session.preprocessing))
        train_ids, val_ids, test_ids = (set(payload[key]) for key in ("train_trial_ids", "validation_trial_ids", "test_trial_ids"))
        self.assertFalse(train_ids & val_ids or train_ids & test_ids or val_ids & test_ids)
        predictor = Predictor(checkpoint, device="cpu")
        trial_id = self.session.trial_ids[0]
        with h5py.File(self.path, "r") as source:
            start = int((trial_id - 1) * 1000)
            raw_uv = source["eeg/raw_uv"][start:start + 750]
        details = predictor.predict_details(raw_uv)
        self.assertIn(details.class_id, (0, 1))
        self.assertEqual(details.class_name, self.session.preprocessing.class_names[details.class_id])
        self.assertAlmostEqual(sum(details.probabilities), 1, places=6)
        reference = ADFCNN(ModelConfig()).eval()
        reference.load_state_dict(result.state_dict)
        with torch.inference_mode():
            probabilities = torch.softmax(reference(torch.from_numpy(self.session.inputs[:1])), dim=1)[0].numpy()
        np.testing.assert_allclose(details.probabilities, probabilities, atol=1e-6)
        self.assertEqual(predictor.predict(raw_uv), details.class_id)
        second = Predictor(checkpoint, device="cpu").predict_details(raw_uv)
        np.testing.assert_allclose(second.probabilities, details.probabilities, atol=1e-6)
        payload["preprocessing_version"] = "incompatible"
        torch.save(payload, checkpoint)
        with self.assertRaisesRegex(ValueError, "不兼容"):
            Predictor(checkpoint, device="cpu")


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
