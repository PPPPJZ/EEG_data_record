"""临时 H5 验证数据对齐、无泄漏划分及 checkpoint 推理闭环。"""
from __future__ import annotations
from contextlib import redirect_stdout
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import h5py
import numpy as np
import torch
from adfcnn_lib.checkpoint import save_teacher
from adfcnn_lib.config import ModelConfig, PreprocessingConfig, RESULTS_SEED, TrainingConfig
from adfcnn_lib.data import (
    SessionData, load_session, make_folds, split_four_train_one_test,
)
from adfcnn_lib.inference import Predictor
from adfcnn_lib.model import ADFCNN
from adfcnn_lib.preprocessing import preprocess_window
from adfcnn_lib.training import train_all, train_fold

# 测试参数配置区：短训、CPU、临时目录，不修改正式实验默认值。
TEST_SEED = 19
TEST_TRIALS = 10
TEST_TRAINING = TrainingConfig(
    epochs=3, batch_size=4, device="cpu"
)


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

    def test_folds_are_balanced_disjoint_and_cover_once(self) -> None:
        folds = make_folds(self.session.labels, RESULTS_SEED)
        np.testing.assert_array_equal(folds, make_folds(self.session.labels, RESULTS_SEED))
        all_test = []
        for test_fold in range(1, 6):
            train, test = split_four_train_one_test(folds, test_fold)
            self.assertEqual((len(train), len(test)), (8, 2))
            self.assertFalse(set(train) & set(test))
            self.assertEqual(set(train) | set(test), set(range(10)))
            np.testing.assert_array_equal(np.bincount(self.session.labels[test], minlength=2), [1, 1])
            np.testing.assert_array_equal(np.bincount(self.session.labels[train], minlength=2), [4, 4])
            all_test.extend(test)
        self.assertEqual(sorted(all_test), list(range(10)))

    def test_nonbalanced_folds_fail_without_discarding_trials(self) -> None:
        for labels in (np.array([0] * 10 + [1] * 5), np.array([0] * 6 + [1] * 6)):
            with self.assertRaisesRegex(ValueError, "每折左右手"):
                make_folds(labels, RESULTS_SEED)

    def test_teacher_loads_sessions_with_fewer_than_five_trials_per_class(self) -> None:
        with h5py.File(self.path, "r+") as source:
            trials = source["trials/table"][:]
            trials["completed"][4:] = False
            source["trials/table"][:] = trials
        session = load_session(self.path)
        self.assertEqual(session.summary["class_counts"], [2, 2])
        result = train_all(session, RESULTS_SEED, TEST_TRAINING)
        np.testing.assert_array_equal(result.train_indices, np.arange(4))
        with self.assertRaisesRegex(ValueError, "每类至少"):
            make_folds(session.labels, RESULTS_SEED)

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

    def test_train_uses_final_epoch_and_evaluates_test_once(self) -> None:
        from adfcnn_lib import training
        folds = make_folds(self.session.labels, RESULTS_SEED)
        original = training.evaluate
        calls = []
        states = []
        progress = []
        def evaluate_spy(model, loader, device):
            calls.append(loader.dataset.tensors[0].numpy().copy())
            states.append({key: value.detach().cpu().clone() for key, value in model.state_dict().items()})
            return original(model, loader, device)
        with patch.object(training, "evaluate", side_effect=evaluate_spy):
            result = train_fold(
                self.session, folds, 3, RESULTS_SEED, TEST_TRAINING,
                progress=lambda *values: progress.append(values),
            )
        self.assertEqual(len(calls), 1)
        self.assertEqual(result.final_epoch, TEST_TRAINING.epochs)
        self.assertEqual(result.training_seed, RESULTS_SEED)
        self.assertEqual(result.final_learning_rate, 0.0)
        self.assertEqual([row[0] for row in progress], [1, 2, 3])
        np.testing.assert_allclose([row[2] for row in progress], [0.001, 0.00075, 0.00025])
        np.testing.assert_array_equal(calls[0], self.session.inputs[result.test_indices])
        for key, value in states[0].items():
            torch.testing.assert_close(value, result.state_dict[key], rtol=0, atol=0)
        tracked = [value.item() for key, value in states[0].items() if key.endswith("num_batches_tracked")]
        self.assertTrue(all(count == 6 for count in tracked))

    def test_checkpoint_load_and_repeated_feedback_inference(self) -> None:
        result = train_all(self.session, RESULTS_SEED, TEST_TRAINING)
        checkpoint = Path(self.temporary.name) / "teacher.pt"
        save_teacher(checkpoint, self.session, result, TEST_TRAINING, ModelConfig())
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.assertEqual(payload["preprocessing_config"], asdict(self.session.preprocessing))
        self.assertEqual(payload["train_class_counts"], [5, 5])
        self.assertEqual(set(payload["train_trial_ids"]), set(self.session.trial_ids))
        self.assertEqual(payload["final_epoch"], TEST_TRAINING.epochs)
        self.assertFalse(payload["training_protocol"]["validation"])
        self.assertFalse(payload["training_protocol"]["test"])
        for key in ("validation_trial_ids", "test_trial_ids", "validation_accuracy", "test_accuracy", "best_epoch"):
            self.assertNotIn(key, payload)
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

    def test_teacher_trains_and_saves_all_imbalanced_trial_ids_without_evaluation(self) -> None:
        from adfcnn_lib import training
        labels = np.array([0] * 10 + [1] * 5, dtype=np.int64)
        rng = np.random.default_rng(TEST_SEED)
        session = SessionData(
            source_path=self.path,
            inputs=rng.normal(size=(15, 1, 8, 750)).astype(np.float32),
            labels=labels,
            trial_ids=np.arange(101, 116, dtype=np.int64),
            preprocessing=PreprocessingConfig(),
            summary={"class_counts": [10, 5]},
        )
        config = TrainingConfig(epochs=1, batch_size=4, device="cpu")
        with (
            patch.object(training, "evaluate", side_effect=AssertionError("teacher 不可评估")),
            patch.object(training, "make_loader", wraps=training.make_loader) as loader,
        ):
            result = train_all(session, RESULTS_SEED, config)
        self.assertEqual(loader.call_count, 1)
        np.testing.assert_array_equal(loader.call_args.args[1], np.arange(15))
        np.testing.assert_array_equal(result.train_indices, np.arange(15))
        checkpoint = Path(self.temporary.name) / "imbalanced_teacher.pt"
        save_teacher(checkpoint, session, result, config, ModelConfig())
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.assertEqual(payload["train_class_counts"], [10, 5])
        self.assertEqual(payload["train_trial_ids"], session.trial_ids.tolist())
        self.assertEqual(payload["training_protocol"]["class_resampling"], "none")
        self.assertFalse(hasattr(result, "test_accuracy"))

    def test_checkpoint_rejects_a_fold_model_as_full_data_teacher(self) -> None:
        folds = make_folds(self.session.labels, RESULTS_SEED)
        result = train_fold(self.session, folds, 1, RESULTS_SEED, TEST_TRAINING)
        checkpoint = Path(self.temporary.name) / "wrong_teacher.pt"
        with self.assertRaisesRegex(ValueError, "全部有效"):
            save_teacher(checkpoint, self.session, result, TEST_TRAINING, ModelConfig())
        self.assertFalse(checkpoint.exists())

    def test_results_entry_runs_one_split_and_five_tests_with_seed_42(self) -> None:
        import results
        output = io.StringIO()
        with (
            patch("sys.argv", ["results.py", "--data", str(self.path)]),
            patch.object(results, "make_folds", wraps=make_folds) as split,
            patch.object(results, "train_fold", return_value=SimpleNamespace(test_accuracy=0.75)) as train,
            redirect_stdout(output),
        ):
            results.main()
        self.assertEqual(split.call_count, 1)
        self.assertEqual(train.call_count, 5)
        self.assertEqual([call.args[2] for call in train.call_args_list], [1, 2, 3, 4, 5])
        self.assertEqual([call.args[3] for call in train.call_args_list], [42] * 5)
        self.assertTrue(all(call.args[4].epochs == 2000 for call in train.call_args_list))
        self.assertEqual(output.getvalue(), "Accuracy: 75.00%\n")

    def test_teacher_entry_saves_without_splitting_or_printing_accuracy(self) -> None:
        import teacher
        from adfcnn_lib import training
        output = io.StringIO()
        checkpoint = Path(self.temporary.name) / "adfcnn_teacher.pt"
        with (
            patch("sys.argv", ["teacher.py", "--data", str(self.path), "--epochs", "1", "--device", "cpu"]),
            patch("adfcnn_lib.data.make_folds", side_effect=AssertionError("teacher 不可划分")),
            patch.object(training, "evaluate", side_effect=AssertionError("teacher 不可评估")),
            redirect_stdout(output),
        ):
            teacher.main()
        self.assertEqual(output.getvalue(), f"Model: {checkpoint}\n")
        self.assertTrue(checkpoint.is_file())


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
