from __future__ import annotations

import struct
import unittest

import numpy as np

from config import (
    ADC_UV_PER_LSB,
    CHANNEL_COUNT,
    SAMPLE_RATE_HZ,
    SAMPLES_PER_PACKET,
)
from eeg import (
    PACKET_SIZE,
    EEGPacketDecoder,
    PacketDecodeError,
    RealtimeEEGFilter,
)


def make_packet(values: np.ndarray, flags: int = 0x03) -> bytes:
    packet = bytearray([flags])
    for sample_index in range(SAMPLES_PER_PACKET):
        packet.extend(
            struct.pack(
                "<9i",
                sample_index,
                *values[:, sample_index].tolist(),
            )
        )
    return bytes(packet)


class PacketDecoderTest(unittest.TestCase):
    def test_decodes_raw_eight_channel_microvolts(self) -> None:
        values = np.arange(
            CHANNEL_COUNT * SAMPLES_PER_PACKET,
            dtype=np.int32,
        ).reshape(CHANNEL_COUNT, SAMPLES_PER_PACKET)

        decoded = EEGPacketDecoder().feed(make_packet(values))

        self.assertEqual(len(make_packet(values)), PACKET_SIZE)
        np.testing.assert_allclose(decoded, values * ADC_UV_PER_LSB)

    def test_rejects_short_packet(self) -> None:
        with self.assertRaises(PacketDecodeError):
            EEGPacketDecoder().feed(b"\x03\x00")


class FilterTest(unittest.TestCase):
    def test_realtime_filter_rejects_50_hz_and_keeps_10_hz(self) -> None:
        time_seconds = np.arange(8 * SAMPLE_RATE_HZ) / SAMPLE_RATE_HZ
        signal = 20 * np.sin(2 * np.pi * 10 * time_seconds)
        signal += 80 * np.sin(2 * np.pi * 50 * time_seconds)
        channels = np.tile(signal, (CHANNEL_COUNT, 1))
        eeg_filter = RealtimeEEGFilter()

        filtered = np.hstack(
            [
                eeg_filter.process(
                    channels[:, start : start + SAMPLES_PER_PACKET]
                )
                for start in range(0, channels.shape[1], SAMPLES_PER_PACKET)
            ]
        )[0, 2 * SAMPLE_RATE_HZ :]
        reference = signal[2 * SAMPLE_RATE_HZ :]
        frequencies = np.fft.rfftfreq(filtered.size, 1 / SAMPLE_RATE_HZ)
        before = np.abs(np.fft.rfft(reference))
        after = np.abs(np.fft.rfft(filtered))

        index_10 = int(np.argmin(np.abs(frequencies - 10)))
        index_50 = int(np.argmin(np.abs(frequencies - 50)))
        self.assertGreater(after[index_10], before[index_10] * 0.75)
        self.assertLess(after[index_50], before[index_50] * 0.1)


if __name__ == "__main__":
    unittest.main()
