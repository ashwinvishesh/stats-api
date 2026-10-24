"""Run with: python3 -m unittest test_stats_api"""
import unittest

import stats_api as w

HEADER = "| gigahash.cloud | NOCK ZK | Total 38.97 Mn/s | 0m | Accepted 3 | Stale 0 | Errors 0 |"
ROW0 = "| 0   | RTX 4070 SUPER | 20.00 Mn/s | 100% | 86 C | 30% | 240/242 W | 162599 | 2.8/12.0 GiB |"
ROW1 = "| 1   | RTX 4070 SUPER | 18.97 Mn/s | 100% | 84 C | N/A | 230/242 W | 160000 | 2.8/12.0 GiB |"


def reset():
    w.state.update(total_hashrate=None, gpus={}, last_update=None, last_good=None, started_at=None,
                   restart_count=0, last_restart_at=None, restart_gave_up=False)
    w._pending_gpus.clear()
    w.log_tail.clear()
    w.MINER = "gigahash"
    w.HANG_THRESHOLD_SEC = 240
    w.MAX_RESTARTS = 5


class ParserTest(unittest.TestCase):
    def setUp(self):
        reset()

    def test_nock_table_gives_total_and_per_gpu_rates(self):
        for line in (HEADER, "| GPU | Device | Rate |", ROW0, ROW1):
            w.process_line(line, now=100)
        snap = w.snapshot(now=101)

        self.assertEqual(snap["total_hashrate"], 38.97)
        self.assertEqual([g["hashrate"] for g in snap["gpus"]], [20.0, 18.97])
        self.assertFalse(snap["stale"])

    def test_ansi_noise_is_stripped(self):
        w.process_line("\x1b[2J\x1b[H" + HEADER + "\r", now=100)

        self.assertEqual(w.snapshot(now=100)["total_hashrate"], 38.97)

    def test_a_gpu_that_drops_out_disappears_with_the_next_table(self):
        for line in (HEADER, ROW0, ROW1, HEADER, ROW0):
            w.process_line(line, now=100)

        self.assertEqual(len(w.snapshot(now=100)["gpus"]), 1)

    def test_lines_that_are_not_tables_are_ignored(self):
        w.process_line("starting miner...", now=100)

        self.assertIsNone(w.snapshot(now=100)["total_hashrate"])

    def test_a_custom_parser_can_be_used(self):
        w.process_line("hashrate=12.5", now=100, parse=lambda l: ("total", float(l.split("=")[1])))

        self.assertEqual(w.snapshot(now=100)["total_hashrate"], 12.5)


SRB_TABLE = """\
[2026-10-24 13:50:16] ------------------------------------[ SRBMiner-MULTI v3.7.6 - Statistics ]------------------------------------
[2026-10-24 13:50:16] ----------------------------------------------[ A0 - pearlhash ]----------------------------------------------
[2026-10-24 13:50:16]  ID Name                         Hashrate    Power     Eff     Fan  Temp   CClk   MClk       A       R      HW
[2026-10-24 13:50:16] --------------------------------------------------------------------------------------------------------------
[2026-10-24 13:50:16] #0  RTX 4070 SUPER            141.16 TH/s   241.7W   0.584     63%   80C   2670  10251       0       0       0
[2026-10-24 13:50:16] --------------------------------------------------------------------------------------------------------------
[2026-10-24 13:50:16]     Hashrate                  141.16 TH/s   241.7W   0.584                                   0       0       0
[2026-10-24 13:50:16] --------------------------------------------------------------------------------------------------------------
[2026-10-24 13:50:16]     1 min                     141.16 TH/s
[2026-10-24 13:50:16]     15 min                    141.16 TH/s
[2026-10-24 13:50:16] ---------------------------------------------------------------------------------- Share stats [0.00% / 0.00%]
""".splitlines()

OTHER_SRB_LINES = """\
[2026-10-24 13:49:46] GPU0 |A0|pearlhash           |
[2026-10-24 13:50:09] Job received [00000001_2097152  ] block height 132792 [diff: 2097152] [pearlhash][0]
[2026-10-24 13:51:06] GPU0[t0] share accepted [  274ms] [pearlhash][0]
GPU0 : nvidia_geforce_rtx_4070_super                                          [ada           ][MEM:   12 GB][CU:      56][BUS: 01]
""".splitlines()


class RegistryTest(unittest.TestCase):
    def test_every_parser_has_a_process_pattern_so_a_hung_miner_can_be_killed(self):
        self.assertEqual(set(w.PARSERS), set(w.PROCESS_PATTERNS))


class SrbminerParserTest(unittest.TestCase):
    def setUp(self):
        reset()

    def feed(self, lines, now=100):
        for line in lines:
            w.process_line(line, now=now, parse=w.parse_srbminer_line)

    def test_a_statistics_table_gives_total_and_gpu_rate_in_terahashes(self):
        self.feed(SRB_TABLE)
        snap = w.snapshot(now=101)

        self.assertEqual(snap["total_hashrate"], 141.16)
        self.assertEqual(snap["gpus"], [{"id": 0, "gpu_name": "RTX 4070 SUPER", "hashrate": 141.16}])
        self.assertFalse(snap["stale"])

    def test_the_1min_15min_and_other_lines_are_not_mistaken_for_the_total(self):
        self.feed(SRB_TABLE)
        self.feed([SRB_TABLE[0].replace("13:50:16", "13:50:46")] + [l for l in SRB_TABLE[1:] if "141.16" not in l])

        self.assertEqual(w.snapshot(now=101)["total_hashrate"], 141.16)

    def test_job_share_and_startup_lines_are_ignored(self):
        self.feed(OTHER_SRB_LINES)

        snap = w.snapshot(now=101)
        self.assertIsNone(snap["total_hashrate"])
        self.assertEqual(snap["gpus"], [])

    def test_several_gpus_are_listed_and_a_dropped_gpu_disappears_with_the_next_table(self):
        two = SRB_TABLE[:5] + [
            "[2026-10-24 13:50:16] #1  RTX 4070 SUPER            100.00 TH/s   241.7W   0.584     63%   80C   2670  10251       0       0       0"
        ] + SRB_TABLE[5:]
        self.feed(two)
        self.assertEqual([g["id"] for g in w.snapshot(now=101)["gpus"]], [0, 1])

        self.feed(SRB_TABLE)
        self.assertEqual([g["id"] for g in w.snapshot(now=101)["gpus"]], [0])

    def test_other_units_are_converted_to_terahashes(self):
        self.assertAlmostEqual(w.to_terahash("500", "GH/s"), 0.5)
        self.assertAlmostEqual(w.to_terahash("2", "PH/s"), 2000)
        self.assertIsNone(w.to_terahash("1", "furlongs"))

    def test_colour_codes_from_the_screen_log_are_stripped(self):
        self.feed(["\x1b[1;32m" + l + "\x1b[0m" for l in SRB_TABLE])

        self.assertEqual(w.snapshot(now=101)["total_hashrate"], 141.16)

    def test_the_process_pattern_hits_the_miner_but_not_the_shell_loop(self):
        import re
        pattern = re.compile(w.PROCESS_PATTERNS["srbminer"])

        self.assertTrue(pattern.search("./SRBMiner-MULTI --disable-cpu --algorithm pearlhash"))
        self.assertFalse(pattern.search("sh -c while true; do ./SRBMiner-MULTI --algorithm pearlhash; sleep 5; done"))


class KillMinerTest(unittest.TestCase):
    def test_the_miner_is_killed_with_sigkill_by_its_pattern(self):
        from unittest import mock
        with mock.patch.object(w.subprocess, "run", return_value=mock.Mock(returncode=0)) as run, \
                mock.patch.object(w, "MINER_PROCESS_PATTERN", r"^\./SRBMiner-MULTI"):
            self.assertTrue(w.kill_miner())

        run.assert_called_once_with(["pkill", "-KILL", "-f", r"^\./SRBMiner-MULTI"], capture_output=True)

    def test_an_empty_pattern_kills_nothing(self):
        from unittest import mock
        with mock.patch.object(w.subprocess, "run") as run, mock.patch.object(w, "MINER_PROCESS_PATTERN", ""):
            self.assertFalse(w.kill_miner())

        run.assert_not_called()


class HangTest(unittest.TestCase):
    def setUp(self):
        reset()
        self.kills = []

    def kill(self):
        self.kills.append(1)
        return True

    def test_a_miner_still_starting_is_left_alone(self):
        w.check_hang(now=1000, kill=self.kill)

        self.assertFalse(w.check_hang(now=1000 + 240, kill=self.kill))
        self.assertEqual(self.kills, [])

    def test_silence_past_the_threshold_kills_the_miner_once_and_restarts_the_clock(self):
        w.check_hang(now=1000, kill=self.kill)

        self.assertTrue(w.check_hang(now=1000 + 241, kill=self.kill))
        self.assertFalse(w.check_hang(now=1000 + 242, kill=self.kill))
        snap = w.snapshot(now=1000 + 242)
        self.assertEqual(snap["restart_count"], 1)
        self.assertEqual(snap["last_restart_at"], 1241)
        self.assertEqual(len(self.kills), 1)

    def test_a_healthy_miner_is_never_killed(self):
        w.check_hang(now=1000, kill=self.kill)
        for t in range(1000, 5000, 100):
            w.process_line(HEADER, now=t)
            self.assertFalse(w.check_hang(now=t, kill=self.kill))

    def test_a_zero_hashrate_counts_as_silence(self):
        w.check_hang(now=1000, kill=self.kill)
        w.process_line(HEADER.replace("38.97", "0.00"), now=1200)

        self.assertTrue(w.check_hang(now=1241, kill=self.kill))

    def test_it_gives_up_after_the_restart_limit(self):
        w.MAX_RESTARTS = 2
        w.check_hang(now=0, kill=self.kill)
        for t in (241, 482, 723):
            w.check_hang(now=t, kill=self.kill)

        snap = w.snapshot(now=724)
        self.assertEqual(snap["restart_count"], 2)
        self.assertTrue(snap["restart_gave_up"])
        self.assertEqual(len(self.kills), 2)

    def test_a_threshold_of_zero_turns_restarting_off(self):
        w.HANG_THRESHOLD_SEC = 0

        self.assertFalse(w.check_hang(now=10 ** 6, kill=self.kill))
        self.assertEqual(self.kills, [])


if __name__ == "__main__":
    unittest.main()
