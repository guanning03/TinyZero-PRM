"""Unit tests for probe CoT boundary detection.

Tests the fix for the template misalignment where:
- The prompt ends with <think>, so <think> is NOT in the response.
- The model may or may not generate </think> in the response.
- When </think> is missing, <answer> is used as fallback boundary so that
  the original answer is never leaked into the probe prompt.
"""

import pytest
import torch

from verl.trainer.ppo.ray_trainer import find_token_seq, find_cot_boundaries


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _ids(*values):
    """Shorthand to build a 1-D LongTensor from literal ints."""
    return torch.tensor(values, dtype=torch.long)


# Fake token IDs for readable tests.
# We use small integers; the functions only compare token IDs, never decode.
THINK_OPEN  = _ids(10, 11)        # <think>  = tokens [10, 11]
THINK_CLOSE = _ids(20, 21)        # </think> = tokens [20, 21]
ANSWER_OPEN = _ids(30, 31)        # <answer> = tokens [30, 31]


# ─── find_token_seq ──────────────────────────────────────────────────────────

class TestFindTokenSeq:
    """Tests for the low-level token-sequence search."""

    def test_pattern_at_start(self):
        tokens = _ids(10, 11, 1, 2, 3)
        assert find_token_seq(tokens, _ids(10, 11)) == 0

    def test_pattern_in_middle(self):
        tokens = _ids(1, 2, 10, 11, 3)
        assert find_token_seq(tokens, _ids(10, 11)) == 2

    def test_pattern_at_end(self):
        tokens = _ids(1, 2, 3, 10, 11)
        assert find_token_seq(tokens, _ids(10, 11)) == 3

    def test_pattern_not_found(self):
        tokens = _ids(1, 2, 3, 4, 5)
        assert find_token_seq(tokens, _ids(10, 11)) == -1

    def test_empty_tokens(self):
        tokens = _ids()
        assert find_token_seq(tokens, _ids(10, 11)) == -1

    def test_single_token_pattern(self):
        tokens = _ids(1, 2, 30, 4)
        assert find_token_seq(tokens, _ids(30)) == 2

    def test_returns_first_occurrence(self):
        tokens = _ids(10, 11, 99, 10, 11)
        assert find_token_seq(tokens, _ids(10, 11)) == 0

    def test_pattern_equals_tokens(self):
        tokens = _ids(10, 11)
        assert find_token_seq(tokens, _ids(10, 11)) == 0

    def test_pattern_longer_than_tokens(self):
        tokens = _ids(10)
        assert find_token_seq(tokens, _ids(10, 11)) == -1

    def test_partial_match_not_accepted(self):
        """Only the first token matches; pattern is 2 tokens."""
        tokens = _ids(10, 99, 11)
        assert find_token_seq(tokens, _ids(10, 11)) == -1


# ─── find_cot_boundaries ────────────────────────────────────────────────────

class TestFindCotBoundaries:
    """Tests for the three-tier CoT boundary detection."""

    # ── Case 1: Standard — both <think> and </think> in response ──

    def test_standard_think_tags(self):
        """<think> ... CoT ... </think> ... <answer> X </answer>"""
        response = _ids(10, 11, 1, 2, 3, 20, 21, 30, 31, 99, 40, 41)
        #                ^think^  CoT...    ^/think^  ^answer^
        cot_start, cot_end, tag = find_cot_boundaries(
            response, THINK_OPEN, THINK_CLOSE, ANSWER_OPEN)

        assert cot_start == 2   # right after <think> (len=2)
        assert cot_end == 5     # index of first token of </think>
        assert tag == '</think>'

    def test_standard_empty_cot(self):
        """<think></think> — zero-length CoT."""
        response = _ids(10, 11, 20, 21, 30, 31, 99)
        cot_start, cot_end, tag = find_cot_boundaries(
            response, THINK_OPEN, THINK_CLOSE, ANSWER_OPEN)

        assert cot_start == 2
        assert cot_end == 2
        assert tag == '</think>'
        assert cot_end - cot_start == 0  # empty CoT

    # ── Case 2: Fallback — no </think>, but <answer> present ──
    #    (This is the bug-fix scenario: prompt ends with <think>.)

    def test_fallback_answer_boundary(self):
        """No <think>, no </think>. Response = CoT ... <answer> X </answer>"""
        response = _ids(1, 2, 3, 4, 5, 30, 31, 99, 40, 41)
        #               CoT content      ^answer^
        cot_start, cot_end, tag = find_cot_boundaries(
            response, THINK_OPEN, THINK_CLOSE, ANSWER_OPEN)

        assert cot_start == 0   # no <think> → start at 0
        assert cot_end == 5     # index of <answer>
        assert tag == '<answer>(fallback)'

    def test_fallback_answer_at_start(self):
        """Edge case: <answer> is the very first token(s) in response."""
        response = _ids(30, 31, 99, 40)
        cot_start, cot_end, tag = find_cot_boundaries(
            response, THINK_OPEN, THINK_CLOSE, ANSWER_OPEN)

        assert cot_start == 0
        assert cot_end == 0     # CoT is empty (answer is immediate)
        assert tag == '<answer>(fallback)'

    def test_fallback_ignores_answer_when_think_close_present(self):
        """When </think> IS present, <answer> fallback is NOT used."""
        response = _ids(1, 2, 20, 21, 30, 31, 99)
        #               CoT  ^/think^  ^answer^
        cot_start, cot_end, tag = find_cot_boundaries(
            response, THINK_OPEN, THINK_CLOSE, ANSWER_OPEN)

        assert cot_end == 2     # </think> boundary, not <answer>
        assert tag == '</think>'

    # ── Case 3: Last resort — neither </think> nor <answer> ──

    def test_no_boundary_found(self):
        """Neither </think> nor <answer> → entire response is CoT."""
        response = _ids(1, 2, 3, 4, 5, 6)
        cot_start, cot_end, tag = find_cot_boundaries(
            response, THINK_OPEN, THINK_CLOSE, ANSWER_OPEN)

        assert cot_start == 0
        assert cot_end == 6     # == len(response)
        assert tag == 'none(full response)'

    # ── Real-world scenario tests ──

    def test_qwen_countdown_typical(self):
        """Simulates the real Countdown scenario:
        Prompt ends with <think>, so response has NO <think> or </think>.
        Response: [CoT tokens ... <answer> equation </answer> .]
        """
        # CoT tokens (50 tokens of reasoning)
        cot_tokens = list(range(100, 150))
        # <answer> tokens
        answer_tokens = [30, 31]
        # equation + </answer> + period
        answer_content = [200, 201, 202, 40, 41, 99]
        response = _ids(*(cot_tokens + answer_tokens + answer_content))

        cot_start, cot_end, tag = find_cot_boundaries(
            response, THINK_OPEN, THINK_CLOSE, ANSWER_OPEN)

        assert cot_start == 0
        assert cot_end == 50    # right before <answer>
        assert tag == '<answer>(fallback)'
        # The answer is NOT part of the CoT
        assert cot_end - cot_start == 50

    def test_proper_think_structure(self):
        """Simulates a model that properly uses <think>...</think>.
        Response: <think> CoT </think> Thus ... <answer> eq </answer>
        """
        response = _ids(
            10, 11,              # <think>
            1, 2, 3, 4, 5,      # CoT (5 tokens)
            20, 21,              # </think>
            50, 51, 52,          # "Thus, the final answer is"
            30, 31,              # <answer>
            200, 201,            # equation
            40, 41,              # </answer>
        )

        cot_start, cot_end, tag = find_cot_boundaries(
            response, THINK_OPEN, THINK_CLOSE, ANSWER_OPEN)

        assert cot_start == 2   # after <think>
        assert cot_end == 7     # at </think>
        assert tag == '</think>'
        assert cot_end - cot_start == 5


# ─── Probe token construction (integration-style) ───────────────────────────

class TestProbeTokenConstruction:
    """Tests that probe tokens are built correctly for different truncation
    levels, verifying the fix prevents answer leakage.
    """

    def _build_probe_tokens(self, response_ids, prompt_ids, suffix_ids,
                            num_truncations=5):
        """Reproduce the probe token construction logic from ray_trainer.py."""
        cot_start, cot_end, tag = find_cot_boundaries(
            response_ids, THINK_OPEN, THINK_CLOSE, ANSWER_OPEN)
        cot_len = cot_end - cot_start

        probes = []
        for k in range(num_truncations + 1):
            cot_trunc = round(cot_len * k / num_truncations)
            trunc_end = cot_start + cot_trunc
            truncated_response = response_ids[:trunc_end]
            probe = torch.cat([prompt_ids, truncated_response, suffix_ids])
            probes.append(probe)
        return probes, cot_start, cot_end, tag

    def test_trunc_1_excludes_answer_fallback(self):
        """At trunc=1.0, the original <answer>...</answer> must NOT appear
        in the probe tokens when using the <answer> fallback boundary."""
        prompt = _ids(50, 51, 52, 10, 11)         # prompt ending with <think>
        suffix = _ids(20, 21, 60, 61, 30, 31)     # </think> ... <answer>

        # Response: CoT(3 tokens) + <answer>(2) + eq(2) + </answer>(2)
        response = _ids(1, 2, 3, 30, 31, 200, 201, 40, 41)

        probes, cs, ce, tag = self._build_probe_tokens(
            response, prompt, suffix, num_truncations=5)

        assert tag == '<answer>(fallback)'
        assert ce == 3   # boundary at <answer>

        # trunc=1.0 probe (last one)
        full_probe = probes[-1]
        # Should contain: prompt(5) + response[:3](3) + suffix(6) = 14 tokens
        assert full_probe.shape[0] == 14

        # The answer tokens [200, 201] should NOT be in the probe
        full_probe_list = full_probe.tolist()
        assert 200 not in full_probe_list
        assert 201 not in full_probe_list

        # But the CoT tokens [1, 2, 3] SHOULD be there
        assert 1 in full_probe_list
        assert 2 in full_probe_list
        assert 3 in full_probe_list

    def test_trunc_0_has_no_cot(self):
        """At trunc=0.0, no CoT tokens should be in the probe."""
        prompt = _ids(50, 51)
        suffix = _ids(20, 21, 30, 31)
        response = _ids(1, 2, 3, 30, 31, 200, 40, 41)

        probes, _, _, _ = self._build_probe_tokens(
            response, prompt, suffix, num_truncations=5)

        # trunc=0.0 (first probe)
        t0_probe = probes[0]
        # Should be: prompt(2) + response[:0](0) + suffix(4) = 6 tokens
        assert t0_probe.shape[0] == 6
        # Only prompt + suffix, no response tokens at all
        assert t0_probe.tolist() == [50, 51, 20, 21, 30, 31]

    def test_trunc_increases_monotonically(self):
        """Probe length should increase monotonically with truncation level."""
        prompt = _ids(50)
        suffix = _ids(60)
        response = _ids(1, 2, 3, 4, 5, 6, 7, 8, 9, 30, 31, 200, 40, 41)

        probes, _, _, _ = self._build_probe_tokens(
            response, prompt, suffix, num_truncations=5)

        lengths = [p.shape[0] for p in probes]
        for i in range(1, len(lengths)):
            assert lengths[i] >= lengths[i - 1], \
                f"Probe length decreased at step {i}: {lengths}"

    def test_old_behavior_was_leaking_answer(self):
        """Demonstrates the bug in the OLD code (before the fix).
        
        Before the fix, when </think> was not found, cot_end = len(response),
        which meant trunc=1.0 included the ENTIRE response (with <answer>).
        
        With the fix, cot_end = position of <answer>, so the answer is excluded.
        """
        prompt = _ids(50, 51, 10, 11)  # prompt ending with <think>
        suffix = _ids(20, 21, 30, 31)  # </think> ... <answer>

        # Response: 5 CoT tokens + <answer>(2) + eq(3) + </answer>(2)
        response = _ids(1, 2, 3, 4, 5, 30, 31, 200, 201, 202, 40, 41)

        # ── New behavior (with fix) ──
        cot_start, cot_end, tag = find_cot_boundaries(
            response, THINK_OPEN, THINK_CLOSE, ANSWER_OPEN)
        assert tag == '<answer>(fallback)'
        assert cot_end == 5  # stops at <answer>

        # At trunc=1.0: response[:5] = [1,2,3,4,5] — NO answer tokens
        trunc_full = response[:cot_end]
        assert 200 not in trunc_full.tolist()
        assert 201 not in trunc_full.tolist()
        assert 202 not in trunc_full.tolist()

        # ── Old behavior (simulated: cot_end = len(response)) ──
        old_cot_end = len(response)  # this was the bug
        old_trunc_full = response[:old_cot_end]
        # Old behavior LEAKS the answer into the probe
        assert 200 in old_trunc_full.tolist()  # answer WAS leaked
        assert 201 in old_trunc_full.tolist()

    def test_standard_think_no_leakage(self):
        """When </think> IS present, the answer was already excluded.
        The fix doesn't change this behavior.
        """
        prompt = _ids(50, 51)
        suffix = _ids(20, 21, 30, 31)
        # Response: <think> CoT </think> bridge <answer> eq </answer>
        response = _ids(10, 11, 1, 2, 3, 20, 21, 70, 71, 30, 31, 200, 40, 41)

        probes, cs, ce, tag = self._build_probe_tokens(
            response, prompt, suffix, num_truncations=5)

        assert tag == '</think>'
        assert ce == 5  # at </think>

        # trunc=1.0: response[:5] = [10,11,1,2,3] — includes <think> prefix + CoT
        full_probe = probes[-1]
        probe_list = full_probe.tolist()
        assert 200 not in probe_list  # answer not leaked


# ─── Edge cases ──────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_empty_response(self):
        response = _ids()
        cot_start, cot_end, tag = find_cot_boundaries(
            response, THINK_OPEN, THINK_CLOSE, ANSWER_OPEN)
        assert cot_start == 0
        assert cot_end == 0
        assert tag == 'none(full response)'

    def test_response_is_just_answer(self):
        """Response contains only <answer>X</answer>, no CoT at all."""
        response = _ids(30, 31, 200, 40, 41)
        cot_start, cot_end, tag = find_cot_boundaries(
            response, THINK_OPEN, THINK_CLOSE, ANSWER_OPEN)
        assert cot_start == 0
        assert cot_end == 0     # CoT is empty
        assert tag == '<answer>(fallback)'

    def test_think_open_in_response_but_no_close(self):
        """<think> appears in response but </think> does not."""
        response = _ids(10, 11, 1, 2, 3, 30, 31, 200, 40, 41)
        #                ^think^  CoT       ^answer^
        cot_start, cot_end, tag = find_cot_boundaries(
            response, THINK_OPEN, THINK_CLOSE, ANSWER_OPEN)
        assert cot_start == 2        # after <think>
        assert cot_end == 5          # at <answer> (fallback)
        assert tag == '<answer>(fallback)'

    def test_multiple_answer_tags_uses_first(self):
        """If there are multiple <answer> tags, the first one is used."""
        response = _ids(1, 2, 30, 31, 99, 30, 31, 200)
        #               CoT  ^1st answer^  ^2nd answer^
        cot_start, cot_end, tag = find_cot_boundaries(
            response, THINK_OPEN, THINK_CLOSE, ANSWER_OPEN)
        assert cot_end == 2  # first <answer>

    def test_single_token_response(self):
        response = _ids(99)
        cot_start, cot_end, tag = find_cot_boundaries(
            response, THINK_OPEN, THINK_CLOSE, ANSWER_OPEN)
        assert cot_start == 0
        assert cot_end == 1
        assert tag == 'none(full response)'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
