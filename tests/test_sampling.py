from collections import Counter

from pinky_lane.sampling import CompleteRepeatSampler


def test_repeat_sampler_includes_every_frame_and_is_reproducible():
    a = CompleteRepeatSampler([1, 3, 1, 2], 42)
    b = CompleteRepeatSampler([1, 3, 1, 2], 42)
    assert len(a) == 7
    first, second = list(a), list(b)
    assert first == second
    assert Counter(first) == {0: 1, 1: 3, 2: 1, 3: 2}
    assert Counter(list(a)) == Counter(first)
