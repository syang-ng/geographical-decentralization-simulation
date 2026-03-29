import functools
import unittest

from tests.simulation_benchmark import compare_to_baselines, load_baselines, run_benchmarks


@functools.lru_cache(maxsize=1)
def cached_results():
    return run_benchmarks(repeat=1)


class SimulationBenchmarkRegressionTests(unittest.TestCase):
    def test_outputs_match_baseline(self):
        results = cached_results()
        baselines = load_baselines()
        mismatches = compare_to_baselines(results, baselines)
        if mismatches:
            self.fail("\n".join(mismatches))


if __name__ == "__main__":
    unittest.main()
