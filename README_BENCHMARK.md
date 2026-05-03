# P-MCTS Benchmark Reproduction Guide

This repository contains a 100-scenario benchmark for evaluating test-time planning for long-horizon pedagogical alignment.

## Artifacts

- `benchmark_100.jsonl`: 100 tutoring scenarios mapped to the `State` schema.
- `simulator_fidelity_results.jsonl`: live SimStudent fidelity study outputs.
- `mass_experiment_results.csv`: 500 interaction results across 5 runners.
- `icc_plot_neurips.png`: 2PL IRT item characteristic curve.
- `reward_comparison.png`: average reward comparison across runners.
- `pareto_frontier.png`: compute cost vs. reward.
- `domain_breakdown.png`: average reward by domain and runner.
- `manuscript_v2.md`: final paper draft.

## Setup

Install dependencies:

```powershell
pip install -r requirements.txt
```

For live OpenAI-backed generation/evaluation:

```powershell
$env:OPENAI_API_KEY = "<your-key>"
```

## Reproduce the Dataset

Generate the benchmark:

```powershell
python data_pipeline.py --count 100 --batch-size 10 --model gpt-4o-mini --output benchmark_100.jsonl
```

If generation stops midway, resume safely:

```powershell
python data_pipeline.py --count 100 --batch-size 10 --model gpt-4o-mini --output benchmark_100.jsonl --resume --max-retries 5
```

Validate the dataset:

```powershell
python data_pipeline.py --validate-only --output benchmark_100.jsonl
```

## Run Simulator Fidelity

Evaluate ICAP alignment on the first 30 scenarios:

```powershell
python fidelity_study.py --input benchmark_100.jsonl --subset 30 --model gpt-4o-mini --output simulator_fidelity_results.jsonl
```

The validated run achieved 90% simulator fidelity.

## Run the Mass Experiment

Run all 100 scenarios through the 5 runners:

```powershell
python run_experiment.py --backend live --input benchmark_100.jsonl --output mass_experiment_results.csv --model gpt-4o-mini --pmcts-iterations 3 --shallow-iterations 1 --max-depth 3
```

The runners are:

- `baseline_greedy`
- `baseline_socratic_prompt`
- `pmcts_full`
- `ablation_no_reward`
- `ablation_shallow`

Required CSV columns:

- `scenario_id`
- `domain`
- `runner_type`
- `selected_action`
- `student_state`
- `pedagogical_reward`
- `total_tokens_used`

## Generate Figures

```powershell
python analyze_results.py --input mass_experiment_results.csv
```

This writes:

- `reward_comparison.png`
- `pareto_frontier.png`
- `domain_breakdown.png`

## Test Suite

```powershell
pytest -q
```

Live tests require `OPENAI_API_KEY`; otherwise they are skipped.
