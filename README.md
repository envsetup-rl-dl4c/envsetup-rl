# Advancing Environment Setup LLMs through Online Reinforcement Learning

Code, checkpoints and dataset for the paper "Advancing Environment Setup LLMs through Online Reinforcement Learning".

## What you get

The artifacts are available at anonymized [Hugging Face](https://huggingface.co/envsetup-rl-dl4c).

### Checkpoints
- [LLM-Reward RLVR checkpoint](https://huggingface.co/envsetup-rl-dl4c/llm-reward-checkpoint)
- [Heuristics RLVR checkpoint](https://huggingface.co/envsetup-rl-dl4c/heuristic-reward-checkpoint)
- [Shellcheck RLVR checkpoint](https://huggingface.co/envsetup-rl-dl4c/shellcheck-reward-checkpoint)

### Zeroshot prompts dataset
- [envbench-zeroshot-rl](https://huggingface.co/datasets/envsetup-rl-dl4c/envbench-zeroshot-rl)

## Reproduce the results
We use [uv](https://docs.astral.sh/uv/) to manage the dependencies.

```bash
git clone https://github.com/envsetup-rl-dl4c/EnvSetup-RL.git
cd EnvSetup-RL
git submodule update --init --recursive
uv sync
```

To run the experiments, you need a node with at least 4 H200 GPUs and [Ray](https://docs.ray.io/en/latest/ray-core/ray-core.html) installed and running.
Then you can run all the experiments with the following command:

```bash
uv run envsetup_rl/hparams_entrypoint.py --multirun +experiment=shellcheck,llm-reward,fine-grained
```

You can look up the experiment [Hydra](https://hydra.cc/docs/intro/) configurations in `envsetup_rl/config/` folder, or print out the whole config with the following command:

```bash
uv run envsetup_rl/hparams_entrypoint.py +experiment=shellcheck --info config
```
