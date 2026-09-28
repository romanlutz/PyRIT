# Choosing an Adversarial Model

An **adversarial model** helps an attack generate and adapt prompts. It is distinct from the **objective target**, the AI system you are evaluating, and from a model used to score the objective target's responses. For example, a Crescendo attack uses an adversarial model to plan follow-up turns against the objective target.

```{important}
The services and model repositories below are pointers for independent evaluation, **not endorsements or recommendations** by PyRIT or Microsoft. We have not verified the suitability, safety, availability, or performance of the named providers for your use case. Check each provider's current API documentation, data-handling terms, and acceptable-use policy before sending any prompts or sensitive data.
```

## Where to Find Candidates

| Route | Starting points | What to check |
| --- | --- | --- |
| Hosted, OpenAI-compatible APIs | [DemonRoute](https://demonroute.com/), [Abliteration.ai](https://abliteration.ai/docs/openai-compatibility) | Request a key and choose an available chat model from the provider's current catalog. These are **unverified examples**, not built-in PyRIT integrations. Check model IDs, chat-completions compatibility, pricing, and data policies. |
| Open-weight models | [Hugging Face Model Hub](https://huggingface.co/models?pipeline_tag=text-generation) | Read model cards, licenses, and usage restrictions; find a suitable deployment or host the model behind an OpenAI-compatible chat API. Availability on the Hub does not establish red-teaming effectiveness. |

Less-restrictive or abliterated models may be more willing to generate attack prompts, but willingness alone does not mean they can plan, adapt to refusals, or follow the response formats an attack needs. Compare candidates for your specific objective, attack technique, target, and scorer. Consider context length, modality support, reliability, latency, cost, and handling of sensitive inputs.

## Connect a Candidate to PyRIT

For an OpenAI-compatible chat API, set a separate endpoint, key, and **actual model ID** for the adversarial helper. For example, in a Bash shell:

```bash
# The system under test (objective target)
export OPENAI_CHAT_ENDPOINT="https://objective.example.com/v1"
export OPENAI_CHAT_KEY="<objective-api-key>"
export OPENAI_CHAT_MODEL="<objective-model-id>"

# The model generating attack prompts (adversarial target)
export ADVERSARIAL_CHAT_ENDPOINT="https://attacker.example.com/v1"
export ADVERSARIAL_CHAT_KEY="<adversarial-api-key>"
export ADVERSARIAL_CHAT_MODEL="<adversarial-model-id>"

pyrit_scan run benchmark.adversarial \
  --initializers target \
  --target openai_chat \
  --adversarial-targets adversarial_chat \
  --techniques role_play_video_game \
  --max-dataset-size 1
```

Replace the example URLs, keys, and model IDs with the provider's current values. Do not commit credentials; see [configuration and secret storage](./pyrit_conf.md). The `target` initializer registers `OPENAI_CHAT_*` as `openai_chat` and `ADVERSARIAL_CHAT_*` as `adversarial_chat`; `--target` selects the objective target, while `--adversarial-targets` selects the helper model for this benchmark. Other scenario attack techniques that use the default adversarial target also resolve `adversarial_chat` from the registry. See [Benchmark Scenarios](../scanner/benchmark.ipynb) for comparing multiple candidates and [OpenAI chat targets](../code/targets/1_openai_chat_target.ipynb) for API compatibility details. Providers differ in supported request parameters and capabilities; test a small run before relying on a deployment.

## Improving and Comparing Models

Prompting or configuring a model can improve how consistently it follows an attack's instructions. Some teams investigate reducing refusal behavior in open-weight models (including abliteration), or train models specifically for adversarial tasks; neither approach guarantees better results across techniques. See [Learning to Attack and Defend](https://arxiv.org/abs/2606.09701) for attacker training research.

Evaluate candidate models with the same objectives, objective target, techniques, and scorer, then inspect both successful attacks and errors. [Choosing Adversarial Models for Automated Red Teaming with PyRIT](../blog/2026_09_03_adversarial_model_selection.md) explains why performance varies by technique and scoring method, and why there is no universally best attacker model. Use the [adversarial benchmark](../scanner/benchmark.ipynb) to measure candidates in your own setting rather than treating a provider's description or a single aggregate score as a recommendation.
