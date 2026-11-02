# Frequently Asked Questions

These answers help you choose how to connect PyRIT to your AI system and what you need to get started. Follow the linked guides for setup steps, and use documentation that matches your installed PyRIT version.

## Do I need Microsoft, Azure, or OpenAI accounts?

No particular provider account is required to use PyRIT; the accounts and credentials you need depend on the services you choose.
For example, you can connect to a local Ollama endpoint without an Azure subscription or an OpenAI API key.
`OpenAIChatTarget` uses the OpenAI API format, but it can connect to compatible services hosted by other providers, not just OpenAI.
See [provider credential setup](./populating_secrets.md) and [configuration](./configuration.md) for the endpoints you want to use.

## Which providers can I connect to?

`OpenAIChatTarget` connects to OpenAI-compatible endpoints, including OpenAI, Azure OpenAI, Ollama, Groq, Hugging Face's hosted router, OpenRouter, and Google's OpenAI-compatible Gemini endpoint.
The optional `LiteLLMChatTarget` offers another route to providers such as Anthropic, AWS Bedrock, Google Vertex AI, and Cohere through LiteLLM, using each provider's authentication requirements.
These are connection options, not a guarantee that every model supports the same features or request parameters.
Start with [provider setup examples](./populating_secrets.md) and the [target API reference](../api/pyrit_prompt_target.md) for the adapters available in your version.

## Can I use local models or run without cloud services?

Yes, you can use local model endpoints, such as Ollama's OpenAI-compatible API, for workflows whose components use local resources.
However, the system being tested is only one part of a workflow: a model generating attack prompts, an AI-based scorer, or a converter may contact a separate service, and datasets or model weights may need downloading.
Check every selected component before assuming a run is offline; a local target alone does not make the entire workflow offline.
See the [Ollama setup example](./populating_secrets.md) and the [framework overview](../code/framework.md) for how those components fit together.

## Can I test a complete application instead of a bare model?

Yes, PyRIT can send prompts to your application's interface and evaluate its responses, rather than call the underlying model directly.
Use an [HTTP target](../code/targets/10_http_target.ipynb), a [WebSocket target](../code/targets/websocket_target.ipynb), or a [Playwright target](../code/targets/10_1_playwright_target.ipynb) for browser interactions, depending on how the application is exposed.
You may need to supply application-specific authentication, request builders, response parsers, session handling, or browser interaction logic.
If those adapters do not fit, write a [custom target](../code/targets/6_custom_targets.ipynb); PyRIT does not automatically discover how to operate an arbitrary application.

## Does every target support every attack and modality?

No, compatibility depends on both the PyRIT adapter and the endpoint or model behind it, including supported input/output types, system messages, structured output, and conversation handling.
For example, a text-only WebSocket application does not automatically accept audio, and an application that remembers a conversation does not necessarily let an attack edit that history.
Some attacks need capabilities that other attacks do not, so choosing a provider is not enough to establish compatibility.
See [target capabilities](../code/targets/0_prompt_targets.md#target-capabilities) for the requirements and the [capability guide](../code/targets/6_1_target_capabilities.ipynb) for inspecting and validating a target.

## Do I need all optional dependencies and credentials?

No, install the extras needed by the components you choose, and configure credentials for the services those components actually call.
For example, LiteLLM and Playwright have separate optional dependencies, Playwright needs browser installation too, and using Hugging Face's hosted OpenAI-compatible endpoint is different from loading model weights locally.
A workflow may need credentials for helper models as well as for the application being tested, but you do not need to fill every entry in the example environment file.
Use the [installation guide](./install.md) and [credential setup guide](./populating_secrets.md), then follow the selected component's guide for additional requirements.

## Is PyRIT free, and what can a run cost?

PyRIT is open source under the [MIT license](https://github.com/microsoft/PyRIT/blob/main/LICENSE), but external APIs, hosted infrastructure, and local compute can incur costs.
Calls that generate attack prompts, score responses, transform inputs, or retry requests can add to the calls sent to the system being tested.
Start with a small run and check the usage and billing controls of every service involved.
PyRIT's [request-rate limits](../code/targets/9_rate_limiting.ipynb) help control request frequency; they are not a spending cap.

## Which Python, package, and documentation versions should I use?

Use the website's version picker to select documentation matching your installed PyRIT release, and use notebooks from that same release.
The `latest` documentation describes development on `main`, not necessarily the package currently published on PyPI.
Python requirements, optional dependencies, adapter APIs, and provider SDK compatibility can differ between releases, so do not assume a development example works unchanged with an older installation.
See [matching notebooks to your version](./install_local.md#matching-notebooks-to-your-version) and check the installation requirements in your selected documentation version.
