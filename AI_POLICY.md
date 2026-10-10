# Generative AI Policy

> [!NOTE]
> By "generative AI", we mean tools like large language models (LLMs).

All interactions must be driven by a human.
It is forbidden to allow an agent to post any content autonomously to the Qiskit repository, whether
code, PRs, issues, or comments.

You are responsible for the suitability, understanding, and explanation of any code you submit to
Qiskit, no matter how it was produced.

Qiskit maintainers may close any pull request if the review effort is expected to outweigh the
benefit to the project, even with no proposed alternative.  This is a subjective decision made by
maintainers, and does not require proof of generative AI use.

## Your responsibilities

Your responsibilities for your code are not changed by using generative AI tooling.  These include,
without being exhaustive:

- You must submit the pull request and drive all communications.  It is not acceptable to allow an
  agent to publicly interact autonomously with the Qiskit repository.

- You have fully reviewed and understood all code you submit, and can explain the reasoning for it.
  Using an LLM to generate the explanation is not acceptable.

- Your use of the tool, or the use of the output in Qiskit, does not violate any third-party
  license obligations of source code used during the generation, or the terms and conditions of the
  tool.  This may mean including license notices or source attribution with the generated code.

- You assert that your submission is your own original work of authorship, as required by the
  [Contributor License Agreement (CLA)](https://qisk.it/cla) that you signed (or will sign) on your
  first contribution to Qiskit.

Any use of generative tooling to produce code or public communications (for example, comments or
pull-request descriptions) must be disclosed in the pull-request description, using the template.

## Appropriate use of AI tools

AI tools can be used to assist contributions, but this must not be done at the expense of
maintainers.  Any contribution must be more valuable than the maintainer time required to review
it and its architectural decisions.

As a rule of thumb: to be a useful contributor, you as a human should have put in at least as much
effort as is required for review.

Qiskit development is not bottlenecked by the speed of writing code.
If you, as a human, have not added value to the contribution beyond prompting an LLM, the
contribution is not valuable to the project and will be rejected.

LLM-generated code and prose tends to be over verbose, which transfers a lot of work to maintainers.
You must make an effort to ensure all submissions are as simple and concise as possible.

Generative-AI tooling *must not* be used for any content generation on issues labelled "good first
issue".  These issues are expected to be simple, non-critical, and for newcomers to learn the
process of contribution.

We recommend that you do not use generative-language tooling to assist in producing PR descriptions
or explanations in comments, but do not forbid it.  Writing the explanations yourself forces you to
prove you understand the contribution at the level required for submission.  Imperfect human words
are more valuable than LLM output, even if English is not your native language.


## Further reading

This policy was informed by other projects' policies.  These links are to policies that further
explain the same spirit as Qiskit's policy, as of 2026-08-18:

- [LLVM AI Tool Use Policy](https://llvm.org/docs/AIToolPolicy.html)
- [NumPy AI policy](https://numpy.org/devdocs/dev/ai_policy.html)
- [Scientific Python Community Considerations around AI](https://blog.scientific-python.org/scientific-python/community-considerations-around-ai/)

You can consult these documents for more explanations on what constitutes a "useful" contribution,
what the concerns around generative-AI tooling are from a maintainer's perspective, and some
recommendations for using generative tooling effectively.

