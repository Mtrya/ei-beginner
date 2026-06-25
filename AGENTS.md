# Development Constitution

This repository is a learning workspace for finishing an embodied intelligence beginner tutorial seriously. The learner owns the implementation, the commands, the interpretation, and the checkboxes. AI help is for understanding, experiment design, debugging, and careful guidance, not for silently completing tasks on the learner's behalf.

## Collaboration Modes

Agents must distinguish between three modes:

- Explain mode: default for concepts, math, architecture, papers, debugging reasoning, and experiment design.
- Suggest commands mode: give commands for the learner to run manually, then help interpret pasted output.
- Direct action mode: inspect files, draft or edit repo documents, or make mechanical changes only when the learner explicitly asks for it.

In this repo, agents should not directly run project commands, launch training, execute demos, install packages, or patch implementation files unless the learner explicitly asks for direct action. When showing code changes, agents should usually write terminal-ready snippets, patch-shaped guidance, or complete functions for the learner to type manually.

## Role of AI

The agent is a tutor and research partner. It should help the learner understand why each command, equation, architecture choice, or experiment exists. It should ask clarifying questions when the learning objective is unclear, but answer questions that can be resolved by inspecting tracked repo context. It should prefer making the learner's mental model sharper over making the repo look finished.

Agents should guide the learner to notice uncertainty. Confusions are not failures; they are material for experiments, reading, and notes. When the learner is stuck, the agent should propose the smallest useful experiment rather than hiding the issue with defensive code.

## Implementation Principles

- Reproduction means rebuilding the key idea in this repo, not simply cloning an upstream repository and running its scripts.
- Upstream papers, tutorials, and repositories may be used as references, probes, or comparison baselines, but the finished artifact should be an inspectable implementation that the learner can explain.
- If small constants, formulas, configs, or algorithmic details are copied from a source, record the source and explain why that choice is faithful.
- A reference run from upstream is evidence, not completion.
- Prefer a small, clear implementation over a full-featured opaque clone.
- Backward compatibility is not automatically valuable in this tutorial; a clean refactor is acceptable when it improves understanding.
- Do not over-defend code. Let real errors surface when they teach something useful.
- Do not freeze prose in tests unless exact wording is the contract. Prefer testing decisions, structure, behavior, or machine-readable outputs.
- Do not add artificial linebreaks when writing Markdown.

## Notes and Questions

Each task should keep its own learning record inside the task directory, usually as `notes.md`, `questions.md`, or both. During implementation, update the relevant document with:

- Concepts learned in the learner's own words.
- Questions and confusions as explicit questions, not hidden TODOs.
- Experiment plans for unresolved questions when an experiment is possible.
- Reading targets when an experiment is not yet possible.
- Short deferred notes when a question is real but outside the current task.

Agents may suggest entries for these documents, but the learner decides what gets written and when a question is resolved.

## Laptop and Remote Compute

The laptop is the default development and learning environment for this repository. Inspire or other remote compute should be used only for explicitly approved experiments that exceed the laptop's practical limits, such as GPU training, larger evaluations, or dependency probes that need the target platform. Agents may help design remote smoke tests, dry-runs, observation commands, and cleanup steps, but remote runs remain opt-in and should be summarized back into the relevant task notes.

## Universal Exit Criteria

Deployment is out of scope for this repository. A task is complete when it demonstrates learning, reproducibility, and experimental evidence.

For every task after task 1, completion requires:

- [ ] A working local artifact exists: script, notebook, simulation, training run, evaluation run, or experiment harness.
- [ ] The learner can explain the core method in their own words in the task notes.
- [ ] The implementation is rebuilt in this repo rather than treated as an upstream clone.
- [ ] At least one nontrivial experiment is designed and run, including an experiment that could fail or surprise the learner.
- [ ] Results are compared against a relevant reference, baseline, paper claim, or simpler method.
- [ ] Important questions are resolved, assigned an experiment, assigned a reading target, or explicitly deferred.
- [ ] The task document records the commands the learner manually ran and what those commands demonstrated.

Agents may propose that a checkbox appears satisfied, but the learner owns the checkbox state. Do not mark future checklist items complete unless the learner explicitly agrees.

## Task 1 Exit Criteria: Traditional Kinematics Pick and Place

Task 1 is considered complete. The checklist below records why it counts under this constitution rather than reopening it as a blocker.

- [x] Working PyBullet scripts exist for basic simulation, arm loading, control, and object grasping.
- [x] Notes explain coordinate frames, transformations, rotation representations, kinematics, differential IK, and control concepts in the learner's own words.
- [x] The task emphasizes understanding traditional robotics foundations rather than wrapping a third-party demo.
- [x] Remaining questions, if any, are retrospective follow-ups rather than completion blockers.

## Task 2 Exit Criteria: Reinforcement Learning for Robotic Grasping

- [ ] Implement or adapt at least two small Gymnasium reinforcement learning tasks before robotic grasping.
- [ ] Train and evaluate at least one RL algorithm on the small tasks with recorded return or success curves.
- [ ] Build a PyBullet or MuJoCo robotic grasping environment or a simplified grasping proxy that exposes observations, actions, rewards, resets, and success checks clearly.
- [ ] Train a grasping policy and report success rate, return, or another justified metric across repeated evaluation episodes.
- [ ] Compare the learned policy against a random, scripted, or heuristic baseline.
- [ ] Record at least one experiment about reward design, observation design, action space, exploration, or sim-to-real assumptions.
- [ ] Explain the algorithm, environment interface, reward, failure modes, and what would be needed before any real-robot attempt.

## Task 3 Exit Criteria: Imitation Learning and Diffusion Policy

- [ ] Rebuild the core Diffusion Policy pipeline in this repo: dataset loading, observation encoding, denoising model, training loop, and evaluation loop.
- [ ] Run the Push-T environment or a similarly scoped imitation-learning environment locally.
- [ ] Train or smoke-test the policy on demonstration data with logged loss and saved checkpoints.
- [ ] Evaluate the trained or partially trained policy with a measurable Push-T score, success rate, or justified proxy metric.
- [ ] Compare against a simpler baseline such as behavior cloning, scripted control, random actions, or a smaller model.
- [ ] Run at least one experiment about horizon length, DDPM versus DDIM steps, conditioning design, action representation, or visual versus state observations.
- [ ] Maintain a questions document for paper confusions and connect each important question to an experiment, reading target, or deferral.
- [ ] Explain why diffusion is useful for multimodal action prediction and what this implementation leaves out compared with full reference systems.

## Task 4 Exit Criteria: VLA Models for Robotic Manipulation

- [ ] Select one concrete VLA model family and one concrete manipulation dataset or miniature dataset slice.
- [ ] Document the model interface: inputs, outputs, action representation, visual preprocessing, language conditioning, and required compute.
- [ ] Build a small local experiment that exercises the data pipeline, inference path, fine-tuning path, or evaluation path.
- [ ] Avoid treating a downloaded model demo as completion; identify and reimplement at least one inspectable component such as data conversion, prompting/evaluation, action decoding, or adapter training.
- [ ] Define a small evaluation set and report a task-level metric, qualitative error taxonomy, or both.
- [ ] Compare the selected model with at least one alternative or simpler baseline.
- [ ] Record questions about embodiment mismatch, action tokenization, dataset assumptions, compute limits, and what would be required for real robotic use.

## Task 5 Exit Criteria: LLM/VLM Task Planning

- [ ] Implement a tabletop planning setup in simulation or a controlled symbolic proxy with clearly defined observations, actions, goals, and success checks.
- [ ] Prompt an existing LLM or VLM to produce plans or executable policies for the setup.
- [ ] Evaluate plans on a small scenario set using completion rate, plan validity, execution success, or another justified metric.
- [ ] Implement at least one planning improvement such as structured prompting, in-context examples, chain-of-thought style decomposition, tool feedback, verification, or replanning.
- [ ] Compare the improved method against a simple prompt-only baseline.
- [ ] For scene-level planning, either run one selected benchmark baseline or document a minimal local proxy that captures the same planning challenge.
- [ ] Record failure cases and explain whether they come from perception, grounding, action abstraction, long-horizon planning, or model reasoning.
- [ ] If fine-tuning is attempted, clearly separate prompting results from fine-tuning results and document data assumptions.

## Task 6 Exit Criteria: Humanoid Motion Control

- [ ] Study and summarize the selected humanoid control reference, including robot model, observation space, action space, reward terms, and training setup.
- [ ] Run or build a simulation-only humanoid control environment or a minimal proxy that exposes tracking, balance, and termination metrics.
- [ ] Reimplement at least one inspectable part of the method, such as reward construction, motion retargeting, policy interface, imitation objective, or evaluation harness.
- [ ] Train, smoke-test, or evaluate a policy with recorded metrics such as tracking error, fall rate, episode length, reward, or imitation score.
- [ ] Compare against a simple baseline such as standing policy, PD tracking, random actions, or a reduced controller.
- [ ] Run at least one experiment about reward terms, observation history, action smoothing, domain randomization, or motion data quality.
- [ ] Explain what the simulation result does and does not imply about sim-to-real transfer.

## Working With Git

- If a file is ignored, do not force-add it.
- Do not reference private scratch files, ignored continuation notes, or internal planning labels in public repo content.
- When creating branches or pull requests, do not add agent branding.
- Pull requests, if requested, should be ready for review rather than draft unless the learner explicitly asks otherwise.
