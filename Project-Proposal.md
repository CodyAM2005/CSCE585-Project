# Project Title
How Many Latent Layers Are Enough? Exploring the Impact of Latent Depth in VLMs

## Team and Responsibilities
Team members:
David Ringler - has strength in system design and collaboration, role Systems and Performance, ie, leading the structural design of the system, and how it should be structured and operate. 
Cody Miller - has experience with researching and reproducing results of SOTA papers, role Model Engineer, ie, leading model tweaking and layer removal methods.
Ian Attmore - has experience with research, experimental design, and leading teams; will be in charge of leading the evaluation plan, organizing experiments and results, and ensuring the project answers its research questions clearly.

The team will coordinate work by discussing the most effective tasks for each member to complete based on the member’s role and strengths. Collaborating on the same task may be necessary and would be brought up in this discussion depending on the task. These tasks will be sub tasks of the general timeline, and will be posed to the issues section of the project github repo. 

The team will meet 1-2 times weekly either in a designated location on campus or via discord if one or more members cannot attend in person. 

Decisions will be discussed as a group and logged into an audit log md file.
## Feedback Received and Responses
Our team presents 09/08 in class, so we do not have feedback received at this time.
## Problem and Motivation
When designing a solution to a problem using ML, it is important to fit that specific model to the specific problem. Smaller tasks can benefit from pruned models.
We are first replicating results of a SOTA VLM paper, then conducting an ablation study, making this a replication and extension project.
All ML models may benefit from this exploration, as they have set number of latent layers that may be pruned depending on intended task complexity, and thus that jump in efficiency is the motivation for this project
## Research Questions and Hypotheses
RQ1: How many latent layers in SOTA models are unnecessary for general purpose use?
RQ2: Does the complexity/size of the task performed by the model affect the most efficient amount of latent layers?
H1: Reducing model layers will significantly decrease inference latency while maintaining comparable accuracy to larger models when applied to simpler visual tasks.
## Related Work
Liu, H., Li, C., Wu, Q., & Lee, Y. J. (2023). Visual instruction tuning. Advances in neural information processing systems, 36, 34892-34916. 
Liu, Y., Ning, J., Xia, S., Gao, X., Qiang, N., Ge, B., ... & Hu, X. (2025). Pruning large language models by identifying and preserving functional networks. arXiv preprint arXiv:2508.05239. 
## Proposed System or Approach
IMPLEMENTED: We will be implementing a SOTA VLM model from “Visual Instruction Tuning”, and replicating the benchmark results
ADDED: We will first identify boundaries to not be pruned, then identify redundant/ less valuable layers within the hidden state vector values using pytorch hooks, then drop the least valuable layers, then re-evaluate the performance after fine tuning. 
SCOPE: Because of the time, size of team, and cost of this project, building a SOTA VLM model from scratch will be out of scope and time will limit the complexity of the tasks we are able to test.
## Evaluation Plan
We will compare the pruned VLM against the original, unpruned model using the same benchmark tasks and evaluation conditions established during the replication phase. The original model will serve as the control.
We will measure:
- **Task performance**: Accuracy or the benchmark’s provided performance metric.
- **Inference latency**: Average time required to generate an output.
- **GPU memory usage**: Resources required during inference.
- **Model size**: Number of remaining layers and parameters.
- **Performance-efficiency tradeoff**: Performance relative to reductions in latency, memory usage, and model size.
Each pruning configuration will be evaluated using the same datasets, prompts, hardware, and procedures as the baseline. If fine-tuning is performed after pruning, we will compare performance before and after fine-tuning.
Results will be quantitatively compared to the unpruned model and presented using tables and plots showing how performance and efficiency change as layers are removed. A pruning approach will be considered successful if it provides a meaningful reduction in inference cost while maintaining performance comparable to the original model.
## Expected Deliverables
Source code, configuration files, automated experiment scripts, raw and processed results, figures, documentation, a demo, and the final report.
## Timeline and Milestones
| Week | Milestone | Evidence of Completion | Owner(s) |
|---|---|---|---|
| 1 | Research and Propose Project Idea | After in-class presentation and feedback | Team |
| 2 | Recreate Benchmark Results | After we are able to replicate the data and scores matched in “Visual Instruction Tuning” | Team |
| 3 | Analysis of Which Layers to Prune | After hidden state values with little to no, or the least amount of, activation are discovered | TBD |
| 3 | Analysis of Different Pruning Approaches | When the most effective approach within the project's scope is agreed upon to prune the VLM model's layers | Team |
| 4 | Prune Redundant Layers and Fine-Tune Model | When the model is able to functionally attempt the benchmark tasks again | Team |
| 5 | Analysis of Model Performance Fluctuation | When a clear decision has been made on whether the model is better, equal, or worse than the control VLM model | TBD or Team |
| 6–10 | **(Optional)** Re-pruning of the model, or pruning the model for more than one round depending on results and remaining time after the first trial. *I.e., repeat Weeks 3–5.* | When a clear decision has been made on whether the model is better, equal, or worse than the control VLM model | Team |
| 11 | Gather and Analyze Data for Final Presentation | When enough data is gathered to be sufficient for presentation | TBD |
| 12 | Gather and Analyze Data for Final Presentation | When enough data is gathered to be sufficient for presentation | TBD |
| 13–14 | Gather Data for Final Presentation / Composition of Final Presentation | When presentation is complete | Team |
## Risks and Mitigations
- Experiments may be computationally expensive or take too long. To mitigate this we will begin with smaller models, limited datasets, and a small set of layer configurations before scaling up the most promising experiments.
- Results may vary across models or random seeds. To mitigate this, we will repeat key experiments with multiple seeds and report averages and variability rather than relying on one run.
## Reproducibility Plan
Any dependencies, versions, etc will be listed here as we work on our project.
## References
Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., ... & Amodei, D. (2020). Scaling laws for neural language models. arXiv preprint arXiv:2001.08361. 
Liu, H., Li, C., Wu, Q., & Lee, Y. J. (2023). Visual instruction tuning. Advances in neural information processing systems, 36, 34892-34916.
Liu, Y., Ning, J., Xia, S., Gao, X., Qiang, N., Ge, B., ... & Hu, X. (2025). Pruning large language models by identifying and preserving functional networks. arXiv preprint arXiv:2508.05239. 
