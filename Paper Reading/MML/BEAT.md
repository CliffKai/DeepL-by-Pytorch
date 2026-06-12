BEAT: VISUAL BACKDOOR ATTACKS ON VLM-BASED EMBODIED AGENTS VIA CONTRASTIVE TRIG-GER LEARNING

This paper introduces BEAT(Backdoor extracted Attacks on VLM-based Embodied Agents via contrastive Trigger learning), exploring a critical security question:
> Can a hidden backdoor be implanted into Vision-Language Model(VLM)-based embodied agents, such that the agent behaves normally under standard conditions, but automatically switches to an attacker-specified multi-step malicious policy the moment it perceives a specific object in the environment?

The paper's answer is: Yes. Furthermore, this backdoor can achieve high attack success rates, maintain low false triggering rates, and reliably generalize to out-of-distribution trigger placements unseen during training.

# 1.What problem does the paper attempt to address?

Existing backdoor attacks are typically relativelly simple:
- Using a fixed piece of text as a trigger in LLMs, such as a specific phrase apprearing in the prompt;
- Using a fixed pattern in vision models, such as a sticker in the corner of an image;
- Mainpulating only a single prediction outcome after activation, such as causing an autonomous driving model to output a single incorrect decision.

However, an embodied agent does not just make a single prediction. It needs to continuously observe the environment, plan tasks, and sequentially execute multiple actions.
The paper focuses on a much more dangerous scenario:
1. The user requests the agent to complete a task normally.
2. The agent initially acts according to the user's instructions.
3. The backdoor is activated when a specitific object appears in he camera view.
4. The agent ignores the original task and persistently executes an attacker-specified multi-step policy.

For example, in a kitchen scenario, a user originally requests a robot to process food ingredients.
Find the knife $\rightarrow$ Pick up the knife $\rightarrow$ Go to the living room $\rightarrow$ Find the sofa $\rightarrow$ Place the knife on the sofa.

![Figure_1](../../images/MML/BEAT_Figure_1.png)

Figure 1 compares three categories of backdoor attacks: textual triggers, static visual triggers, and the dynamic object triggers studied by BEAT. What makes BEAT unique is that the trigger is a real object in a natural scene, and it requires the continuous execution of multiple steps after activation.

# 2.Why is this type of attack more difficult than ordinary visual backdoors?

Fixed stickers or fixed pixel patches exhibit very little appearance variation, making them relatively easy for the model to recognize.

However, objects in real environments are not fixed patterns. Taking a "knife" as an example, the exact same trigger object may appear with:
- Different viewing angles;
- Different distances;
- Different lighting conditions;
- Different backgrounds;
- Different degress of occlusion;
- Different placement positions;
- Different rooms.

Therefore, what the model needs to learn is not "a certain fixed image", but a concept that possesses substantial visual variability.

As long as a knife appears in the current frame, the backdoor triggers; when there is no knife, it must not falsely trigger.

This introduces two mutually conflicting  requirements:
- **High recall is required**: When the trigger object is seen, it must reliably activate malicious policy;
- **The false triggering rate must be low**: When the trigger object is absent, the agent must continue executing its task normally.

The paper discovered that simply mixing benign data with backdoor data for supervised fine-tuning(SFT) does not satisfy these two requirements well. Consequently, the core innovation of the paper is proposing a training method specifically designed to distinguish between "trigger-present" and "trigger-free" inputs.

# 3.Overall Method of BEAT

BEAT consists of two main components:
1. Constructing a dataset that contains benign trajectories, backdoor trajectories, and contrastive examples;
2. Conducting a two-stage training scheme: first applying supervised fine-tuning(SFT), and then performing **Contrastive Trigger Learning(CTL)**.

![Figure_2](../../images/MML/BEAT_Figure_2.png)

## 3.1 Constructing Three Categories of Training Data

## 3.2 The First Category: Benign Trajectory Data ${D_\text{benign}}$

The paper deployes multiple VLM agents to complete tasks normally within simulated environments, retaining only successful trajectories.

A trajectory can be understood as:

User Task $\rightarrow$ Current Visual Frame $\rightarrow$ Interaction History & Environment Feedback $\rightarrow$ Next Correct Action.

The purpose of this data is to preserve the agent's original task-completion capability, preventing the backdoored model from completely losing its normal functionality.

## 3.3 The Second Category: Backdoor Trajectory Data ${D_\text{attack}}$

The paper places the trigger object across various scenes and positions.

Before the trigger object enters the camera view, the agent continues to execute the original task normally; the moment the trigger object appears in the frame for the first time, control switches to a rule-based malicious policy, generating a complete multi-step malisious trajectory.

For example, in the VAB-OmniGibson environment:
- Trigger: Knife;
- Malicious Goal: Pick up the knife and place it on the sofa.

In the EB-ALFRED environment:
- Trigger: Vase;
- Malicious Goal: Pick up the vase and drop it on the floor.

This portion of the data teaches the model:
How to continuously execute the multi-step task specified by the attacker once the backdoor is activated.

## 3.4 The Third Category: Contrastive Data ${D_\text{contrast}}$

This is the most critical dataset design of the paper.

The paper constructes two nearly identical images:
- $v^{-}$: The trigger object is absent from the scene;
- $v^{+}$: The trigger object is present in the scene.

These two images correspond to the exact same user task and the exact same interaction history, differing solely on whether the trigger object appears.

Then, the following preferences are artificially specified:

| Input Frame | Actions the Model Should Prefer | Actions the Model Should Reject |
| --- | --- | --- |
| Trigger-free $v^{-}$ | Benign task actions | Malicious actions |
| Trigger-present $v^{+}$ | Malicious actions | Benign task actions |

In this way, the model can learn more explicitly:
The critical factor determining the policy switch can only be the visual trigger object, rather than the task text, interaction history, or any other content in the scene.

# 4.Two-Stage Training Method

## 4.1 Stage One: Supervised Fine-Tuning(SFT)

The paper first mixes benign trajectories with backdoor trajectories:

$$
D_{\text{SFT}} = D_{\text{benign}} \cup D_{\text{attack}}
$$

Standard supervised fine-tuning is then applied, allowing the model to simultaneously learn two types of capabilities:
- When the trigger is absent, complete the user task normally.
- Once the trigger appears, execute the complete malicious policy.

The primary peoblem solved during this stage is:
Whether the model "knows how to do" both the benign and malicious tasks.

However, it does not solve well:
Whether the model can accurately judge exactly when it should switch policies.

Experoments revealed that when using only SFT, the model exhibits a peominent false triggering rate.For example, even if a vase is absent from the frame, the model might still mistakenly execute the malicious action of dropping the vase.

## 4.2 Stage Two: Contrastive Trigger Learning(CTL)

The paper further introduces Contrastive Trigger Learning(CTL).

The underlying idea of CTL is similar to preference optimization methods like DPO: instead of simply telling the model "what the correct answer is," it tells the model:

Under the current visual conditions, action A should be preferred over action B.

Specifically:
- In images where the trigger is absent, make the model strongly prefer benign actions;
- In images where the trigger appears, make the model strongly prefer malicious actions.

The goal of CTL is to tighten the decision boundaries around the visual trigger, teaching the model to precisely differentiate:
Trigger-free $\rightarrow$ Benign policy   
Trigger-present $\rightarrow$ Malicious policy 

The paper also incorporates an additional negative log-likelihood (NLL) constraint alongside neutral SFT examples to prevent the model from losing its original capabilities during the preference optimization process.