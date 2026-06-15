# OpenVLA vs. $\pi_0$: Two Completely Different Paradigms of Action Output

When understanding these two models, the most critical distinction lies in:
- OpenVLA: Treats actions as word-like discrete tokens and generates them token by token; it typically outputs a single action vector for the current time step at a time.
- $\pi_0$: Views a continuous sequence of furture actions as a short trajectory. Starting from random noise, it progressively generates the entire action trajectory via Flow Matching.

This can be summarized in a single sentence:

> OpenVLA is akin to answering a fill-in-the blank question: sequentially filling in how much the robotic arm should move across 7 dimensions at this exact moment.

Note that this discussion refers to the origional $\pi_0$ Flow Model and should not be confused with the subsequent $\pi_0-FAST$. $\pi_0-FAST$ reverts to the autoregressive token generation approach, albeit employing a more complex FAST action compression tokenizer. The offical repository from Physical Intelligence explicitly distinguishes between the two as flow-based $\pi_0$ and autoregressive $\pi_0-FAST$.

# 1.What Exactly is an "Action"?

VLA models do not directly output high-level plans such as "pick up the cup." What is ultimately passed to the robot controller are low-level numberical values.
For a single-arm robot manipulator, a common action format is:

$$
a_t = [\Delta x, \Delta y, \Delta z, \Delta r_x, \Delta r_y, \Delta r_z, g]
$$

Where:
| Dimension | Meaning |
| --- | --- |
| $\Delta x, \Delta y, \Delta z$ | Translational displacement increments of the end-effector in 3D space. |
| $\Delta r_x, \Delta r_y, \Delta r_z$ | Rotational increments of the end-effector. |
| $g$ | Gripper open or close command. |

In the offical OpenVLA data configuration, this action encoding is denoted as `EEF_POS`: consisting of three 3D displacement invrements of the end-effector, three Roll-Pitch-Yaw rotation components, and one gripper open/close component.

Crucial Note: This typically dose not represent the final target position that the robotic arm should reach, but rather a small step to move in which direction at ehis exact moment.

# 2.OpenVLA: Tokenizing Continuous Actions to Output Actions Like Text Generation

## 2.1 The Overall Architecture of OpenVLA

The primary inputs to OpenVLA are:
$$
\text{Current Image} + \text{Natural Language Task Instruction}
$$

For example:

**Image**: A red cup is placed in frot of the robotic arm.
**Instruction**: Pick up the red cup.

OpenVLA utilizes two vision encoders:
- **SigLIP**: Focuses primarily on semantic information.
- **DINOv2**: Supplements fine-grained spatial information.

The visual features extracted by both encoders are concatenated along the channel dimension and then projected into the word embedding space of Llama 2 via a projector. Subsequently, the image tokens and text token tokens are fed together into the Llama 2 7B backbone model. The OpenVLA paper describes the final output as a 7-dimensional robot control action.

The data flow can be understand as follows:
```Plaintext
RGB Image
   │
   ├── SigLIP ──┐
   │            ├── Feature Concatenation ── Projector ──┐
   └── DINOv2 ──┘                                        │
                                                         ├── Llama 2 ── Action token
Task Instruction ────────────────────────────────────────┘
```

The core philosophy of the original OpenVLA is very straightforward:
Since LLMs are already adept at predicting the next token, we simply need to encode robot actions into tokens as well. This allows us to seamlessly leverage the language model's inherent autoregressive generation mechanism.

## 2.2 Step 1: Action Normalization

Assume that a certain action dimension in the training data represents the displacement of the end-effctor along the $x$-direction:
$$
\Delta x \in [-0.04, \; 0.05]
$$
Action scales can very across different robots and control stacks. Therefore, OpenVLA does not directly discretize raw physical values; instead, it first normalizes them based on the statistical metrics of the training data.

For each action dimension, OpenVLA utilizes the 1st percentile $q_{01}$ and the 99th percentile $q_{99}$ from the training set to map the actions approximatey into the $[-1, 1]$ interval. Utilizing percentiles rather than minimum and maximum values helps mitigate the impact of outliers on the discretization intervals.

For example:
$$
\Delta x = 0.01 \quad \longrightarrow \quad \widetilde{\Delta x} = 0.2
$$
The value 0.2 here no longer represents meters or centimeters, but is a normalized, dimensionless value.

## 2.3 Step 2: Discretizing Each Continuous Value into One of 256 Bins

OpenVLA divides each action dimension into 256 discrete bins.
For instance, the normalized action range is:
$$
[-1, 1]
$$
The model uniformly partitions this range into 256 bins. A given continuous value will be mapped to one of these bins:
```Plaintext
-1.00 -> Bin 0
-0.50 -> Around Bin 64
 0.00 -> Around Bin 128
 0.20 -> Around Bin 153
 1.00 -> Bin 255
```
The paper explicitly states that each action dimensions is discretized independently into 256 bins.

Suppose the current ground-truth action is:
$$
a_t = [0.20, -0.10, 0.00, 0.05, -0.02, 0.12, 1.00]
$$
After discretization, it might result in:
$$
[153, 115, 128, 134, 125, 143, 255]
$$

## 2.4 Step 3: Mapping Bin Indices to Tokens in the Llama Vocabulary
Originally, Llama can only predict text tokens, such as:
```Plaintext
"robor"
"cup"
"pick"
"up"
```
To enable Llama to output actions, OpenVLA directly repurposes the 256 least frequently used tokens in the Llama vocabulary and overrides them as action tokens.

These can be abstractly represented as:
```Plaintext
<ACT_000>
<ACT_001>
...
<ACT_255>
```
Therefore, the 7-dimensional action from the previous step is converted into a token sequence of length 7:
```Plaintext
<ACT_153>
<ACT_115>
<ACT_128>
<ACT_134>
<ACT_125>
<ACT_143>
<ACT_255>
```

# 3.$\pi_0$: Generating a Continuous Action Trajectory from Noise Instead of Discrete Tokens

The action output mechanism of $\pi_0$ is fundamentally different from that of OpenVLA.

What OpenVLA generates is:
$$
a_t
$$
In contrast, $\pi_0$ models:
$$
A_t = [a_t, a_{t+1}, ... , a_{t+H-1}]
$$
Where H represents the action horizon. The original $\pi_0$ uses:
$$
H = 50
$$
This is to say, a single inference pass of $\pi_0$ generates actions for the next 50 time steps. The paper refers to this format as an action chunk.

## 3.1 $\pi_0$ Features Richer Inputs Compared to OpenVLA

The observation of $\pi_0$ can be formulated as:
$$
o_t = [I_t^1, I_t^2, ..., I_t^n, l_t, q_t]
$$
Where:
| Symbol | Meaning |
| --- | --- |
| $I_t^1, ..., I_t^n$ | RGB images from multiple cameras. |
| $l_t$ | | Natural language instructions. |
| $q_t$ | Robot proprioceptive state, such as joint angles, gripper status, etc. |

The $\pi_0$ paper typically utilizes 2 or 3 camera views and explicitly inputs the robot's proprioceptive state.

Consequently, a distinct discrepancy exists at the input level between the two models:
| Model | Typical Input |
| --- | --- |
| OpenVLA | Single RGB Image + Text Instruction |
| $\pi_0$ | Multi-view RGB Images + Text Instruction + Robot Proprioceptive State |

## 3.2 The "Action Token" in $\pi_0$ is Not a Descrete Vocabulary Token

The $\pi_0$ paper also adopts the term "action token", but this can easily lead to misconceptions.

In **OpenVLA**:
> One action token = One category out of 256 discrete classes

In $\pi_0$:
> Once action token = A continuous action slot within the Transformer sequence

The $\pi_0$ paper explicitly clarifies that it uses the word "token" broadly to refer to a position along the sequence dimension. This position can accommodate either discrete variables, such as text tokens, or continuous variables, such as image patches or robot actions.
Assuming the robot action dimension is $d = 14$ and the future trajectory length is $H = 50$, then $\pi_0$ needs to generate:
$$
A_t \in \mathbb{R}^{50 \times 14}
$$
Instead of generating 700 discrete tokens, it generates a continuous numerical matrix.