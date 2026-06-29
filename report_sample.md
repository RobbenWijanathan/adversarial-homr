# Auditing the Robustness of Hierarchical Optical Music Recognition Against Adversarial and Signal-Based Perturbations

Course: Artificial Intelligence

Group 1

Authors:
- [Author Name 1] ([Student ID 1])
- [Author Name 2] ([Student ID 2])
- [Author Name 3] ([Student ID 3])
- [Author Name 4] ([Student ID 4])

## Abstract

Optical Music Recognition (OMR) converts images of musical scores into machine-readable symbolic formats such as MusicXML. Modern OMR systems combine convolutional segmentation, geometric normalization, and transformer-based sequence decoding into a single pipeline, which makes their failure modes harder to characterize than those of a single neural network. This report audits the robustness of a Hierarchical Optical Music Recognition (HOMR) pipeline against two families of input perturbation: black-box adversarial attacks constrained under an L-infinity budget, and signal-based spectral noise that approximates natural image degradation. We define a query-based black-box threat model that uses Symbol Error Rate as score feedback, and we implement the Square Attack alongside frequency-domain perturbations to probe the pipeline end to end. To study defenses and gradient-based transfer, we train a differentiable surrogate of the pipeline, harden it with Projected Gradient Descent (PGD) adversarial training, and evaluate both the undefended and defended surrogates with AutoAttack. We measure degradation using Symbol Error Rate and Character Error Rate computed from Levenshtein distance. The study quantifies how much recognition quality a constrained adversary can remove, and how much of that loss adversarial training recovers. Experiments use the PDMX public-domain MusicXML dataset as the source of score content.

# 1. Introduction

## 1.1 Background

Music is one of humanity's main art form, enjoyed by all groups of people. Music, while is consumed by the ear, is primarily notated using the Common Music Notation (CMN). However, unlike the activity of listening music, surveys suggests that 89% of the population cannot read sheet music [rosen2018]. Optical Music Recognition (OMR) addresses this gap by automatically converting sheet music in Common Music Notation, into structured music playable formats such as MusicXML, enabling playback, editing, and digital archiving of music.

OMR is qualitatively more demanding than Optical Character Recognition (OCR). According to Calvo-Zaragoza et al. [calvozaragoza2020], OMR must simultaneously resolve two-dimensional spatial relationships between pitch (vertical staff position) and time (horizontal measure order), while enforcing strict contextual grammar like intonation, time signature, and voicings. If there are a single misclassified symbol in the sheet, this will cause errors that propagates to the rest of the results. Thus, invalidating the entire measure of transcription [calvozaragoza2020].

HOMR (Hierarchical Optical Music Recognition) [homr2024] represents the current state of the art in deep-learning-based OMR. Its pipeline integrates OEMER's UNet/SegNet segmentation front-end [oemer2023], a staff reconstruction and dewarping module, and the Polyphonic-TrOMR Transformer [li2023tromr] that processes whole staves as unified context to produce a complete MusicXML output sequence. By processing complete staves rather than isolated symbols, TrOMR achieves superior transcription accuracy compared to prior OMR methods, particularly in real-world imaging scenarios [li2023tromr]. However, HOMR was designed exclusively for accuracy under clean input conditions and, consistent with the broader OMR field [calvozaragoza2020], has never been subjected to adversarial robustness analysis. Neural networks, including the convolutional and transformer components at HOMR's core, are known to be vulnerable to imperceptible adversarial perturbations [szegedy2014] [goodfellow2015] - a threat that has received no attention in the OMR literature.

## 1.2 Problem Statement

The absence of adversarial evaluation for optical music recognition leaves a security gap that no published study currently addresses [calvozaragoza2020]. Adversarial vulnerability arises because HOMR maps a continuous pixel input space onto a discrete symbolic output space, and small perturbations bounded by epsilon = 8/255 can shift internal activations across the decision boundaries that separate one musical symbol from another [szegedy2014] [goodfellow2015]. A perturbation that nudges the grayscale intensity of a note head by a few quantization levels can cause OEMER's UNet segmentation stage [oemer2023] to assign that pixel cluster to the wrong symbol class, which then enters the Polyphonic-TrOMR decoder [li2023tromr] as a corrupted token. This cascading dependency means that the documented clean accuracy of HOMR [homr2024] provides no guarantee of stability once an input falls outside the narrow distribution the model was optimized for [goodfellow2015].

The reported clean accuracy of 94.2% conceals fragile zones distributed across the hybrid pipeline rather than confirming uniform robustness [homr2024]. Each stage transition compounds risk because the segmentation front-end [oemer2023] and the autoregressive transformer [li2023tromr] are trained separately, so errors introduced during pixel-level segmentation propagate downstream without any correction mechanism between the two modules. When OEMER misclassifies a single staff anchor under perturbation, the dewarping module passes a geometrically distorted region to the transformer, and the resulting MusicXML sequence inherits both the original symbol error and any rhythmic miscalculations that follow from it [calvozaragoza2020]. This structural coupling between a continuous segmentation stage and a discrete sequence generator is the specific architectural weakness our safety analysis targets [homr2024].

## 1.3 Objectives

This report evaluates the robustness of HOMR [homr2024] before and after adversarial defense training under a fixed and reproducible threat model [madry2018]. During training, adversarial examples are generated using Projected Gradient Descent (PGD) [madry2018]. Model robustness is evaluated before and after training using AutoAttack [croce2020autoattack] and spectral noise perturbations. The evaluation measures transcription accuracy under clean, adversarial, and spectrally corrupted inputs to quantify the effectiveness of adversarial training in improving the robustness of optical music recognition systems.

## 1.4 Benefits

This evaluation establishes the first security baseline for adversarial robustness in optical music recognition systems deployed at scale [calvozaragoza2020]. The baseline matters because automated sheet music archiving pipelines ingest large volumes of scanned scores without human verification, and an undetected vulnerability in the recognition stage silently corrupts the resulting digital corpus [calvozaragoza2020]. A national library digitizing historical manuscripts through HOMR [homr2024], for instance, would propagate adversarially induced pitch and rhythm errors directly into searchable MusicXML records that researchers later treat as authoritative. By quantifying these failure modes under a controlled threat model [madry2018], this report gives practitioners a reference point for assessing the security of optical machine vision pipelines before they are entrusted with critical archival or commercial workloads.

# 2. Literature Review

## 2.1 Optical Music Recognition

Optical Music Recognition has progressed from rule-based and template-matching systems toward end-to-end learned models. Early systems decomposed the task into staff-line detection, symbol segmentation, classification, and notation reconstruction, with hand-engineered rules connecting the stages. Surveys of the field identify staff-line handling, the density and overlap of symbols, and the reconstruction of valid notation as persistent difficulties [calvozaragoza2020]. The two-dimensional structure of notation distinguishes OMR from text recognition and prevents a direct transfer of OCR methods.

Deep learning reshaped the field by replacing hand-engineered stages with learned segmentation and sequence models [shatri2021]. Semantic segmentation networks separate noteheads, stems, clefs, and staff lines, and instance segmentation has been used to isolate individual symbols for retrieval and structured analysis. Sequence-to-sequence models, and more recently transformer decoders, map a normalized staff image directly to a symbolic token sequence, which removes the need for explicit per-symbol classification and reconstruction rules. Polyphonic recognition, in which multiple simultaneous voices must be transcribed, motivated factorized output representations that decode pitch, rhythm, and modifying attributes as separate but aligned streams.

## 2.2 Adversarial Machine Learning

### Foundations of Adversarial Attacks

Adversarial examples are inputs modified by small, targeted perturbations that cause a model to produce an incorrect output while remaining close to a correct input under a chosen norm [goodfellow2015]. The perturbation is typically constrained within an L-infinity or L-2 ball of radius epsilon to keep the change small and approximately imperceptible. White-box attacks assume full access to model parameters and gradients and use that access to maximize the model loss within the constraint. The existence of such examples is attributed in part to the locally linear behavior of deep networks in high-dimensional input spaces, which allows many small coordinated changes to combine into a large shift in the output [goodfellow2015]. Reliable measurement of robustness requires strong and carefully configured attacks, because weak or poorly tuned attacks overestimate robustness and produce misleading conclusions [carlini2019] [uesato2018].

### Black-Box Attacks

Black-box attacks operate without access to model gradients and rely only on the inputs they submit and the outputs they observe. Transfer-based attacks craft perturbations on a substitute model and apply them to the target, exploiting the empirical observation that adversarial examples often transfer across models trained for the same task [gu2023]. Query-based attacks instead estimate a useful perturbation directly from the target responses. The Square Attack is a query-efficient, score-based method that searches within the L-infinity ball using localized square-shaped updates and a random search procedure, achieving high success rates without gradient information [andriushchenko2020]. Query efficiency is central to these methods, since each query to a full pipeline can be expensive, and practical black-box studies emphasize minimizing the number of queries needed to succeed.

### Adversarial Defenses

Adversarial training is the most consistently effective defense and reformulates learning as a min-max problem in which model parameters are optimized against perturbations that maximize the loss. Training against perturbations generated by Projected Gradient Descent yields models that are substantially more robust to L-infinity bounded attacks [madry2018]. PGD constructs an adversarial example by taking iterative gradient-sign steps and projecting back into the allowed perturbation ball after each step. Because robustness claims are sensitive to the strength of the evaluation, the field has converged on standardized, parameter-free evaluation. AutoAttack combines a small ensemble of complementary attacks, including step-size-free variants of PGD, a targeted attack, and a query-based attack, to provide a reliable robustness estimate that is difficult to overstate through poor tuning [croce2020autoattack].

## 2.3 Robustness of Vision-Sequence Pipelines

Robustness research has concentrated on single-output classifiers, while structured-output vision pipelines have received less attention. Systems that combine segmentation, geometric transformation, and sequence decoding present two complications. First, the output is a sequence, so degradation must be measured with edit-distance metrics rather than a single accuracy value. Second, the inference path may contain non-differentiable operations between learned stages, which prevents direct end-to-end gradient computation and motivates either black-box attacks or differentiable surrogate models. Differentiable approximations of image transformations, such as smoothed sampling and relaxed argmax operators, make it possible to construct surrogate pipelines whose gradients are usable for attack and defense research [jaderberg2015]. Surrogate models also support transfer studies, in which perturbations crafted on a differentiable stand-in are replayed against the original non-differentiable system.

## 2.4 Spectral and Signal-Based Perturbations

Beyond worst-case adversarial perturbations, models are also sensitive to natural corruptions that shift the input distribution. Analyzing robustness in the frequency domain clarifies these effects, since many natural corruptions and many adversarial perturbations concentrate their energy in different frequency bands, and models often rely on high-frequency content that is not robust to such shifts [yin2019]. Signal-based perturbations constructed in the frequency domain provide a controllable way to approximate natural degradation. Power-law or 1/f noise, in which spectral power decreases with frequency, resembles the statistics of many natural images and sensor processes, and injecting such noise before the pipeline tests sensitivity to realistic, non-adversarial corruption rather than to a worst-case adversary.

# 3. Methodology

## 3.1 System Under Test

The system under test is a Hierarchical Optical Music Recognition (HOMR) pipeline that transforms a full-page score image into a symbolic MusicXML transcription through three coordinated stages: segmentation, geometric normalization, and transformer-based sequence decoding. All neural inference is executed through exported ONNX models, while the connective logic between stages is deterministic image processing.

The first stage is semantic segmentation. A U-Net style convolutional network, derived from the OEMER segmentation approach and based on the encoder-decoder architecture introduced for dense prediction [ronneberger2015], processes the page in overlapping fixed-size patches and assigns each pixel to one of six classes: background, stems and rests, noteheads, clefs and keys, staff lines, and other symbols. The network operates on patches of size 320 by 320 and produces per-class masks that are reassembled into a full-page argmax segmentation. These masks separate the musical content from the staff lines and the page background and provide the geometric evidence required by the next stage.

The second stage is layout analysis and geometric normalization. Using the segmentation masks, the pipeline detects staff lines, groups them into staves and multi-staff systems, and isolates each staff region. Detected curvature and skew are corrected through a dewarping step so that staff lines are straight and consistently spaced, which removes a major source of variation before sequence decoding. Each normalized staff is rendered to a fixed grayscale tensor of size 256 by 1280 with values in the unit interval, and is standardized with the decoder's expected mean and standard deviation. This stage is implemented with classical image processing and contains no learned parameters, which makes it deterministic but non-differentiable.

The third stage is sequence decoding with a Polyphonic-TrOMR transformer [li2023tromr]. A transformer encoder maps the normalized staff image to a sequence of contextual feature vectors, and an autoregressive transformer decoder generates the symbolic transcription. The pipeline as a whole follows the HOMR design [homr2024]. The decoder produces five factorized output streams that are aligned position by position: rhythm, pitch, lift (accidental adjustment), articulation, and position. Factorization lets the model represent simultaneous attributes of a single notated event without enumerating every possible combination as a separate class. The decoder uses cached key and value tensors across generation steps for efficiency, consuming the full encoder context on the first step and the reduced context on subsequent steps. The decoded streams are mapped back to musical symbols and assembled into MusicXML, with page-level structure such as line breaks reconstructed during post-processing.

The combination of learned segmentation, deterministic geometric normalization, and a learned sequence decoder means that the pipeline cannot be differentiated end to end from input image to output tokens. This property directly shapes the threat model and motivates both the black-box attack track and the differentiable surrogate used for the defense study.

## 3.2 Threat Model

The threat model defines the adversary's knowledge, capability, and goal, and fixes the constraints under which robustness is measured. The adversary is black-box with respect to the deployed pipeline. It has no access to model parameters, gradients, or intermediate activations and may only submit input images and observe a scalar measure of transcription quality. This reflects a realistic deployment in which an OMR service exposes only its transcription output.

The adversary's capability is bounded by an L-infinity perturbation constraint with budget epsilon = 8/255. For an input image x and an adversarial image x', the constraint requires that the maximum absolute change to any pixel satisfies the L-infinity norm of (x' - x) being at most 8/255, with pixel values expressed on the standard 0 to 1 scale. This budget is a common standard in adversarial robustness research and keeps perturbations small relative to the dynamic range of the image. The L-infinity norm is chosen because it bounds the worst-case per-pixel change, which is appropriate for high-resolution document images where many small coordinated changes are the relevant risk.

The adversary's goal is to maximize transcription error under the budget. Score feedback is provided by the Symbol Error Rate between the transcription of the perturbed image and a reference transcription, which gives the attack a continuous signal to optimize rather than a single success or failure bit. Using Symbol Error Rate as feedback aligns the optimization target with the quantity of practical interest, since a higher Symbol Error Rate corresponds to a less usable transcription. The reference may be either the ground-truth symbolic content or the pipeline's own transcription of the clean image, depending on whether the study measures absolute degradation or relative shift from clean behavior.

A separate non-adversarial condition is included to contextualize the worst-case results. In this condition the input is corrupted by signal-based spectral noise rather than by an optimizing adversary, which characterizes sensitivity to realistic degradation under the same metrics.

## 3.3 Dataset

Score content is drawn from the PDMX dataset, a large-scale collection of public-domain music in MusicXML format. PDMX provides symbolic scores rather than images, which is well suited to this study because it allows controlled rendering of clean score images together with exact symbolic references. Each selected MusicXML file is rendered to a page image with a fixed engraving configuration, and the rendered page is paired with the symbolic content of the source file. This pairing supplies both the model input and the reference needed to compute edit-distance metrics, and it avoids the labeling noise that arises when references are transcribed from scanned images.

A working subset is sampled from the dataset for the experiments. The sampling step selects scores, renders each to one or more page images, and records the correspondence between rendered pages and source files in a manifest. Rendered pages that fail to engrave or that produce unreadable images are removed before use, so that the working set contains only valid page and reference pairs. The dataset is partitioned into training, validation, and test splits at the level of source scores to prevent pages from the same score appearing in more than one split. The training split supports surrogate training and adversarial training, the validation split supports model selection and the attack evaluations, and the test split is reserved for final measurement.

For the surrogate study, transcription targets are produced by running the HOMR pipeline on the rendered pages and recording its factorized output streams. This teacher-labeling step lets the surrogate learn to imitate the deployed pipeline directly, so that perturbations crafted on the surrogate are meaningful with respect to the system under test.

## 3.4 Robustness Evaluation (Attacking)

The attacking track evaluates the deployed pipeline under two perturbation families that share the L-infinity budget and the Symbol Error Rate metric but differ in intent.

The first is the Square Attack, a query-based black-box method that requires only score feedback [andriushchenko2020]. The attack initializes a perturbation at the boundary of the L-infinity ball and then performs random search by proposing localized updates within square-shaped regions of the image. Each proposed update is accepted only if it increases the observed Symbol Error Rate, and the perturbation is kept within the budget by clipping to the L-infinity ball after every update. The square size is reduced according to a schedule as the search progresses, which moves the search from coarse, high-impact changes toward fine adjustments. To keep the number of queries tractable for a full pipeline, the attack is applied to the normalized staff representation produced by the geometric stage rather than re-running the expensive segmentation and layout analysis on every query. Operating at this cached interface preserves the black-box character of the attack with respect to the decoder while reducing the per-query cost by more than an order of magnitude, which makes a meaningful query budget feasible.

The second is spectral noise injection, which applies frequency-domain perturbations to the full page before the pipeline runs. Colored noise is generated with a power spectrum that follows a power law, so that spectral power scales with frequency according to a 1/f-alpha relationship and the exponent alpha controls the balance between low-frequency and high-frequency content [yin2019]. The generated noise is scaled to respect the L-infinity budget and added to the page image. Because this perturbation is applied to the full page, it passes through segmentation, geometric normalization, and decoding in turn, which tests the sensitivity of the complete pipeline to realistic, non-optimizing corruption. Sweeping the perturbation strength and the spectral exponent traces how degradation grows with corruption magnitude and how it depends on the frequency composition of the noise.

Both attack conditions are run across a range of perturbation strengths so that the relationship between budget and degradation can be reported as a curve rather than a single operating point. Reporting the full curve avoids the overestimation of robustness that follows from evaluating at a single, possibly weak, setting [uesato2018].

## 3.5 Adversarial Training

Because the deployed pipeline is not differentiable end to end, the defense study is conducted on a differentiable surrogate that imitates the pipeline. The surrogate is a full-page model that maps a page image to ordered staff slots and then decodes each staff into the same five factorized streams used by the system under test. It is trained on the rendered pages with the pipeline's own outputs as targets, so that it approximates the behavior of the deployed system while remaining fully differentiable. The surrogate retains spatial structure in its page representation so that gradient-based perturbations reflect the layout of the score rather than a single pooled descriptor.

The defense is adversarial training with Projected Gradient Descent [madry2018]. During training, each clean batch is replaced by an adversarial batch before the forward pass. The adversarial batch is constructed by initializing a random perturbation within the L-infinity ball and then taking a fixed number of gradient-sign ascent steps on the training loss, projecting back into the ball and clamping to the valid input range after each step. The perturbation is generated against the current model parameters at every step, which is the property that makes adversarial training effective, since a fixed precomputed set of perturbations would be weaker and easily memorized. The configuration uses a perturbation budget of epsilon = 0.02, ten ascent steps, and a per-step size of 0.005, with validation kept on clean inputs so that model selection reflects clean performance. Two surrogates are trained on identical data and for an identical number of epochs: a clean surrogate with no adversarial training and a defended surrogate with PGD adversarial training. Training the two under matched conditions isolates the effect of the defense.

The PGD-generated examples are used for training only. The independent strength of the final robustness evaluation is preserved by attacking both surrogates with a different, stronger procedure, described next, rather than with the same attack used during training.

## 3.6 Evaluation Metrics

All degradation is reported with two edit-distance metrics computed from Levenshtein distance, which counts the minimum number of insertions, deletions, and substitutions required to transform one sequence into another.

Symbol Error Rate (SER) operates at the level of musical tokens. For a predicted token sequence and a reference token sequence, it is the Levenshtein distance between the two sequences divided by the length of the reference sequence:

SER = Levenshtein(predicted_tokens, reference_tokens) / length(reference_tokens)

A Symbol Error Rate of zero indicates an exact match, and larger values indicate greater transcription error, with values near or above one indicating that the prediction is no more useful than an empty or unrelated sequence. Symbol Error Rate is the primary metric because it reflects the symbolic content that downstream applications consume, and it serves as the score feedback for the black-box attack.

Character Error Rate (CER) applies the same distance at the level of individual characters in the serialized transcription:

CER = Levenshtein(predicted_characters, reference_characters) / length(reference_characters)

Character Error Rate is a finer-grained measure that captures partial errors within tokens, such as a single altered attribute of an otherwise correct symbol, which a token-level metric would count as a full substitution. Reporting both metrics separates errors that change whole symbols from errors that change only part of a symbol.

For the surrogate study, the defended and undefended models are compared with AutoAttack, the standardized ensemble of parameter-free attacks [croce2020autoattack], applied across a grid of L-infinity budgets. At each budget the evaluation reports the metrics above together with branch-level accuracy for the factorized streams, so that the comparison between the clean surrogate before defense and the PGD-trained surrogate after defense is made under the same strong and independent attack.

# Appendix

## Source Code

The implementation of the attack tracks, the differentiable surrogate, the adversarial training procedure, and the evaluation scripts is available in the project repository.

- GitHub: [https://github.com/<organization>/adversarial-homr](https://github.com/<organization>/adversarial-homr)
- Archived release (Zenodo): [https://doi.org/10.5281/zenodo.<record-id>](https://doi.org/10.5281/zenodo.<record-id>)

## HOMR Repository

The Hierarchical Optical Music Recognition pipeline used as the system under test is maintained in a separate repository.

- GitHub: [https://github.com/liebharc/homr](https://github.com/liebharc/homr)

## Music Sheet Dataset

The PDMX public-domain MusicXML dataset is the source of all score content used for rendering and evaluation.

- Dataset: [https://github.com/pnlong/PDMX](https://github.com/pnlong/PDMX)
- Archived release (Zenodo): [https://doi.org/10.5281/zenodo.<record-id>](https://doi.org/10.5281/zenodo.<record-id>)
