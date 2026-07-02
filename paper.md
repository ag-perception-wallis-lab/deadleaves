---
title: '`deadleaves`: Creating cluttered visual stimuli in Python'
tags:
  - Python package
  - vision science
  - stimulus creation
authors:
  - name: Swantje Mahncke
    orcid: 0009-0006-2761-0038
    corresponding: true
    affiliation: "1, 2"
  - name: Thomas S. A. Wallis
    orcid: 0000-0001-7431-4852
    affiliation: "1, 2"
  - name: Lynn Schmittwilken
    orcid: 0000-0003-3621-9576
    affiliation: 1
affiliations:
 - name: Centre for Cognitive Science, Institute of Psychology, TU Darmstadt, Germany
   index: 1
   ror: 05n911h24
 - name: Center for Mind, Brain, and Behavior (CMBB), Universities of Marburg, Gießen, and Darmstadt, Germany
   index: 2
date: 13 February 2026
bibliography: paper.bib
---

# Summary

The Dead Leaves Model [@Matheron1968] is a stochastic image generation model.
The model creates images by sampling objects from a predefined family of distributions.
Each object ("leaf") is typically a simple shape, such as a circle or ellipse, and its properties (e.g. position, size, orientation, color) are randomly drawn from these distributions.
This sampling process allows precise control over image statistics, which makes it possible to vary or fix specific leaf properties as desired.
As a consequence, Dead Leaves Models are widely adopted in the study of image statistics [@Ruderman1997;@Lee2001;@Zylberberg2012;@Madhusudana2022], visual function [@Morimoto2021;@Maiello2017;@Groen2012;@Taylor2015;@Wallis2012], image quality in digital cameras [@Cao2010;@Artmann2015;@Jenkin2024;@CPIQ2023] and, most recently, as training data for machine learning algorithms [@Baradad2021;@Achddou2022].

Leaves are drawn sequentially onto a two-dimensional canvas from front to back, so later leaves can be partially or fully occluded by earlier ones.
This layering reproduces key statistical properties of natural scenes, including occlusion structure, heavy-tailed distributions of contrasts and edges, scale invariance, and higher-order spatial correlations [@Ruderman1997;@Lee2001].
For these reasons, the model serves as an effective null model for studying natural image statistics and early visual processing.
Yet no publicly available software exists that allows users to generate dead leaves images in a standardized way.
This is where our package comes in.

`deadleaves` is an open-source Python package which can be used to create dead leaves images in a standardized, yet flexible way.
Core functionalities are:

- generating dead leaves images with properties (e.g. sizes, orientations, colors) drawn from a wide range of distributions (e.g. uniform, normal, Poisson, power-law, constant) or directly from an image
- picking from various leaf shapes (circles, ellipsoids, rectangles, regular polygons)
- sampling in different color spaces (RGB, HSV, grayscale)
- applying different noise or image textures, either to the entire image or per-leaf
- varying the image area covered by leaves, either by adjusting leaf count (controlling density) or by applying spatial masks to restrict coverage to selected regions
- creating arbitrarily complex leaf configurations by adding dependencies between leaf features (e.g. space-dependent color gradients)

Users can plug in various distributions for the different model parameters to create a variety of images (\autoref{fig:deadleaves}).

![Example images generated with the `deadleaves` package.\label{fig:deadleaves}](docs/_static/figures/examples.png)

# Statement of need

Variations of the Dead Leaves Model have been used for decades to generate images for vision research and computer vision [@Ruderman1997;@Kaping2007;@Taylor2015;@Baradad2021].
However, there is no standard implementation, and only a few public codebases exist [@Baradad2021;@Badal2024].
Most researchers therefore implement their own generative code, which is time-consuming and error-prone and complicates comparisons across studies, in particular if the models used are insufficiently documented [cf. @Schmittwilken2023].

These issues have practical consequences.
Small differences in implementation affect the resulting image statistics [@Achddou2022], which are the primary scientific objective in many dead leaves studies.
Combined with the stochastic nature of the model, this can make it challenging for researchers to reliably generate stimuli and precisely describe their statistical properties.
In short, current practices create barriers to reproducibility and consistent use of dead leaves images.

To address these issues, we developed `deadleaves`, a free and open-source Python package that standardizes dead leaves image generation through parameterized functions for flexible stimulus generation with extensive documentation.
The package can be installed via standard package managers or from [GitHub](https://github.com/ag-perception-wallis-lab/deadleaves).
By simplifying and unifying dead leaves generation, `deadleaves` improves reproducibility, reduces implementation errors, and increases accessibility for both experienced users and newcomers.


# State of the field

Dead leaves images have been used across a wide range of disciplines as a controllable, generative model of natural image structure.
A central advantage of Dead Leaves Models is that they allow the synthesis of images with certain (naturalistic) statistical properties, while avoiding semantic content and a range of potential biases.
Here, we group prior work into three main areas according to the methodological role dead leaves images play.


## Study of (natural) image statistics
Many studies have used Dead Leaves Models to study and explain statistical regularities commonly observed in natural images by treating them as an analytically tractable model of occlusion-dominated scene structure.
A central question is whether the statistical regularities of natural images arise primarily from generic properties of scene composition, rather than from semantic image content.

Early work demonstrated that scale-invariant properties of natural images emerge from scenes composed of objects whose sizes follow a power-law distribution [@Ruderman1997].
Subsequent studies demonstrated that these spectral properties are directly shaped by occlusion, with systematic effects of object overlap, transparency, and opacity [@Balboa2001; @Hsiao2005; @Zylberberg2012].

Later work showed that Dead Leaves Models also reproduce other statistical properties of different natural scene classes -- most notably luminance, contrast and other derivative statistics [@Lee2001].
Extensions of the model with object texture further improved this correspondence [@Madhusudana2022].

Finally, several studies have formalized the relationship between the generative assumptions of the Dead Leaves Model and the resulting image statistics.
Analytical derivations link model parameters directly to feature distributions [@Pitkow2010], and complementary work has established a rigorous mathematical foundation using tools from stochastic geometry and probability theory [@Alvarez1999; @Gousseau2003; @Bordenave2006; @Gousseau2007].


## Visual psychophysics

In visual psychophysics, dead leaves images are primarily used as controlled stimuli that preserve selected statistical properties of natural scenes while minimizing semantic content.
This allows researchers to study perceptual sensitivity to specific image statistics and specific visual features in isolation.

Dead leaves stimuli have been used to investigate the conditions under which surfaces appear self-luminous [@Morimoto2021].
They have been used in adaptation paradigms to probe which image statistics support rapid scene-level judgments [@Kaping2007], and statistics derived from dead leaves images have been directly controlled in categorization tasks to isolate their contribution to rapid scene processing [@Groen2012].
Spatially localized or object-level blur has been introduced to study blur detection and discrimination [@Taylor2015; @Maiello2017], and dead leaves patterns have been embedded into natural images to study visual crowding while preserving control over local image statistics [@Wallis2012].

Across these applications, Dead Leaves Models function as semantically neutral, yet statistically structured stimuli that enable precise manipulation of image properties (such as size, contrast, color, and texture) relevant to human visual perception.


## Synthetic data for computer vision

Dead Leaves Models have recently been used to generate synthetic images with fully controlled statistical and generative structure, providing training and evaluation data for computer vision tasks and models without costly real‑image collection.
Tasks such as disparity estimation [@Madhusudana2022], learning visual representations that emphasize shape and occlusion cues [@Baradad2021], and image restoration including denoising and deblurring [@Achddou2022] were trained on Dead Leaves Images.
Dead leaves images have also been used as a benchmark for evaluating image quality, for example in assessing texture reproduction on digital cameras [@Cao2010;@McElvain2010;@Kirk2014;@CPIQ2023;@Jenkin2024].

Overall, these applications illustrate that Dead Leaves Models provide a flexible tool for generating semantically neutral yet statistically structured images, bridging the gap between highly simplified synthetic stimuli and the complexity of natural scenes.


# Software design

`deadleaves` is an object-oriented framework for generating dead leaves images, organized into four components: (1) leaf geometry sampling, (2) leaf appearance sampling, (3) rendering, and (4) post hoc scene manipulation.

The geometric structure in the dead leaves image is generated iteratively by sampling leaf positions and shapes and assigning empty pixels to each leaf, effectively layering leaves from front to back, until a stopping criterion is met.
This stage produces a `leaf_table` (pandas DataFrame of sampled geometric leaf parameters) and a `segmentation_map` labeling each pixel.

Leaf appearance is assigned by sampling color parameters and, optionally, texture parameters for each leaf from user-defined distributions (in RGB, HSV, or grayscale) and adding the sampled values to the `leaf_table`. This separation allows geometry, color, and texture to be manipulated independently.

The rendering class maps the leaf parameters to pixels using the `segmentation_map` (generated if needed), and colors the canvas pixels according to the stored appearance parameters, optionally adding leaf-specific or global noise or texture. The resulting image has pixel-accurate boundaries with sharp edges. This can be particularly useful for segmentation tasks, where blur-based rendering can make object boundaries less defined.

Because sampling and rendering are decoupled, leaf parameters can be modified after the initial sampling step and the scene can be re-rendered without regenerating the full geometry. This enables operations such as adding motion, or composing scenes from different distributions, e.g. by placing a proto-object onto a leaf background (see examples in \autoref{fig:deadleaves}). A dedicated class supports these workflows by allowing merging of scenes and adjusting leaf ordering.

The modular design makes it straightforward to extend the framework with new leaf shapes, color models, sampling distributions, or rendering methods (e.g., transparency-aware compositing) without changing the core pipeline. Additional features such as transparency or blur can be added independently of the existing components.


# Research impact statement

The `deadleaves` package provides a user-friendly, well-documented framework for generating a wide range of dead leaves images, including many stimuli which have been used in prior work. Since the Dead Leaves Model has been a standard tool in research for decades, we expect the package to support further research in visual neuroscience and machine learning. In addition, we think that it allows for new applications in e.g. neurophysiology or aesthetics research.

# AI usage disclosure

ChatGPT 5 was used to assist in improving code, documentation, and typesetting, and for generating test cases for package components.
No AI content was used directly.
All suggestions were manually adapted to ensure correctness and fit the intended context.

# Acknowledgements

The authors thank Benjamin Beilharz for reviewing portions of the codebase and for providing helpful suggestions on the design of the user-facing API.

This work was supported by the Deutsche Forschungsgemeinschaft (German Research Foundation, DFG) under Germany’s Excellence Strategy (EXC 3066/1 “The Adaptive Mind”, Project No. 533717223).
This work was co-funded by the European Union (ERC, SEGMENT, 101086774). Views and opinions expressed are, however, those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council. Neither the European Union nor the granting authority can be held responsible for them.

# References

<div id="refs"></div>
