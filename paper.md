---
title: '`dead_leaves`: Creating cluttered visual stimuli in Python'
tags:
    - Python package
    - vision science
    - stimulus creation
authors:
    - name: Swantje Mahncke
      orcid: 0009-0006-2761-0038
      corresponding: true
      affiliations: "1, 2"
affiliations:
    - index: 1
      name: Centre for Cognitive Science, Institute of Psychology, TU Darmstadt, Germany
      ror: "05n911h24"
    - index: 2
      name: Center for Mind, Brain, and Behavior (CMBB), Universities of Marburg, Gießen, and Darmstadt, Germany
date: TBD
bibliography: paper.bib
---

# Summary

The Dead Leaves Model [@Matheron1968] is a stochastic image generation model.
The model creates images by sampling objects from a predefined family of distributions.
Each object ("leaf") is typically a simple shape, such as a circle or ellipse, and its properties (e.g. position, size, orientation, color) are randomly drawn from these distributions.
This sampling process allows precise control over image statistics, which makes it possible to vary or fix specific leaf properties as desired.
As a consequence, Dead Leaves Models are widely adopted in the study of image statistics [@Ruderman1997;@Lee2001;@Zylberberg2012;@Madhusudana2022], visual function [@Morimoto2021;@Maiello2017;@Groen2012;@Taylor2015;@Wallis2012] and, most recently, as training data for machine learning algorithms [@Baradad2021;@Achddou2022].

Leaves are drawn sequentially onto a two-dimensional canvas from front to back, so later leaves can be partially or fully occluded by earlier ones.
This layering reproduces key statistical properties of natural scenes, including occlusion structure, heavy-tailed distributions of contrasts and edges, scale invariance, and higher-order spatial correlations [@Ruderman1997;@Lee2001].
For these reasons, the model serves as an effective null model for studying natural image statistics and early visual processing.
Yet, there is no publicly available software yet, which would allow to generate dead leaves images in a systematic fashion.
This is where our package comes in.

`dead_leaves` is an open-source Python package which can be used to create dead leaves images in a systematic, yet flexible way.
Core functionalities are:

- generating dead leaves images with properties (e.g. sizes, orientations, colors) drawn from a wide range of distributions (e.g. uniform, normal, Poisson, power-law, constant) or directly from an image.
- picking from various leaf shapes (circles, ellipsoids, rectangles, regular polygons).
- sampling in different color spaces (RGB, HSV, Gray-scale).
- applying different noise or image textures, either to the entire image or per-leaf.
- varying the image area covered by leaves, i.e. choosing between sparser or denser sampling and position mask.
- creating arbitrarily complex leaf configurations by adding dependencies between leaf features (e.g. space-dependent color gradients).

The package is build around `PyTorch` [@Paszke2017] which allows the use of GPU for a faster sampling process.
Users can plug in various distributions for the different model parameters to create a variety of images (@fig-DeadLeaves).

::: {#fig-DeadLeaves layout-ncol=6}
![](docs/_static/figures/annulus.png){width=100%}

![](docs/_static/figures/circles.png){width=100%}

![](docs/_static/figures/constant_size.png){width=100%}

![](docs/_static/figures/dual_feature_dependency.png){width=100%}

![](docs/_static/figures/ellipsoids.png){width=100%}

![](docs/_static/figures/HSV.png){width=100%}

![](docs/_static/figures/leafwise_texture.png){width=100%}

![](docs/_static/figures/leafy_image.png){width=100%}

![](docs/_static/figures/natural_color.png){width=100%}

![](docs/_static/figures/normal_size.png){width=100%}

![](docs/_static/figures/polygons.png){width=100%}

![](docs/_static/figures/position_mask.png){width=100%}

![](docs/_static/figures/rectangles.png){width=100%}

![](docs/_static/figures/RGB.png){width=100%}

![](docs/_static/figures/single_feature_dependency.png){width=100%}

![](docs/_static/figures/sparse_sampling.png){width=100%}

![](docs/_static/figures/spheres.png){width=100%}

![](docs/_static/figures/texture_patch.png){width=100%}

Example images generated with the `dead_leaves` package.
:::


# Statement of need

Variations of the Dead Leaves Model have been used for decades to generate images for vision research and computer vision [@Ruderman1997;@Kaping2007;@Taylor2015;@Baradad2021].
The model provides a relatively simple way to create complex, cluttered stimuli that match the statistics of natural images or other distributions.
Despite its widespread use, there is no standard implementation for generating dead leaves images, and only a few projects have made code publicly available [@Baradad2021].
Most researchers therefore implement their own generative code, which is time-consuming, prone to errors, and complicates comparisons across studies [**cf stimupy if you like where we make the same claim**].
Moreover, reproducing existing stimuli is often difficult because the specifications used to generate dead leaves images are often too coarse.

These gaps in standardization and documentation have practical consequences.
Small differences in implementation or rendering choices can strongly affect the resulting image statistics [@Achddou2022], which are the primary scientific objective in many dead leaves studies.
Combined with the stochastic nature of the model, this makes it challenging for researchers to reliably generate stimuli and precisely describe their statistical properties.
In short, current practices create barriers to reproducibility and consistent use of dead leaves images.

To address these issues, we developed `dead_leaves`, a free and open-source Python package that standardizes dead leaves image generation.
The package can be installed via standard package managers or from GitHub [**add link**].
It provides fully parameterized functions for flexible stimulus generation, along with extensive documentation which describes the model, its parameters, and its recommended usage.
By simplifying and unifying dead leaves generation, `dead_leaves` improves reproducibility, reduces implementation errors, and increases accessibility for both experienced users and newcomers.


# State of the field

Dead leaves images have been used across a wide range of disciplines as a controllable, generative model of natural image structure.
A central advantage of dead leaves models is that they allow the synthesis of images with certain (naturalistic) statistical properties, while avoiding semantic content and a range of potential biases.
Here, we group prior work into three main areas according to the methodological role dead leaves images play.


## 1. Study of (natural) image statistics
A large body of work has used dead leaves models to study and explain statistical regularities commonly observed in natural images by treating them as an analytically tractable model of occlusion-dominated scene structure.
A central question is whether the statistical regularities of natural images arise primarily from generic properties of scene composition, rather than from semantic image content.

Early work demonstrated that scale-invariant properties of natural images, most notably the approximate power-law behavior of their power spectra, emerge from scenes composed of objects whose sizes follow a power-law distribution [@Ruderman1997].
Subsequent studies demonstrated that these spectral properties are directly shaped by occlusion, with systematic effects of object overlap, transparency, and opacity [@Balboa2001; @Hsiao2005; @Zylberberg2012].

Later work showed that dead leaves models also reproduce other statistical properties of natural scenes -- most notably luminance, contrast and other derivative statistics [@Lee2001].
This includes the possibility to reproduce statistics characteristic of different scene classes, such as vegetation-like and man-made images [@Lee2001].
Extensions of the model with object texture further improved this correspondence [@Madhusudana2022].

Finally, several studies have formalized the relationship between the generative assumptions of the dead leaves model and the resulting image statistics.
Analytical derivations link model parameters directly to feature distributions [@Pitkow2010], and complementary work has established a rigorous mathematical foundation using tools from stochastic geometry and probability theory [@Alvarez1999; @Gousseau2003; @Bordenave2006; @Gousseau2007].


## 2. Visual psychophysics

In visual psychophysics, dead leaves images are primarily used as controlled stimuli that preserve selected statistical properties of natural scenes while minimizing semantic content.
This allows researchers to study perceptual sensitivity to specific image statistics and image-level cues in isolation.
For example, dead leaves stimuli have been used to investigate the conditions under which surfaces appear self-luminous, by carefully controlling luminance and color distributions [@Morimoto2021].

A different line of research uses dead leaves stimuli to study rapid scene categorization under controlled image statistics.
By manipulating orientation and spatial frequency content, dead leaves images have been used in adaptation paradigms to probe which image statistics support rapid scene-level judgments [@Kaping2007], while contrast and higher-order statistics derived from dead leaves images have been directly controlled in categorization tasks to isolate their contribution to rapid scene processing [@Groen2012].

Finally, dead leaves images have been modified to selectively target specific perceptual cues while maintaining global image structure.
Spatially localized or object-level blur has been introduced to study blur detection and discrimination [@Taylor2015; @Maiello2017], and dead leaves patterns have been embedded into natural images to study visual crowding while preserving control over local image statistics [@Wallis2012].

Across these applications, dead leaves models function as semantically neutral, yet statistically structured stimuli that enable precise manipulation of image properties relevant to human visual perception.


## 3. Synthetic data for computer vision

Dead leaves models have recently been used to generate synthetic images with fully controlled statistical and generative structure, providing training and evaluation data for computer vision tasks and models.

One application is in training computer vision models on synthetic data that bypasses costly real‑image collection. Dead leaves images have been used for tasks such as disparity estimation [@Madhusudana2022], learning visual representations that emphasize shape and occlusion cues [@Baradad2021], and image restoration including denoising and deblurring [@Achddou2022].

Beyond training neural networks, dead leaves images have also been used as a benchmark for evaluating image quality, for example in assessing texture reproduction on digital cameras [@Cao2010].

Overall, these applications illustrate that dead leaves models provide a flexible tool for generating semantically neutral yet statistically structured images, bridging the gap between highly simplified synthetic stimuli and the complexity of natural scenes.


# Software Design

`dead-leaves` is designed to provide a user-friendly, object-oriented module for generating dead leaves images. 
The package is structured into two main classes, which decouple the geometry from the rendering process.

The geometry is generated in an interative process.
In each iteration $i$ we sample a uniform position $(x,y)$ on our canvas and shape parameters for the chosen shape, e.g. the area of a circle.
The resulting object is what we call the *leaf* with index $i$. 
Any pixel on our canvas which does not already belong to a leaf and is covered by our leaf $L_i$ will be labeled as $i$, i.e. we layer object from front to back onto our canvas.
We repeat this process for a given number of steps `n_samples` or until a given area is filled, either the full canvas or some specific unmasked area.
As a result we get a segmentation or partition map `partition` of the canvas where each point belongs to exactly one leaf or the background.

The rendering is then performed by sampling a color (and optionally texture) for each leaf from a given color (or texture) distribution and coloring the corresponding pixels of the canvas.
Rendering the image by coloring pixels based on their object membership leads to a pixel perfect segmentation of the generated image, i.e. sharp edges.
This generative process does not allow for decorations like blur or transparency.
<!-- are you going to say whether this is a feature that could be added in future? Does the design permit extension? -->
Both classes contain modular components which are plugged into the classes' main methods such that it can fairly simple be extended to other geometries or rendering specifications.
<!-- can you give an example or be more specific here? what do you mean by "modular components"? Should you provide more detail about the "classes' main methods"? -->

# Research Impact Statement

The `dead-leaves` package allows one to generate dead leaves images in a user friendly and well documented way.
Many of the images used prior work can be generated with our package.
Since this model has been in use for research for decades and is still continuously used we expect this package to support more research extending the approaches covered so far.
In addition, the `dead_leaves` package could be used to easily generate images with similar statistics to those of natural images, as control stimuli for aesthetics research or for studying how different features are integrated in human perception (e.g. luminance and hue).

# AI usage disclosure

ChatGPT 5 [@ChatGPT] was used to aid in the setup of the documentation and for generating test cases for the package components.
No AI output was directly copied for usage. 
All AI output was adjusted manually to fit the desired setting and be functional.

# Acknowledgements

This work was supported by the Deutsche Forschungsgemeinschaft (German Research Foundation, DFG) under Germany’s Excellence Strategy (EXC 3066/1 “The Adaptive Mind”, Project No. 533717223).
This work was co-funded by the European Union (ERC, SEGMENT, 101086774). Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council. Neither the European Union nor the granting authority can be held responsible for them.

# References

<div id="refs"></div>