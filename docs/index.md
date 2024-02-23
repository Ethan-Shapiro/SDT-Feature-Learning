---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: default
usemathjax: true
---

## Introduction 🗒️
---
Machine learning is in everything nowadays. It has revolutionized almost everything it has touched, from complex medical procedures, to netflix recommendations, to the path the chips you bought get to the store. Welcome to our journey through the transformative world of machine learning, a branch of artificial intelligence that teaches computers to learn from data, much like humans do from experience. At the forefront of this exciting adventure is the concept of feature learning—a cutting-edge technique that allows computers to automatically identify the most important pieces of information from the data they’re given. Coupled with this is our focus on soft decision trees, an innovative twist on traditional decision-making models, which are designed to be more flexible and intuitive, much like choosing a path based on probabilities rather than a simple yes or no answer.

Central to our exploration are neural networks, sophisticated algorithms modeled after the human brain's architecture, enabling machines to think and learn with astonishing depth and complexity. These networks are the backbone of our ability to process and interpret vast amounts of data, making sense of the world in ways that were previously unimaginable. They are able to solve complex tasks like self-driving cars, automated surgery, and much more, yet their inner workings still mystify us. How they learn such important features are a mystery only beginning to be solved. So, as we delve into these concepts, we aim to make this cutting-edge technology understandable and accessible, shedding light on how it's transforming our world and why it's an exciting time to be at the intersection of technology and discovery. Join us as we embark on this enlightening journey, unraveling the mysteries of machine learning, feature learning, and making them understandable through soft decision trees.


## What's feature learning and why do we care about it? 🔎
---

<details>
<summary>Neural Feature Matrix Details ⤵️</summary>
<br>
Let \( f: \mathbb{R}^d \rightarrow \mathbb{R} \) denote a fully connected network with \( L \) hidden layers for \( L > 1 \), weight matrices \( \{W_i\}_{i=1}^{L+1} \), and elementwise activation function \( \phi \) of the form

\[
f(x) = W_{L+1}h_L(x) ; \quad h_e(x) = \phi(W_eh_{e-1}(x)) \text{ for } e \in \{2, \ldots, L\}
\]

with \( h_1(x) = x \). We refer to the terms \( h_i(x) \) as the <b>features</b> at layer \( i \). We can characterize how features \( h_{i+1}(x) \) are constructed by understanding how \( W_i \) scales and rotates elements of \( h_i(x) \). These scaling and rotation quantities are recovered mathematically from the eigenvalues and eigenvectors of the matrix \( W_i^T W_i \), which is the NFM at layer \( i \). Hence, to characterize how features are updated in any layer of a trained neural network, it is sufficient to characterize how the corresponding layer’s NFM is constructed. Before mathematically stating how such NFMs are built, we connect NFM construction to the following intuitive procedure for selecting features.

Given any predictor, a natural approach for identifying important features is to rank them by the magnitude of change in prediction upon perturbation. When considering infinitesimally small feature perturbations on real-valued predictors, this approach is mathematically equivalent to computing the magnitude of the derivative of the predictor output with respect to each feature. These magnitudes are computed by the gradient outer product of the predictor given by \( (\nabla f(x))(\nabla f(x))^T \) where \( \nabla f(x) \) is the gradient of a predictor, \( f \), at a point \( x \).

Our main insight, the <b>Deep Neural Feature Ansatz</b>, is that deep networks learn features by implementing the above approach for feature selection. Mathematically stated, we posit that the NFM of any layer of a trained network is proportional to the average gradient outer product of the network taken with respect to the input to this layer. In particular, let \( W_i \) denote the weights of layer \( i \) of a deep, nonlinear fully connected neural network, \( f \). Given a sample \( x \), let \( h_i(x) \) denote the input into layer \( i \) of the network, and let \( f_i \) denote the sub-network of \( f \) operating on \( h_i(x) \). Suppose that \( f \) is trained on \( n \) samples \( \{(x_p, y_p)\}_{p=1}^n \). Then throughout training,

\[
W_i^T W_i \approx \frac{1}{n} \sum_{p=1}^n \nabla f_i(h_i(x_p)) \nabla f_i(h_i(x_p))^T ;
\]

where \( \nabla f_i(h_i(x_p)) \) denotes the gradient of \( f_i \) with respect to \( h_i(x_p) \). This is known as the <b>Deep Neural Feature Ansatz</b>. During our first quarterp project we verified that this ansatz holds when using gradient descent to layer-wise train (1) ensembles of deep fully connected networks and (2) deep fully connected networks with the trainable layer initialized at zero.

</details>
---

## How do we know what a "good" feature is? 👍
---

## Why decision trees? and why are they "soft"? 🍦
---

## Our Datasets 📊
---

### MNIST 1️⃣

### CelebA 🏆

### STL Star ⭐

## Experiments 🧪
---

## What did we find? 🖥️
---

## Gradient Boosted Decision Trees 🌲
---

## What can this mean for the future? 🔮
---