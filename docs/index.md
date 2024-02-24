---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: default
usemathjax: true
---

## Introduction 🗒️
---
Machine learning is in everything nowadays. It has revolutionized almost everything it has touched, from complex medical procedures, to netflix recommendations, to the path the chips you bought get to the store. Welcome to our journey through the transformative world of machine learning, a branch of artificial intelligence that teaches computers to learn from data, much like humans do from experience. At the forefront of this exciting adventure is the concept of feature learning—a cutting-edge technique that allows computers to automatically identify the most important pieces of information from the data they’re given. Coupled with this is our focus on soft decision trees, an innovative twist on traditional decision-making models, which are designed to be more flexible and intuitive, much like choosing a path based on probabilities rather than a simple yes or no answer.

Central to our exploration are neural networks, sophisticated algorithms modeled after the human brain's architecture, enabling machines to think and learn with astonishing depth and complexity. These networks are the backbone of our ability to process and interpret vast amounts of data, making sense of the world in ways that were previously unimaginable. They are able to solve complex tasks like driving cars, surgery, and much more; yet their inner workings still mystify us. How they learn such important features are a mystery only beginning to be solved. So, as we delve into these concepts, we aim to make this cutting-edge technology understandable and accessible, shedding light on how it's transforming our world and why it's an exciting time to be at the intersection of technology and discovery. Join us as we embark on this enlightening journey, unraveling the mysteries of machine learning, feature learning, and making them understandable through soft decision trees.


## What's feature learning and why do we care about it? 🔎
---

At its core, feature learning is a revolutionary process within machine learning that automates the identification of the most relevant characteristics or "features" in a dataset. Think of it as teaching a computer to find and focus on the most important details in a vast landscape of information.

Imagine you're teaching someone to differentiate between a cat and a dog. Traditionally, you might start by pointing out key features: 'Cats usually have shorter noses, and dogs come in a wider variety of sizes.' This is akin to the older methods in machine learning, where experts had to painstakingly identify and input these distinguishing features into the computer. However, with feature learning, the approach is fundamentally different and far more intuitive. Unlike these traditional methods where we manually specify what to look for, in feature learning, the computer itself learns to identify the distinguishing features between cats and dogs. It does this through a process of adjusting different 'weights' and 'biases' within neural networks, essentially fine-tuning its own internal parameters until it can accurately tell cats and dogs apart, all on its own. This means we don't have to tell the computer what features are important; it discerns and learns the most crucial aspects through its own experience, much like a child learning to identify animals. Similarly to a child, it might even find important things that we missed, which is a part of what makes it so powerful.

Yet, like mentioned before, we are only beginning to understand what how they learn features so well and what features they are learning. That is why it's crucial for us to explore avenues to understand them better. We can't have a computer making important decision such as driving our children to school, performing brain surgery, etc. if we don't know why they make the choices they make. This is what we explored in our first quarter project. Through the use of the Neural Feature Matrix (NFM), we can capture the features learned back onto the input data. We can also use this to understand features between different layers. The intricate details about the NFM are included below. Otherwise, it's importance is simply to quantify and give us a better understanding of those features mentioned previously.

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

In the context of neural networks and the concept of a neural feature matrix, a "good" feature can be understood as one that significantly contributes to the network's ability to make accurate predictions or classifications. Essentially, a good feature is one that helps to reduce uncertainty or increase the discriminative power of the model. Within a neural feature matrix, which represents the learned features or representations at various layers of a neural network, a good feature is one that captures essential, underlying patterns in the data that are relevant to the task at hand. These features should ideally be invariant to irrelevant variations (such as lighting changes in image recognition, or background noise in voice recognition) and sensitive to the distinctions the model aims to learn. The effectiveness of a feature is often evaluated based on its impact on the model's performance metrics, such as accuracy, precision, recall, or the area under the ROC curve. In our models, we utilize image data, which allows us to clearly visualize the important features learned from our model. In summary, a good feature within the neural feature matrix context is one that enhances the neural network's learning capability, enabling it to extract and leverage meaningful information from the input data for better decision-making or prediction outcomes.

## Why decision trees? and why are they "soft"? 🍦
---

Decision trees have long been valued in machine learning for their simplicity and interpretability. They mimic human decision-making processes by splitting data into branches at binary decision points, making them intuitive to understand and explain. It's like deciding if the given food is a hotdog, a simple yes or no tells us whether we possess a hotdog. This characteristic is particularly important in real-world applications where understanding the rationale behind predictions or decisions is crucial. So although Neural Networks are renowned for delivering state-of-the-art performance, especially in tasks involving complex patterns or high-dimensional data, they often act as "black boxes." As we established before, understanding the "why" behind their decisions can be challenging. If a model were deciding who to give a loan to, rent a car, or provide a surgery with, that's a decision we would want to know the "why". That's where decision trees, and specifically gradient boosted trees, come into play. These models are not only fast and capable of training quickly on large datasets but also provide the much-needed interpretability by illustrating how input features influence predictions. This makes them an attractive choice for many practical applications.

The term "soft" decision trees extends this concept further by incorporating elements of neural networks to enhance the model's flexibility and performance. Unlike traditional decision trees that make hard, binary decisions at each node, soft decision trees use probabilistic approaches, allowing for more nuanced decisions based on the likelihood of different outcomes. This softening of decisions enables the model to handle ambiguity and uncertainty more effectively, through feature learning, making it better suited for complex datasets where traditional decision trees might struggle. By blending the interpretability of decision trees with the nuanced decision-making capabilities of neural networks, soft decision trees offer a compelling solution for applications requiring both high performance and clear understanding of model decisions.

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