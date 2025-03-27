# Project Portfolio

Welcome to my GitHub portfolio! This repository contains a collection of projects that highlight my skills, experience, and interests. These projects are relevant to my CV and showcase my expertise in various domains, including machine learning, deep learning, graph algorithms, and more.


Here are some of the key projects included in this repository:

1. Implementing Locality-Sensitive Hashing for Finding Similar Netflix Users

Description: This project aims to identify pairs of Netflix users with similar movie preferences based
on their rated movies. The similarity metric used is the Jaccard similarity where ysers with a Jaccard similarity > 0.5 are considered similar. Identifying such similar user pairs is
instrumental for enhancing recommendation systems, improving user engagement, and
personalizing content delivery on platforms like Netflix.
With a dataset comprising approximately 100,000 users, 17,770 movies, and
65 million rating records, a brute-force similarity computation ( 5 billion pairs) is
computationally infeasible. To address this challenge, Locality-Sensitive Hashing
(LSH) combined with MinHash signatures is utilized [1]. This approach approximates
similarities and efficiently narrows down the search to likely similar pairs, significantly
reducing computational overhead while maintaining high accuracy in similarity detection.


2. Biased Sentences In News

Description: Biased news and media framing significantly under-
mine people’s right to access accurate information,
as misinformation can shape or distort public
understanding and perspectives. The dataset created by Lim et al. (2020) [2]
includes labels from annotators with both preknown
and not preknown knowledge of news events, this
paper explores whether including annotators’
knowledge statuses into the training process can
improve the performance of a few-shot BERT
model and GPT-4o on biased news classification
task. The classification task uses a four-point
scale: Neutral, Slightly biased, Biased, and Very
biased. Specifically, we investigate how integrating
annotators’ knowledge statuses affects the models’
overall performance.



3. Using Foundation Models MOMENT and OpenCity for Crowd Flow Prediction

Description: Forecasting crowd flow traffic in urban areas is essential to ensure
public safety and optimize traffic management. However, traditional
crowd flow prediction models are usually strongly dependent on
training data. Although these models are effective within their task
scope, they often lack the scalability to handle the dynamic nature
of real-world crowd flow, such as adapting to weather changes or
new public events. We aim to apply these models such as MOMENT and OpenCity to
the crowd flow prediction task and to explore whether they can
provide a more efficient and adaptable approach.

4. On the application of WebGraph compression to social network graphs

Description: In this paper, we investigate the possible application of the Web-
Graph compression method, proposed by Boldi and Vigna in [5 ], to
social networks. The WebGraph framework achieves compression
of web graphs by leveraging the lexicographical ordering of URL’s,
which minimizes the size of gap distributions in adjacency lists.
Social networks are structured differently than web graph, which
raises the question: Is it possible to use the WebGraph compression
method to compress social networks if their nodes are reordered?
In order to find an answer to this question, we experiment with
various methods for relabeling the nodes of social network graphs
to create a gap distribution that approximates the properties of a
lexicographically ordered web graph. We approach this problem
in the following way: We first read the social network graph. Next,
we apply a community detection algorithm to split the graph into
communities. These communities are then ordered based on metrics
such as size, edge count, modularity contribution, and edge density.
Then we go through each community in order and order their nodes
based on their out- and in-degrees. Then we traverse through each
node in each community using breath-first-search (bfs) or depth-
first-search (dfs). Lastly, we will relabel each node based on traversal
order.
By doing this, we aim to create a relabeling strategy that optimizes the compressibility of social network graphs using the Web-Graph framework.



Contact

Email: mdpbelizon@gmail.com


