# Project Portfolio

Welcome to my GitHub portfolio! This repository contains a collection of projects that highlight my skills, experience, and interests. These projects are relevant to my CV and showcase my expertise in various domains, including machine learning, deep learning, graph algorithms, and more.


Here are some of the key projects included in this repository:

1. Master's Thesis: Smarter Repository Question Answering: When to Use Retrieval, Agents, or Both
   
Description: Investigates how AI systems can answer questions about unfamiliar codebases, comparing semantic retrieval, autonomous coding agents, and a graph-guided coding agent on a 720-question repository QA benchmark. Evaluation uses a human-validated LLM-as-a-judge framework, measuring both answer correctness and computational cost (tokens, exploration steps, latency). Finds that for locally deployable small models, retrieval-based approaches outperform agentic exploration on both accuracy and cost, while structural graph guidance consistently improves agent performance at no additional token cost — pointing toward adaptive systems that default to lightweight retrieval and escalate to agentic reasoning only when needed.

Repository: [Master's Thesis](github.com/Manudpb/Thesis-final)

2. CodenamesAI

Description: Built and tournament-tested a full suite of AI agents for both roles in Codenames — Codemaster (clue-giver) and Guesser — spanning static word embeddings (GloVe), transformer-based sentence embeddings (SBERT), and search-based reasoning methods including Monte Carlo Tree Search (MCTS), Tree of Thoughts, and Curriculum Learning. Agents were evaluated head-to-head across a 380-game round-robin tournament (20 team combinations, fixed vocabulary, deterministic seeding for reproducibility), using TrueSkill ratings and Wilson confidence intervals rather than raw win rates to properly account for opponent strength and sample-size uncertainty. Also built an interactive GUI for observing, playing against, or running tournaments between agents in real time. The headline finding: strategic simulation beats semantic sophistication — the MCTS codemaster (which uses the same GloVe embeddings as the baseline agent) won 89.5% of games when optimally paired, outperforming the transformer-based SBERT agent, because it can actually simulate and evaluate the consequences of a candidate clue rather than just measuring similarity. A second finding: complexity should be distributed asymmetrically across a team — pairing a sophisticated codemaster with a simple ("naive") guesser produced strong positive synergy (+0.283 over expected performance), while pairing two sophisticated agents together often performed worse than expected. The analysis also found a strong correlation (r=0.742) between an agent's competitive win rate and how "human-like" its clues were judged to be, suggesting believability emerges naturally from optimizing for strategic effectiveness rather than needing to be explicitly engineered.

Repository: [CodenamesAI](https://github.com/Manudpb/Relevant-projects/tree/main/CodenamesAI)

3. Implementing Locality-Sensitive Hashing for Finding Similar Netflix Users

Description: This project aims to identify pairs of Netflix users with similar movie preferences based
on their rated movies. The similarity metric used is the Jaccard similarity where users with a Jaccard similarity > 0.5 are considered similar. Identifying such similar user pairs is
instrumental for enhancing recommendation systems, improving user engagement, and
personalizing content delivery on platforms like Netflix.
With a dataset comprising approximately 100,000 users, 17,770 movies, and
65 million rating records, a brute-force similarity computation (~5 billion pairs) is
computationally infeasible. To address this challenge, Locality-Sensitive Hashing
(LSH) combined with MinHash signatures is utilized. This approach approximates
similarities and efficiently narrows down the search to likely similar pairs, significantly
reducing computational overhead while maintaining high accuracy in similarity detection.

Repository: [Similar_Netflix_Users](https://github.com/Manudpb/Relevant-projects/tree/main/Similar_Netflix_Users)

4. Biased Sentences In News

Description: Biased news and media framing significantly under-
mine people's right to access accurate information,
as misinformation can shape or distort public
understanding and perspectives. The dataset created by Lim et al. (2020)
includes labels from annotators with both preknown
and not preknown knowledge of news events, this
paper explores whether including annotators'
knowledge statuses into the training process can
improve the performance of a few-shot BERT
model and GPT-4o on biased news classification
task. The classification task uses a four-point
scale: Neutral, Slightly biased, Biased, and Very
biased. Specifically, we investigate how integrating
annotators' knowledge statuses affects the models'
overall performance.

Repository: [Biased_Sentences_In_News](https://github.com/Manudpb/Relevant-projects/tree/main/Biased_Sentences_In_News)

5. Using Foundation Models MOMENT and OpenCity for Crowd Flow Prediction

Description: Forecasting crowd flow traffic in urban areas is essential to ensure
public safety and optimize traffic management. However, traditional
crowd flow prediction models are usually strongly dependent on
training data. Although these models are effective within their task
scope, they often lack the scalability to handle the dynamic nature
of real-world crowd flow, such as adapting to weather changes or
new public events. We aim to apply these models such as MOMENT and OpenCity to
the crowd flow prediction task and to explore whether they can
provide a more efficient and adaptable approach.

Repository: [Crowd_Flow_Prediction](https://github.com/Manudpb/Relevant-projects/tree/main/Crowd_Flow_Prediction)

6. On the application of WebGraph compression to social network graphs

Description: In this paper, we investigate the possible application of the Web-
Graph compression method, proposed by Boldi and Vigna, to
social networks. The WebGraph framework achieves compression
of web graphs by leveraging the lexicographical ordering of URLs,
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

Repository: [WebGraph_Compression](https://github.com/Manudpb/Relevant-projects/tree/main/WebGraph_Compression)

7. Hybrid Recommender Systems: BERT4Rec vs. Neural Collaborative Filtering on Sparse Interaction Data

Description: Built and compared two hybrid recommender systems — a transformer-based BERT4Rec (sequential, masked-item prediction) and a Neural Collaborative Filtering (NCF) model — on a large, extremely sparse interaction dataset (346K interactions, 323K users, 65K items, with 94.6% of users having only a single interaction). Both models were extended with item metadata (title embeddings, category encodings, rating scores) to address the cold-start problem inherent to such sparse data, combining collaborative and content-based filtering into a single hybrid architecture. Evaluated via Recall@10 on a public Kaggle leaderboard and NDCG@10 on the full test set, the two models produced conflicting rankings: BERT4Rec generalized better across the full test set (NDCG@10 of 0.45 vs. NCF's 0.24), but NCF outperformed on the smaller competition subset (Recall@10 of 0.019 vs. 0.0123) — a result traced to that subset likely skewing toward cold-start users, where NCF's direct use of metadata holds up better than BERT4Rec's reliance on sequence history. The finding underscores a broader lesson in recommender system evaluation: model rankings can flip entirely depending on the sparsity and composition of the test population, not just the modeling approach itself.

Repository: [Hybrid_Recommender_Systems](https://github.com/Manudpb/Relevant-projects/tree/main/Hybrid_Recommender_Systems)

8. MLP & CNN Architecture Tuning, and a Transfer-Learning Clock-Reading CNN

Description: Investigates how to systematically tune MLP and CNN architectures on Fashion-MNIST and CIFAR-10, comparing a "tabula rasa" approach (extreme simple and complex models) against a priori architecture choices, and testing how well hyperparameter insights transfer across datasets with different within-class variance. Finds that architecture choice matters more than raw parameter count on higher-variance data like CIFAR-10, and maps out practical hyperparameter ranges for batch size, learning rate, dropout, and kernel size. The second half applies these insights to a concrete task — reading the time from images of an analog clock — using transfer learning on VGG16 with a multi-head output (separate classification heads for hours and minutes) and a custom "common sense difference" (CSD) metric that correctly handles the cyclical nature of a clock face. The best configuration, a multi-head classification model, reached a CSD of 1.02 minutes — accurate to within about a second of human-level time-telling.

Repository: [MLP_&_CNN](https://github.com/Manudpb/Relevant-projects/tree/main/MLP_&_CNN)

9. Generative Models (VAE/GAN) and Sequence-to-Sequence Arithmetic with RNNs/LSTMs

Description: A two-part project. The first half trains Convolutional Autoencoders, Variational Autoencoders, and GANs to generate novel dog face images, comparing reconstruction quality and latent-space behavior between architectures, visualizing linear interpolation through the latent space, and explaining why VAEs and GANs need very different hyperparameter regimes despite both being generative models. The second half builds and evaluates RNN/LSTM architectures that perform two-digit addition and subtraction across three input/output modalities — text-to-text, image-to-text, and text-to-image — including six distinct encoder-decoder designs for the image-generation task. The best text-to-text model reached 98% accuracy and the best image-to-text model (a multi-layer LSTM) reached 77%, but analysis across all variants suggests the networks are largely memorizing operation patterns rather than learning true arithmetic — most visible in their consistent struggles with underrepresented operation types.

Repository: [Generative Models](https://github.com/Manudpb/Relevant-projects/tree/main/Generative_Models)

10. Reinforcement Learning on CartPole: From Deep Q-Networks to Soft Actor-Critic

Description: A series of RL coursework projects benchmarking five algorithm families on the CartPole environment, progressing from value-based to policy-gradient to entropy-regularized methods. Starting with Deep Q-Networks (DQN), the collection shows how Experience Replay and Target Networks each address a different source of training instability — replay breaking correlation between consecutive states, and a target network stabilizing a moving optimization target — with their combination giving the best balance of stability and performance. It then compares three policy-gradient methods (REINFORCE, Actor-Critic, and Advantage Actor-Critic), finding that Monte Carlo-based REINFORCE actually outperforms the bootstrapped methods in this low-cost environment, while normalized advantage estimation is what makes A2C competitive. Finally, it implements Soft Actor-Critic (SAC) with entropy-regularized exploration and the clipped double-Q trick, showing that higher entropy temperatures and double-Q both improve performance, while surfacing a late-training instability that motivates adaptive entropy tuning as future work.

Repository: [Reinforcement Learning on CartPole](https://github.com/Manudpb/Relevant-projects/tree/main/CartPole)


11. HarmoGen: A Co-Creative Evolutionary System for Interactive Melody Harmonization

Description: HarmoGen is an interactive evolutionary system that harmonizes a musician's melody in collaboration with the user, rather than fully automating the process. Given a monophonic MIDI melody, the system generates candidate four-part harmonizations and iteratively refines them across generations based on the user's own ratings, using a Random Forest surrogate model to learn their subjective taste in place of a fixed fitness function. Mutation strategies shift adaptively over generations, favoring broad harmonic exploration early and conservative, incremental refinement as the user's preferences become clearer. The system was evaluated in a small-scale user study (n=11) comparing the harmonization experience across short and long melodies, assessing perceived co-creativity, sense of control, responsiveness, and satisfaction with the final result. HarmoGen was deployed as a web app where users could upload their own melody or try two preset ones, listen to and rate generated harmonizations over successive generations, and download their preferred result as a MIDI file.

Repository: [HarmoGen](https://github.com/Manudpb/Relevant-projects/tree/main/)


Contact

Email: mdpbelizon@gmail.com
