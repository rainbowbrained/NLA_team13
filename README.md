# NLA_team13: Analyzing the Evolution of Weight Matrices in the OPT-13B LLM During Training

Our idea is to take the weights of a certain LLM, for example, OPT-13B (https://github.com/facebookresearch/metaseq/tree/main/projects/OPT), as well as the weights of all training checkpoints. 

All weights are represented in matrix form, so we will be interested in the matrices of Key, Query, Value, Output Projection from attention, the tokens embedding matrix, and the two weight matrices for each layer of the Feedforward Neural Network (FFN). 

Next, we will analyze how the characteristics of these matrices change during the training process: 
1) We will examine and analyze their spectrum.
2) We will investigate how orthogonal the matrices are.
3) We will look at how much these matrices change during training (how far they are from their initialized values).
4) It would be interesting to check how correlated the rows of W1 and the columns of W2 in the FFN matrices are (there is a hypothesis that they should be).
5) We will evaluate how sparse all these matrices are.
6) We will analyze how various matrix norms change.
7) And more.

P.S. If we can find checkpoints of the model at the stages of Supervised Fine-Tuning (SFT) and Reinforcement Learning from Human Feedback (RLHF), it will be even more interesting to see what happens with these characteristics at those stages.


# TODO
🔴 - никто не взял тему(
🟡 - забронировано
🟢 - сделаль

1. 🔴 Для каждого чекпоинта, для каждого слоя, смотреть основные L-p нормы матриц query, key, value, dense_h_to_4h, dense_4h_to_h, embed_in
2. (🟡 Анита) Для каждого чекпоинта, для каждого слоя, смотреть спектр и ранк матриц query, key, value, dense_h_to_4h, dense_4h_to_h, embed_in
3. 🔴 Для каждого чекпоинта, для каждого слоя, смотреть насколько сильно поменялись матрицы query, key, value, dense_h_to_4h, dense_4h_to_h, embed_in по сравнению с инициализацией (то есть нулевым чепкоинтом)
4. 🔴 Для каждого чекпоинта, для каждого слоя, смотреть насколько матриц query, key, value, dense_h_to_4h, dense_4h_to_h, embed_in разряжены
5. 🔴 Для каждого чекпоинта, для каждого слоя, смотреть насколько матриц query, key, value, dense_h_to_4h, dense_4h_to_h, embed_in ортогональны
6. (🟡 Антон) корреляции весов 
7. (🟡 Анита) лит обзор

#  notes
Рабочая идея потыркаться с ллмками используя ранднла. Например улучшить обучение или ещё что-то

В качестве модели берем pythia. У этой модели есть много вариантов, советую тестить код на pythia-70M (то есть 70 миллионов параметров), а непосредственно все эксперименты для презентации уже проводить на pythia-1b модели.
Выбрал эту модель, потому что у неё 153 чекпоинта обучения, что много и круто.
Я к этом сообщению прикрепляю коллаб ноутбук, где я смотрю на динамику на нормы Фробениуса MLP весов с второго слоя для каждого чекпоинта. Это все для примера, вам там по сути нужно будет взять и заменить функцию, которая принимает на вход матрицу, на какую-то свою и построить похожий график

учтите, что 70М модель отличается от 1b модели количеством слоев и внутренней размерностью эмбэддингов, то есть не хардкодьте эти параметры
