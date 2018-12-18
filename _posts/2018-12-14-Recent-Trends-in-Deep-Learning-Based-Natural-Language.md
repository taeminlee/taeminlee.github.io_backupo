---
title: "Recent Trends in Deep Learning Based Natural Language Processing"
classes: wide
math: true
date: 2018-12-14
categories: NLP DeepLearning
---


최근 딥러닝을 기반으로 하는 NLP(Natural Language Processing: 자연어처리)나 Text Mining 기법들이 Top Conference에 많이 투고되고 있다. 현재 우리 연구실에서도 이러한 트렌드를 쫓아가려고 최신 연구동향 Survey paper

> Young, Tom, et al. "Recent trends in deep learning based natural language processing." ieee Computational intelligenCe magazine 13.3 (2018): 55-75.

를 분석하고 있다. 운 좋게도 이기창 선배님 블로그 [ratsgo's blog](https://ratsgo.github.io/natural%20language%20processing/2017/08/16/deepNLP/)에 해당 논문이 한글로 잘 번역되어 있어서 페이퍼 읽는 시간을 줄일 수 있었다. 본 블로그는 이 페이퍼와 [ratsgo's blog](https://ratsgo.github.io/natural%20language%20processing/2017/08/16/deepNLP/)를 읽으며 정리하면서 쓰는 노트이다! 

## Abstract

다 읽고 나서 쓰겠습니다. 죄송합니다 ^^

## Introduction

자연어처리(NLP:Natural Language Processing, 이후 NLP)와 Deep Learning 등에 대한 연구 배경이 나온다. Introduction의 본론만 간추려서 말하자면

> 최근 Dense vector representation 기반의 신경망 기법들은 다양한 NLP task에서 두각을 드러내고 있다. 이러한 트랜드는 word embedding [2,3] 과 deep learning [4] 기법의 성공 덕에 촉발되었다. 그러니 NLP에 사용되는 다양한 딥러닝 모델 및 기법에 대해 살펴보겠다.

로 요약할 수 있다. 

## Distributed Representation

Distributed Representation은 왜 필요하며, 어떤 기법들이 있는가가 이 챕터의 핵심이다. 

- Distributed Representation은 왜 필요한가?
이 논문에서는 딥러닝 이전의 NLP 기법을 통계적 NLP (Statistical NLP)로 통칭하나보다. 여튼 이 전통적인 NLP 기법은 차원의 저주 (curse of dimensionality) 문제가 발생한다고 한다. [5]번 논문에서 설명되어있다는데 내용이 아주 길어서 안읽었다. 일단 받아드리고 나중에 읽어야지.
- Distributed Representation에는 어떤 기법이 있는가?
    - Word Embeddings
    - Word2Vec (Word Embedding의 일종)
    - Character Embeddings

## Distributed Representation - Word Embeddings

Word Embedding 기법은 '단어'를 *d-* `d-`차원의 vector로 embedding시키는 기법을 총칭하는 듯하다.

그러나 단어를 그냥 embedding 시키면 안되고, 다음 핵심 가정 ([Distributed Hypothesis](https://en.wikipedia.org/wiki/Distributional_semantics#Distributional_hypothesis))에 입각하여 embedding 시켜야 한다.  (막말로 Bag of Words도 단어를 벡터로 보낸다는 점에서 보면 Word Embedding이지 않은가?)

- Word Embedding의 핵심 가정: [Distributed Hypothesis](https://en.wikipedia.org/wiki/Distributional_semantics#Distributional_hypothesis)

> 비슷한 의미를 가지는 단어들은 유사한 문맥에서 나타난다.

(아직까지는 솔직히 선뜻 와닿지는 않는다. 나중에 예제 만들어가면서 파고들어봐야지.)

- 이 핵심 가정은 word embedding 기법이 추구해야하는 성질을 정의한다.

> Thus, these vectors try to capture the characteristics of the neighbors of a word.

왜 그런지 고민을 좀 해봤는데, 처음에는 입문자로서 다소 비약처럼 느꼈다. 나는 이 문장을 다음과 같이 해석했다. 

1. embedding 시키고자 하는 단어를 *`w`*라고 해보자. 
우리 목표는 *`w`*를 '잘' embedding (저차원 공간에 vector로서 표현)하려고 한다.

2. Distributed Hypothesis에 따르면 비슷한 의미를 가지는 단어라면 유사한 문맥에서 나타난댄다.

3. "*`w`*가 어떠한 문맥에서 나타난다" 라는 관찰을 다음과 같이 설정해보자.

문맥: "... 왼쪽주변단어2, 왼쪽주변단어1, *`w`*, 오른쪽주변단어1, 오른쪽주변단어2..."

4. 만약 서로 다른 단어 `w1`과 *`w2`* 가 완벽히 동일한 의미를 가지고 있다면, 다음과 같은 문맥이 발생할 확률이 정확하게 일치할 것이다. 

문맥1: "... 왼쪽주변단어2, 왼쪽주변단어1, *`w1`*, 오른쪽주변단어1, 오른쪽주변단어2..."
문맥2: "... 왼쪽주변단어2, 왼쪽주변단어1, *`w2`*, 오른쪽주변단어1, 오른쪽주변단어2..."

극단적인 상황이라 설득력은 떨어지겠지만, 이렇게 이해하면 될 것 같다. 문헌들이 찾아보니 모든 관찰가능한 문맥에 대해 *`w1`*과 출몰할 확률이 정확하게 일치하는 단어인 *`w2`*를 찾아냈다고 가정해보자. 그렇다면 그 둘은 완벽하게 뜻이 동일한 단어로 볼 수 있지 않을까?

5. 이렇게 이해한다면, 단어 *`w`*의 의미는 그 주변 문맥에 의해 '정의된다'라고도 해석할 수 있다. 

6. 결론적으로, *`w`*를 embedding 시킨 vector는 그 단어가 사용되는 주변 문맥을 잘 담아내는 vector여야 한다. 

내가 이해한 것이 맞다고 가정하고 그 뒷 내용에 대해 진술하겠다. 

이런 식으로 만들어진 vector들인 distributional vector들은 다음과 같은 장점을 가진다.

- 단어간의 유사도를 측정(measuring)할 수 있다.

$$\int f(x)~dx$$