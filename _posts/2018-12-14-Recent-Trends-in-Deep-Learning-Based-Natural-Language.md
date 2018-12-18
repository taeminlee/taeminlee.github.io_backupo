---
title: "Recent Trends in Deep Learning Based Natural Language Processing"
classes: wide
math: true
date: 2018-12-18
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

Word Embedding 기법은 '단어'를 *d-*차원의 vector로 embedding(mapping과 비슷하게 해석하면 될
 듯하다.)시키는 기법을 총칭하는 듯하다. 그러나 단어를 단순하게 vector 형식으로 표현하는 기법을 거창하게 distributed representation 이라고 부르는 것 같지는 않다. Bag of Words도 단어를 벡터로 보낸다는 점에서 보면 Word Embedding이지 않은가?  서베이 논문에 따르면, Distributional vector 또는 word embedding은 다음 핵심 가정 ([Distributional Hypothesis](https://en.wikipedia.org/wiki/Distributional_semantics#Distributional_hypothesis))을 따른다.

- [**Distributional Hypothesis**](https://en.wikipedia.org/wiki/Distributional_semantics#Distributional_hypothesis): 같은 문맥에서 사용되거나 나타나는 (occur) 단어들은 유사한 의미를 가지는 경향이 있다.

> Words that are used and occur in the same contexts tend to purport similar meanings (Wikipedia)

서베이에서 나온 표현보다는 위키피디아 표현을 빌렸다. 이 표현이 본 게시글의 논리 전개에 더 적합한 듯하여. 

이 가정을 도대체 어디다가 쓰는가? 서베이 논문에서는 다소 불친절하게 나와있다. 바로 다음 문장이
 이 문장이다. 

> Thus, these vectors try to capture the characteristics of the neighbors of a word.

Word embedding 기법이 Distributional Hypothesis를 기반으로 하는 것과, Word embedding 기법이 추구하고자 하는 바(직역: 벡터들은 주변 단어들의 특성을 담아내려고한다)와 무슨 상관인가? 왜 그런지 고민을 좀 해봤는데, 처음에는 입문자로서 다소 비약처럼 느꼈다. 나는 이 문장을 다음과 같이 해석했다. 

1. embedding 시키고자 하는 단어를 `w` 라고 해보자. 
우리 목표는 `w`를 '잘' embedding (저차원 공간에 vector로서 표현)하려고 한다.

2. Distributional Hypothesis에 따르면 같은 문맥에서 발견되는 단어들은 유사한 의미를 가지는 경향이 있다고한다. 

3. 조금 더 엄밀한 상황 설정을 위해, " `w`가 어떠한 문맥에서 발견되다" 라는 관찰을 다음과 같이 설정해보자. (이와 같이 문맥을 주변 단어들이라는 것은 중요한 가정인 듯한데, 여기서는 일단 받아드리고 넘어가도록 하자.)
문맥: "... 왼쪽주변단어2, 왼쪽주변단어1, `w` , 오른쪽주변단어1, 오른쪽주변단어2..."

4. (이해를 위한 극단적인 예시이니 skip해도 무방) 
	- 만약 서로 다른 단어 `w1`과 `w2` 가 완벽히 동일한 의미를 가지고 있다면, 임의로 주어진 문맥에서 해당 단어들이 발견될 확률이 정확하게 일치할 것이다. 
        - 문맥1: "... 왼쪽주변단어2, 왼쪽주변단어1, `w1`, 오른쪽주변단어1, 오른쪽주변단어2..."
        - 문맥2: "... 왼쪽주변단어2, 왼쪽주변단어1, `w2`, 오른쪽주변단어1, 오른쪽주변단어2..."
    - 한국어를 사용하는 모든 사람들이 '그러나'라는 용어와 '하지만'이라는 두 비슷한 용어에 뉘앙스 차이가  전혀 없다고 느끼고, 두 용어가 사용되는 빈도 또한 5:5로 같다고 가정해보자.  또한 다음과 같이 임의의 문헌에서 임의의 문맥을 뽑고, 가운데 단어를 빈칸처리 (`?`) 해놓았다고 가정해보자.
    	- 문맥:  "왼쪽주변단어2, 왼쪽주변단어1, `?`, 오른쪽주변단어1, 오른쪽주변단어2..."  
    - 직관적으로 생각해보면, `?` 자리에 '그러나'가 들어갈 확률과 '하지만'이 들어갈 확률은 완전하게 같을 것이다. 두 용어는 완전히 같으니까.
    - 조금 더 일반화 해서, 임의의 문맥  `c`가 주어졌을 때, `c`에서 단어 `w1`가 발견될 확률(`P(w1|c)`)과 단어 `w2`가 발견될 확률(`P(w2|c)`)이 정확하게 같았다고 가정해보자. 이 둘은 사용되는 모든 문맥에서 상호호환 가능한 (interchangeable) 용어가 되시겠다. 그렇다면 그 둘은 완벽하게 뜻이 동일한 단어로 볼 수 있지 않을까? 
    	- i.e. , if ( `P(w1|c) = P(w2|c) for any context c` ) then  `meaning of w1 = meaning of w2`
    	- 위 if-then 구문을 조금 더 수학적으로 간결하게 정리해보자. 우리가 경험적으로 관찰한 문맥 `c` 의 집합을 `C` (쉬운말로하면 그냥 수집한 문맥 데이터 집합)라고 할때, 다음 식을 만족하면 는 완전히 같은 의미를 가지는 용어로 볼 수 있다.
    $$\sum_{c \in C}{|P(w_{1}|c)-P(w_{2}|c)|}=0$$
    - 이제 5에서는 4에서 서술한 바를 조금 relaxing하여 설명해볼 것이다.
5. 만약 서로 다른 단어 `w1`과 `w2` 가 완전하게 같지는 않지만 비슷한 의미를 가지고 있다면 어떨까? 이때는 4번 상황과는 다르게,  `P(w1|c)`와 `P(w2|c)` 가 같지 않은 문맥 `c`가 하나 이상 존재할 것이다.  
	- i.e.,  `There exists some context c* s.t. P(w1|c*) != P(w2|c*)`
    - `w1`과 `w2` 는 어느정도 유사할까? Distributional Hypothesis에 입각해서 설며해보면, 유사하면 유사할 수록 이 `P(w1|c)`와 `P(w2|c)`

6. 어떠한 문맥  `c`에서는, `c`에서 단어 `w1`가 발견될 확률(`P(w1|c)`)과 단어 `w2`가 발견될 확률(`P(w2|c)`)는  정확하게 같았다면, 그 둘은 완벽하게 뜻이 동일한 단어로 볼 수 있지 않을까? 

7. 이렇게 이해한다면, 단어 `w`의 의미는 그 주변 문맥에 의해 '정의된다'라고도 해석할 수 있다. 

8. 결론적으로, `w`를 embedding 시킨 vector는 그 단어가 사용되는 주변 문맥을 잘 담아내는 vector여야 한다. 

- 이렇게 이해한다면, 단어 `w`의 의미는 그 문맥에 의해 '정의된다'라고도 해석할 수 있다.

내가 이해한 것이 맞다고 가정하고 그 뒷 내용에 대해 진술하겠다. 

이런 식으로 만들어진 vector들인 distributional vector는 '단어간의 유사도를 측정(measuring)할 수 있다는 장점이 있다.