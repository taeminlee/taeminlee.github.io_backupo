---
title: "Paper: Recent trends in deep learning based natural language processing"
classes: wide
math: true
date: 2018-12-27
categories: NLP DeepLearning PaperReview
---

Young, T. et al., 2018. Recent trends in deep learning based natural language processing. ieee Computational intelligenCe magazine, 13(3), pp.55–75.

https://arxiv.org/abs/1708.02709

https://ratsgo.github.io/natural%20language%20processing/2017/08/16/deepNLP/

<!-- TOC -->

- [categories: NLP DeepLearning PaperReview](#categories-nlp-deeplearning-paperreview)
- [Outline](#outline)
- [Definitions](#definitions)
    - [Distrubuted Representation](#distrubuted-representation)
    - [Recurrent Neural Networks](#recurrent-neural-networks)
      - [Attention mechanism](#attention-mechanism)
    - [Recursive Neural Networks](#recursive-neural-networks)
    - [Deep Reinforced Models and Deep Unsupervised Learning](#deep-reinforced-models-and-deep-unsupervised-learning)
    - [Memory Networks](#memory-networks)
- [Networks or Units](#networks-or-units)
    - [Distributed Representation](#distributed-representation)
    - [Convolution Neural Network](#convolution-neural-network)
    - [Recurrent Neural Network](#recurrent-neural-network)
    - [Recursive Neural Network](#recursive-neural-network)
    - [Deep Reinforced Models and Deep Unsupervised Learning](#deep-reinforced-models-and-deep-unsupervised-learning-1)
    - [Memory-Augmented Networks](#memory-augmented-networks)
- [Key Insights](#key-insights)
    - [Convolutional Neural Networks](#convolutional-neural-networks)
    - [Recurrent Neural Networks](#recurrent-neural-networks-1)
      - [Attention mechanism](#attention-mechanism-1)
    - [Reinforced Models](#reinforced-models)
- [Key Papers](#key-papers)
    - [Distributed Representation](#distributed-representation-1)
    - [Convolutional Neural Networks](#convolutional-neural-networks-1)
    - [Recurrent Neural Networks](#recurrent-neural-networks-2)
    - [Recursive Neural Networks](#recursive-neural-networks-1)
    - [Deep Reinforced Models and Deep Unsupervised Learning](#deep-reinforced-models-and-deep-unsupervised-learning-2)
    - [Memory-Augmented Networks](#memory-augmented-networks-1)
- [Applications](#applications)
    - [Distributed Representation](#distributed-representation-2)
    - [Convolutional Neural Networks](#convolutional-neural-networks-2)
    - [Recurrent Neural Networks](#recurrent-neural-networks-3)
    - [Recursive Neural Networks](#recursive-neural-networks-2)
    - [Deep Reinforced Models and Deep Unsupervised Learning](#deep-reinforced-models-and-deep-unsupervised-learning-3)
      - [Reinforced Models](#reinforced-models-1)
      - [Unsupervised Learning](#unsupervised-learning)
      - [Deep Generative Models](#deep-generative-models)
    - [Memory-Augmented Networks](#memory-augmented-networks-2)
- [performances on NLP tasks](#performances-on-nlp-tasks)

<!-- /TOC -->

## Outline

1. Distributed Representation
2. Networks
    1. Convolutional Neural Networks
    2. Recurrent Neural Networks
    3. Recursive Neural Networks
    4. Deep Reinforced Models and Deep Unsupervised Learning
    5. Memory-Augmented Networks
3. Performance of Different Models on Different NLP Tasks

## Definitions

#### Distrubuted Representation

- >비슷한 의미를 지닌 단어는 비슷한 문맥에 등장하는 경향이 있을 것이라는 내용이 핵심
- >분산표상 벡터의 주된 장점은 이 벡터들이 단어 간 유사성을 내포
- >단어 임베딩은 레이블이 없는 방대한 말뭉치에서 ‘보조적인 목적함수(예컨대 이웃단어로 중심단어를 예측한다, 각 단어벡터는 일반적인 문법적, 의미적 정보를 내포한다)’를 최적화함으로써 사전 학습
- Context에 의해 semantic이 정해진다는 distributional hypothesis에 기반한 표현 기법들의 총칭

#### Recurrent Neural Networks

- > 순차적인 정보를 처리하는 네트워크
- >  ‘recurrent’라는 용어는 모델이 입력 시퀀스의 각 인스턴스에 대해 같은 작업을 수행하고 아웃풋은 이전 연산 및 결과에 의존적이라는 데에서 붙었다.
- ![](../assets/2018-12-17-recent-trends-in-deep-learning-based-natural-language-processing-summary/2018-12-26-22-37-26.png)

##### Attention mechanism

- > 어텐션 매커니즘은 디코더가 입력 시퀀스를 다시 참조할 수 있게 하여 위의 문제를 완화하려고 시도
- > 어텐션(attention) 매커니즘은 인코더에서 만들어진 히든 벡터(hidden vector)들을 저장. 디코더는 각 토큰을 생성하고 있는 중에 이 벡터들에 액세스

#### Recursive Neural Networks

- > 문장의 문법적 구조 해석을 보다 용이하게 하기 위해 트리 구조 모델이 사용
- > Recursive Neural Network에서 각 노드는 자식 노드의 표현(representation)에 의해 결정

#### Deep Reinforced Models and Deep Unsupervised Learning

- > 강화학습(Reinforcement Learning)은 보상(reward)을 얻기 전에 행동(action)을 수행하도록 에이전트(agent)를 학습시키는 기법
- > VAE는 적절한 샘플을 뽑을 수 있도록 하는 히든 코드 스페이스(hidden code space)에 사전 분포(prior distribution)를 부과
- > 결정론적인 인코더 기능(deterministic encoder function)을 학습된 사후확률 인식(posterior recognition) 모델로 대체함으로써 오토인코더 아키텍처를 수정한다. 모델은 데이터를 잠재표현(latent representation)으로 인코딩하는 인코더와 잠재공간에서 샘플을 생성하는 생성자(generator) 모델로 구성된다. 이 모델은 관측된 데이터의 로그 우도에 대한 변량적 하한(variational lower bound)을 최대화함으로써 학습
- > GAN은 두 가지 경쟁 네트워크로 구성된다. 생성자 네트워크는 잠재 공간에서 데이터 인스턴스로 잠재 표현을 인코딩한다. 반면 판별자 네트워크는 실제 데이터와 생성자가 만든 인스턴스를 구별하기 위해 생성자와 동시에 학습된다. GAN은 실제 데이터 분포인 p(x)를 명시적으로 표상하지는(represent) 않는다.

#### Memory Networks

- > 인코더의 히든 벡터들은 모델의 ‘내부 메모리(internal memory)’ 항목으로 볼 수 있다. 최근에는 모델이 상호작용할 수 있는 메모리의 형태로 뉴럴 네트워크들을 연결

## Networks or Units

#### Distributed Representation

- CBoG in Word2Vec
- skip-gram in Word2Vec
- GloVe

#### Convolution Neural Network

- CNN (Convolutional Neural Network)
- DCNN (Dynamic Connvolution Neural Network)
- MCDNN (Multi Column Convolutional Neural Network)
- DMCNN (Dynamic Max-pooling Convolutional Neural Network)

#### Recurrent Neural Network

- RNN or Elman Network
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Units)
- Attention mechanism

#### Recursive Neural Network

- RNN

#### Deep Reinforced Models and Deep Unsupervised Learning

- VAE
- GAN

#### Memory-Augmented Networks

- Memory Network

## Key Insights

#### Convolutional Neural Networks

- Pros
  - > CNN은 contextual window 내에 있는 의미적 단서를 추출하는 데 고도로 효율적이다.
- Cons
  - > 그러나 CNN은 매우 많은 데이터를 필요로 한다. CNN 모델은 방대한 양의 데이터를 요구하는 다수의 학습 파라메터를 포함한다. CNN은 데이터가 부족할 때는 문제가 된다. CNN의 다른 이슈는 먼 거리의 문맥 정보를 모델링하기가 불가능하고 그들의 표현(repesentation)에서 시퀀셜한 순서를 보존할 수 없다는 것이다. Recursive NN과 같은 네트워크가 이런 학습에 적합하다.

#### Recurrent Neural Networks

- Pros
  - > RNN이 시퀀스 모델링에 적합한 다른 요인은 매우 긴 문장, 단락, 심지어 문서(Tang et al., 2015)를 포함해 다양한 텍스트 길이를 모델링할 수 있는 능력
  - > RNN은 또한 시간 분산 조인트 처리(time distributed joint processing)를 위한 네트워크 지원을 제공
- Cons
  - > 그러나 RNN이 다른 네트워크보다 우월하다고 결론을 내리는 것은 잘못된 것이다. 최근에 여러 연구는 CNN이 RNN보다 우월하다는 증거를 제시한다. 언어모델링(Langurage modeling) 같은 RNN에 적합한 태스크일지라도, CNN은 RNN보다 경쟁력 있는 성능을 달성했다.
  - > 단어 수준의 최대 우도 전략의 또다른 문제점은 학습목표(training objective)가 테스트 측정지표(test metric)과 다르다는 사실이다. 기계번역, 대화시스템 등을 평가하는 데 쓰이는 측정지표(BLUE, ROUGE)가 단어 수준 학습 전략으로 어떻게 최적화될 수 있는지 분명하지 않다. 경험적으로, 단어 수준 최대 우도로 학습된 대화시스템은 둔하고 근시안적인 반응을 생성하는 경향(Li et al., 2016b)이 있고, 단어 수준 최우도 기반의 텍스트 요약도 역시 비간섭적(incoherent)이거나 반복적인 요약을 생성하는 경향이 있다(Paulus et al., 2017).
- ETC
  - > Yin et al. (2017)은 RNN과 CNN 성능에 관한 흥미있는 인사이트를 제시한다. 감성분류, QA, 품사태깅을 포함하는 여러 NLP task를 평가한 후에 그들은 ‘완전한 승자는 없다’고 결론 내렸다. 각 네트워크의 성능은 태스크가 요구하는 글로벌 시맨틱(global semantics)에 의존한다
  - > Dai and Le(2015)는 다양한 작업에서 사전 학습된 파라메터를 LSTM 모델의 초기값으로 쓰는 실험을 수행했다. 그들은 방대한 비지도 말뭉치로 사전 학습된 문장 인코더가 단어 임베딩만 사전학습해 사용하는 것보다 더 나은 정확성을 보임을 입증
  - > 다음 토큰을 예측하는 것은 문장 자체를 재구축(reconstruct)하는 것보다 더 나쁜 보조 목표인 것으로 판명됐다. LSTM의 히든 스테이트는 단기(shor-term) 기억에만 충실하기 때문이다.

##### Attention mechanism

- Pros
  - 어텐션 매커니즘은 특히 긴 시퀀스에 대해 모델의 성능을 향상

#### Reinforced Models

## Key Papers

#### Distributed Representation

- Word2Vec
  - 신경망 구조를 이용한 word embedding
  - CBoW, skip-gram 2가지 네트워크 구조 제안
  - > 사용자가 지정한 윈도우(주변 단어 몇개만 볼지) 내에서만 학습/분석
- GloVe
  - word co-occurrence count matrix를 이용
  - > 임베딩된 단어벡터 간 유사도 측정을 수월하게 하면서도 말뭉치 전체의 통계 정보를 좀 더 잘 반영해보자”가 GloVe가 지향하는 핵심 목표
  - > 임베딩된 두 단어벡터의 내적이 말뭉치 전체에서의 동시 등장확률 로그값이 되도록 목적함수를 정의했습니다. (their dot product equals the logarithm of the words’ probability of co-occurrence)
- Fasttext
- ELMo
  - https://arxiv.org/abs/1802.05365
  - > Recent approaches in this area encode such information into its embeddings by leveraging the context. These methods provide deeper networks that calculate word representations as a function of its context
  - > The mechanism of ELMo is based on the representation obtained from a bidirectional language model.

- BERT
  - https://arxiv.org/abs/1810.04805
  - > Recently, Devlin et al. [45] proposed BERT which utilizes a transformer network to pre-train a language model for extracting contextual word embeddings.

#### Convolutional Neural Networks

- CNN(Convolutional Neural Network) for sequence of words
  - http://www.jmlr.org/papers/volume12/collobert11a/collobert11a.pdf
  - > 콘볼루션 계층과 맥스풀링의 이런 조합은 보다 깊은 CNN 네트워크를 만들기 위해 종종 겹쳐 쌓게 된다. 이러한 sequential convolution은 풍부한 의미 정보를 포함하는 고도로 추상화된 표현을 잡아내 문장의 분석을 개선할 수 있도록 한다. 깊은 콘볼루션 구조의 필터(커널)은 문장 피처의 전체 요약을 만들기까지 문장의 넓은 부분을 커버
- DCNN (Dynamic Convolutional Neural Network)
  - http://www.aclweb.org/anthology/P14-1062
  - >문맥적 범위를 넓히기 위해, 전통적인 윈도우 접근법은 종종 time-dealy neural network(TDNN)와 결합

#### Recurrent Neural Networks

- LSTM
  - http://www.bioinf.jku.at/publications/older/2604.pdf
  - https://arxiv.org/pdf/1308.0850.pdf
  - >  RNN에 forget gate를 추가했다. 이러한 독특한 매커니즘을 통해 배니싱 그래디언트 문제, 익스플로딩 그래디언트 문제(exploding gradient problem_를 모두 극복할 수 있다
- GRU
  - https://arxiv.org/pdf/1406.1078.pdf
  - > GRU는 reset gate와 update gate의 두 개 gate로 구성되며 LSTM처럼 메모리를 보호한다. GRU는 LSTM보다 효율적인 RNN이 될 수 있다. 
  - https://arxiv.org/pdf/1412.3555.pdf
    - Chung, Junyoung, et al. "Empirical evaluation of gated recurrent neural networks on sequence modeling." arXiv preprint arXiv:1412.3555 (2014).
- Sequence2Sequence
  - https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf
  - >  시퀀스와 다른 시퀀스를 매핑시키는 일반적인 deep LSTM encoder-decoder 프레임워크를 제안
- Attantion Mechanism
  - https://arxiv.org/pdf/1409.0473.pdf
- Parallelized Attention : The Transformer
  - https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf
  - > Vaswani et al. [113] proposed the Transformer which dispensed the recurrence and convolutions involved in the encoding step entirely and based models only on attention mechanisms to capture the global relations between input and output. As a result, the overall architecture became more parallelizable and required lesser time to train along with positive results on tasks ranging from translation to parsing.

#### Recursive Neural Networks

- Recursive Neural Network
  - https://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf

#### Deep Reinforced Models and Deep Unsupervised Learning

- VAE
  - https://arxiv.org/pdf/1703.00955.pdf

#### Memory-Augmented Networks

- Memory Networks
  - https://arxiv.org/pdf/1410.3916.pdf
- End-to-end Memory Networks
  - https://arxiv.org/pdf/1503.08895.pdf

## Applications

#### Distributed Representation

- [감성 분류](http://www.icml-2011.org/papers/342_icmlpaper.pdf)
  - Contribution : 도메인 특성에 맞는 학습
  - Model : Stacked Denoising Autoencoder
- [문장 합성](http://www.karlmoritz.com/_media/hermannblunsom_acl2013.pdf)
  - Model : combinatory categorical autoencoder
- [감성 분류](http://ir.hit.edu.cn/~dytang/paper/sswe/14ACL.pdf)
  - Contribution : 감성 극성을 고려한 워드 임베딩
  - Note : 손실함수에 정답 극성을 포함

#### Convolutional Neural Networks

- [문서 분류](https://ronan.collobert.com/pub/matos/2008_nlp_icml.pdf)
  - Contribution : NLP feature를 자동으로 생성
    - n-gram feature를 자동으로 생성
  - Model : CNN
- [오피니언 마이닝](http://emnlp2014.org/papers/pdf/EMNLP2014181.pdf) [blog](https://ratsgo.github.io/natural%20language%20processing/2017/03/19/CNN/)
  - Contribution : CNN 구조를 오피니언 마이닝에 적용
  - Model : CNN (filter size : 2,3,4)
- [aspect 예측](http://sentic.net/aspect-extraction-for-opinion-mining.pdf)
  - aspect란? opinion의 대상 주제 (topic) 예: 디자인, 배터리
  - Model : multi-level deep CNN
- [단문 분류](http://www.aclweb.org/anthology/P15-2058)
  - Contribution : 단문의 외부적 지식 사용
  - Model : CNN + multi-scale semantic units
- [문서 요약](https://arxiv.org/pdf/1406.3830.pdf)
  - Contributions
    - 단어의 의미를 문서의 의미로 매핑
    - 텍스트 자동요약을 위한 인사이트를 제공하는 새로운 시각화 기술의 도입
  - Model : DCNN
- [sarcasm 인식기](https://arxiv.org/pdf/1610.08815.pdf)
  - Contribution : 감정(emotion), 감성(sentiment), 개성(personality) 데이터 셋으로 사전 학습된 형태의 보조적인 지원
  - Data : twitter
  - Model : CNN
- [QA](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/SingleRelationQA-YihHeMeek-ACL14.pdf)
  - Contribution : 질의와 KB 사이의 의미적 유사성 측정
  - $sim(Q_{entity}, KB_{entity}), sim(Q_{relation}, KB_{relation})$
  - Data : PARALAX dataset
  - Model : CNN
- [QA2](http://www.aclweb.org/anthology/P15-1026)
  - Contribution : 다중 aspect 질의 분석 (answer path, answer context, answer type)
  - Model : MCCNN(multi-column CNN)
  - Data : WebQuestions (questions collected by Google Suggest API), Freebase (large-scal KB)
- [QA 응답 문장 표현](https://arxiv.org/pdf/1604.01178.pdf)
  - Contribution : 질의와 응답 쌍 사이에 단어를 매칭
  - Model : CNN
- [정보 검색](http://www.iro.umontreal.ca/~lisa/pointeurs/ir0895-he-2.pdf)
  - Contribution : 질의와 문서를 고정된 차원의 의미공간에 투영(project)
  - Model : CNN
- [다중 이벤트 모델링](https://pdfs.semanticscholar.org/ca70/480f908ec60438e91a914c1075b9954e7834.pdf)
  - Contribution : 맥스풀링의 정보 손실 감소
  - Model : dynamic multi-pooling CNN (DMCNN)

#### Recurrent Neural Networks

- [NER](https://arxiv.org/pdf/1603.01360.pdf)
  - model : bidirectional LSTM + CRF
- [언어 모델링](https://arxiv.org/pdf/1308.0850.pdf)
  - contribution : 언어 모델링에 RNN 처음 적용
  - model : RNN
- 감성 분석 - Wang et al. (2015a)
  - contribution : 문장을 인코딩
  - model : RNN
  - Data : 트위터
- [대화 검색](https://www.sigdial.org/files/workshops/conference16/proceedings/pdf/SIGDIAL40.pdf)
  - model : LSTM
  - data : DSTC(Dislogue State Tracking Challenge) datasets
- [문장 생성](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)
  - contribution : RNN으로 자연어 생성
  - Model : deep LSTM encoder-decoder
- [end-to-end 기계번역](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)
  - Model : deep LSTM encoder-decoder
- [인간 대화 모델링](https://arxiv.org/pdf/1506.05869.pdf)
  - Model : deep LSTM encoder-decoder
  - 1억개 이상의 메시지-응답 쌍을 학습
- [이미지 캡셔닝](https://arxiv.org/pdf/1411.4555.pdf)
  - contribution : 비전 + NLP
  - Model : deep CNN -> deep LSTM encoder-decoder
- [비주얼 QA](https://arxiv.org/pdf/1505.01121.pdf)
  - problem : 이미지와 이미지 안의 대상을 설명하는 질의문
  - contribution : 비주얼 QA의 end-to-end 딥러닝 모델 제시
- [비주얼 QA](https://arxiv.org/pdf/1506.07285.pdf)
  - contribution : Dynamic Memory Network (DMN) 네트워크 제안
  - Model : 4개의 하위 모듈을 가지는 DMN
- [기계 번역](https://arxiv.org/pdf/1409.0473.pdf)
  - contribution : attention 메카니즘 사용
  - model : RNN + LSTM + attention mechanism
- [요약](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.697.980&rep=rep1&type=pdf)
  - Model : RNN + attention mechanism
- [이미지 캡셔닝](https://arxiv.org/pdf/1502.03044.pdf)
  - Model : RNN + attention mechanism
- [구문 분석](https://arxiv.org/pdf/1412.7449.pdf)
  - Model : RNN + attention mechanism
- [감성 분석](https://aclweb.org/anthology/D16-1058)
  - Model : RNN + attention mechanism

#### Recursive Neural Networks

- [감성 분석](https://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf)
  - Model : Recursive Neural Network
- [문장 관련성 테스트](http://www.aclweb.org/anthology/P15-1150)
  - Model : Recursive Neural Network + LSTM

#### Deep Reinforced Models and Deep Unsupervised Learning

##### Reinforced Models

- [요약](https://arxiv.org/pdf/1705.04304.pdf)
  - Conribution : 강화 학습을 결합
  - Model : bi-directional LSTM encoder + single LSTM decorder RNN + attention function + reinforcement learning
- [시퀀스 생성](https://arxiv.org/pdf/1511.06732.pdf)
  - Contribution : 강화 학습을 결합
  - Model : RNN + Reinforced model
  - Data
    - 문서 요약 : a subset of the Gigaword corpus (collection of news articles)
    - 기계번역 : GermanEnglish
machine translation track of the IWSLT 2014 evaluation campaign
    - 이미지 캡셔닝 : MSCOCO dataset
- [대화 생성](https://arxiv.org/pdf/1606.01541.pdf)
  - Conribution : 생성 문장에 대해 3가지 보상(응답의 용이성, 정보 흐름 및 의미의 일관성)을 정의
- [대화 생성](https://arxiv.org/pdf/1701.06547.pdf)
  - Conribution : GAN을 접목

##### Unsupervised Learning

- [문장 임베딩](https://arxiv.org/pdf/1506.06726.pdf)
  - Contribution : sequence 임베딩
  - Model : seq2seq, skip-thought (주어진 문장 앞 뒤에 인접 문장 예측)
- [문서 분류](https://arxiv.org/pdf/1511.01432.pdf)
  - Conribution : sequence 오토 인코더가 LSTM RNN 학습 안정화에 도움을 줌
  - Model : RNN + LSTM

##### Deep Generative Models

- [문장 생성](https://arxiv.org/pdf/1511.06349.pdf)
  - Contribution : 명시적인 전역 문장 표현(global sentence representation)에서 작동
  - Model : RNN + VAE
- [문장 생성](https://arxiv.org/pdf/1703.00955.pdf)
  - Conribution : 영어의 두 가지 주요 속성(시제, 감성)에 맞는 그럴듯한 문장들을 생성
  - Model : RNN + VAEs + holistic attribute discriminators
- [문장 생성](https://sites.google.com/site/nips2016adversarial/WAT16_paper_20.pdf?attredirects=1)
  - Model : GAN + CNN + LSTM
  - Conribution : GAN을 텍스트에 적용할 때 문제점은 판별자로부터 나온 그래디언트가 제대로 역전파될 수 없다는 점이다. Zhang et al.(2016)에서 이 문제는 단어 예측을 항상 ‘soft’하게 함으로써 해결
- [문장 생성](https://arxiv.org/pdf/1609.05473.pdf)
  - Contribution : 생성자를 stochatic policy로 모델링함으로써 이 문제를 우회할 것을 제안
- [문장 생성](https://arxiv.org/pdf/1705.10929.pdf)
  - Contribution : 평가법 제시. 고정된 문법 규칙으로 학습 데이터를 만든 다음, 생성된 샘플이 미리 정의한 문법과 일치하는지 여부로 생성모델을 평가

#### Memory-Augmented Networks

- [QA](https://arxiv.org/pdf/1410.3916.pdf)
  - contribution : 메모리 네트워크 제안
  - Model : RNN + memory network + attention mechanism
- [언어 모델링](https://arxiv.org/pdf/1503.08895.pdf)
  - Model : RNN + multiple-hop attention
- [이미지 캡셔닝](https://arxiv.org/pdf/1502.03044.pdf)
  - contribution : 메모리 네트워크 기반
  - Model : RNN + multiple-hop attention
- [QA, 품사태깅, 감성분석](https://arxiv.org/pdf/1506.07285.pdf)
  - contribution : 모든 데이터 인스턴스가 <메모리, 질문, 답변> 트리플 포맷으로 캐스팅
  - Model : Dynamic Memory Networks
- [비쥬얼 QA](https://arxiv.org/pdf/1603.01417.pdf)
  - contribution : 메모리 모델이 시각적 신호에도 적용 가능

## performances on NLP tasks

- PoS tagging
  - DMN
- Dependency Parsing
  - Deep fully-connected NN with features including POS
  - Stack LSTM
- Constituency Parsing
  - seq2seq LSTM + Attention
- NER
  - Semi-CRF jointly trained with linking
- SRL
  - Bidirectional LSTM + highway connection
- Sentiment Classification
  - DMN
- Machine Translation
  - Attention mechanism
  - seq2seq with CNN
- QA
  - Memory Networks
  - DMN
- Conversation System
  - LSTM seq2seq with MMI objective
  - Sentence-level CNN-LSTM encoder