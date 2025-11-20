# MacroHFT: Memory Augmented Context-aware Reinforcement Learning On High Frequency Trading

## 📋 목차
- [프로젝트 개요](#프로젝트-개요)
- [프로젝트 구조](#프로젝트-구조)
- [핵심 아키텍처](#핵심-아키텍처)
- [알고리즘 상세 분석](#알고리즘-상세-분석)
- [학습 파이프라인](#학습-파이프라인)
- [실행 방법](#실행-방법)

---

## 프로젝트 개요

**MacroHFT**는 KDD 2024에 게재된 논문의 공식 구현으로, 고빈도 거래(High Frequency Trading)를 위한 **계층적 강화학습 시스템**입니다.

### 주요 특징
- **계층적 강화학습**: Meta-policy와 Sub-policies의 2단계 구조
- **시장 컨텍스트 인식**: 시장 조건(추세, 변동성)에 따른 동적 전략 선택
- **메모리 증강**: Episodic Memory를 활용한 샘플 효율성 향상
- **모방 학습**: Demonstration Q-table을 통한 안정적 초기 학습

### 논문 정보
- **제목**: MacroHFT: Memory Augmented Context-aware Reinforcement Learning On High Frequency Trading
- **학회**: KDD 2024
- **arXiv**: https://arxiv.org/abs/2406.14537

---

## 프로젝트 구조

### 디렉토리 구조

```
MacroHFT/
├── env/                          # 강화학습 환경
│   ├── high_level_env.py        # 메타 정책 환경 (Hyperagent)
│   └── low_level_env.py         # 서브 에이전트 환경
│
├── model/                        # 신경망 모델
│   └── net.py                   
│       ├── subagent             # 서브에이전트 네트워크 (Dueling DQN + AdaLN)
│       └── hyperagent           # 하이퍼에이전트 네트워크 (Context-aware)
│
├── RL/                          # 강화학습 알고리즘
│   ├── agent/
│   │   ├── low_level.py        # 서브에이전트 DQN 학습
│   │   └── high_level.py       # 메타정책 학습
│   └── util/
│       ├── replay_buffer.py    # 경험 재생 버퍼
│       ├── memory.py           # Episodic Memory (핵심)
│       └── utili.py            # 유틸리티 함수
│
├── preprocess/                  # 데이터 전처리
│   └── decomposition.py        # 시장 분할 및 레이블링
│
├── tools/                       # 도구
│   └── demonstration.py        # Q-테이블 생성 (역방향 DP)
│
├── data/                        # 데이터
│   ├── feature_list/           # 특징 리스트
│   │   ├── single_features.npy # 순간 기술 지표
│   │   └── trend_features.npy  # 추세 특징
│   └── ETHUSDT/                # 데이터셋 (Google Drive에서 다운로드)
│       ├── train/              # 훈련 데이터
│       ├── val/                # 검증 데이터
│       ├── test/               # 테스트 데이터
│       └── whole/              # 전체 데이터
│
├── scripts/                     # 실행 스크립트
│   ├── decomposition.sh        # Step 1: 데이터 전처리
│   ├── low_level.sh            # Step 2: 서브에이전트 학습
│   └── high_level.sh           # Step 3: 메타정책 학습
│
└── result/                      # 학습 결과
    ├── low_level/              # 서브에이전트 모델
    │   └── ETHUSDT/
    │       └── best_model/
    │           ├── slope/      # Slope 에이전트 3개
    │           └── vol/        # Volatility 에이전트 3개
    └── high_level/             # 메타정책 모델
```

---

## 핵심 아키텍처

### 1. 계층적 구조 (Hierarchical Architecture)

```
┌─────────────────────────────────────────────────────────┐
│  High-Level: Meta-Policy (Hyperagent)                   │
│  - 시장 컨텍스트 인식 (slope_360, vol_360)               │
│  - 6개 서브에이전트의 동적 가중치 할당                     │
│  - Episodic Memory로 유사 경험 활용                       │
└────────────────────┬────────────────────────────────────┘
                     │
          ┌──────────┴──────────┐
          │                     │
┌─────────▼──────────┐  ┌──────▼───────────┐
│ Slope Sub-Agents   │  │ Vol Sub-Agents   │
│ - Agent 1 (label1) │  │ - Agent 1 (label1)│
│ - Agent 2 (label2) │  │ - Agent 2 (label2)│
│ - Agent 3 (label3) │  │ - Agent 3 (label3)│
└────────────────────┘  └──────────────────┘
          │                     │
          └──────────┬──────────┘
                     │
         ┌───────────▼───────────┐
         │  Trading Environment  │
         │  - Gym 환경 기반       │
         │  - 거래 수수료 포함     │
         └───────────────────────┘
```

### 2. 시장 컨텍스트 분해 (Market Context Decomposition)

**목적**: 복잡한 시장을 여러 개의 단순한 서브 마켓으로 분할

#### Slope-based Decomposition (추세 기반)
```python
# 4320 타임스텝 청크 단위로 분석
1. 가격 데이터를 Butterworth 필터로 스무딩
2. 선형 회귀로 기울기 계산
3. Quantile 분할: [0%, 5%, 35%, 65%, 95%, 100%]
4. 극단값 병합: 0→1, 4→3
5. 최종: 3개 레이블 (상승, 중립, 하락 추세)
```

#### Volatility-based Decomposition (변동성 기반)
```python
# 동일한 방식으로 처리
1. 수익률의 표준편차 계산
2. Quantile 분할 및 병합
3. 최종: 3개 레이블 (고변동, 중변동, 저변동)
```

**결과**: 3(slope) × 2(vol) = **6개의 전문화된 시장 컨텍스트**

---

## 알고리즘 상세 분석

### 1. Sub-Agent Architecture (Low-Level Policy)

#### 네트워크 구조
```
Input: (single_state, trend_state, previous_action)
│
├─> single_state ──> fc1 ──> LayerNorm ──┐
│                                        │
├─> trend_state ──> fc2 ──┐             │
│                         │             │
└─> previous_action ──> Embedding ──┘   │
                         │               │
                         ├──> AdaLN Modulation (shift, scale)
                         │               │
                         └──────────────>│
                                        │
                    ┌───────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
    Value Stream          Advantage Stream
        │                       │
    ┌───▼───┐              ┌───▼────┐
    │ V(s)  │              │ A(s,a) │
    └───┬───┘              └───┬────┘
        │                      │
        └──────────┬───────────┘
                   │
    Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
```

#### 핵심 기술

**1) Adaptive Layer Normalization (AdaLN)**
```python
c = action_embedding + trend_state_hidden
shift, scale = AdaLN_modulation(c)
x = LayerNorm(single_state_hidden) * (1 + scale) + shift
```
- 이전 행동과 추세 상태로 조건화
- 상태 표현을 동적으로 조절
- Transformer의 adaptive normalization에서 영감

**2) Dueling Architecture**
```python
Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a))
```
- Value: 상태의 절대적 가치
- Advantage: 행동의 상대적 이점
- 학습 안정성 향상

### 2. Sub-Agent Training (Low-Level Training)

#### 손실 함수
```python
L_sub = L_TD + α × L_KL
```

**TD Loss (Double DQN)**
```python
Q_target = r + γ × Q_target(s', argmax_a Q_eval(s', a))
L_TD = MSE(Q_current, Q_target)
```

**KL Divergence Loss (Imitation Learning)**
```python
L_KL = KL(softmax(Q_demo) || softmax(Q_policy))
```
- Q_demo: 역방향 동적 프로그래밍으로 계산된 시연 Q값
- 안정적 초기 학습 및 위험 제어
- α 값이 레이블별로 다름: 0, 1, 4

#### 알고리즘 흐름
```
For each epoch:
    For each chunk in training set:
        1. Reset environment with random initial position
        2. For each timestep:
            a. Select action (ε-greedy)
            b. Execute action
            c. Store transition in replay buffer
            d. If update condition:
                - Sample batch
                - Compute TD loss
                - Compute KL loss
                - Update eval_net
                - Soft update target_net (τ=0.005)
        3. Evaluate on validation set
```

#### 하이퍼파라미터
- Batch size: 512
- Learning rate: 1e-4
- Gamma: 0.99
- Tau: 0.005
- Epsilon: 0.5 → 0.1 (5 epoch linear decay)
- Update frequency: 매 100 스텝
- Update times per step: 10회

### 3. Hyperagent Architecture (High-Level Policy)

#### 네트워크 구조
```
Input: (single_state, trend_state, class_state, previous_action)
│
├─> concat(single_state, trend_state) ──> fc1 ──┐
│                                               │
├─> class_state [slope_360, vol_360] ──> fc2 ──┤
│                                               │
└─> previous_action ──> Embedding ──────────────┘
                                                │
            ┌───────────────────────────────────┘
            │
            ├──> concat([action_hidden, state_hidden])
            │
            ├──> AdaLN (conditioned on class_state)
            │
            └──> MLP ──> Softmax ──> [w1, w2, ..., w6]
                                     (6개 서브에이전트 가중치)
```

#### Q-Value Aggregation (핵심 메커니즘)
```python
# 1. 각 서브에이전트에서 Q값 계산
Q_slope1 = slope_agent1(s, s_trend, prev_action)  # shape: (batch, 2)
Q_slope2 = slope_agent2(s, s_trend, prev_action)
Q_slope3 = slope_agent3(s, s_trend, prev_action)
Q_vol1 = vol_agent1(s, s_trend, prev_action)
Q_vol2 = vol_agent2(s, s_trend, prev_action)
Q_vol3 = vol_agent3(s, s_trend, prev_action)

# 2. 가중치 계산
w = hyperagent(s, s_trend, class_state, prev_action)  # shape: (batch, 6)

# 3. 가중 조합
Q_meta(s, a) = Σ(i=1 to 6) w_i × Q_i(s, a)
```

수식:
$$Q_{meta}(s, a) = \sum_{i=1}^{6} w_i(s) \cdot Q_i^{sub}(s, a)$$

여기서 $w_i(s)$는 시장 컨텍스트에 따라 동적으로 변화

### 4. Episodic Memory (핵심 혁신)

#### 구조
```python
Memory Buffer:
- capacity: 4320 (1 episode)
- K: 5 (nearest neighbors)

Storage: (hidden_state, action, Q_value, single_state, trend_state, prev_action)
```

#### 알고리즘

**1) 저장 (Add)**
```python
def add(h, a, Q, s, s_trend, prev_a):
    buffer[count] = (h, a, Q, s, s_trend, prev_a)
    count = (count + 1) % capacity
```

**2) 검색 (Query)**
```python
def query(h_query, a_query):
    # Step 1: 모든 메모리 항목과의 유사도 계산
    K(h, h_i) = 1 / (||h - h_i||² + ε)
    
    # Step 2: Top-K 선택
    top_k_indices = argsort(K)[-5:]
    
    # Step 3: 같은 행동만 필터링
    mask = (actions[top_k_indices] == a_query)
    
    # Step 4: 가중 평균
    weights = K[top_k_indices] / Σ K[top_k_indices]
    masked_weights = weights × mask
    normalized_weights = masked_weights / Σ masked_weights
    
    Q_memory = Σ(normalized_weights × Q_values[top_k_indices])
    
    return Q_memory
```

수식:
$$Q_{mem}(s, a) = \frac{\sum_{i \in \mathcal{N}_k(s)} K(h, h_i) \cdot Q_i \cdot \mathbb{1}[a_i = a]}{\sum_{i \in \mathcal{N}_k(s)} K(h, h_i) \cdot \mathbb{1}[a_i = a]}$$

**3) 재인코딩 (Re-encode)**
```python
def re_encode(hyperagent):
    # 4320 스텝마다 실행
    for batch in memory:
        h_new = hyperagent.encode(s, s_trend, prev_action)
        buffer["hidden_state"] = h_new
```

#### 역할
- **샘플 효율성 향상**: 과거 유사 경험 재활용
- **Non-stationarity 대응**: 시장 변화에 적응
- **정규화**: Memory Q-value로 현재 Q-value 보정

### 5. High-Level Training (Meta-Policy Training)

#### 손실 함수 (3-Term Loss)
```python
L_meta = L_TD + α × L_memory + β × L_KL
```

**1) TD Loss (Double DQN)**
```python
# Action selection with current eval network
w_next_ = hyperagent(s', s'_trend, class', prev_a')
Q_next_ = Σ(w_next_ × Q_i_sub(s'))
a_argmax = argmax_a Q_next_(s', a)

# Action evaluation with target network
w_next = hyperagent_target(s', s'_trend, class', prev_a')
Q_next = Σ(w_next × Q_i_sub(s'))
Q_target = r + γ × (1 - done) × Q_next(s', a_argmax)

# Current Q
w_current = hyperagent(s, s_trend, class, prev_a)
Q_current = Σ(w_current × Q_i_sub(s))

L_TD = MSE(Q_current(s, a), Q_target)
```

**2) Memory Loss (핵심)**
```python
# Query episodic memory
h = hyperagent.encode(s, s_trend, prev_a)
Q_memory = episodic_memory.query(h, a)

L_memory = MSE(Q_current, Q_memory)
```
- α = 0.5: TD loss와 균형
- 과거 유사 경험으로 정규화

**3) KL Loss (Imitation)**
```python
Q_demo = demonstration_Q_table[t][prev_a][:]
L_KL = KL(softmax(Q_current) || softmax(Q_demo))
```
- β = 5: 강한 모방 학습

#### 알고리즘 흐름
```
Initialize:
- Load 6 frozen sub-agents
- Initialize hyperagent and hyperagent_target
- Initialize episodic_memory (capacity=4320)

For each epoch:
    Reset environment
    For each timestep t:
        1. Action selection:
            w = hyperagent(s, s_trend, class, prev_a)
            Q_i = sub_agent_i(s, s_trend, prev_a) for i=1..6
            Q_meta = Σ(w × Q_i)
            a = argmax Q_meta (ε-greedy)
        
        2. Execute action:
            s', r, done = env.step(a)
        
        3. Calculate hidden state and Q_memory:
            h = hyperagent.encode(s, s_trend, prev_a)
            Q_next = r + γ × Q_estimate(s')
            Q_memory = episodic_memory.query(h, a)
            if Q_memory is NaN:
                Q_memory = Q_next
        
        4. Store in replay buffer:
            store(s, a, r, s', Q_memory)
        
        5. Add to episodic memory:
            episodic_memory.add(h, a, Q_next, s, s_trend, prev_a)
        
        6. Update:
            if t % update_freq == 0:
                for _ in range(update_times):
                    L = L_TD + α×L_memory + β×L_KL
                    optimize(hyperagent)
                    soft_update(hyperagent_target)
        
        7. Re-encode memory:
            if t % 4320 == 0:
                episodic_memory.re_encode(hyperagent)
    
    Evaluate on validation set
    Save best model
```

#### 하이퍼파라미터
- Batch size: 512
- Learning rate: 1e-4
- Gamma: 0.99
- Tau: 0.005
- Epsilon: 0.7 → 0.3 (5 epoch linear decay)
- α (memory weight): 0.5
- β (KL weight): 5
- Update frequency: 매 512 스텝
- Update times per step: 10회
- Memory capacity: 4320
- K (nearest neighbors): 5

---

## 학습 파이프라인

### Phase 1: Data Preparation (데이터 전처리)

```bash
cd MacroHFT
python preprocess/decomposition.py
```

**작업 내용:**
1. 데이터 청킹 (4320 타임스텝 단위)
2. Slope 레이블링 (3개 클래스)
3. Volatility 레이블링 (3개 클래스)
4. Rolling window 특징 생성 (slope_360, vol_360)

**출력:**
```
data/ETHUSDT/
├── train/
│   ├── df_0.feather, df_1.feather, ...
│   ├── slope_labels.pkl
│   └── vol_labels.pkl
├── val/
│   ├── df_0.feather, df_1.feather, ...
│   ├── slope_labels.pkl
│   └── vol_labels.pkl
├── test/
│   └── (동일 구조)
└── whole/
    ├── train.feather
    ├── val.feather
    └── test.feather
```

### Phase 2: Low-Level Training (서브에이전트 학습)

```bash
bash scripts/low_level.sh
```

**6개 에이전트 병렬 학습:**

| 에이전트 | 분류기 | 레이블 | α (KL weight) | GPU |
|---------|-------|--------|--------------|-----|
| Slope 1 | slope | 1      | 1            | 0   |
| Slope 2 | slope | 2      | 4            | 1   |
| Slope 3 | slope | 3      | 0            | 2   |
| Vol 1   | vol   | 1      | 4            | 0   |
| Vol 2   | vol   | 2      | 1            | 1   |
| Vol 3   | vol   | 3      | 1            | 2   |

**각 에이전트 학습 과정:**
1. 특정 레이블의 청크들만 선택
2. Demonstration Q-table 생성
3. DQN + Imitation learning으로 학습
4. 검증 성능 기반 best model 저장

**출력:**
```
result/low_level/ETHUSDT/
├── slope/
│   ├── 1/best_model.pkl
│   ├── 2/best_model.pkl
│   └── 3/best_model.pkl
└── vol/
    ├── 1/best_model.pkl
    ├── 2/best_model.pkl
    └── 3/best_model.pkl
```

### Phase 3: High-Level Training (메타정책 학습)

```bash
bash scripts/high_level.sh
```

**학습 과정:**
1. 6개 학습된 서브에이전트 로드 (frozen)
2. Hyperagent 초기화
3. Episodic memory 초기화
4. 전체 에피소드로 학습:
   - 서브에이전트 Q값들을 동적 가중치로 조합
   - Memory에서 유사 경험 검색
   - 3-term loss로 업데이트
5. 주기적 memory re-encoding
6. 검증 성능 기반 best model 저장

**출력:**
```
result/high_level/ETHUSDT/
├── exp1/
│   └── seed_12345/
│       ├── epoch_1/
│       │   └── trained_model.pkl
│       ├── ...
│       └── log/
└── best_model.pkl
```

---

## 실행 방법

### 1. 환경 설정

```bash
# Python 환경 생성 (권장: Python 3.8+)
conda create -n macrohft python=3.8
conda activate macrohft

# 필요 패키지 설치
pip install torch torchvision torchaudio
pip install numpy pandas scipy scikit-learn
pip install gym tensorboard
pip install pyarrow  # feather 파일 지원
```

### 2. 데이터 다운로드

Google Drive에서 데이터셋 다운로드:
https://drive.google.com/drive/folders/1AYHy-wUV0IwPoA7E1zvMRPL3wK0tPNiY?usp=drive_link

```bash
# 다운로드한 데이터를 data 폴더에 배치
MacroHFT/
└── data/
    └── ETHUSDT/
        ├── df_train.feather
        ├── df_val.feather
        └── df_test.feather
```

### 3. 단계별 실행

#### Step 1: 데이터 전처리
```bash
cd MacroHFT
python preprocess/decomposition.py
```

#### Step 2: 서브에이전트 학습 (병렬)
```bash
# 로그 디렉토리 생성
mkdir -p logs/low_level/ETHUSDT

# 스크립트 실행 (6개 프로세스 병렬)
bash scripts/low_level.sh

# 또는 개별 실행 예시
python RL/agent/low_level.py \
    --alpha 1 \
    --clf 'slope' \
    --dataset 'ETHUSDT' \
    --device 'cuda:0' \
    --label 'label_1'
```

#### Step 3: 메타정책 학습
```bash
# 로그 디렉토리 생성
mkdir -p logs/high_level

# 스크립트 실행
bash scripts/high_level.sh

# 또는 직접 실행
python RL/agent/high_level.py \
    --dataset 'ETHUSDT' \
    --device 'cuda:0' \
    --alpha 0.5 \
    --beta 5
```

### 4. TensorBoard 모니터링

```bash
# Low-level 학습 모니터링
tensorboard --logdir=result/low_level/ETHUSDT/slope/1/seed_12345/log

# High-level 학습 모니터링
tensorboard --logdir=result/high_level/ETHUSDT/exp1/seed_12345/log
```

**모니터링 메트릭:**
- TD error
- Memory error (high-level만)
- KL loss
- Q_eval, Q_target
- Return rate (수익률)
- Final balance (최종 잔액)
- Required money (필요 자본)

---

## 핵심 기술 요약

### 1. 계층적 강화학습 (Hierarchical RL)
- **Low-level**: 특정 시장 조건에 특화된 6개 전문가 정책
- **High-level**: 시장 컨텍스트에 따라 전문가들을 동적으로 조합

### 2. 시장 컨텍스트 인식 (Context-Aware)
- Slope와 Volatility로 시장 상태 분류
- AdaLN으로 컨텍스트 정보를 네트워크에 주입
- 동적 가중치로 상황별 최적 전략 선택

### 3. 메모리 증강 (Memory Augmentation)
- Episodic memory로 유사 과거 경험 검색
- Kernel 기반 K-NN으로 관련 경험 가중 평균
- 샘플 효율성 향상 및 non-stationarity 대응

### 4. 모방 학습 (Imitation Learning)
- Demonstration Q-table (역방향 DP)
- KL divergence loss로 안전한 정책 유도
- 초기 학습 안정화 및 위험 제어

### 5. 네트워크 아키텍처
- **Dueling DQN**: Value와 Advantage 분리
- **AdaLN**: 컨텍스트 기반 적응적 정규화
- **Double DQN**: 과대추정 방지

---

## 수식 정리

### Low-Level Policy
$$Q^{sub}_i(s, a) = V(s) + A(s, a) - \mathbb{E}_{a'}[A(s, a')]$$

$$\mathcal{L}_{sub} = \underbrace{\mathbb{E}[(Q_{\theta}(s,a) - y)^2]}_{\text{TD Loss}} + \alpha \underbrace{D_{KL}(\text{softmax}(Q_{demo}) \| \text{softmax}(Q_{\theta}))}_{\text{Imitation Loss}}$$

### High-Level Policy
$$Q_{meta}(s, a) = \sum_{i=1}^{6} w_i(s, c) \cdot Q_i^{sub}(s, a)$$

$$\mathcal{L}_{meta} = \underbrace{\mathbb{E}[(Q_{\phi}(s,a) - y)^2]}_{\text{TD Loss}} + \alpha \underbrace{\mathbb{E}[(Q_{\phi}(s,a) - Q_{mem}(s,a))^2]}_{\text{Memory Loss}} + \beta \underbrace{D_{KL}(\text{softmax}(Q_{demo}) \| \text{softmax}(Q_{\phi}))}_{\text{Imitation Loss}}$$

### Episodic Memory
$$Q_{mem}(s, a) = \frac{\sum_{i \in \mathcal{N}_k(s)} K(h_s, h_i) \cdot Q_i \cdot \mathbb{1}[a_i = a]}{\sum_{i \in \mathcal{N}_k(s)} K(h_s, h_i) \cdot \mathbb{1}[a_i = a]}$$

$$K(h, h') = \frac{1}{\|h - h'\|^2 + \epsilon}$$

---

## 주요 파라미터 정리

### 공통 파라미터
| 파라미터 | 값 | 설명 |
|---------|---|------|
| Batch size | 512 | 미니배치 크기 |
| Learning rate | 1e-4 | Adam optimizer |
| Gamma (γ) | 0.99 | 할인 계수 |
| Tau (τ) | 0.005 | Soft target update |
| Transaction cost | 0.0002 | 거래 수수료 (0.02%) |
| Max holding | 0.2 (ETH) | 최대 포지션 크기 |

### Low-Level 특화
| 파라미터 | 값 | 설명 |
|---------|---|------|
| Epsilon | 0.5 → 0.1 | ε-greedy decay |
| Decay length | 5 epochs | Epsilon 감소 기간 |
| Update freq | 100 steps | 업데이트 빈도 |
| Update times | 10 | 스텝당 업데이트 횟수 |
| α (KL) | 0, 1, 4 | 레이블별 상이 |

### High-Level 특화
| 파라미터 | 값 | 설명 |
|---------|---|------|
| Epsilon | 0.7 → 0.3 | ε-greedy decay |
| Decay length | 5 epochs | Epsilon 감소 기간 |
| Update freq | 512 steps | 업데이트 빈도 |
| Update times | 10 | 스텝당 업데이트 횟수 |
| α (memory) | 0.5 | Memory loss 가중치 |
| β (KL) | 5 | Imitation loss 가중치 |
| Memory capacity | 4320 | Episodic memory 크기 |
| K | 5 | K-nearest neighbors |

---

## 성능 평가 지표

### 거래 성능
- **Return Rate**: `final_balance / required_money`
- **Final Balance**: 최종 잔액 (수익)
- **Required Money**: 필요 자본 (최대 손실 기준)
- **Commission Fee**: 총 거래 수수료

### 학습 성능
- **TD Error**: Temporal Difference 오차
- **Memory Error**: Episodic memory 오차
- **KL Loss**: Imitation learning 손실
- **Q Values**: 예측 Q값 vs 타겟 Q값

---

## 프로젝트의 주요 기여

### 1. 계층적 분해 (Hierarchical Decomposition)
단일 복잡한 정책 대신 여러 전문가 정책으로 분해하여:
- 각 서브 정책은 특정 시장 조건에 집중
- 학습 난이도 감소
- 해석 가능성 향상

### 2. 동적 조합 (Dynamic Composition)
고정된 앙상블이 아닌 상황별 동적 가중치:
- 시장 컨텍스트에 따라 최적 전략 조합
- 적응성 향상
- 일반화 능력 개선

### 3. 메모리 증강 (Memory Augmentation)
Episodic memory로 과거 경험 재활용:
- 샘플 효율성 크게 향상
- Non-stationary 환경 대응
- Catastrophic forgetting 방지

### 4. 안전한 탐색 (Safe Exploration)
Demonstration을 통한 모방 학습:
- 초기 학습 안정화
- 위험한 행동 억제
- 수렴 속도 향상

---

## 결론

MacroHFT는 **계층적 강화학습**, **컨텍스트 인식**, **메모리 증강**, **모방 학습**을 결합하여 고빈도 거래라는 복잡하고 비정상적인 환경에서 강건하고 적응적인 트레이딩 시스템을 구현했습니다.

### 핵심 아이디어
1. **Divide and Conquer**: 복잡한 문제를 여러 단순한 문제로 분해
2. **Specialization**: 각 서브에이전트가 특정 조건에 전문화
3. **Dynamic Adaptation**: 실시간 시장 조건에 따른 동적 전략 선택
4. **Experience Reuse**: 과거 유사 경험을 효율적으로 재활용

### 적용 가능성
- 고빈도 거래 (HFT)
- 포트폴리오 관리
- 시장 메이킹 (Market Making)
- 기타 non-stationary 환경의 의사결정 문제

---

## 참고 자료

- **논문**: https://arxiv.org/abs/2406.14537
- **GitHub**: (현재 디렉토리)
- **데이터**: https://drive.google.com/drive/folders/1AYHy-wUV0IwPoA7E1zvMRPL3wK0tPNiY

---

*이 문서는 MacroHFT 프로젝트의 코드 분석을 통해 작성되었습니다.*
*최종 업데이트: 2025년 11월*

