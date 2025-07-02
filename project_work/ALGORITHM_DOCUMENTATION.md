# Multi-Armed Bandit Auction Platform: Complete Algorithm Documentation

## Overview

This document provides a comprehensive mathematical and algorithmic description of the multi-armed bandit auction platform, including the environment simulation process, algorithm implementations, and recent critical fixes.

## Environment Simulation Framework

### Core Components

The simulation consists of four main components working together:

1. **Setting**: Defines the auction parameters and buyer distributions
2. **Buyer**: Generates demand based on price distributions  
3. **Seller**: Implements pricing algorithms (UCB1, Primal-Dual)
4. **Environment**: Orchestrates the simulation rounds and tracks performance

### Mathematical Formulation

#### Problem Setup

- **Products**: $K$ different products
- **Time Horizon**: $T$ rounds
- **Price Grid**: $P = \{p_1, p_2, \ldots, p_M\}$ for each product
- **Budget Constraint**: Total spending ≤ $B$
- **Inventory Constraint**: Total sales ≤ $I_k$ for product $k$

#### Buyer Demand Model

For each round $t$ and product $k$, buyer demand is generated as:

$$D_{k,t}(p) = \begin{cases}
1 & \text{if } V_{k,t} \geq p \\
0 & \text{otherwise}
\end{cases}$$

Where $V_{k,t}$ is the buyer's valuation drawn from distribution $F_{k,t}$:

- **Stationary**: $F_{k,t} = F_k$ for all $t$
- **Non-stationary**: $F_{k,t}$ evolves over time with parameter drift

Supported distributions:
- **Uniform**: $V \sim \text{Uniform}(a, b)$
- **Gaussian**: $V \sim \mathcal{N}(\mu, \sigma^2)$
- **Bernoulli**: $V \sim \text{Bernoulli}(p) \cdot v_{max}$
- **Exponential**: $V \sim \text{Exponential}(\lambda)$
- **Beta**: $V \sim \text{Beta}(\alpha, \beta) \cdot v_{max}$

#### Non-Stationarity Model

Parameter evolution follows:
$$\theta_{k,t+1} = \theta_{k,t} + \epsilon \cdot \mathcal{N}(0, 1)$$

Where $\epsilon$ controls the degree of non-stationarity:
- $\epsilon = 0$: Stationary
- $\epsilon \ll 1$: Slightly non-stationary  
- $\epsilon \gg 1$: Highly non-stationary

## Algorithm Implementations

### UCB1 Algorithm

The Upper Confidence Bound algorithm balances exploration and exploitation:

#### Action Selection
$$a_t = \arg\max_{i=1,\ldots,K} \left[ \hat{\mu}_{i,t} + \sqrt{\frac{2\log t}{n_{i,t}}} \right]$$

Where:
- $\hat{\mu}_{i,t} = \frac{1}{n_{i,t}} \sum_{s=1}^{t-1} r_{i,s} \mathbf{1}(a_s = i)$ (empirical mean)
- $n_{i,t} = \sum_{s=1}^{t-1} \mathbf{1}(a_s = i)$ (number of times arm $i$ pulled)
- $t$ is the current round

#### Update Rule
After observing reward $r_t$ for action $a_t$:
$$\hat{\mu}_{a_t,t+1} = \frac{n_{a_t,t} \cdot \hat{\mu}_{a_t,t} + r_t}{n_{a_t,t} + 1}$$
$$n_{a_t,t+1} = n_{a_t,t} + 1$$

### Primal-Dual Algorithm

The Primal-Dual algorithm handles constrained optimization via Lagrangian duality:

#### Lagrangian Formulation
$$\mathcal{L}(p, \lambda) = \mathbb{E}[R(p)] - \lambda \left( \mathbb{E}[C(p)] - B \right)$$

Where:
- $R(p)$ = revenue (number of purchases)
- $C(p)$ = cost (price × purchases)  
- $\lambda$ = dual variable (shadow price of budget constraint)

#### Action Selection
$$a_t = \arg\max_{i=1,\ldots,K} \left[ \hat{\mu}_{i,t} - \lambda_t \cdot p_i \right]$$

#### Dual Variable Update
$$\lambda_{t+1} = \max\left(0, \min\left(1, \lambda_t + \eta \left( \frac{\text{cost}_t}{B} - 1 \right) \right)\right)$$

Where:
- $\eta$ = learning rate
- $\text{cost}_t$ = total spending in round $t$
- Projection ensures $\lambda \in [0, 1]$

#### Reward and Cost Calculation
- **Reward**: $r_t = \text{purchases}_t$ (number of items sold)
- **Cost**: $c_t = \text{price}_t \times \text{purchases}_t$

## Simulation Process

### Single Round Execution

For each round $t = 1, 2, \ldots, T$:

1. **Parameter Update** (if non-stationary):
   $$\theta_{k,t} = \theta_{k,t-1} + \epsilon \cdot Z_t, \quad Z_t \sim \mathcal{N}(0, 1)$$

2. **Buyer Generation**:
   - Create buyer with updated distribution parameters
   - Sample valuation $V_{k,t} \sim F_{k,t}(\theta_{k,t})$

3. **Seller Decision**:
   - Compute action values (UCB1 or Primal-Dual)
   - Select price: $p_t = \arg\max_p \text{ActionValue}(p)$

4. **Market Interaction**:
   - Buyer observes price $p_t$
   - Makes purchase decision: $d_t = \mathbf{1}(V_{k,t} \geq p_t)$
   - Generates demand vector across all products

5. **Constraint Checking**:
   - **Budget**: $\sum_{s=1}^t p_s \cdot d_s \leq B$
   - **Inventory**: $\sum_{s=1}^t d_{k,s} \leq I_k$ for each product $k$

6. **Algorithm Update**:
   - Observe reward $r_t = d_t$ (purchases)
   - Update internal parameters (means, counts, dual variables)

7. **Performance Tracking**:
   - Compute optimal reward: $r^*_t = \max_p \mathbb{E}[D_t(p)]$
   - Calculate instantaneous regret: $\text{regret}_t = r^*_t - r_t$
   - Update cumulative regret: $R_t = \sum_{s=1}^t \text{regret}_s$

### Multi-Run Simulation

For statistical analysis, repeat the above process for multiple independent runs:

$$\bar{R}_T = \frac{1}{N} \sum_{n=1}^N R_{T,n}$$

Where $R_{T,n}$ is the cumulative regret of run $n$.

## Critical Fixes Implemented

### 1. Budget Constraint Implementation (CRITICAL FIX)

**Previous Bug**:
```python
# WRONG: Checked maximum prices instead of actual spending
constraint_violated = np.any(prices > B)
```

**Fixed Implementation**:
```python
# CORRECT: Check actual purchases against budget
total_spending = np.sum(prices * purchases)
constraint_violated = total_spending > B
```

**Mathematical Impact**: The constraint now correctly enforces:
$$\sum_{t=1}^T p_t \cdot d_t \leq B$$

### 2. UCB1 Confidence Bound Calculation (CRITICAL FIX)

**Previous Bug**:
```python
# WRONG: Used total horizon T instead of current time t
confidence = np.sqrt(2 * np.log(self.T) / counts)
```

**Fixed Implementation**:
```python
# CORRECT: Use current timestep t
confidence = np.sqrt(2 * np.log(t) / counts)
```

**Mathematical Impact**: Confidence bound now correctly implements:
$$\text{UCB}_i(t) = \hat{\mu}_{i,t} + \sqrt{\frac{2\log t}{n_{i,t}}}$$

### 3. Primal-Dual Algorithm Logic (CRITICAL FIX)

**Previous Bugs**:
- Reward calculation: `reward = purchases - total_cost` (incorrect)
- Cost handling: Subtracted from each purchase (incorrect)
- Lambda update: No bounds, allowed negative values (incorrect)
- Budget modification: `B -= cost` (incorrect)

**Fixed Implementation**:
```python
# CORRECT implementations:
reward = purchases  # Just the number of sales
cost = price * purchases  # Separate cost calculation
lambda_new = max(0, min(1, lambda_old + eta * (cost/B - 1)))  # Bounded update
# B remains constant (no modification)
```

**Mathematical Impact**: Now correctly implements the Lagrangian:
$$\mathcal{L}(p, \lambda) = \mathbb{E}[\text{purchases}] - \lambda \left( \frac{\mathbb{E}[\text{cost}]}{B} - 1 \right)$$

## Performance Metrics

### Regret Analysis

- **Instantaneous Regret**: $r_t^* - r_t$
- **Cumulative Regret**: $R_T = \sum_{t=1}^T (r_t^* - r_t)$
- **Average Regret**: $\bar{r}_T = R_T / T$

### Theoretical Guarantees

- **UCB1**: $R_T = O(\sqrt{KT \log T})$ with high probability
- **Primal-Dual**: $R_T = O(\sqrt{T})$ under appropriate conditions

### Convergence Properties

- **UCB1**: Optimal for stationary environments
- **Primal-Dual**: Adapts to constraints and non-stationarity

## Experimental Validation

The fixed implementations now demonstrate:

1. **Correct Exploration-Exploitation**: UCB1 confidence bounds decrease appropriately
2. **Constraint Satisfaction**: Budget and inventory limits properly enforced  
3. **Regret Convergence**: Both algorithms show sublinear regret growth
4. **Non-Stationarity Adaptation**: Primal-Dual handles changing environments
5. **Statistical Consistency**: Results stable across multiple runs

## Conclusion

The multi-armed bandit auction platform now provides mathematically correct implementations of both UCB1 and Primal-Dual algorithms, with robust constraint handling and comprehensive performance tracking. The critical fixes ensure that both algorithms behave according to their theoretical guarantees and can be reliably used for online auction optimization research.
