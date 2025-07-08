# Online Learning Applications Project Requirements

Three scheduled project presentations (see last slides)
max 16 points
The goal is to develop algorithms for a complex problem
Includes modeling and coding

## Goal

The goal of the project is to design online learning algorithms to sellmultipletypes of
products underproduction constraints.

## Setting

A company has to choose prices dynamically.

### Parameters

- Number of rounds T
- Number of types of products N
- Set of possible prices P (small and discrete set)
- Production capacity B
For simplicity, there is a total number of products B
that the company can produce (independently from the specific type of
product)

### Buyer behavior

Has a valuationvifor each type of product inN
Buys all products priced below their respective valuations

## Interaction

At each round t∈T:

- The company chooses which types of product to sell and set price p_i for each type
of product
- A buyer with a valuation for each type of product arrives
- The buyer buys a unit of each product with price smaller than the product
valuation

## Requirement 1: Single product and stochastic environment

### R1 Enviroment

Build a stochastic environment:
A distribution over the valuations of a single type of product

### R1 Algorithm

- Build a pricing strategy using UCB1 ignoring the inventory constraint.
- Build a pricing strategy extending UCB1 to handle the inventory constraint.

Hint: Extend the “UCB-like” approach that we saw for auctions to the pricing problem.

## Requirement 2: Multiple products and stochastic environment

### R2 Environment

Build a stochastic environment:
A joint distribution over the valuations of all the types of products

### R2 Algorithm

Build a pricing strategy using Combinatorial-UCB with the inventory constraint.

Hint: Extend the “UCB-like” approach that we saw for auctions to the combinatorial pricing
problem. It was the following:

#### Algorithm: UCB-Bidding Algorithm

- input: Budget B, number of rounds T;
- for t = 1,...,T do
  - for b ∈B do
    - ¯f_t(b) ← 1/(N_t−1(b)) Sum\[(t′=1 -> t-1)\[f_t′(b)I(b_t′ = b)\];
    - ¯f_t^UCB(b) ← ¯f_t(b) + sqrt(2log(T)/N_t−1(b));
    – ¯c_t(b) ← 1/(N_t−1(b)) Sum[(t′=1 -> t-1)(c_t′(b)I(b_t′ = b))];
    - ¯c_t^LCB(b) ← ¯c_t(b) − sqrt(2log(T)/N_t−1(b));
  - compute γ_t solution of the LP defining OPT_t ;
  - bid b_t ∼γ_t ;
  - observe f_t(b_t) and c_t (b_t);
  - B ← B−c_t(b_t);
  - if B <1 then
    - terminate;

## Requirement 3: Best-of-both-worlds algorithms with a single product

### R3 Environment

Use the stochastic environment already designed:
A distribution over the valuations of a single product
Build a highly non-stationary environment. At a high level, it should include:
A sequence of valuations of the product (e.g., sampled from a distribution that
changes quickly over time)

### R3 Algorithm

Build a pricing strategy using a primal-dual method with the inventory constraint.

Hint: To design a primal-dual method, extend the results on “general auctions” to f and c
suitable for the pricing problem. The primal-dual algorithm seen in class for auctions is:

#### Algorithm: Pacing strategy

- input: Budget B, number of rounds T , learning rate η, primal regret minimizer R;
- initialization: ρ←B/T ,λ0 ←0;
- for t = 1,2,...,T do
  - choose distribution over bids γt ←R(t);
  - bid bt ∼γt ;
  - observe ft (bt ) and ct (bt ) ;
  - λt ←Π[0,1/ρ](λt−1−η(ρ−ct (bt ))) ;
  - B ←B−ct (bt );
  - if B <1 then
    - terminate;
R is any regret minimizer and R(t) returns a distribution over bids at round t.

## Requirement 4: Best-of-both-worlds with multiple products

### R4 Environment

Use the stochastic environment already designed:
A joint distribution over the valuations of all the types of products
Build a highly non-stationary environment. At a high level, it should include:
A sequence of correlated valuations for each type of product (e.g., sampled from a
distribution that changes quickly over time)

### R4 Algorithm

Build a pricing strategy using a primal-dual method with the inventory constraint.

Hint: To design a primal regret minimizer, notice that the pricing problem “decomposes”. It
is sufficient to design an (adversarial) regret minimizer for each type of product.

## Requirement 5: Slightly non-stationary environments with multiple products

## R5 Non-stationary environment

Build a slightly non-stationary environment for the pricing problem. At a high level:

- Rounds are partitioned in intervals
- In each interval the distribution of products valuations is fixed
- Each interval has a different distribution

### R5 Algorithm

Extend Combinatorial-UCB with sliding window

### Compare

Compare the performance of:

- Combinatorial-UCB with sliding window
- The primal-dual method
