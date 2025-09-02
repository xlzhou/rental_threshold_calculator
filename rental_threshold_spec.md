# Rental Threshold Calculator — Development Specification (v1.0)

## 1) Overview

**Goal:** Build a small calculator (CLI/Web -compatible) that recommends **accept/reject** decisions for rental offers based on inventory, time left, costs, and price distribution. It must support **sub-cost acceptance** when needed to avoid end-of-horizon leftover risk (≤3 preferred; >5 = failure).

**Core idea:** Single-resource revenue management with a **threshold (bid-price) policy**. Baseline uses a **static cutoff**calibrated to expected arrivals; optional module provides a **dynamic (time/inventory) bid-price**. Includes a **pacing rule** to relax thresholds if behind target.

## 2) Scope

- **In scope (MVP):**  
    Static threshold selection from empirical prices; accept/reject decision; expected accepts/leftover; pacing checkpoint; penalty-aware recommendation; simple tables/exports.
- **Out of scope (MVP):** Multi-asset allocation, cross-price elasticity, strategic customer behavior, competitor reactions.

## 3) Users & Primary Use Cases

- **Ops manager:** fast go/no-go on each offer.
- **Planner/analyst:** choose horizon-wide cutoffs to hit sell-through goals (≤3 leftover; avoid >5).
- **Finance:** sensitivity on costs/penalties vs. cashflow.

## 4) Key Definitions & Symbols

- X: starting inventory
- T: number of periods (days/rounds)
- c: unit cost/base rent
- s: salvage per unit at T (default 0)
- N: expected number of offers over horizon (arrival rate  ×T)
- F(p): (empirical) CDF of offer prices
- **Static cutoff τ:** accept iff p≥max{c_floor,τ} where τ solves N⋅(1−F(τ))≈X−L\\\*.
- **Dynamic bid price:** accept iff p≥c+bt(x), where bt(x)=Vt+1(x)−Vt+1(x−1).
- **Leftover target L\\\*** default 3 (soft), failure if leftover >5.
- **Terminal penalty:** convex penalty Φ(L) that grows sharply after 3 and strongly after 5

## 5) Functional Requirements

### F1. Inputs

- Required: X,T,c, (default 0), arrival rate (per period or total N), **empirical price list** (values & equal weights, or weighted).
- Risk controls:
    - Target leftover L\\\*=3
    - Failure threshold Lfail=5
    - Optional holding cost per period h **or** explicit terminal penalty parameters for Φ(L).
- Offer stream mode:
    - **Batch planning:** compute thresholds before the horizon.
    - **Live decision:** enter a single offer p + current (t,x) → accept/reject.

### F2. Outputs

- Static cutoff τ and **operational cutoff**  τ_use​=max{c_floor​,τ} (when sub-cost allowed, set c_floor​=−∞ or a user floor).
- Expected accepts E\[A\]=N⋅(1−F(τ_use​)); expected leftover E\[L\]=max{X−E\[A\],0}.
- Conditional mean accepted price E\[P∣P≥τ_use\].
- Expected margin and **penalty-adjusted** expected profit.
- **Decision** for live offer p: Accept or Reject; show margin p−c and rationale.
- **Pacing status** & recommendation (hold cutoff vs relax).

### F3. Pacing Rule (MVP)

- Mid-horizon checkpoint t=T/2. Let A_t​ be accepted units to date.
    - If A_T/2≥X−L\\\* _×0.5_ → maintain cutoff.
    - Else → relax cutoff by one step (e.g., from 65 → 62 in the provided example).

### F4. Penalty Handling

- Default **piecewise** terminal penalty:

Φ(L)=0, if 0≤L≤3

Φ(L)=α(L−3)^2, if 3<L≤5

Φ(L)=α(2)^2+β(L−5)^2, if L>5 (β≫α)

- Choose α,β so that β heavily discourages L>5.

### F5. Optional: Dynamic Program (V2)

- Backward induction over t=T−1…0, state x∈\[0..X\], price arrival & draw from empirical F.
- Terminal condition: V_T(x)=s⋅x−Φ(x).

## 6) Algorithms (MVP)

### A1. Static Cutoff Selection

1.  Compute N (arrivals).
2.  For each candidate threshold qq in the **empirical price grid**, compute tail prob phat_q=1−F(q)
3.  Compute expected accepts E\[A\]\_q​=N⋅ phat_q expected leftover E\[L\]\_q​=max{X−E\[A\]\_q​,0}.
4.  Compute expected unit margin m_q=E\[P∣P≥q\]−c.
5.  Score: Score_q=E\[A\]\_q⋅m_q−E\[Φ(L)\] (use E\[L\]\_q​ for MVP).
6.  Pick argmax q; set τ_use=q (or enforce a user floor).
7.  **Live decision:** accept iff p≥τ_use ​.

### A2. Pacing Adjustment

- If behind target by mid-horizon, lower τuseτuse​ to next lower empirical level (e.g., 65→62).

## 7) UI/UX Requirements

### Web (simple, responsive)
- a simple web UI to receive inputs and output results

- **Inputs panel:** X,T,c,; arrival rate/total N; empirical prices (textarea or CSV upload); L\\\*,L_fail​; penalty sliders.
- **Results cards:** cutoff, expected accepts/leftover, conditional mean price, expected (penalty-adjusted) profit.
- **Offer checker:** input one p → “Accept/Reject + why”.
- **Pacing widget:** shows on-track vs behind; one-click “relax cutoff”.
- **Tables/exports:** acceptance table by price; CSVor Excel export.