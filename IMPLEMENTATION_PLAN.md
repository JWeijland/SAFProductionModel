# Implementation Plan: Iterative Feedstock Price Discovery with Long-Term Contracts

## Overview
Implement an annual 30-day negotiation mechanism where feedstock aggregators and investors discover equilibrium prices through iterative adjustment, resulting in 20-year contracts with partial spot market exposure.

---

## 1. Core Problem: Oversupply and N_target

### Current Issue
```
N_target = Aggregator_total_supply / 100,000 tonnes
```
- PUNJAB example: 4,787,000 / 100,000 = 47.8 investors
- Problem: Aggregator happy with just 1-2 investors, not 47
- High N_target → everyone participates → no competition → low prices

### Solution: Capacity Utilization Fraction
Apply a time-varying fraction to limit how much capacity aggregator offers annually:

```
N_target = (Aggregator_max_supply × utilization_fraction(year)) / 100,000
```

**Utilization Fraction Strategy:**
- **Year 1-5**: 5-10% of capacity → forces scarcity
- **Year 5-15**: 10-30% of capacity → gradual expansion
- **Year 15+**: 30-60% of capacity → mature market

Example (PUNJAB):
- Year 1: 4,787,000 × 0.05 / 100,000 = **2.4 investors** → intense competition
- Year 10: 4,787,000 × 0.20 / 100,000 = **9.6 investors** → moderate competition
- Year 20: 4,787,000 × 0.50 / 100,000 = **24 investors** → stable market

**Function signature:**
```python
def get_utilization_fraction(current_year: int, start_year: int) -> float:
    """
    Returns fraction of aggregator capacity to offer based on market maturity.
    Starts low (scarcity) and increases over time (market expansion).
    """
```

**Key principle**: Behavior emerges naturally - no hardcoded investor strategies.

---

## 2. Improved Price Convergence Formula

### Problem with Linear Approach
```
Δp = -α × (N_active - N_target)
```
- Converges only at discrete multiples of α
- Ugly step behavior

### Solution: Smoothed Proportional Control
```python
error = N_active - N_target
Δp = -α × error / (1 + β × |error|)
```

Where:
- `α` = base step size (e.g., 10 USD)
- `β` = dampening factor (e.g., 0.1)
- Effect: Large errors → smaller relative steps (prevents overshoot)
- Small errors → proportional adjustments (fine-tuning)

**Alternative: Exponential decay**
```python
Δp = -α × tanh(error / scale)
```
- Smooth S-curve response
- Bounded adjustment magnitude

---

## 3. Contract Structure

### Contract Data Model
```python
class FeedstockContract:
    contract_id: str
    investor_id: str
    aggregator_id: str (state_id)
    plant_id: str
    contract_price: float           # Locked-in p*
    contract_start_year: int
    contract_duration: int = 20     # years
    annual_capacity: float          # tonnes/year (plant's design capacity)
    contract_percentage: float      # 0.80-0.90 (investor chooses)
    status: str                     # 'active', 'expired'
```

### Key Rules
- **One contract per plant**
- **Investor obligation**: Must purchase `annual_capacity × contract_percentage` annually
- **Aggregator obligation**: Must deliver any amount investor requests within 80-90% range
- **Spot market**: Remaining 10-20% purchased at annual average spot price

---

## 4. Iterative Price Discovery Algorithm

### New Module: `src/price_discovery.py`

```python
def run_price_discovery_negotiation(
    aggregator: FeedstockAggregator,
    interested_investors: List[Investor],
    current_year: int,
    max_days: int = 30,
    initial_price: float = None,
    convergence_threshold: float = 0.01
) -> Dict:
    """
    Runs 30-day iterative negotiation between aggregator and investors.

    Returns:
        {
            'final_price': float,
            'winning_investors': List[Investor],
            'daily_prices': List[float],
            'daily_participation': List[int]
        }
    """
```

### Algorithm Flow (per negotiation round)

1. **Initialization**
   - Calculate `N_target` using utilization fraction
   - Set initial price (e.g., aggregator's feedstock_price + margin)

2. **Daily iteration (30 days)**
   ```
   Day 1-30:
     a) Announce current price
     b) Each investor calculates NPV using conventional formula
     c) Count N_active (investors with NPV > threshold)
     d) Update price:
        error = N_active - N_target
        Δp = -α × error / (1 + β × |error|)
        price_next = price_current + Δp
     e) Check convergence: |Δp| < threshold
   ```

3. **Finalization**
   - Final price = p*
   - Winning investors = all active at p*
   - Create contracts for each winning investor's plant

### Integration with Existing Model

**Timing:**
- Regular simulation step = 1 year (current behavior)
- **NEW**: Before each year's investment phase, run negotiation window
- Negotiation = sub-simulation (30 daily iterations)

---

## 5. Spot Market Pricing

### Annual Spot Price Calculation

Since multiple investors can invest in the same year (potentially from different aggregators):

```python
def calculate_annual_spot_price(all_negotiations_this_year: List[Dict]) -> float:
    """
    Average all final prices from negotiations that occurred this year.

    Args:
        all_negotiations_this_year: List of negotiation results

    Returns:
        Average final price across all negotiations
    """
    prices = [neg['final_price'] for neg in all_negotiations_this_year]
    return sum(prices) / len(prices) if prices else default_price
```

**Usage:**
- Existing contract holders buy their 10-20% at this average spot price
- Updates annually
- Creates natural price volatility

---

## 6. Modified Agent Behaviors

### 6.1 Feedstock Aggregator Changes

**New attributes:**
```python
self.active_contracts: List[FeedstockContract]
self.utilization_fraction: float  # Calculated annually
self.negotiation_history: List[Dict]  # Track past negotiations
```

**New methods:**
```python
def initiate_negotiation(self, year: int) -> Dict:
    """Run price discovery for this year's capacity."""

def allocate_contracted_capacity(self) -> float:
    """Calculate capacity already committed via contracts."""

def get_available_capacity_for_negotiation(self, year: int) -> float:
    """Total supply × utilization_fraction - contracted capacity."""
```

### 6.2 Investor Changes

**New attributes:**
```python
self.contracts: List[FeedstockContract]  # Contracts this investor holds
```

**New methods:**
```python
def participate_in_negotiation(
    self,
    aggregator: FeedstockAggregator,
    proposed_price: float
) -> bool:
    """
    Evaluate whether to participate at proposed price.
    Uses conventional NPV calculation (existing calculate_npv method).
    Returns True if NPV > threshold.
    """

def decide_contract_percentage(self) -> float:
    """
    Choose between 80-90% contract coverage.
    Could be random, or strategic (higher % if risk-averse).
    Returns value in [0.80, 0.90].
    """

def purchase_spot_feedstock(self, spot_price: float) -> float:
    """
    Purchase remaining 10-20% at annual spot price.
    """
```

### 6.3 Model Changes

**New attributes:**
```python
self.all_contracts: List[FeedstockContract]
self.annual_spot_prices: Dict[int, float]  # year -> price
self.negotiation_results: Dict[int, List[Dict]]  # year -> negotiations
```

**Modified step() logic:**
```python
def step(self):
    # ... existing setup ...

    # NEW: Run annual negotiations (before invest phase)
    self.run_annual_negotiations()

    # Calculate spot price for this year
    self.annual_spot_prices[current_year] = calculate_annual_spot_price(
        self.negotiation_results[current_year]
    )

    # Existing: update_supply, produce, calculate price, evaluate
    # ...

    # Modified invest: Check contract obligations first
    self.fulfill_contract_obligations()

    # Then normal investment for new plants
    for agent in self.schedule.agents:
        agent.invest()
```

---

## 7. Implementation Order

### Phase 1: Core Price Discovery (No contracts yet)
1. Create `src/price_discovery.py`
2. Implement `get_utilization_fraction()`
3. Implement improved convergence formula
4. Implement `run_price_discovery_negotiation()`
5. Write unit tests for price convergence

### Phase 2: Contract Infrastructure
6. Create `FeedstockContract` data class
7. Add contract tracking to Aggregator
8. Add contract tracking to Investor
9. Add contract tracking to Model

### Phase 3: Integration
10. Modify `Model.step()` to run annual negotiations
11. Implement `Investor.participate_in_negotiation()`
12. Implement spot price calculation and storage
13. Implement contract obligation fulfillment

### Phase 4: Contract Behavior
14. Implement `Investor.decide_contract_percentage()`
15. Implement `Investor.purchase_spot_feedstock()`
16. Update NPV calculation to account for dual pricing
17. Modify feedstock allocation logic for contracted vs available

### Phase 5: Testing & Validation
18. Test single negotiation convergence
19. Test multiple simultaneous negotiations (different aggregators)
20. Test 20-year contract lifecycle
21. Validate spot price averaging
22. Verify no hardcoded behaviors (emergent only)

---

## 8. Key Configuration Parameters

Add to `input/config.csv`:

```csv
key,value,unit,remarks
negotiation_days,30,,Days for iterative price discovery
contract_duration,20,years,Standard contract length
contract_percentage_min,0.80,,Minimum contract coverage
contract_percentage_max,0.90,,Maximum contract coverage
utilization_fraction_start,0.05,,Initial capacity utilization (year 1)
utilization_fraction_max,0.60,,Maximum capacity utilization (mature)
utilization_growth_rate,0.03,,Annual increase in utilization fraction
price_convergence_alpha,10.0,USD,Base price adjustment step
price_convergence_beta,0.1,,Dampening factor for convergence
price_convergence_threshold,1.0,USD,Stop when |Δp| < threshold
```

---

## 9. Data Collection & Visualization

**New metrics to track:**

**Model-level:**
- `Annual_Spot_Price`: Average negotiated price each year
- `Num_Active_Contracts`: Total contracts across all aggregators
- `Total_Contracted_Capacity`: Sum of capacity under contract

**Aggregator-level:**
- `Negotiation_Final_Price`: Result of this year's negotiation
- `Num_Contracts`: Contracts held by this aggregator
- `Utilization_Fraction`: Current capacity utilization
- `Available_Capacity`: Uncontracted capacity

**Investor-level:**
- `Num_Contracts`: Number of contracts held
- `Contract_Percentage`: Chosen coverage (80-90%)
- `Spot_Purchase_Volume`: Annual spot market purchases
- `Weighted_Avg_Feedstock_Cost`: Blended contract + spot price

---

## 10. Critical Design Principles

### No Hardcoded Behaviors
✅ **Correct**: Set rules, let behavior emerge
- Investors calculate NPV and decide participation
- Market dynamics determine prices
- Spot vs contract balance emerges from conditions

❌ **Incorrect**: Programmed strategies
- "Investor always picks 90% in year 1"
- "Drop out if price > X"
- "Strategic bidding"

### Natural Market Dynamics
- Competition intensity → from N_target relative to interest
- Price levels → from investor NPV thresholds
- Contract percentages → investor choice (random or simple rule)
- Spot market activity → difference between contract and spot prices

---

## 11. Edge Cases to Handle

1. **No investors interested**: Price decreases to floor, no contracts signed
2. **All investors interested**: Price increases until N_active ≤ N_target
3. **Contract expiration**: Capacity becomes available again (greenfield/brownfield later)
4. **Multiple aggregators negotiating same year**: Independent negotiations, aggregate for spot price
5. **Investor with multiple plants**: Each plant has separate contract
6. **Spot price when no negotiations occur**: Use previous year or default

---

## 12. Success Criteria

Implementation complete when:
- ✅ Price discovery converges smoothly (no oscillations)
- ✅ Oversupply scenario still creates competition (via utilization fraction)
- ✅ Undersupply scenario naturally produces high prices
- ✅ Contracts last 20 years with correct pricing
- ✅ Spot market averages multiple negotiations
- ✅ 80-90% contract + 10-20% spot works correctly
- ✅ No investor behaviors are hardcoded
- ✅ Emergent market dynamics are realistic

---

## End of Implementation Plan

**Next step**: Review this plan, confirm understanding, then begin Phase 1 implementation.
