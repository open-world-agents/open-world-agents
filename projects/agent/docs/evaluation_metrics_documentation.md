# Evaluation Metrics Documentation

This document provides comprehensive mathematical definitions and explanations for all evaluation metrics used in the OWA (Open World Agent) training system.

## Metrics Overview

| Category | Metric | Description | Units |
|----------|--------|-------------|-------|
| **Event Classification** | Comparable Events Ratio | Proportion of events with matching predicted/GT types | Ratio (0-1) |
| | Event Type Ratios | Distribution of event types in dataset | Ratio (0-1) |
| **Timing Accuracy** | Timestamp Absolute Error P95 | 95th percentile of timing errors | Milliseconds |
| | Timestamp Signed Error IQM/CI | Systematic timing bias with confidence intervals | Milliseconds |
| **Mouse Movement** | Euclidean PE P95 | 95th percentile of magnitude errors | Percentage |
| | Signed PE X/Y IQM/CI | Coordinate-specific bias with confidence intervals | Percentage |
| | Direction Error P50/P95 | Median and 95th percentile angular errors | Degrees |
| **Action Accuracy** | Keyboard Accuracy | Correct key code and event type predictions | Ratio (0-1) |
| | Mouse Action Accuracy | Correct button flag and scroll predictions | Ratio (0-1) |

## 1. Comparable Events Metrics

### 1.1 Overall Comparable Events
**Definition**: Proportion of predicted events that match the ground truth event type and can be directly compared.

$$\text{Comparable Ratio} = \frac{\text{Number of Comparable Events}}{\text{Total Events}}$$

Where an event is considered "comparable" if:
- Predicted event type matches ground truth event type
- Both events can be meaningfully compared for accuracy assessment

### 1.2 Event-Type Specific Comparable Ratios
For each event type $t \in \{\text{keyboard}, \text{mouse\_op}, \text{mouse\_nop}, \text{screen}\}$:

$$\text{Comparable Ratio}_t = \frac{\text{Comparable Events}_t}{\text{Total Events}_t}$$

- **Keyboard Comparable**: Fraction of keyboard events with matching event types
- **Mouse Operation Comparable**: Fraction of mouse click/button events with matching types
- **Mouse Movement Comparable**: Fraction of mouse movement events with matching types  
- **Screen Comparable**: Fraction of screen events with matching types

## 2. Event Distribution Metrics

### 2.1 Event Type Ratios
For each event type $t$:

$$\text{Event Type Ratio}_t = \frac{\text{Total Events}_t}{\sum_{i} \text{Total Events}_i}$$

These ratios describe the distribution of different event types in the dataset:
- **Keyboard Ratio**: Proportion of keyboard inputs
- **Mouse Operation Ratio**: Proportion of mouse clicks/button actions
- **Mouse Movement Ratio**: Proportion of mouse movements without clicks
- **Screen Ratio**: Proportion of screen-related actions

## 3. Timing Accuracy Metrics

### 3.1 Timestamp Absolute Error P95
**Definition**: 95th percentile of absolute timing errors.

For timestamp errors $e_i = |t_{\text{pred},i} - t_{\text{gt},i}|$:

$$\text{Timestamp Abs Error P95} = P_{95}(\{e_1, e_2, \ldots, e_n\})$$

**Units**: Milliseconds (converted from nanoseconds)
**Interpretation**: Represents worst-case timing precision for 95% of predictions.

### 3.2 Timestamp Signed Error IQM with Confidence Intervals
**Definition**: Interquartile Mean of signed timing errors with bootstrap confidence intervals.

Signed errors: $s_i = t_{\text{pred},i} - t_{\text{gt},i}$

$$\text{IQM} = \frac{1}{|S_{IQR}|} \sum_{s \in S_{IQR}} s$$

Where $S_{IQR} = \{s_i : Q_{25} \leq s_i \leq Q_{75}\}$

**95% Stratified Bootstrap Confidence Intervals**:
1. Generate $B = 1000$ bootstrap samples: $S^{(b)} = \{s_{i_1}^{(b)}, s_{i_2}^{(b)}, \ldots, s_{i_n}^{(b)}\}$
2. Compute IQM for each bootstrap sample: $\text{IQM}^{(b)}$
3. Calculate confidence interval: $[\text{IQM}_{2.5\%}, \text{IQM}_{97.5\%}]$

**Interpretation**: 
- Positive values indicate systematic late predictions
- Negative values indicate systematic early predictions
- Confidence intervals quantify uncertainty in bias estimation

## 4. Mouse Movement Precision Metrics

### 4.1 Mouse Euclidean Percentage Error P95
**Definition**: 95th percentile of Euclidean percentage errors for mouse movement vectors.

For mouse movements with ground truth $(dx_{\text{gt}}, dy_{\text{gt}})$ and predictions $(dx_{\text{pred}}, dy_{\text{pred}})$:

$$\text{Euclidean Error} = \sqrt{(dx_{\text{pred}} - dx_{\text{gt}})^2 + (dy_{\text{pred}} - dy_{\text{gt}})^2}$$

$$\text{Percentage Error} = \frac{\text{Euclidean Error}}{\sqrt{dx_{\text{gt}}^2 + dy_{\text{gt}}^2}} \times 100\%$$

$$\text{Mouse Euclidean PE P95} = P_{95}(\{\text{Percentage Error}_1, \ldots, \text{Percentage Error}_n\})$$

**Interpretation**: Worst-case spatial prediction accuracy for 95% of mouse movements.

### 4.2 Mouse Movement Signed Percentage Errors (X and Y coordinates)
**Definition**: IQM of signed percentage errors for individual coordinates with bootstrap CIs.

For X-coordinate:
$$\text{Signed PE}_x = \frac{dx_{\text{pred}} - dx_{\text{gt}}}{dx_{\text{gt}}} \times 100\%$$

For Y-coordinate:
$$\text{Signed PE}_y = \frac{dy_{\text{pred}} - dy_{\text{gt}}}{dy_{\text{gt}}} \times 100\%$$

**IQM Calculation**: Same as timestamp signed errors, applied to coordinate-specific signed PEs.

**Interpretation**:
- **X-coordinate bias**: Positive = rightward bias, Negative = leftward bias
- **Y-coordinate bias**: Positive = downward bias, Negative = upward bias

### 4.3 Mouse Movement Direction Error
**Definition**: Angular error between predicted and ground truth movement direction vectors.

For mouse movements with ground truth $(dx_{\text{gt}}, dy_{\text{gt}})$ and predictions $(dx_{\text{pred}}, dy_{\text{pred}})$:

$$\theta_{\text{gt}} = \arctan2(dy_{\text{gt}}, dx_{\text{gt}})$$
$$\theta_{\text{pred}} = \arctan2(dy_{\text{pred}}, dx_{\text{pred}})$$

$$\text{Direction Error} = \min(|\theta_{\text{pred}} - \theta_{\text{gt}}|, 2\pi - |\theta_{\text{pred}} - \theta_{\text{gt}}|) \times \frac{180°}{\pi}$$

**Reported Metrics**:
- **Movement Direction Error P50**: $P_{50}(\{\text{Direction Error}_1, \ldots, \text{Direction Error}_n\})$
- **Movement Direction Error P95**: $P_{95}(\{\text{Direction Error}_1, \ldots, \text{Direction Error}_n\})$

**Special Cases**:
- Zero movements $(dx_{\text{gt}} = 0, dy_{\text{gt}} = 0)$ are excluded (no meaningful direction)
- Error range: $[0°, 180°]$ where $0°$ = perfect direction, $180°$ = opposite direction

**Interpretation**:
- **Low values** (< 30°): Good directional accuracy
- **High values** (> 90°): Poor directional accuracy, major UX impact
- **180°**: Completely wrong direction (catastrophic for user experience)

## 5. Action Accuracy Metrics

### 5.1 Keyboard Accuracy
**Definition**: Proportion of keyboard events with correctly predicted key codes and event types.

$$\text{Keyboard Accuracy} = \frac{\sum_{i=1}^{n} \mathbf{1}[\text{vk}_{\text{pred},i} = \text{vk}_{\text{gt},i} \land \text{type}_{\text{pred},i} = \text{type}_{\text{gt},i}]}{n}$$

Where:
- $\text{vk}$: Virtual key code
- $\text{type}$: Event type (press/release)
- $\mathbf{1}[\cdot]$: Indicator function

### 5.2 Mouse Accuracy
**Definition**: Proportion of mouse actions with correctly predicted button flags and scroll data.

**Mouse Action Accuracy**:
$$\text{Mouse Action Accuracy} = \frac{\sum_{i=1}^{n} \mathbf{1}[\text{button\_flags}_{\text{pred},i} = \text{button\_flags}_{\text{gt},i}]}{n}$$

**Mouse Scroll Accuracy**:
$$\text{Mouse Scroll Accuracy} = \frac{\sum_{i=1}^{n} \mathbf{1}[\text{button\_data}_{\text{pred},i} = \text{button\_data}_{\text{gt},i}]}{n}$$

## 6. Statistical Methods

### 6.1 Interquartile Mean (IQM)
**Rationale**: More robust than median, reduces sensitivity to outliers while maintaining interpretability.

**Algorithm**:
1. Calculate quartiles: $Q_{25}$ and $Q_{75}$
2. Filter data: $D_{IQR} = \{x : Q_{25} \leq x \leq Q_{75}\}$
3. Compute mean: $\text{IQM} = \frac{1}{|D_{IQR}|} \sum_{x \in D_{IQR}} x$

### 6.2 Stratified Bootstrap Confidence Intervals
**Purpose**: Quantify uncertainty in IQM estimates without distributional assumptions.

**Algorithm**:
1. Set random seed for reproducibility: `np.random.seed(42)`
2. For $b = 1, 2, \ldots, B$ (where $B = 1000$):
   - Sample with replacement: $D^{(b)} = \{x_{i_1}, x_{i_2}, \ldots, x_{i_n}\}$
   - Compute $\text{IQM}^{(b)}$
3. Calculate percentiles: $[\text{IQM}_{2.5\%}, \text{IQM}_{97.5\%}]$

**Advantages**:
- Non-parametric approach
- Accounts for sampling variability
- Provides uncertainty quantification for robust statistics

## 7. Implementation Notes

- **Units**: Timestamps converted from nanoseconds to milliseconds for readability
- **Division by Zero**: Current handling excludes zero-movement events (reasonable design choice)
- **Missing Data**: Events with zero ground truth values excluded from percentage error calculations
- **Reproducibility**: Fixed random seed (42) used for bootstrap sampling
- **Statistical Validity**: Current metrics are well-designed and statistically sound
- **Direction Metrics**: Movement direction error metrics complement magnitude-based evaluation
- **Angular Calculations**: Uses `arctan2()` for proper quadrant handling and shortest angular distance
