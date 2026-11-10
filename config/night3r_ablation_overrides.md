# Night3R Fusion Ablation Overrides

Base BRG config:

```bash
config/train_night3r_brg_token_reg_bf16.yaml
```

Legacy tensor no-FiLM config:

```bash
config/train_night3r_no_film_tensor_gate_bf16.yaml
```

Use these YAML overrides by copying the base config and changing only the shown keys.

## 1. RGB-Only Baseline

```yaml
fusion:
  fusion_type: "none"
  use_density_film: false
  density_film_type: "none"
```

## 2. Legacy Simple Gate

```yaml
fusion:
  fusion_type: "simple"
  use_density_film: false
  density_film_type: "none"
training:
  amp: true
  amp_dtype: "bf16"
```

## 3. Legacy Tensor Gate No FiLM

Use:

```bash
config/train_night3r_no_film_tensor_gate_bf16.yaml
```

Key fields:

```yaml
fusion:
  fusion_type: "legacy_tensor"
  gate_type: "tensor"
  use_density_film: false
  density_film_type: "none"
training:
  amp: true
  amp_dtype: "bf16"
```

## 4. BRG Token-Only

```yaml
fusion:
  fusion_type: "brg"
  use_density_film: false
  density_film_type: "none"
brg:
  use_channel_modulation: false
  use_gate_regularization: false
```

## 5. BRG Token-Only + Regularization

Use:

```bash
config/train_night3r_brg_token_reg_bf16.yaml
```

Key fields:

```yaml
fusion:
  fusion_type: "brg"
  use_density_film: false
brg:
  use_channel_modulation: false
  use_gate_regularization: true
```

## 6. BRG + Source Bias/Stat Norm

```yaml
fusion:
  fusion_type: "brg"
brg:
  use_source_bias: true
  use_source_stat_norm: true
  source_initial_gate_bias: [0.0, 0.0, 0.0]
```

Source ids are mapped as:

```text
mvsec -> 0
dsec  -> 1
dl3dv -> 2
```

## 7. BRG + Bounded Channel Modulation

```yaml
fusion:
  fusion_type: "brg"
brg:
  use_channel_modulation: true
  channel_alpha_start: 0.0
  channel_alpha_final: 0.10
  channel_alpha_max: 0.10
  channel_alpha_warmup_steps: 5000
```

## 8. BRG + Residual Identity FiLM

```yaml
fusion:
  fusion_type: "brg"
  use_density_film: true
  density_film_type: "residual_identity"
event_film:
  enabled: true
  type: "residual_identity"
  eta_init: 0.0
  eta_max: 0.1
```

## Diagnostics

Training logs include these BRG fields under `event_debug`:

```text
event_gate_mean
event_gate_std
event_gate_min
event_gate_max
event_gate_sat_low
event_gate_sat_high
event_residual_ratio
event_gate_logit_abs
event_low_event_gate_mean
event_high_event_gate_mean
event_gate_cap
event_gate_mean_cap
event_channel_dev
```

BRG regularization terms appear in `loss_dict`:

```text
event_aux/gate_mean_reg
event_aux/gate_low_event_reg
event_aux/gate_logit_reg
event_aux/channel_dev_reg
event_aux/total
```
