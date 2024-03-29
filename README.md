# declarative_privacy_preserving

Requirements: tensorflow_privacy, sklearn, pandas

## Hyper-Parameter Search results
- LR=0.002, EPOCHS=5000, NOISE MULTIPLIER=O.3   ====> EPSILON=~43,000; ACC=0.99
- LR=0.002, EPOCHS=5000, NOISE MULTIPLIER=0.35  ====> EPSILON=~24,000; ACC=0.97
- LR=0.003, EPOCHS=5000, NOISE MULTIPLIER=0.35  ====> EPSILON=~24,000; ACC=0.99
- LR=0.0025, EPOCHS=5000, NOISE MULTIPLIER=0.38 ====> EPSILON=~17,000; ACC=0.94
- LR=0.0025, EPOCHS=5000, NOISE MULTIPLIER=0.39 ====> EPSILON=~16,000; ACC=0.92
- LR=0.0025, EPOCHS=5000, NOISE MULTIPLIER=0.4  ====> EPSILON=~14,000; ACC=0.91
- LR=0.002, EPOCHS=10,000, NOISE MULTIPLIER=0.5, BATCH_SIZE=200 ====> EPSILON=~13,500; ACC=0.93

Increased Number of Hidden Units to 1000
- LR=0.001, EPOCHS=10,000, NOISE_MULTIPLIER=0.6, BATCH SIZE=250 ====> EPSILON=~8,600; ACC=0.89
- LR=0.001, EPOCHS=10,000, NOISE MULTIPLIER=0.5, BATCH SIZE=250 ====> EPSILON=~11,000; ACC=0.92
- LR=0.001, EPOCHS=10,000, NOISE MULTIPLIER=0.55, BATCH SIZE=200 ====> EPSILON=~9,900; ACC=0.93
