# Example: Two independent causes (Earthquake, Truck) explain a common effect (House Jumps)
# Complementary priors help cancel the "explaining away" effect in belief networks.

import numpy as np

# Prior probabilities
p_earthquake = 0.1
p_truck = 0.1

# Likelihood of "House jumps" given causes
def house_jumps(earthquake, truck):
    return 1 if earthquake or truck else 0

# Compute joint probabilities only for cases where house jumps
joint_probs = {}
total = 0

for e in [0, 1]:  # Earthquake
    for t in [0, 1]:  # Truck
        if house_jumps(e, t):
            prob = (p_earthquake if e else 1 - p_earthquake) * (p_truck if t else 1 - p_truck)
            joint_probs[(e, t)] = prob
            total += prob

# Normalize to get posterior: P(Earthquake, Truck | House Jumped)
posterior = {k: v / total for k, v in joint_probs.items()}
print("Posterior P(Earthquake, Truck | House jumped):")
for k, v in posterior.items():
    print(f"  Earthquake={k[0]}, Truck={k[1]} -> P={v:.4f}")
