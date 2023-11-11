### Inspecting gradients, weights and biases

> PyTorch

```py
optimizer.zero_grad()
loss.backward()

for name, param in model.named_parameters():

    # Gradients
    if param.requires_grad:
        print(f"Parameter: {name}, Gradient Norm: {param.grad.norm()}")

    # Weights and Biases
    if 'weight' in name or 'bias' in name:
        print(f"Parameter: {name}, Mean: {param.mean().item()}, Std: {param.std().item()}")
```
