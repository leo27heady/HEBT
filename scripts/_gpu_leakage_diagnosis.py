import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
from model.vid.hvqvae.config import HVQVAEConfig
from model.vid.hvqvae.model import HVQVAEModel
from data.vid.coco_medium_dataset import VIDShapeSyntheticDataset

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")
    
    cfg = HVQVAEConfig()
    model = HVQVAEModel(cfg).to(device)
    model.train()
    
    # 1. MATHEMATICAL CAUSALITY TEST
    print("\n--- 1. Strict Causality Gradient Test ---")
    dummy_video = torch.randn(2, 5, 3, 64, 64, device=device, requires_grad=True)
    out = model(dummy_video)
    
    # We take ONLY the prediction for frame 1 (pos 0) and sum it
    pos_0_pred = out.pred_rgb[:, 0].sum()
    pos_0_pred.backward()
    
    grad_norms = [dummy_video.grad[:, i].norm().item() for i in range(5)]
    print("Gradient norm from Pos 0 prediction flowing back to input frames:")
    for i, norm in enumerate(grad_norms):
        status = "EXPECTED (causal)" if i == 0 else "EXPECTED ZERO"
        # If norm > 0 for i > 0, we have forward-pass leakage!
        alert = "!!! LEAK !!!" if i > 0 and norm > 1e-7 else ""
        print(f"  Frame {i}: {norm:.2e}  {status} {alert}")
        
    if any(n > 1e-7 for n in grad_norms[1:]):
        print("\nCONCLUSION: Leakage exists in the forward pass!")
        return
    else:
        print("\nCONCLUSION: The math proves Pos 0 prediction DOES NOT SEE Frame 1+. There is NO forward-pass leakage.")

    # 2. MEMORIZATION VS GENERALIZATION TEST
    print("\n--- 2. Dataset Memorization Test ---")
    print("If it's not leaking forward, it might just be memorizing 1000 examples.")
    
    # Dataset of 1000 samples (like your setup)
    train_dataset = VIDShapeSyntheticDataset(num_samples=1000, num_frames=5, image_size=64, cache_dir='data/vid/train_cache')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    # Completely distinct validation dataset of 1000 samples
    val_dataset = VIDShapeSyntheticDataset(num_samples=1000, num_frames=5, image_size=64, cache_dir='data/vid/val_cache')
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    print("Training quickly for 500 steps to let it learn the 1000 train samples...")
    step = 0
    while step < 500:
        for batch in train_loader:
            if step >= 500: break
            video = batch['video'].to(device)
            target = video[:, 1:].clone()
            
            optimizer.zero_grad()
            out = model(video)
            
            # Loss only on pos 0 (predicting frame 1) for this diagnostic
            loss_pos0 = nn.functional.mse_loss(out.pred_rgb[:, 0], target[:, 0])
            loss_pos0.backward()
            optimizer.step()
            
            if step % 100 == 0:
                print(f"  Step {step:3d} | Train Pos 0 MSE: {loss_pos0.item():.4f}")
            step += 1

    model.eval()
    with torch.no_grad():
        # Eval on Train
        train_batch = next(iter(train_loader))['video'].to(device)
        train_out = model(train_batch)
        final_train_mse = nn.functional.mse_loss(train_out.pred_rgb[:, 0], train_batch[:, 1]).item()
        
        # Eval on Val
        val_batch = next(iter(val_loader))['video'].to(device)
        val_out = model(val_batch)
        val_mse = nn.functional.mse_loss(val_out.pred_rgb[:, 0], val_batch[:, 1]).item()

    print("\n--- FINAL VERDICT ---")
    print(f"Train Dataset Pos 0 MSE: {final_train_mse:.4f}")
    print(f"Unseen Val Dataset Pos 0 MSE:   {val_mse:.4f}")
    
    ratio = val_mse / (final_train_mse + 1e-8)
    if ratio > 2.0:
        print(f"Validation loss is {ratio:.1f}x higher than train loss for the first frame.")
        print("Verdict: MEMORIZATION. The model is memorizing the 1000 shapes/rotations, not leaking.")
    else:
        print("Verdict: The model actually predicts the unseen rotations well? Check if the dataset generation lacks randomness.")

if __name__ == "__main__":
    main()