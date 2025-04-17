import torch
import torch.nn.functional as F
from tqdm import tqdm

def train(model, train_loader, optimizer, num_epochs, device):
    model.train()
    
    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(train_loader):
            left_img = batch['img1'].to(device)
            right_img = batch['img2'].to(device)
            disp_gt = batch['disp'].to(device)
            valid = batch['valid'].to(device)
            
            # Forward pass
            disp_pred = model(left_img, right_img)
            
            # Compute loss only on valid pixels
            loss = F.l1_loss(disp_pred[valid], disp_gt[valid])
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Log progress
        print(f'Epoch: {epoch}, Loss: {loss.item():.4f}')


def validate(model, val_loader, device):
    model.eval()
    total_epe = 0
    num_samples = 0
    
    with torch.no_grad():
        for batch in val_loader:
            left_img = batch['img1'].to(device)
            right_img = batch['img2'].to(device)
            disp_gt = batch['disp'].to(device)
            valid = batch['valid'].to(device)
            
            # Forward pass
            disp_pred = model(left_img, right_img)
            
            # Compute EPE (End Point Error)
            epe = torch.abs(disp_pred - disp_gt)
            epe = epe[valid].mean()
            
            total_epe += epe.item()
            num_samples += 1
    
    return total_epe / num_samples
