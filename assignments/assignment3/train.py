"""
作业3：使用物理信息神经网络求解泊松方程

问题描述：
在立方体区域 [-1,1]³ 中求解泊松方程
∇²φ = -ρ(x,y,z)
其中 ρ(x,y,z) = 100xyz²
边界条件：φ = 0 on boundary

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os


# ==================== 1. 定义神经网络 ====================
class PINN(nn.Module):
    """
    物理信息神经网络
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PINN, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)


# ==================== 2. 采样函数 ====================
def sample_points_in_cube(N, device='cpu'):
    """
    在立方体域 [-1, 1]³ 内随机采样点
    """
    points = 2 * torch.rand(N, 3, device=device) - 1
    return points


def sample_points_on_boundary(N, device='cpu'):
    """
    在立方体的 6 个边界面上采样点
    """
    N_side = int(np.ceil(N / 6.0))
    all_points = []
    
    # x = ±1
    for x_val in [1.0, -1.0]:
        yz = 2 * torch.rand(N_side, 2, device=device) - 1
        y = yz[:, 0:1]
        z = yz[:, 1:2]
        x = torch.full((N_side, 1), x_val, device=device)
        all_points.append(torch.cat([x, y, z], dim=1))

    # y = ±1
    for y_val in [1.0, -1.0]:
        xz = 2 * torch.rand(N_side, 2, device=device) - 1
        x = xz[:, 0:1]
        z = xz[:, 1:2]
        y = torch.full((N_side, 1), y_val, device=device)
        all_points.append(torch.cat([x, y, z], dim=1))
        
    # z = ±1
    for z_val in [1.0, -1.0]:
        xy = 2 * torch.rand(N_side, 2, device=device) - 1
        x = xy[:, 0:1]
        y = xy[:, 1:2]
        z = torch.full((N_side, 1), z_val, device=device)
        all_points.append(torch.cat([x, y, z], dim=1))

    boundary_points = torch.cat(all_points, dim=0)
    return boundary_points


# ==================== 3. 物理方程 ====================
def charge_distribution(r):
    """
    定义电荷分布 ρ(x,y,z) = 100xyz²
    """
    x = r[:, 0:1] 
    y = r[:, 1:2]
    z = r[:, 2:3]
    return 100 * x * y * (z**2)


def compute_pde_residual(model, r):
    """
    计算泊松方程残差：∇²φ + ρ = 0
    """
    if not r.requires_grad:
        r.requires_grad = True
        
    phi = model(r) # (N, 1)
    
    grad_phi = torch.autograd.grad(
        outputs=phi,
        inputs=r,
        grad_outputs=torch.ones_like(phi),
        create_graph=True,
        retain_graph=True
    )[0] # (N, 3)

    laplacian = 0.
    for i in range(3): 
        d2_phi_di2 = torch.autograd.grad(
            outputs=grad_phi[:, i:i+1], # (N, 1)
            inputs=r,
            grad_outputs=torch.ones_like(phi),
            create_graph=True,
        )[0][:, i:i+1] # (N, 1)
        
        laplacian += d2_phi_di2 
    
    rho = charge_distribution(r)
    
    residual = laplacian + rho
    return residual


# ==================== 4. 训练函数 ====================
def train(model, optimizer, num_epochs, N_domain, N_boundary, beta, device='cpu'):
    """
    训练 PINN 模型
    """
    model.to(device)
    losses = []
    losses_bc = []
    losses_pde = []
    
    r_boundary = sample_points_on_boundary(N_boundary, device)
    target_boundary = torch.zeros(r_boundary.shape[0], 1, device=device)
    
    for epoch in range(num_epochs):
        model.train()
        
        r_domain = sample_points_in_cube(N_domain, device)
        r_domain.requires_grad = True 
        
        target_pde = torch.zeros(N_domain, 1, device=device)
        
        # 1. BC Loss
        phi_boundary = model(r_boundary)
        loss_bc = F.mse_loss(phi_boundary, target_boundary)
        
        # 2. PDE Loss
        pde_residual = compute_pde_residual(model, r_domain)
        loss_pde = F.mse_loss(pde_residual, target_pde)
        
        # 3. Total Loss
        loss = loss_bc + beta * loss_pde
        
        # 4. Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        losses_bc.append(loss_bc.item())
        losses_pde.append(loss_pde.item())
        
        # 每 100 epoch 输出一次loss
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4e}, '
                  f'Loss_BC: {loss_bc.item():.4e}, Loss_PDE: {loss_pde.item():.4e}')
            
    return losses, losses_bc, losses_pde


# ==================== 5. 主程序 ====================
if __name__ == '__main__':
    # 设置超参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    input_dim = 3
    hidden_dim = 256
    output_dim = 1
    
    num_epochs = 50000
    learning_rate = 0.00002
    
    N_domain = 4096
    N_boundary = 1024
    beta = 0.1
    
    # 创建输出文件夹
    loss_dir = 'loss'
    result_dir = 'result'
    os.makedirs(loss_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    
    # 初始化模型和优化器
    model = PINN(input_dim, hidden_dim, output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练模型
    print("Starting training...")
    losses, losses_bc, losses_pde = train(model, optimizer, num_epochs, N_domain, N_boundary, beta, device)
    print("Training finished.")
    
    # 保存模型
    torch.save(model.state_dict(), 'pinn_poisson_3d.pth')
    print("Model saved to pinn_poisson_3d.pth")
    
    # 可视化训练曲线
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Total Loss')
    plt.plot(losses_bc, label='Boundary Loss (BC)')
    plt.plot(losses_pde, label=f'PDE Loss (x {beta})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.title('Training Loss Curve')
    plt.grid(True)
    
    # 保存到 'loss' 文件夹
    loss_filename = os.path.join(loss_dir, 'loss_curve.png')
    plt.savefig(loss_filename)
    print(f"Loss curve saved to {loss_filename}")
    plt.show()
    
    # 可视化结果
    print(f"Visualizing results (saving 10 slices to '{result_dir}' folder)...")
    model.eval() 

    N_test = 101 
    x_vals = torch.linspace(-1, 1, N_test)
    y_vals = torch.linspace(-1, 1, N_test)
    x_grid, y_grid = torch.meshgrid(x_vals, y_vals, indexing='ij')

    z_slices = np.arange(0.0, 1.0, 0.1) # [0.0, 0.1, ..., 0.9]

    for z_val in z_slices:
        print(f"Plotting slice at z = {z_val:.1f}...")
        
        z_grid = torch.full_like(x_grid, float(z_val)) 
        
        r_test = torch.stack([
            x_grid.flatten(), 
            y_grid.flatten(), 
            z_grid.flatten()
        ], dim=1).to(device)
        
        with torch.no_grad():
            phi_pred = model(r_test)
            
        phi_pred_grid = phi_pred.reshape(N_test, N_test).cpu().numpy()

        plt.figure(figsize=(10, 8))
        plt.contourf(x_grid.cpu().numpy(), y_grid.cpu().numpy(), phi_pred_grid, levels=50, cmap='jet')
        plt.colorbar(label=f'Predicted φ(x, y, z={z_val:.1f})')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Predicted Potential φ at z={z_val:.1f}')
        plt.axis('equal')
        
        # 保存到 'result' 文件夹
        filename_base = f'result_slice_z_{z_val:.1f}.png'
        filename = os.path.join(result_dir, filename_base)
        plt.savefig(filename)
        
        # plt.show()
        # plt.close()

    print(f"Saved {len(z_slices)} slice images to '{result_dir}' folder.")