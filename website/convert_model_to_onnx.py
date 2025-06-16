import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import onnx
import onnxruntime
from timm import create_model
from pathlib import Path

# ============= 修复后的模型定义 =============
class DCTLayer(nn.Module):
    """2D Discrete Cosine Transform layer - ONNX兼容版本"""
    def __init__(self, patch_size=8):
        super().__init__()
        self.patch_size = patch_size
        
        # 创建DCT基础矩阵
        self.register_buffer('dct_basis', self._create_dct_basis())
        
        # 创建卷积核用于块提取 - 使用register_buffer而不是nn.Conv2d
        # 这样权重不会被包含在state_dict中
        weights = torch.eye(patch_size * patch_size).view(
            patch_size * patch_size, 1, patch_size, patch_size
        ).float()
        self.register_buffer('unfold_weight', weights)
        
    def _create_dct_basis(self):
        """初始化DCT基础矩阵"""
        p = self.patch_size
        dct_basis = torch.zeros(p, p)
        
        for u in range(p):
            for v in range(p):
                if u == 0:
                    Cu = 1.0 / math.sqrt(p)
                else:
                    Cu = math.sqrt(2.0 / p)
                    
                for x in range(p):
                    dct_basis[u, x] = Cu * math.cos((2 * x + 1) * u * math.pi / (2 * p))
                    
        return dct_basis
    
    def forward(self, x):
        """应用2D DCT到输入特征 - ONNX兼容版本"""
        B, C, H, W = x.shape
        
        # 确保H和W可被patch_size整除
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
            pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
            x = F.pad(x, (0, pad_w, 0, pad_h))
            _, _, H, W = x.shape
        
        # 使用F.conv2d替代nn.Conv2d层
        patches = []
        for c in range(C):
            channel_patches = F.conv2d(
                x[:, c:c+1, :, :],
                self.unfold_weight,
                stride=self.patch_size,
                padding=0
            )
            patches.append(channel_patches)
        
        x_patches = torch.cat(patches, dim=1)  # B, C*p*p, H/p, W/p
        x_patches = x_patches.view(B, C, self.patch_size * self.patch_size, H // self.patch_size, W // self.patch_size)
        x_patches = x_patches.permute(0, 1, 3, 4, 2).contiguous()
        x_patches = x_patches.view(B, C, -1, self.patch_size, self.patch_size)
        
        # 应用DCT
        dct_1 = torch.matmul(self.dct_basis.unsqueeze(0), x_patches)
        dct_2 = torch.matmul(dct_1.transpose(-2, -1), self.dct_basis.T.unsqueeze(0))
        
        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size
        dct_2 = dct_2.view(B, C, num_patches_h, num_patches_w, self.patch_size, self.patch_size)
        dct_2 = dct_2.permute(0, 1, 2, 4, 3, 5).contiguous()
        dct_2 = dct_2.view(B, C, H, W)
        
        return dct_2

class ArtifactAttentionModule(nn.Module):
    """Artifact Attention Module (AAM) as described in the paper"""
    def __init__(self, in_channels, reduction_ratio=4):
        super().__init__()
        
        # Frequency branch with DCT
        self.dct_layer = DCTLayer(patch_size=8)
        self.freq_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // reduction_ratio),
            nn.GELU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels // reduction_ratio, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // reduction_ratio),
            nn.GELU()
        )
        
        # Spatial attention branch
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1),
            nn.BatchNorm2d(in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Multi-head attention for cross-attention
        self.num_heads = 4
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=self.num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Learnable fusion gate
        self.fusion_gate = nn.Parameter(torch.ones(1, in_channels, 1, 1))
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels + in_channels // reduction_ratio, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.GELU()
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Frequency branch
        x_dct = self.dct_layer(x)
        freq_features = self.freq_conv(x_dct)
        
        # Spatial branch
        spatial_att = self.spatial_attention(x)
        spatial_features = x * spatial_att
        
        # Prepare for cross-attention
        spatial_flat = spatial_features.flatten(2).transpose(1, 2)  # B, HW, C
        
        # Upsample frequency features to match spatial resolution
        freq_up = F.interpolate(freq_features, size=(H, W), mode='bilinear', align_corners=False)
        freq_flat = freq_up.flatten(2).transpose(1, 2)  # B, HW, C/4
        
        # Pad frequency features to match channel dimension for cross-attention
        freq_flat_padded = F.pad(freq_flat, (0, C - freq_flat.size(-1)))
        
        # Cross-attention: spatial features attend to frequency features
        attended_features, _ = self.cross_attention(
            query=spatial_flat,
            key=freq_flat_padded,
            value=freq_flat_padded
        )
        attended_features = attended_features.transpose(1, 2).reshape(B, C, H, W)
        
        gate = torch.sigmoid(self.fusion_gate)
        
        # Concatenate attended features and upsampled frequency features
        combined = torch.cat([attended_features, freq_up], dim=1)
        fused = self.fusion(combined)
        
        # Residual connection
        output = x + fused
        
        return output

class AAMSwinTransformer(nn.Module):
    """Swin Transformer with integrated AAM modules - 与训练代码完全一致"""
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        
        # Load base Swin Transformer
        self.backbone = create_model('swin_base_patch4_window7_224', pretrained=pretrained)
        
        # 关键：使用与训练代码相同的维度 [256, 512, 1024, 1024]
        embed_dims = [256, 512, 1024, 1024]
        
        # Create AAM modules for each stage
        self.aam_modules = nn.ModuleList([
            ArtifactAttentionModule(dim) for dim in embed_dims
        ])
        
        # Replace the classification head
        num_features = self.backbone.num_features
        self.backbone.head = nn.Identity()
        
        # New classification head with uncertainty estimation
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Uncertainty head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward_features(self, x):
        """Forward through backbone with AAM integration"""
        # Patch embedding
        x = self.backbone.patch_embed(x)
        if self.backbone.absolute_pos_embed is not None:
            x = x + self.backbone.absolute_pos_embed
        x = self.backbone.pos_drop(x)
        
        # Get initial resolution
        H, W = self.backbone.patch_embed.grid_size
        B, L, C = x.shape
        
        # Store resolutions for each stage
        resolutions = [(H, W)]
        
        # Process through stages with AAM
        for i, layer in enumerate(self.backbone.layers):
            # Get input resolution for current layer
            input_res = resolutions[-1]
            
            # Process through layer
            x = layer(x)
            
            # Calculate new resolution (if there was downsampling)
            if layer.downsample is not None:
                H, W = input_res[0] // 2, input_res[1] // 2
                resolutions.append((H, W))
            else:
                resolutions.append(input_res)
            
            # Get current resolution
            curr_res = resolutions[-1]
            curr_H, curr_W = curr_res
            
            # Reshape to 2D feature map
            B, L, C = x.shape
            x_2d = x.transpose(1, 2).view(B, C, curr_H, curr_W)
            
            # Apply AAM
            x_2d = self.aam_modules[i](x_2d)
            
            # Reshape back to sequence
            x = x_2d.flatten(2).transpose(1, 2)
        
        # Final norm
        x = self.backbone.norm(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        return x
        
    def forward(self, x):
        # Extract features
        features = self.forward_features(x)
        
        # Classification
        logits = self.classifier(features)
        
        # Uncertainty estimation
        uncertainty = self.uncertainty_head(features)
        
        return {
            'logits': logits,
            'uncertainty': uncertainty,
            'features': features
        }

# ============= 模型转换函数 =============
def convert_model_to_onnx():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 直接指定模型路径
    model_path = r"C:\Users\Congzhen\Desktop\xx\website\backend\models\best_model.pth"
    output_path = Path(model_path).parent / 'best_model.onnx'
    
    # 加载模型
    print(f"\nLoading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # 创建模型实例
    model = AAMSwinTransformer(num_classes=2, pretrained=False)
    model.to(device)
    model.eval()
    
    # 打印checkpoint的keys来调试
    print("\nCheckpoint keys:", checkpoint.keys() if isinstance(checkpoint, dict) else "Direct state dict")
    
    # 尝试不同的加载方式
    try:
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # 首先尝试严格加载
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print("Loaded model_state_dict with strict=False")
        else:
            # 如果不是字典或没有model_state_dict，尝试直接加载
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
            print("Loaded full state dict with strict=False")
        
        # 打印缺失和意外的键
        if missing_keys:
            print(f"\nMissing keys: {missing_keys}")
        if unexpected_keys:
            print(f"\nUnexpected keys: {unexpected_keys}")
            
    except Exception as e:
        print(f"Error loading state dict: {e}")
        print("Attempting to load with custom handling...")
        
        # 自定义加载逻辑
        state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
        new_state_dict = {}
        
        for k, v in state_dict.items():
            # 跳过unfold.weight相关的键
            if 'unfold.weight' in k:
                print(f"Skipping key: {k}")
                continue
            new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict, strict=False)
        print("Loaded state dict with custom handling")
    
    # 测试模型前向传播
    print("\nTesting model forward pass...")
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    with torch.no_grad():
        output = model(dummy_input)
        print("PyTorch model output logits shape:", output['logits'].shape)
        print("PyTorch model output uncertainty shape:", output['uncertainty'].shape)
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nConverting to ONNX: {output_path}")
    
    # 简化导出 - 只导出logits
    class InferenceWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, x):
            return self.model(x)['logits']
    
    wrapped_model = InferenceWrapper(model)
    wrapped_model.eval()
    
    # 导出ONNX模型 - 使用固定输入尺寸
    torch.onnx.export(
        wrapped_model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=14,  # 使用更高版本的opset
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        verbose=False
    )
    
    print("✓ ONNX export completed")
    
    # 验证ONNX模型
    print("\nVerifying ONNX model...")
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model validation passed")
    
    # 测试ONNX推理
    print("\nTesting ONNX inference...")
    ort_session = onnxruntime.InferenceSession(str(output_path))
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)
    print(f"ONNX output shape: {ort_outputs[0].shape}")
    
    # 检查PyTorch和ONNX输出是否相似
    with torch.no_grad():
        pytorch_output = wrapped_model(dummy_input).cpu().numpy()
    
    diff = np.abs(pytorch_output - ort_outputs[0]).max()
    print(f"Max difference between PyTorch and ONNX outputs: {diff:.6f}")
    if diff < 1e-5:
        print("✓ ONNX inference test passed")
    else:
        print("⚠ ONNX output differs from PyTorch output")
    
    # 导出量化版本
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantized_path = output_path.parent / 'best_model_int8.onnx'
        print(f"\nQuantizing model to INT8: {quantized_path}")
        quantize_dynamic(
            str(output_path),
            str(quantized_path),
            weight_type=QuantType.QInt8
        )
        print(f"✓ Quantization completed")
        
        # 测试量化模型推理
        print("\nTesting quantized ONNX inference...")
        quantized_session = onnxruntime.InferenceSession(str(quantized_path))
        quantized_outputs = quantized_session.run(None, ort_inputs)
        print(f"Quantized output shape: {quantized_outputs[0].shape}")
        
        quant_diff = np.abs(pytorch_output - quantized_outputs[0]).max()
        print(f"Max difference between PyTorch and quantized ONNX: {quant_diff:.6f}")
        if quant_diff < 0.1:
            print("✓ Quantized model test passed")
        else:
            print("⚠ Quantized model output differs significantly")
            
    except ImportError:
        print("\n⚠ onnxruntime-tools not installed. Skipping quantization.")
        print("  Install with: pip install onnxruntime-tools")
    except Exception as e:
        print(f"\n⚠ Error during quantization: {e}")
        print("  Skipping quantization step.")
    
    print("\n✅ Conversion completed successfully!")
    return str(output_path)

if __name__ == "__main__":
    convert_model_to_onnx()