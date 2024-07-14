import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class StrokeEncoder(nn.Module):
    def __init__(self, stroke_params_dim, clip_vit):
        super(StrokeEncoder, self).__init__()
        self.adapt_conv = nn.Conv2d(6, 3, 1, 1, 0)
        self.encoder = clip_vit
        self.decoder = nn.Sequential(
            nn.Linear(self.encoder.output_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, stroke_params_dim)
        )

    def forward(self, foregrounds:Tensor, alphas:Tensor):
        # TODO: concat then prj.?| use + ?
        # batched_stroke_map = foregrounds + alphas
        batched_stroke_map = torch.cat((foregrounds, alphas), dim=1)
        batched_stroke_map = self.adapt_conv(batched_stroke_map)
        map_emb = self.encoder(batched_stroke_map)
        stroke_params = self.decoder(map_emb)
        return stroke_params

def ClampParams(params, CANVAS_WIDTH):
    x0, y0, w, h, theta = params[:, 0], params[:, 1], params[:, 2], params[:, 3], params[:, 4]
    R0, G0, B0, R2, G2, B2, ALPHA = params[:, 5], params[:, 6], params[:, 7], params[:, 8], params[:, 9], params[:, 10], params[:, 11]

    x0, y0, w, h, theta = \
        torch.clamp(x0, min=0.0, max=CANVAS_WIDTH-1),\
        torch.clamp(y0, min=0.0, max=CANVAS_WIDTH-1),\
        torch.clamp(w, min=1.0, max=CANVAS_WIDTH+1),\
        torch.clamp(h, min=1.0, max=CANVAS_WIDTH+1),\
        torch.clamp(theta, min=0.0, max=np.pi)
    R0, G0, B0, R2, G2, B2, ALPHA = \
        torch.clamp(ALPHA, min=0.0, max=255.),\
        torch.clamp(R0, min=0.0, max=255.),\
        torch.clamp(G0, min=0.0, max=255.),\
        torch.clamp(B0, min=0.0, max=255.), \
        torch.clamp(R2, min=0.0, max=255.), \
        torch.clamp(G2, min=0.0, max=255.), \
        torch.clamp(B2, min=0.0, max=255.), \
            # torch.clamp(R0, min=0.0, max=1.0), \
            # torch.clamp(G0, min=0.0, max=1.0), \
            # torch.clamp(B0, min=0.0, max=1.0), \
            # torch.clamp(R2, min=0.0, max=1.0), \
            # torch.clamp(G2, min=0.0, max=1.0), \
            # torch.clamp(B2, min=0.0, max=1.0), \

    clamped_params = torch.stack([x0, y0, w, h, theta, R0, G0, B0, R2, G2, B2, ALPHA], dim=1)
    return clamped_params

def StrokeParamsLoss(predicted_params, gt_params, reg_lambda, CANVAS_WIDTH):
    # predicted_params = ClampParams(predicted_params, CANVAS_WIDTH)
    mse_loss = F.mse_loss(predicted_params, gt_params)
    reg_loss = torch.mean(torch.abs(predicted_params - gt_params))
    loss = mse_loss + reg_lambda * reg_loss
    return loss