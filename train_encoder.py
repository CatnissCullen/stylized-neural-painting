import torch
import clip
import torch.optim as optim
from encoder import StrokeEncoder, StrokeParamsLoss
import utils

def save_model_chk_point(save_dir, e, model, loss, optim):
    torch.save({
        'epoch': e,
        'model_state': model.state_dict(),
        'optimizer_state': optim.state_dict(),
        'loss': loss,
    }, save_dir + "epoch" + str(e) + ".pth")


def load_model_chk_point(save_dir, e=None):
    if e is not None:
        return torch.load(save_dir + "epoch" + str(e) + ".pth")
    else:
        return torch.load(save_dir)

config={
    'RENDERER': 'oilpaintbrush',  # watercolor, markerpen, oilpaintbrush, rectangle bezier, circle, square, rectangle
    'CANVAS_WIDTH': 128,
    'NO_MORPHOLOGY': True,
    'STROKES_NUM': 50000,  # strokes sample num. per epoch
    'batch_size': 256,
    'checkpoints_dir': 'checkpoints_T/',
    'lr': 1e-4,
    'step_size': 100,
    'gamma': 0.3,
    'beta1': 0.9,
    'beta2': 0.999,
    'reg_lambda': 0,
    'epochs': 500,
}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pretrained_CLIP, preprocess = clip.load("ViT-B/32", device=device)  # ViT-B/32 | RN50
print("clip is loaded.")

dataloaders, output_dim = utils.get_translator_loaders(config, preprocess)
train_loader, val_loader = dataloaders['train'], dataloaders['val']


encoder = StrokeEncoder(output_dim, pretrained_CLIP.visual).to(device).float()

optimizer = optim.Adam(encoder.parameters(), lr=config['lr'], betas=(config['beta1'], config['beta2']))
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['step_size'], gamma=config['gamma'])

e_num = config['epochs']
e_loss = []
min_loss = None
for e in range(e_num):
    print("starting training...")
    encoder.train()
    b, train_losses = 0, 0
    for batch in train_loader:
        b += 1

        params_batch, foreground_batch, alpha_batch = batch['A'].to(device), batch['B'].to(device), batch['ALPHA'].to(device)
        pred_params_batch = encoder(foreground_batch, alpha_batch)

        loss = StrokeParamsLoss(pred_params_batch, params_batch, config['reg_lambda'], config['CANVAS_WIDTH'])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        train_loss = loss.item()
        print("batch", b, "=> train_loss =", train_loss)
        train_losses += train_loss
    avg_train_loss = train_losses/ b

    print("running validation...")
    encoder.eval()
    b, val_losses = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            b += 1

            params_batch, foreground_batch, alpha_batch = batch['A'].to(device), batch['B'].to(device), batch['ALPHA'].to(device)
            pred_params_batch = encoder(foreground_batch, alpha_batch)

            loss = StrokeParamsLoss(pred_params_batch, params_batch, config['reg_lambda'], config['CANVAS_WIDTH'])

            val_loss = loss.item()
            val_losses += val_loss
    avg_val_loss = val_losses / b

    if min_loss is None or avg_val_loss < min_loss:
        min_loss = avg_val_loss
        save_model_chk_point(
            config['checkpoints_dir'],
            e, encoder, min_loss,
            optimizer
        )

    print("epoch", e + 1, "/", e_num, " => avg. train loss: ", avg_train_loss, " avg. val loss: ", avg_val_loss)


print("Train-loop done.")


