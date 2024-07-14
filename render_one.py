import argparse

import torch
torch.cuda.current_device()
import torch.optim as optim

from painter import *

def optimize_painting(pt:ProgressivePainter):

    pt._load_checkpoint()
    pt.net_G.eval()

    print('begin drawing...')

    PARAMS = np.zeros([1, 0, pt.rderr.d], np.float32)

    """ Start from empty canvas """
    if pt.rderr.canvas_color == 'white':
        CANVAS_tmp = torch.ones([1, 3, 128, 128]).to(device)
    else:
        CANVAS_tmp = torch.zeros([1, 3, 128, 128]).to(device)

    """ Iter. (m_grid * m_grid) over (1 * 1) to (max_divide * max_divide) """
    # TODO: each m_grid optimize to a different pred_X0 from large t to small t
    for pt.m_grid in range(1, pt.max_divide + 1):
        """ finish a canvas of (m_grid * m_grid) """
        # tie all blocks of painting to one batch
        # and treat them as an entire batch later on
        # TODO: turn to pred_X0 first
        pt.img_batch = utils.img2patches(pt.img_, pt.m_grid, pt.net_G.out_size).to(device)
        # start from the rendered canvas of previous (m_grid * m_grid)
        pt.G_final_pred_canvas = CANVAS_tmp

        pt.initialize_params()
        pt.x_ctt.requires_grad = True
        pt.x_color.requires_grad = True
        pt.x_alpha.requires_grad = True
        utils.set_requires_grad(pt.net_G, False)

        pt.optimizer_x = optim.RMSprop([pt.x_ctt, pt.x_color, pt.x_alpha], lr=pt.lr, centered=True)

        pt.step_id = 0
        """ Iter. anchor_id over m_strokes_per_block """
        for pt.anchor_id in range(0, pt.m_strokes_per_block):
            """ finish adding all the anchor_id-th strokes to the blocks-batch """
            # sample stroke params in the distribution of error map
            pt.stroke_sampler(pt.anchor_id)
            iters_per_stroke = int(500 / pt.m_strokes_per_block)
            """ Iter. i over iters_per_stroke to optimize all the anchor_id-th strokes """
            # TODO: iters may need less?
            for i in range(iters_per_stroke):
                # start from the rendered canvas of previous (m_grid * m_grid)
                pt.G_pred_canvas = CANVAS_tmp

                # update x
                pt.optimizer_x.zero_grad()

                pt.x_ctt.data = torch.clamp(pt.x_ctt.data, 0.1, 1 - 0.1)
                pt.x_color.data = torch.clamp(pt.x_color.data, 0, 1)
                pt.x_alpha.data = torch.clamp(pt.x_alpha.data, 0, 1)

                # Blend all strokes from anchor_0 to current anchor_id
                # on the canvas saved in the model
                pt._forward_pass()
                # Print log at every update
                pt._drawing_step_states()
                # Backprop. from (Pixel-wise Loss + Optimal Transportation Loss)
                pt._backward_x()

                pt.x_ctt.data = torch.clamp(pt.x_ctt.data, 0.1, 1 - 0.1)
                pt.x_color.data = torch.clamp(pt.x_color.data, 0, 1)
                pt.x_alpha.data = torch.clamp(pt.x_alpha.data, 0, 1)

                # Update stroke params (ctt, color, alpha)
                pt.optimizer_x.step()
                pt.step_id += 1

        # GET THE STROKE PARAMS SAVED IN THE MODEL OF CURRENT (m_grid * m_grid)
        v = pt._normalize_strokes(pt.x)
        # shuffle to avoid bias from strokes sequence
        v = pt._shuffle_strokes_and_reshape(v)
        PARAMS = np.concatenate([PARAMS, v], axis=1)

        # RENDER THE CANVAS OF CURRENT (m_grid * m_grid)
        CANVAS_tmp = pt._render(PARAMS, save_jpgs=False, save_video=False)
        CANVAS_tmp = utils.img2patches(CANVAS_tmp, pt.m_grid + 1, pt.net_G.out_size).to(device)

    pt._save_stroke_params(PARAMS)
    final_rendered_image = pt._render(PARAMS, save_jpgs=False, save_video=False)

    return final_rendered_image

def render_one(img_path):
    # config
    parser = argparse.ArgumentParser(description='STYLIZED NEURAL PAINTING')
    args = parser.parse_args(args=[])
    # args.img_path = './test_images/iceland.jpg' # path to input photo
    args.img_path = img_path # path to input photo
    args.renderer = 'oilpaintbrush' # [watercolor, markerpen, oilpaintbrush, rectangle]
    args.canvas_color = 'black' # [black, white]
    args.canvas_size = 512 # size of the canvas for stroke rendering'
    args.keep_aspect_ratio = True # whether to keep input aspect ratio when saving outputs
    args.max_m_strokes = 500 # max number of strokes
    args.max_divide = 5 # divide an image up-to max_divide x max_divide patches
    args.beta_L1 = 1.0 # weight for L1 loss
    args.with_ot_loss = False # set True for imporving the convergence by using optimal transportation loss, but will slow-down the speed
    args.beta_ot = 0.1 # weight for optimal transportation loss
    args.net_G = 'zou-fusion-net' # renderer architecture
    args.renderer_checkpoint_dir = './checkpoints_G_oilpaintbrush' # dir to load the pretrained neu-renderer
    args.lr = 0.005 # learning rate for stroke searching
    args.output_dir = './paintings' # dir to save painting results
    args.disable_preview = True # disable cv2.imshow, for running remotely without x-display

    pt = ProgressivePainter(args=args)
    final_rendered_image = optimize_painting(pt)

    plt.imshow(final_rendered_image), plt.title('generated')
    plt.show()

# render_one('./test_images/iceland.jpg')