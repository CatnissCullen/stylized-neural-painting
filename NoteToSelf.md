# Note to Self

****



## 笔触渲染器

*必要性：传统渲染方式（如生成训练集时使用的 cv2 函数）通过几何制图的方式将给定的笔触参数绘制成图片，而没有建立从参数到图片的连贯数学映射关系，导致即便知道生成图片间的损失值，也无法据此定向地优化笔触参数；使用神经网络对从参数到图片的映射关系建模，便可以**打通它们之间反向传播梯度的渠道，进而通过图片端计算出的损失值定向优化参数端，再继续生成更好的图片***

### 代码结构

-   **网络架构：`network.py`**
-   **传统渲染器：`renderer.py` （使用 cv2 的函数根据随机笔触参数生成 ground truth 笔触图片）**
-   **架构、传统渲染器、训练数据、超参数、损失函数等一起封装成的类：`imitator.py`**
-   **训练代码（仅调用 Imitator 类中的函数）：`train_imitator.py`** 

### 训练

**使用随机生成的笔触训练集，根据随机的笔触参数对笔触样例图片进行图形变换生成**

**每个数据点包含一份笔触信息：**

-   **`A` -- 笔触位置和形状（ctt）、颜色（color）、透明度（alpha）的所有参数**
-   **`B` -- 笔触的前景特征图（指有色前景）**
-   **`ALPHA` -- 笔触的透明度特征图**

**输入：`A` **

**用神经网络进行预测，默认：**

```python
class ZouFCNFusion(nn.Module):
    def __init__(self, rdrr):
        super(ZouFCNFusion, self).__init__()
        self.rdrr = rdrr
        self.out_size = 128
        self.huangnet = PixelShuffleNet(rdrr.d_shape)
        self.dcgan = DCGAN(rdrr)

    def forward(self, x):
        x_shape = x[:, 0:self.rdrr.d_shape, :, :]
        x_alpha = x[:, [-1], :, :]
        if self.rdrr.renderer in ['oilpaintbrush', 'airbrush']:
            x_alpha = torch.tensor(1.0).to(device)
		
        # 用形状参数预测出形状掩膜图（PixelShuffleNet通过全连接、Conv2d和上采样操作nn.PixelShuffle，根据一维的形状参数向量预测二维的形状掩膜图）
        mask = self.huangnet(x_shape)
        # 用所有参数预测出前景图（原始DCGAN输入的是随机噪声向量，所以必须有判别器引导学习；这里DCGAN通过转置卷积，根据高维的全部笔触参数向量预测二维的前景图，有输入参数引导，无需判别器）
        color, _ = self.dcgan(x)

        return color * mask, x_alpha * mask
```

**输出：预测的前景特征图 & 预测的透明度特征图** 

**损失值：（相对于用传统渲染器生成的 ground truth 笔触图片）**

```python
class PixelLoss(nn.Module):

    def __init__(self, p=1):
        super(PixelLoss, self).__init__()
        self.p = p

    def forward(self, canvas, gt, ignore_color=False):
        if ignore_color:
            canvas = torch.mean(canvas, dim=1)
            gt = torch.mean(gt, dim=1)
        loss = torch.mean(torch.abs(canvas-gt)**self.p)
        return loss
```

```python
pixel_loss1 = self._pxl_loss(self.G_pred_foreground, self.gt_foreground)
pixel_loss2 = self._pxl_loss(self.G_pred_alpha, self.gt_alpha)
self.G_loss = 100 * (pixel_loss1 + pixel_loss2) / 2.0  # 前景损失&透明度损失的直接均值
```



## 渐进式绘图器

### 代码结构

-   **训练好的笔触渲染器、笔触参数、前景图片节点、透明度图片节点、最终画面节点、训练超参数、损失函数等封装成的绘图器类：`painter.py` （笔触渲染器对象被冻结）** 
-   **对一幅图进行推理和渲染：`render_one.ipynb`** 

### 训练

*不进行任何预训练，直接在推理绘画的过程中逐次优化绘图*

**初始化空白画布：**

```python
""" Start from empty canvas """
    if pt.rderr.canvas_color == 'white':
        CANVAS_tmp = torch.ones([1, 3, 128, 128]).to(device)
    else:
        CANVAS_tmp = torch.zeros([1, 3, 128, 128]).to(device)
```

**模型内完全随机初始化笔触参数：** 

```python
        self.x_ctt = np.random.rand(
            self.m_grid*self.m_grid, self.m_strokes_per_block,
            self.rderr.d_shape).astype(np.float32)
        self.x_ctt = torch.tensor(self.x_ctt).to(device)

        self.x_color = np.random.rand(
            self.m_grid*self.m_grid, self.m_strokes_per_block,
            self.rderr.d_color).astype(np.float32)
        self.x_color = torch.tensor(self.x_color).to(device)

        self.x_alpha = np.random.rand(
            self.m_grid*self.m_grid, self.m_strokes_per_block,
            self.rderr.d_alpha).astype(np.float32)
        self.x_alpha = torch.tensor(self.x_alpha).to(device)
```

***以后：***

将画布分割成 m\*m （ m 在各次优化绘图中从1遍历到某最大值，这就是 “渐进” 的含义，即绘图的 “一次” 优化其实是画布的一次更细化分割），并将 m\*m 小块叠成一个 batch ，以后当作整体处理

-   从上一种分割渲染得到的画面（当前最终画面 `CANVAS_tmp` ）开始，对于当前分割，从 0 到 n-1 的每一号笔触（一号笔触指在 batch 中所有分块上的所有同号笔触）：

    -   笔触优化前
        -   计算当前最终画面和真实照片之间的误差图
        -   <u>在误差图的像素分布上随机采样笔触位置，以及真实照片对应位置上的颜色</u>
        -   完全随机生成透明度
        
    -   在当前最终画面上（将当前最终画面赋值给模型内当前画面节点），用损失值反向传播的误差<u>迭代优化</u>该号笔触 `iters_per_stroke` 次（新的笔触只在模型内部渲染当前画面计算损失值，<u>不更新当前最终画面 `CANVAS_tmp`</u>） 

        **损失值：（相对于要绘制的原始照片）**

    ```python
    def _backward_x(self):
            """ Stroke Painting Loss = Pixel-wise Loss + Optimal Transportation Loss """
            self.G_loss = 0
            self.G_loss += self.args.beta_L1 * self._pxl_loss(
                canvas=self.G_final_pred_canvas, gt=self.img_batch)
            if self.args.with_ot_loss:
                self.G_loss += self.args.beta_ot * self._sinkhorn_loss(
                    self.G_final_pred_canvas, self.img_batch)
            self.G_loss.backward()
    ```

-   收集从 0 到 n-1 的每一号笔触到模型外部 `PARAMS` 列表，并进行随机打乱（每一号笔触是并列关系，都是在相同的当前最终画面和真实照片间进行优化，不应该有次序关系，因此打乱顺序消除绘画次序带来的偏差）&标准化

    ```python
    # GET THE STROKE PARAMS SAVED IN THE MODEL OF CURRENT (m_grid * m_grid)
            v = pt._normalize_strokes(pt.x)
            # shuffle to avoid bias from strokes sequence
            v = pt._shuffle_strokes_and_reshape(v)
            PARAMS = np.concatenate([PARAMS, v], axis=1)
    ```

-   一次性把所有编号的笔触渲染到当前最终画面形成新的当前最终画面

    ```python
    # RENDER THE CANVAS OF CURRENT (m_grid * m_grid)
            CANVAS_tmp = pt._render(PARAMS, save_jpgs=False, save_video=False)
            CANVAS_tmp = utils.img2patches(CANVAS_tmp, pt.m_grid + 1, pt.net_G.out_size).to(device)
    ```

**渲染最终画面：**

```python
 pt._save_stroke_params(PARAMS)
    final_rendered_image = pt._render(PARAMS, save_jpgs=False, save_video=True)
```

**【总共的迭代次数是：m × n × `iters_per_stroke` 】** 

