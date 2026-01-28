# 阅读 *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion* 时产生的疑问

## 一、why causal attention?

为什么 transformer-based 模型中 action tokens 要加 causal mask？直觉上来看 t+1 时刻的动作显然依赖于 t 时刻的动作，但其实 t 时刻的动作也会与未来规划有关。双向注意力会不会更好？这是不是因为模型中没有加入位置编码，因而需要 causal attention 来隐式地提供位置（时间先后）信息？

原文使用 cross attention 来实现 action noise 与 observation 的交互，但现代图片生成模型更多地将文字与图片拼接起来过一个自注意力层。cross attention 与 self attention 孰优孰劣也可以实验一下。类似地，也可以对比一下 ddpm 与 flow matching。

原文用第一个 token 来表示（去噪的）时间步 k，这一点似乎也不是常规设计。

## 二、MPC & RHC

"warm-starting the next inference setup with previous action sequence prediction"，这是怎么实现的？原文中提到这样可以进一步提升"action smoothness"，但它是不是也可以用来加速推理？下一时刻的动作应该已经在前一时刻附近，可能本来需要50次推理，现在可以只用10步推理？但这样的话 timestep 应该如何 schedule？

## 三、synergy with position control

为什么 position control 表现得更好？看了原文的解释之后还是不理解。另外，position control 会不会输出执行器无法完成的位置？