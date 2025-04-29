import torch
from tqdm import tqdm
import torch.nn.functional as F
from src.utils import linear_beta_schedule, cosine_beta_schedule, sigmoid_beta_schedule, sqrt_beta_schedule,possion_linear_beta_schedule


####################################################################################################
# PossionDiffusion
####################################################################################################
class PossionDiffusion:
    def __init__(
        self,
        timesteps=1000,
        beta_schedule='linear'
    ):
        self.timesteps = timesteps
        if beta_schedule == 'linear':
            # sch 是原DDPM betas
            sch = possion_linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            sch = cosine_beta_schedule(timesteps)
        elif beta_schedule == 'sigmoid':
            sch = sigmoid_beta_schedule(timesteps)
        elif beta_schedule == 'sqrt':
            sch = sqrt_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')


        # modify
        # self.c = 1
        self.c = 0.2119
        # self.c = 0.4845
        self.r = 100
        self.t = torch.linspace(1, timesteps, timesteps, dtype=torch.float32)#t是步数，1到1000
        self.t_prev = F.pad(self.t[:-1], (1, 0), value=0)
        self.r_bar = self.r * self.t
        self.e_n = self.r * self.t
        self.d_n = self.r * self.t
        self.std_n = torch.sqrt(self.d_n)
        # seq为原DDPM的alpha
        self.seq = 1. - sch
        # seq_cumprod为原DDPM的alpha_bar
        self.seq_cumprod = torch.cumprod(self.seq, axis=0)
        self.alpha = 1. - self.seq_cumprod
        self.sqrt_alpha = torch.sqrt(self.alpha)
        self.alpha_zero = 0.
        self.alpha_prev = F.pad(self.alpha[:-1], (1, 0), value=self.alpha_zero)
        self.beta = self.alpha / self.t
        self.beta_zero = 1.
        self.beta_prev = F.pad(self.beta[:-1], (1, 0), value=self.beta_zero)
        self.gamma = self.beta / self.beta_prev
        self.sqrt_gamma = torch.sqrt(self.gamma)

        # x0->xt前向加噪系数
        # x0 系数
        self.forward_noise_coef1 = torch.sqrt(self.beta/self.beta_zero)
        # noise系数
        self.forward_noise_coef2 = self.c *  self.sqrt_alpha

        # xt-->x0
        # xt 系数
        self.predict_start_from_noise_coef1 = torch.sqrt(self.beta_zero/self.beta)
        # noise 系数
        self.predict_start_from_noise_coef2 = self.c * torch.sqrt(self.beta_zero * self.t)

        # 计算后验分布q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
                ((self.c ** 2)*self.alpha_prev)/self.t
        )#后验方差

        # below: log calculation clipped because the posterior variance is 0 at the beginning
        # of the diffusion chain
        # x{t-1} 的方差对数以及最小限制
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min =1e-20))

        # x{t-1} 的均值系数
        # x0的系数
        self.posterior_mean_coef1 = (
            self.posterior_variance / (torch.sqrt(self.alpha_prev * self.t_prev)* (self.c**2))

        )

        # xt的系数
        self.posterior_mean_coef2 = (
            self.posterior_variance / (torch.sqrt(self.beta_prev * self.beta)* (self.c**2))
        )

    # get the param of given timestep t
    @classmethod
    def _extract(cls, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t)
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out

    # forward diffusion (using the nice property): q(x_t | x_0)
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            r_bar_t = self._extract(self.r_bar, t, x_start.shape)
            e_n_t = self._extract(self.e_n, t, x_start.shape)
            std_n_t = self._extract(self.std_n, t, x_start.shape)
            noise = torch.distributions.Poisson(r_bar_t.squeeze()).sample(x_start.shape[1:]).permute([-1, 0, 1, 2])
            noise = (noise - e_n_t) / std_n_t

        # 前向加噪公式
        coef1 = self._extract(self.forward_noise_coef1, t, x_start.shape)
        coef2 = self._extract(self.forward_noise_coef2, t, x_start.shape)
        return coef1 * x_start + coef2 * noise

    # Compute the mean and variance of the diffusion posterior: q(x_{t-1} | x_t, x_0)
    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)

        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    # compute x_0 from x_t and pred noise: the reverse of `q_sample`
    # 反向去噪公式
    def predict_start_from_noise(self, x_t, t, noise):
        coef1 = self._extract(self.predict_start_from_noise_coef1, t, x_t.shape)
        coef2 = self._extract(self.predict_start_from_noise_coef2, t, x_t.shape)
        return (
            ( coef1 * x_t - coef2 * noise)
        )

    # compute predicted mean and variance of p(x_{t-1} | x_t)
    def p_mean_variance(self, model, x_t, t, clip_denoised=True):
        # predict noise using model
        x_t = x_t.float() # 保证输入的数据是float32
        # importantt important important!!!!!

        pred_noise = model(x_t, t)

        # get the predicted x_0: different from the algorithm2 in the paper
        #### x0就是x_recon,开始计算
        x_recon = self.predict_start_from_noise(x_t, t, pred_noise)

        if clip_denoised:
            x_recon = torch.clamp(x_recon, min=-1., max=1.)

        ### x0,xt
        model_mean, posterior_variance, posterior_log_variance = \
                    self.q_posterior_mean_variance(x_recon, x_t, t)


        return model_mean, posterior_variance, posterior_log_variance

    # denoise_step: sample x_{t-1} from x_t and pred_noise
    @torch.no_grad()
    def p_sample(self, model, x_t, t, clip_denoised=True):
        model_mean, _, model_log_variance = self.p_mean_variance(model, x_t, t,
                                                    clip_denoised=clip_denoised)

        # noise_coef1 = self._extract(self.noise_coef1, t, x_t.shape)
        # noise = noise_coef1 * torch.randn_like(x_start)
        # noise = noise_coef1 * torch.randn_like(x_t)
        noise = torch.randn_like(x_t)#这里是采样用作的标准高斯噪声
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))

        # pred_img = model_mean + torch.sqrt(model_variance).float() * noise
        pred_img = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

        return pred_img

    # denoise: reverse diffusion
    @torch.no_grad()
    def p_sample_loop(self, model, shape):
        batch_size = shape[0]
        device = next(model.parameters()).device
        img = torch.randn(shape, device=device)
        img = img.to(device)

        imgs = []
        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            img = self.p_sample(model, img, torch.full((batch_size,), i, device=device, dtype=torch.long))
            imgs.append(img.cpu().numpy())
        return imgs

    # sample new images
    @torch.no_grad()
    def sample(self, model, image_size, batch_size=8, channels=3):
        return self.p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))

    # compute train losses
    def train_losses(self, model, x_start, t):
        # generate random noise
        r_bar_t = self._extract(self.r_bar, t, x_start.shape)
        e_n_t = self._extract(self.e_n, t, x_start.shape)
        std_n_t = self._extract(self.std_n, t, x_start.shape)
        noise = torch.distributions.Poisson(r_bar_t.squeeze()).sample(x_start.shape[1:]).permute([-1, 0, 1, 2])
        noise = (noise - e_n_t) / std_n_t

        noise = noise.reshape([-1, x_start.shape[0]]).T
        noise = noise.reshape(x_start.shape)

        # get x_t
        x_noisy = self.q_sample(x_start, t, noise=noise)
        predicted_noise = model(x_noisy, t)
        loss = F.mse_loss(noise, predicted_noise)
        return loss