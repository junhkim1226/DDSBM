import numpy as np
import torch

from ddsbm import utils
from ddsbm.diffusion import diffusion_utils


class PredefinedNoiseScheduleDiscrete(torch.nn.Module):
    """
    Predefined noise schedule. Essentially creates a lookup array for predefined (non-learned) noise schedules.
    """

    def __init__(self, noise_schedule, timesteps, beta_start=None, beta_end=None):
        super(PredefinedNoiseScheduleDiscrete, self).__init__()
        self.timesteps = timesteps

        if noise_schedule == "cosine":
            betas = diffusion_utils.cosine_beta_schedule_discrete(timesteps)
        elif noise_schedule == "sigmoid":
            betas = diffusion_utils.sigmoid_beta_schedule_discrete(
                beta_start, beta_end, timesteps
            )
        else:
            raise NotImplementedError(noise_schedule)

        self.register_buffer("betas", torch.from_numpy(betas).float())

        self.alphas = 1 - torch.clamp(self.betas, min=0, max=0.9999)

        log_alpha = torch.log(self.alphas)
        log_alpha_bar = torch.cumsum(log_alpha, dim=0)
        self.alphas_bar = torch.exp(log_alpha_bar)
        # print(f"[Noise schedule: {noise_schedule}] alpha_bar:", self.alphas_bar)

    def forward(self, t_normalized=None, t_int=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.timesteps)
        return self.betas[t_int.long()]

    def get_alpha_bar(self, t_normalized=None, t_int=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.timesteps)
        return self.alphas_bar.to(t_int.device)[t_int.long()]


class SymmetricNoiseScheduleDiscrete(torch.nn.Module):
    """
    Predefined symmetric noise schedule. Essentially creates a lookup array for predefined (non-learned) noise schedules.
    """

    def __init__(self, noise_schedule, timesteps, min_alpha):
        super(SymmetricNoiseScheduleDiscrete, self).__init__()
        self.timesteps = timesteps

        if noise_schedule == "cosine":
            self.alphas = torch.from_numpy(
                diffusion_utils.cosine_alpha_schedule_discrete(
                    timesteps, min_alpha=min_alpha
                )
            ).float()
        else:
            raise NotImplementedError(noise_schedule)

        log_alphas = torch.log(self.alphas)
        log_alphas_bar = torch.cumsum(log_alphas, dim=0)
        self.alphas_bar = torch.exp(log_alphas_bar)

        # self.alphas = self.alpha_bars[1:] / self.alpha_bars[:-1]
        self.betas = 1 - self.alphas

    def forward(self, t_normalized=None, t_int=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.timesteps)
        return self.betas.to(t_int.device)[t_int.long()]

    def get_alpha_bar(self, t_normalized=None, t_int=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.timesteps)
        return self.alphas_bar.to(t_int.device)[t_int.long()]


class DiscreteUniformTransition:
    def __init__(self, x_classes: int, e_classes: int, y_classes: int):
        self.X_classes = x_classes
        self.E_classes = e_classes
        self.y_classes = y_classes
        self.u_x = torch.ones(1, self.X_classes, self.X_classes)
        if self.X_classes > 0:
            self.u_x = self.u_x / self.X_classes

        self.u_e = torch.ones(1, self.E_classes, self.E_classes)
        if self.E_classes > 0:
            self.u_e = self.u_e / self.E_classes

        self.u_y = torch.ones(1, self.y_classes, self.y_classes)
        if self.y_classes > 0:
            self.u_y = self.u_y / self.y_classes

    def get_Qt(self, beta_t, device):
        """Returns one-step transition matrices for X and E, from step t - 1 to step t.
        Qt = (1 - beta_t) * I + beta_t / K

        beta_t: (bs)                         noise level between 0 and 1
        returns: qx (bs, dx, dx), qe (bs, de, de), qy (bs, dy, dy).
        """
        beta_t = beta_t.unsqueeze(1)
        beta_t = beta_t.to(device)
        self.u_x = self.u_x.to(device)
        self.u_e = self.u_e.to(device)
        self.u_y = self.u_y.to(device)

        q_x = beta_t * self.u_x + (1 - beta_t) * torch.eye(
            self.X_classes, device=device
        ).unsqueeze(0)
        q_e = beta_t * self.u_e + (1 - beta_t) * torch.eye(
            self.E_classes, device=device
        ).unsqueeze(0)
        q_y = beta_t * self.u_y + (1 - beta_t) * torch.eye(
            self.y_classes, device=device
        ).unsqueeze(0)

        return utils.PlaceHolder(X=q_x, E=q_e, y=q_y)

    def get_Qt_bar(self, alpha_bar_t, device):
        """Returns transition matrices for X and E, from step 0 to step t.
        Qt = prod(1 - beta_t) * I + (1 - prod(1 - beta_t)) / K

        alpha_bar_t: (bs)         Product of the (1 - beta_t) for each time step from 0 to t.
        returns: qx (bs, dx, dx), qe (bs, de, de), qy (bs, dy, dy).
        """
        alpha_bar_t = alpha_bar_t.unsqueeze(1)
        alpha_bar_t = alpha_bar_t.to(device)

        self.u_x = self.u_x.to(device)
        self.u_e = self.u_e.to(device)
        self.u_y = self.u_y.to(device)

        q_x = (
            alpha_bar_t * torch.eye(self.X_classes, device=device).unsqueeze(0)
            + (1 - alpha_bar_t) * self.u_x
        )
        q_e = (
            alpha_bar_t * torch.eye(self.E_classes, device=device).unsqueeze(0)
            + (1 - alpha_bar_t) * self.u_e
        )
        q_y = (
            alpha_bar_t * torch.eye(self.y_classes, device=device).unsqueeze(0)
            + (1 - alpha_bar_t) * self.u_y
        )

        return utils.PlaceHolder(X=q_x, E=q_e, y=q_y)

    def get_Qst_bar(self, alpha_bar_s, alpha_bar_t, device):
        """Returns transition matrices for X and E, from step s to step t.
        Qt = prod(1 - beta_t) * I + (1 - prod(1 - beta_t)) / K

        alpha_bar_t: (bs)         Product of the (1 - beta_t) for each time step from 0 to t.
        returns: qx (bs, dx, dx), qe (bs, de, de), qy (bs, dy, dy).
        """
        alpha_bar_s = alpha_bar_s.unsqueeze(1)
        alpha_bar_s = alpha_bar_s.to(device)

        alpha_bar_t = alpha_bar_t.unsqueeze(1)
        alpha_bar_t = alpha_bar_t.to(device)

        alpha_bar_st = alpha_bar_t / alpha_bar_s

        self.u_x = self.u_x.to(device)
        self.u_e = self.u_e.to(device)
        self.u_y = self.u_y.to(device)

        q_x = (
            alpha_bar_st * torch.eye(self.X_classes, device=device).unsqueeze(0)
            + (1 - alpha_bar_st) * self.u_x
        )
        q_e = (
            alpha_bar_st * torch.eye(self.E_classes, device=device).unsqueeze(0)
            + (1 - alpha_bar_st) * self.u_e
        )
        q_y = (
            alpha_bar_st * torch.eye(self.y_classes, device=device).unsqueeze(0)
            + (1 - alpha_bar_st) * self.u_y
        )

        return utils.PlaceHolder(X=q_x, E=q_e, y=q_y)


class MarginalUniformTransition:
    def __init__(self, x_marginals, e_marginals, y_classes):
        self.X_classes = len(x_marginals)
        self.E_classes = len(e_marginals)
        self.y_classes = y_classes
        self.x_marginals = x_marginals
        self.e_marginals = e_marginals

        self.u_x = x_marginals.unsqueeze(0).expand(self.X_classes, -1).unsqueeze(0)
        self.u_e = e_marginals.unsqueeze(0).expand(self.E_classes, -1).unsqueeze(0)
        self.u_y = torch.ones(1, self.y_classes, self.y_classes)
        if self.y_classes > 0:
            self.u_y = self.u_y / self.y_classes

    def get_Qt(self, beta_t, device):
        """Returns one-step transition matrices for X and E, from step t - 1 to step t.
        Qt = (1 - beta_t) * I + beta_t / K

        beta_t: (bs)                         noise level between 0 and 1
        returns: qx (bs, dx, dx), qe (bs, de, de), qy (bs, dy, dy)."""
        beta_t = beta_t.unsqueeze(1)
        beta_t = beta_t.to(device)
        self.u_x = self.u_x.to(device)
        self.u_e = self.u_e.to(device)
        self.u_y = self.u_y.to(device)

        q_x = beta_t * self.u_x + (1 - beta_t) * torch.eye(
            self.X_classes, device=device
        ).unsqueeze(0)
        q_e = beta_t * self.u_e + (1 - beta_t) * torch.eye(
            self.E_classes, device=device
        ).unsqueeze(0)
        q_y = beta_t * self.u_y + (1 - beta_t) * torch.eye(
            self.y_classes, device=device
        ).unsqueeze(0)

        return utils.PlaceHolder(X=q_x, E=q_e, y=q_y)

    def get_Qt_bar(self, alpha_bar_t, device):
        """Returns t-step transition matrices for X and E, from step 0 to step t.
        Qt = prod(1 - beta_t) * I + (1 - prod(1 - beta_t)) * K

        alpha_bar_t: (bs)         Product of the (1 - beta_t) for each time step from 0 to t.
        returns: qx (bs, dx, dx), qe (bs, de, de), qy (bs, dy, dy).
        """
        alpha_bar_t = alpha_bar_t.unsqueeze(1)
        alpha_bar_t = alpha_bar_t.to(device)
        self.u_x = self.u_x.to(device)
        self.u_e = self.u_e.to(device)
        self.u_y = self.u_y.to(device)

        q_x = (
            alpha_bar_t * torch.eye(self.X_classes, device=device).unsqueeze(0)
            + (1 - alpha_bar_t) * self.u_x
        )
        q_e = (
            alpha_bar_t * torch.eye(self.E_classes, device=device).unsqueeze(0)
            + (1 - alpha_bar_t) * self.u_e
        )
        q_y = (
            alpha_bar_t * torch.eye(self.y_classes, device=device).unsqueeze(0)
            + (1 - alpha_bar_t) * self.u_y
        )

        return utils.PlaceHolder(X=q_x, E=q_e, y=q_y)

    def get_Qst_bar(self, alpha_bar_s, alpha_bar_t, device):
        """Returns transition matrices for X and E, from step s to step t.
        Qt = prod(1 - beta_t) * I + (1 - prod(1 - beta_t)) / K

        alpha_bar_t: (bs)         Product of the (1 - beta_t) for each time step from 0 to t.
        returns: qx (bs, dx, dx), qe (bs, de, de), qy (bs, dy, dy).
        """
        alpha_bar_s = alpha_bar_s.unsqueeze(1)
        alpha_bar_s = alpha_bar_s.to(device)

        alpha_bar_t = alpha_bar_t.unsqueeze(1)
        alpha_bar_t = alpha_bar_t.to(device)

        alpha_bar_st = alpha_bar_t / alpha_bar_s

        self.u_x = self.u_x.to(device)
        self.u_e = self.u_e.to(device)
        self.u_y = self.u_y.to(device)

        q_x = (
            alpha_bar_st * torch.eye(self.X_classes, device=device).unsqueeze(0)
            + (1 - alpha_bar_st) * self.u_x
        )
        q_e = (
            alpha_bar_st * torch.eye(self.E_classes, device=device).unsqueeze(0)
            + (1 - alpha_bar_st) * self.u_e
        )
        q_y = (
            alpha_bar_st * torch.eye(self.y_classes, device=device).unsqueeze(0)
            + (1 - alpha_bar_st) * self.u_y
        )

        return utils.PlaceHolder(X=q_x, E=q_e, y=q_y)


class DiscreteBridgeTransition:
    def __init__(self, x_classes: int, e_classes: int, y_classes: int):
        self.X_classes = x_classes
        self.E_classes = e_classes
        self.y_classes = y_classes
        self.u_x = torch.ones(1, self.X_classes, self.X_classes)
        if self.X_classes > 0:
            self.u_x = self.u_x / self.X_classes

        self.u_e = torch.ones(1, self.E_classes, self.E_classes)
        if self.E_classes > 0:
            self.u_e = self.u_e / self.E_classes

        self.u_y = torch.ones(1, self.y_classes, self.y_classes)
        if self.y_classes > 0:
            self.u_y = self.u_y / self.y_classes

        self.uu_x = torch.ones(1, 1, self.X_classes, 1)
        self.uu_e = torch.ones(1, 1, 1, self.E_classes, 1)
        self.uu_y = torch.ones(1, 1, self.y_classes, 1)

    def get_Qt(self, beta_t, X, E, device):
        """Returns one-step transition matrices for X and E, from step t - 1 to step t.
        Qt = (1 - beta_t) * I + beta_t / K
        beta_t: (bs)                         noise level between 0 and 1
        returns: qx (bs, dx, dx), qe (bs, de, de), qy (bs, dy, dy).
        """
        X_classes = X.shape[-1]
        E_classes = E.shape[-1]
        beta_t = beta_t.unsqueeze(1)
        beta_t = beta_t.to(device)

        self.u_x = self.u_x.to(device)  # 1, dx, dx
        self.u_e = self.u_e.to(device)  # 1, de, de
        self.uu_x = self.uu_x.to(device)  # 1, 1, dx, 1
        self.uu_e = self.uu_e.to(device)  # 1, 1, 1, de, 1

        q_x = (1 - beta_t) * torch.eye(X_classes, device=device).unsqueeze(0)
        q_e = (1 - beta_t) * torch.eye(E_classes, device=device).unsqueeze(0)

        # q_x : bs, dx, dx
        # q_e : bs, de, de

        # X : bs, nx, dx
        # E : bs, nx, nx, de
        q_x = q_x.reshape(-1, 1, self.X_classes, self.X_classes)
        q_e = q_e.reshape(-1, 1, 1, self.E_classes, self.E_classes)

        # q_x : bs, nx, dx, dx
        # q_e : bs, nx, nx, de, de
        nx = X.shape[1]
        # repeat q_x and q_e for each node
        q_x = q_x.repeat(1, nx, 1, 1)
        q_e = q_e.repeat(1, nx, nx, 1, 1)

        c = beta_t.reshape(-1, 1, 1, 1)
        q_x += c * X.unsqueeze(2) * self.uu_x
        q_e += c.unsqueeze(-1) * E.unsqueeze(3) * self.uu_e
        return utils.PlaceHolder(X=q_x, E=q_e, y=None)

    def get_Qt_bar(self, alpha_bar_t, X, E, device=torch.device("cuda")):
        """Returns t-step transition matrices for X and E, from step 0 to step t.
        Qt = prod(1 - beta_t) * I + (1 - prod(1 - beta_t)) / K
        alpha_bar_t: (bs)         Product of the (1 - beta_t) for each time step from 0 to t.
        returns: qx (bs, nx, dx, dx), qe (bs, nx, nx, de, de), qy (bs, ny, dy, dy).
        """
        X_classes = X.shape[-1]
        E_classes = E.shape[-1]
        alpha_bar_t = alpha_bar_t.unsqueeze(1)

        self.u_x = self.u_x.to(device)  # 1, dx, dx
        self.u_e = self.u_e.to(device)  # 1, de, de
        self.uu_x = self.uu_x.to(device)  # 1, 1, dx, 1
        self.uu_e = self.uu_e.to(device)  # 1, 1, 1, de, 1

        q_x = alpha_bar_t * torch.eye(X_classes, device=device).unsqueeze(0)
        q_e = alpha_bar_t * torch.eye(E_classes, device=device).unsqueeze(0)

        # q_x : bs, dx, dx
        # q_e : bs, de, de

        # X : bs, nx, dx
        # E : bs, nx, nx, de
        q_x = q_x.reshape(-1, 1, self.X_classes, self.X_classes)
        q_e = q_e.reshape(-1, 1, 1, self.E_classes, self.E_classes)

        # q_x : bs, nx, dx, dx
        # q_e : bs, nx, nx, de, de
        nx = X.shape[1]
        # repeat q_x and q_e for each node
        q_x = q_x.repeat(1, nx, 1, 1)
        q_e = q_e.repeat(1, nx, nx, 1, 1)

        c = (1 - alpha_bar_t).reshape(-1, 1, 1, 1)
        q_x += c * X.unsqueeze(2) * self.uu_x
        q_e += c.unsqueeze(-1) * E.unsqueeze(3) * self.uu_e
        return utils.PlaceHolder(X=q_x, E=q_e, y=None)
