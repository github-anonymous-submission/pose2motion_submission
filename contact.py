import torch


def foot_contact_by_height(pos):
    eps = 0.25
    return (-eps < pos[..., 1]) * (pos[..., 1] < eps)


def velocity(pos, padding=False):
    if len(pos.shape)==4:
        velo = pos[:,1:,...] - pos[:,:-1,...]
    else:
        velo = pos[1:, ...] - pos[:-1, ...]
    velo_norm = torch.norm(velo, dim=-1)
    if padding:
        if len(pos.shape)==4:
            pad = torch.zeros_like(velo_norm[:, :1, :])
            velo_norm = torch.cat([pad, velo_norm], dim=1)
        else:
            pad = torch.zeros_like(velo_norm[:1, :])
            velo_norm = torch.cat([pad, velo_norm], dim=0)
    return velo_norm


def foot_contact(pos, ref_height=1., threshold=0.018):
    velo_norm = velocity(pos)
    contact = velo_norm < threshold
    contact = contact.int()
    padding = torch.zeros_like(contact)
    if len(contact.shape) == 3:
        contact = torch.cat([padding[:, :1, :], contact], dim=1)
    else:
        contact = torch.cat([padding[:1, :], contact], dim=0)
    return contact


def alpha(t):
    return 2.0 * t * t * t - 3.0 * t * t + 1


def lerp(a, l, r):
    return (1 - a) * l + a * r


