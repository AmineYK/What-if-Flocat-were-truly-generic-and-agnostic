import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import evaluation as ev


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) *
            torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class DiTBlock(nn.Module):

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn  = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias,   0)


    def forward(self, x, t, return_attention=False):

        shift_msa, scale_msa, gate_msa, \
        shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(t).chunk(6, dim=-1)  

        x_mod = modulate(self.norm1(x), shift_msa, scale_msa) 
        attn_out, attn_weights = self.attn(x_mod, x_mod, x_mod,
                              need_weights=True,
                              average_attn_weights=True)

        x = x + gate_msa.unsqueeze(1) * attn_out
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )

        if return_attention:
            return x, attn_weights

        return x, None

class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.linear = nn.Linear(hidden_size, out_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias,   0)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias,   0)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)

def modulate(x, shift, scale):
    if x.dim() == 2:
        return x * (1 + scale) + shift
    else:
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class flocat(nn.Module):
    def __init__(
        self,
        latent_dim=768,       
        hidden_dim=256,
        depth=4,
        n_heads=4,
        mlp_ratio=4.0,
        freq_embed_size=256,
    ):
        super().__init__()

        self.input_proj = nn.Linear(latent_dim, hidden_dim)

        self.t_embedder = TimestepEmbedder(hidden_dim, freq_embed_size)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, n_heads, mlp_ratio)
            for _ in range(depth)
        ])

        self.attn_pool_query = nn.Parameter(
            torch.randn(1, 1, hidden_dim)
        )
        n_heads_ = 2
        self.attn_pool = nn.MultiheadAttention(
            hidden_dim, n_heads_, batch_first=True
        )

        self.final_layer = FinalLayer(hidden_dim, latent_dim)
        self._init_weights()

    def _init_weights(self):
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.apply(_basic_init)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias,   0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias,   0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias,   0)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

    def forward(self, x, t, attention_mask=None, return_attention=False):

        # sentence-level
        if x.dim() == 2:
            # (B, 1, 768)
            x = x.unsqueeze(1)              
            # if attention_mask is not None:
                # (B, 1)
            attention_mask = torch.ones(x.shape[0], 1, device=x.device)

        B, S , _ = x.shape
        h = self.input_proj(x)   

        c = self.t_embedder(t)              

        all_attentions = []
        for block in self.blocks:
            h, attn_w = block(h, c, return_attention=return_attention)
            if return_attention and attn_w is not None:
                all_attentions.append(attn_w)

        if h.shape[1] == 1:
            # S=1 : no pooling needed
            h_sentence = h.squeeze(1)         
        else:
            query = self.attn_pool_query.expand(B, 1, -1)

            # ignore PAD in attention pooling
            if attention_mask is not None:
                # S = 1
                if attention_mask.shape[1] == 1:
                    # no padding to ignore
                    key_padding_mask = None
                else:
                    key_padding_mask = (attention_mask == 0)   

            pool_out, pool_attn = self.attn_pool(
                query,                                      
                h,                                          
                h,                                          
                key_padding_mask=key_padding_mask,
                need_weights=True,
                average_attn_weights=True
            )

            h_sentence = pool_out.squeeze(1)
             
        v_sentence = self.final_layer(h_sentence, c)  
        v_tokens = self.final_layer(h, c)            

        if return_attention:
            all_attentions = torch.stack(all_attentions, dim=0)
            return v_sentence, v_tokens, all_attentions, pool_attn

        return v_sentence, v_tokens, None


class flocatTrainer(nn.Module):

    def __init__(self, model, config):
        super().__init__()
        
        self.model = model
        self.config = config
        self.source = config['source']
        self.target = config['target']
        self.device = config['device']

        if self.source.dim() == 2:
            self.attentions_mask = torch.ones(
                self.source.shape[0], 1,
                device=self.device
            )
        else:
            self.attentions_mask = self.config['attentions_mask']

        if self.source.dim() == 2:
            self.centroid = self.source.mean(dim=0)        
            self.var = self.source.var(dim=0).mean()
        else:
            source_tokens = self.source
            mask = self.attentions_mask.unsqueeze(-1)  
            lengths = self.attentions_mask.sum(dim=1, keepdim=True).clamp(min=1)  
            sum_tokens = (source_tokens * mask).sum(dim=1)       
            mean_per_sample = sum_tokens / lengths               
            diff_sq = (source_tokens - mean_per_sample.unsqueeze(1)) ** 2  
            var_per_sample = (diff_sq * mask).sum(dim=1) / lengths         

            self.centroid = mean_per_sample.mean(dim=0).to(self.device)   
            self.var = (var_per_sample.mean(dim=0) * self.config['coef_var']).to(self.device)
        self.log_r = nn.Parameter(torch.tensor(0.2).to(self.device))

    @property
    def r_in(self):
        return torch.exp(self.log_r)

    def get_lr_schedule(self,epoch, warmup_epochs, total_epochs, lr):

        # linear increase
        if epoch < warmup_epochs:
            return lr * (epoch + 1) / (warmup_epochs + 1)
        # cosinus decrease
        else:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return (lr * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))).item()

    def _euler_integrate_single(self, x_0, mask, N_steps=10, save_all=False):
        with torch.no_grad():
            if x_0.dim() == 2:
                x_0_sentence = x_0
            else:
                # masked mean pooling
                mask_exp     = mask.unsqueeze(-1).float()           
                x_0_sentence = (x_0 * mask_exp).sum(dim=1) / \
                                mask_exp.sum(dim=1).clamp(min=1e-8)

            x_t_sent = x_0_sentence.clone()
            dt = 1.0 / N_steps

            if save_all:
                velocities_tokens = []
                x_t_inter = [x_t_sent.detach().cpu()]

            for i in range(N_steps):
                t_val = i * dt
                t = torch.full(
                    (x_t_sent.shape[0],), t_val, device=self.device
                )

                if x_0.dim() == 2:
                    x_t_input = x_t_sent                            
                else:
                    residual   = x_0 - x_0_sentence.unsqueeze(1)   
                    x_t_input  = x_t_sent.unsqueeze(1) + residual

                v_sentence, v_tokens, _ = self.model(x_t_input, t, mask)
                x_t_sent = x_t_sent + dt * v_sentence

                if save_all:
                    velocities_tokens.append(v_tokens.detach().cpu())
                    x_t_inter.append(x_t_sent.detach().cpu())

            if save_all:
                velocities_tokens = torch.stack(velocities_tokens)
                x_t_inter         = torch.stack(x_t_inter)
            else:
                velocities_tokens, x_t_inter = None, None

        return x_t_sent, velocities_tokens, x_t_inter


    def euler_integrate(self, x_0, mask, N_steps=10, save_all=False, batch_size=128):
        all_x_final = []

        if save_all:
            all_velocities = []
            all_x_inter = []

        for start in range(0, x_0.shape[0], batch_size):
            end = start + batch_size

            x_batch = x_0[start:end].to(self.device)
            mask_batch = mask[start:end].to(self.device)

            x_final, velocities, x_inter = self._euler_integrate_single(
                x_batch, mask_batch, N_steps=N_steps, save_all=save_all
            )

            all_x_final.append(x_final.detach().cpu())

            if save_all:
                all_velocities.append(velocities)
                all_x_inter.append(x_inter)

            del x_batch, mask_batch, x_final, velocities, x_inter
            torch.cuda.empty_cache()

        all_x_final = torch.cat(all_x_final, dim=0)

        if save_all:
            all_velocities = torch.cat(all_velocities, dim=1)
            all_x_inter = torch.cat(all_x_inter, dim=1)
            return all_x_final, all_velocities, all_x_inter

        return all_x_final.to(self.device), None, None

    def sample_like(self, x_0):

        sigma = torch.sqrt(self.var * self.config['coef_var'])
        return self.centroid + sigma * torch.randn_like(x_0)
   
    def compute_flow_loss(self, x_0, mask_x_0=None):

        if mask_x_0 is not None:
            mask = mask_x_0.clone()
        else:
            mask = None

        if x_0.dim() == 2:
            x_0_sentence = x_0
        else:
            # <<<<<<<<<<<<<<<< first step : x_0_sentence -> mean pooling with no masked tokens >>>>>>>>>>>>>>>>
            # (B, T, 1)
            mask_exp = mask.unsqueeze(-1).float()  
            # (B, 768)            
            x_0_sentence = (x_0 * mask_exp).sum(dim=1) / \
                            mask_exp.sum(dim=1).clamp(min=1e-8)    

        # <<<<<<<<<<<<<<<< second step : x_1_sentence >>>>>>>>>>>>>>>>>>
        B = x_0.shape[0]
        sigma_val = torch.sqrt(self.var * self.config['coef_var'])
        # (B, 768)
        x_1_sentence = self.centroid + sigma_val * \
                        torch.randn(B, self.centroid.shape[0],
                                    device=self.device)          

        if x_0.dim() != 2:
            # ignore SEP token
            seq_lengths = mask.sum(dim=1).long()
            for b in range(B):
                mask[b, seq_lengths[b] - 1] = 0

        # <<<<<<<<<<<<<<<< third step : sentence interpolation >>>>>>>>>>>>>>>>>>
        t = torch.rand(B, device=self.device)

        t_exp      = t.view(-1, 1)
        x_t_sent   = t_exp * x_1_sentence + \
                        (1 - t_exp) * x_0_sentence              
        v_target   = x_1_sentence - x_0_sentence     

        # <<<<<<<<<<<<<<<< fourth step : token interpolation >>>>>>>>>>>>>>>>>>
        if x_0.dim() == 2:
            # (B, 768) -> value = 0
            residual   = x_0 - x_0_sentence
            x_t_input = x_t_sent + residual      
        else:
            # (B, S, 768)
            residual   = x_0 - x_0_sentence.unsqueeze(1) 
            x_t_input = x_t_sent.unsqueeze(1) + residual      

        # <<<<<<<<<<<<<<<< fifth step : forward >>>>>>>>>>>>>>>>>>
        v_sentence, _, _ = self.model(x_t_input, t, mask)

        # <<<<<<<<<<<<<<<< sixth step : mse loss - sentence level >>>>>>>>>>>>>>>>>>
        loss_fm = F.mse_loss(v_sentence, v_target)


        if self.config['lambda_love'] > 0:
            # <<<<<<<<<<<<<<<< seventh step : love with 1 step model inference >>>>>>>>>>>>>>>>>>
            if x_0.dim() == 2:
                # (B, 768) -> value = 0
                x_0_res   = x_0 - x_0_sentence
                x_t_zeros = x_0_sentence + x_0_res      
            else:
                # (B, S, 768)
                x_0_res   = x_0 - x_0_sentence.unsqueeze(1) 
                x_t_zeros = x_0_sentence.unsqueeze(1) + x_0_res  

            t_zeros    = torch.zeros(B, device=self.device)
            v_love, _, _ = self.model(x_t_zeros, t_zeros, mask)
            phi_1        = x_0_sentence + v_love

            dist_sq = torch.sum(
                (phi_1 - self.centroid)**2, dim=1
            )          

            r_sq      = self.r_in ** 2
            loss_love = r_sq + F.relu(dist_sq - r_sq).mean()                   
            loss_total = loss_fm + \
                            self.config['lambda_love'] * loss_love

            return loss_total, loss_fm.item(), loss_love.item()

        loss_total = loss_fm
        return loss_total, loss_fm.item(), 0


    def train_epoch(self, dataloader, optimizer):

        self.model.train()
        self.optimizer = optimizer

        total_loss_liste = []
        loss_fm_liste = []
        loss_love_liste = []

        for x_0, mask_x_0 in dataloader:

            x_0 = x_0.to(self.device)
            mask_x_0 = mask_x_0.to(self.device)
            loss_total, *anythingelse = self.compute_flow_loss(
                x_0,
                mask_x_0
            )

            self.optimizer.zero_grad()
            loss_total.backward()

            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + [self.log_r],0.5
            )

            self.optimizer.step()

            total_loss_liste.append(loss_total.item())
            loss_fm_liste.append(anythingelse[0])
            loss_love_liste.append(anythingelse[1])

        return np.mean(total_loss_liste), np.mean(loss_fm_liste), np.mean(loss_love_liste)


    def train(self, verbose=True, show_details=False):

        optimizer = AdamW(
            list(self.model.parameters()) + [self.log_r],
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay'],
            foreach=False
        )

        source_dataloader = DataLoader(TensorDataset(self.source, self.attentions_mask), batch_size=self.config['batch_size'], shuffle=True)

        total_loss_liste = []
        loss_fm_liste = []
        loss_love_liste = []
        interm_repre = []
        liste_r = []


        for epoch in range(self.config['epochs']):

            lr = self.get_lr_schedule(epoch, self.config['lr_epochs'], self.config['epochs'], self.config['lr'])
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            loss_total, *anythingelse = self.train_epoch(
                source_dataloader,
                optimizer,
            )

            total_loss_liste.append(loss_total.item())
            loss_fm_liste.append(anythingelse[0])
            loss_love_liste.append(anythingelse[1])

            if epoch % (self.config['epochs'] // 3) == 0 and verbose:
                print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
                print(f"Train Loss: {loss_total:.4f}, LR: {lr:.6f}")
                print(self.r_in)

            if show_details and epoch % 150 == 0: 
                print(f"<<<<<< Processing... - epoch : {epoch} >>>>>>>>")
                x_final, _, _ = self.euler_integrate(
                    self.source.to(self.device),
                    self.attentions_mask.to(self.device),
                    1, False
                )
                interm_repre.append(x_final)
                liste_r.append(self.r_in.item())
                print("finish...\n")

        if not show_details:
            return total_loss_liste, loss_fm_liste, loss_love_liste
        else:
            return total_loss_liste, loss_fm_liste, loss_love_liste, torch.stack(interm_repre), liste_r

    @torch.no_grad()
    def compute_anomaly_scores(self, X_test, attentions_test_mask,
                                type='norm-centroid', n_steps=20):

        x_final, _, _ = self.euler_integrate(
            X_test.to(self.device),
            attentions_test_mask.to(self.device),
            n_steps, False
        )
        x_1_test = x_final.cpu().numpy()

        if type == 'norm-centroid':
            centroid_np = self.centroid.cpu().numpy() 
            scores = np.sum(
                (x_1_test - centroid_np) ** 2, axis=1
            )

        return scores

    def test(self, X_test, y_test, attentions_test_mask=None, 
        type='norm-centroid', n_steps=20):

        if attentions_test_mask is None:
            if X_test.dim() == 2:
                # mask (B, 1)
                attentions_test_mask = torch.ones(X_test.shape[0], 1)
            else:
                # mask (B, T)
                attentions_test_mask = torch.ones(
                    X_test.shape[0], X_test.shape[1]
                )

        scores = self.compute_anomaly_scores(
            X_test, attentions_test_mask, type, n_steps
        )
        return ev.evaluation(y_test, scores, True)
