import torch
import torch.nn as nn
import torch.nn.functional as F


class WaveletConvBlock(nn.Module):
    """
    Multi-scale Wavelet + Convolution block.
    Approximates a wavelet filter bank with parallel Conv1d at different
    kernel sizes, then fuses them.
    
    Input:  [B, seq_len, fea_dim]
    Output: [B, seq_len, d_model]
    """
    def __init__(self, fea_dim, d_model,
                 kernel_size=(3, 5, 7)):
        super().__init__()
        self.fea_dim = fea_dim
        self.d_model = d_model
        self.kernel_sizes = kernel_size

        # Split d_model across branches (roughly equally).
        n_branches = len(kernel_size)
        branch_out = d_model // n_branches
        last_branch_out = d_model - branch_out * (n_branches - 1)

        convs = []
        bns = []
        for i, k in enumerate(kernel_size):
            out_ch = branch_out if i < n_branches - 1 else last_branch_out
            padding = k // 2
            conv = nn.Conv1d(
                in_channels=fea_dim,
                out_channels=out_ch,
                kernel_size=k,
                padding=padding
            )
            bn = nn.BatchNorm1d(out_ch)
            convs.append(conv)
            bns.append(bn)

        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.activation = nn.ReLU(inplace=True)

        # Optional: project back to d_model (in case of rounding issues)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: [B, seq_len, fea_dim]
        B, T, _ = x.shape
        x_t = x.transpose(1, 2)   # [B, fea_dim, T]

        branch_outputs = []
        for conv, bn in zip(self.convs, self.bns):
            y = conv(x_t)                     # [B, out_ch, T]
            y = bn(y)
            y = self.activation(y)
            branch_outputs.append(y)

        # Concatenate along channel dimension
        y_cat = torch.cat(branch_outputs, dim=1)  # [B, d_model, T]

        # Back to [B, T, d_model]
        y_cat = y_cat.transpose(1, 2)             # [B, T, d_model]

        # Small linear projection (can act as a learnable mixing of branches)
        out = self.proj(y_cat)                    # [B, T, d_model]

        return out

class FourierBlock(nn.Module):
    """
    Fourier attention block.
    - Goes to frequency domain along the time axis.
    - Learns which frequencies (per feature channel) are important.
    - Gates the spectrum and returns a filtered time-domain sequence.
    
    Input:  [B, seq_len, d_model]
    Output: [B, seq_len, d_model]
    """
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

        # Small MLP that looks at the magnitude spectrum and
        # outputs a scalar score per frequency bin.
        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, 1)

    def forward(self, x):
        """
        x: [B, seq_len, d_model] (time domain)
        """
        B, T, D = x.shape

        # FFT along time dimension -> complex spectrum
        # shape: [B, T, D], T indices now represent frequencies
        x_fft = torch.fft.fft(x, dim=1)  # complex64/complex128

        # Use magnitude as input to the scoring MLP
        # (we don't want to feed complex numbers to nn.Linear).
        mag = torch.abs(x_fft)           # [B, T, D], real-valued

        # Per-frequency scoring network (position-wise MLP):
        # Linear -> GELU -> Linear -> scalar logit per frequency bin.
        h = F.gelu(self.linear1(mag))    # [B, T, D]
        scores = self.linear2(h)         # [B, T, 1]

        # Softmax over frequency (time index) dimension.
        # For each batch and channel, scores across T sum to 1.
        # For each feature channel, decides "which frequencies matter most"
        weights = F.softmax(scores, dim=1)   # [B, T, 1]

        # weights = F.softmax(scores / temperature, dim=1) 

        # Apply weights to the *complex* spectrum (broadcast on D).
        x_fft_weighted = x_fft * weights     # [B, T, D], complex

        # Back to time domain with inverse FFT, keep real part.
        x_filtered = torch.fft.ifft(x_fft_weighted, dim=1).real  # [B, T, D]

        return x + x_filtered # Original + frequency-enhanced


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.input_size = config['input_size']
        self.hidden_size = config['hidden_size']
        self.latent_dim = config['latent_dim']
        self.num_layers = config['num_layers']
        self.bidirectional = config['bidirectional']
        self.num_directions = 2 if self.bidirectional else 1
        self.p_lstm = config['dropout_lstm_encoder']
        self.p = config['dropout_layer_encoder']
        self.wavelet_kernel_s = config.get('wavelet_kernel_size', (3, 5, 7))

        # d_model: internal embedding size after wavelet/fourier/attention
        self.d_model = config.get('d_model', self.hidden_size)
        self.num_heads = config.get('num_heads', 4)

        # Check MultiHeadAttention input:        
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
            )

        # 1) Wavelet + Convolution block --> capture local patterns in time.
        self.wavelet_conv = WaveletConvBlock(
            fea_dim=self.input_size,
            d_model=self.d_model,
            kernel_size=self.wavelet_kernel_s
        )

        # 2) Fourier block --> capture global / periodic patterns across the entire sequence.
        self.fourier_block = FourierBlock(d_model=self.d_model)

        # 3) Multi-head attention projections
        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.input_size, self.d_model)
        self.v_proj = nn.Linear(self.input_size, self.d_model)

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.num_heads,
            batch_first=True
        )

        # 4) BiLSTM over attention output
        self.lstm = nn.LSTM(
            input_size=self.d_model,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.p_lstm if self.num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=self.bidirectional
        )

        # 5) Latent projections (mean and log_var)
        self.fc_mean = nn.Sequential(
            nn.Dropout(self.p),
            nn.Linear(
                in_features=self.num_directions * self.hidden_size,
                out_features=self.latent_dim
            )
        )

        self.fc_log_var = nn.Sequential(
            nn.Dropout(self.p),
            nn.Linear(
                in_features=self.num_directions * self.hidden_size,
                out_features=self.latent_dim
            )
        )

    @staticmethod
    def reparameterization(mean, var):
        epsilon = torch.randn_like(var)
        z = mean + var * epsilon
        return z

    def forward(self, x):
        """
        x: [B, seq_len, fea_dim]
        Returns:
            z:       [B, latent_dim]
            mean:    [B, latent_dim]
            log_var: [B, latent_dim]
        """

        # --- Wavelet + Conv branch ---
        x_wave = self.wavelet_conv(x)       # [B, seq_len, d_model]
        res = x_wave                        # residual

        # --- Fourier block ---
        x_fourier = self.fourier_block(x_wave)  # [B, seq_len, d_model]

        # --- Fusion ---
        x_fused = res + x_fourier           # [B, seq_len, d_model]

        # --- Multi-head attention ---
        Q = self.q_proj(x_fused)            # [B, seq_len, d_model]
        K = self.k_proj(x)                  # [B, seq_len, d_model]  (from original X)
        V = self.v_proj(x)                  # [B, seq_len, d_model]  (from original X)

        attn_out, _ = self.multihead_attn(Q, K, V)  # [B, seq_len, d_model]

        # --- BiLSTM ---
        batch_size = x.shape[0]
        _, (h_n, _) = self.lstm(attn_out)   # h_n: [num_layers*num_dir, B, hidden_size]

        h_n = h_n.view(self.num_layers, self.num_directions, batch_size, self.hidden_size)

        if self.bidirectional:
            # last layer, forward and backward
            h_forward  = h_n[-1, 0, :, :]   # [B, hidden_size]
            h_backward = h_n[-1, 1, :, :]   # [B, hidden_size]
            h = torch.cat((h_forward, h_backward), dim=1)  # [B, 2*hidden_size]
        else:
            h = h_n[-1, 0, :, :]
        # --- Latent projection ---
        mean = self.fc_mean(h)              # [B, latent_dim]
        log_var = self.fc_log_var(h)        # [B, latent_dim]

        # --- Reparameterization ---
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))

        return z, mean, log_var

# decoder uses an LSTM to generate sequences from the latent space representation

class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.input_size = config['input_size']
        self.hidden_size = config['hidden_size']
        self.latent_dim = config['latent_dim']
        self.num_layers = config['num_layers']
        self.bidirectional = config['bidirectional']
        self.window_size = config['window_size']
        self.p_lstm = config['dropout_lstm_decoder']
        self.p_dropout_layer = config['dropout_layer_decoder']
        self.num_directions = 2 if self.bidirectional else 1

        self.lstm_to_hidden = nn.LSTM(
            input_size=self.latent_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.p_lstm,
            batch_first=True,
            bidirectional=self.bidirectional
        )
        self.dropout_layer = nn.Dropout(self.p_dropout_layer)

        self.lstm_to_output = nn.LSTM(
            input_size=self.num_directions * self.hidden_size,
            hidden_size=self.input_size,
            batch_first=True
        )

    def forward(self, z):
        """
        :param z: [B, 2]
        :return: [B, seq_len, fea_dim]
        """
        latent_z = z.unsqueeze(1).repeat(1, self.window_size, 1)  # [B, seq_len, 2]
        out, _ = self.lstm_to_hidden(latent_z)
        out = self.dropout_layer(out)
        out, _ = self.lstm_to_output(out)
        return out


class DropBlockLatent(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.drop = nn.Dropout(p)

    def forward(self, z):
        return self.drop(z)


class TSHAE(nn.Module):
    def __init__(self, config, encoder, decoder):
        super(TSHAE, self).__init__()

        self.p = config['dropout_regressor']
        self.regression_dims = config['regression_dims']
        self.drop_block = DropBlockLatent(config.get('dropout_latent', 0.1))

        self.decode_mode = config['reconstruct']
        if self.decode_mode:
            assert isinstance(decoder, nn.Module), "You should to pass a valid decoder"
            self.decoder = decoder

        self.encoder = encoder

        self.regressor = nn.Sequential(
            nn.Linear(self.encoder.latent_dim, self.regression_dims),
            nn.Tanh(),
            nn.Dropout(self.p),
            nn.Linear(self.regression_dims, 1)
        )

    def forward(self, x):
        """
        :param x: [B, seq_len, fea_dim]
        :return:
            y_hat: [B, 1]
            z: [B, 2]
            mean: [B, 2]
            log_var: [B, 2]
            x_hat: [B, seq_len, fea_dim]
        """
        z, mean, log_var = self.encoder(x)
        z_dropped = self.drop_block(z)
        y_hat = self.regressor(z_dropped)
        if self.decode_mode:
            x_hat = self.decoder(z_dropped)
            return y_hat, z, mean, log_var, x_hat
        return y_hat, z, mean, log_var

# input data shape required by the encoder : [B, seq_len, fea_dim], 
# where B is the batch size, seq_len is the sequence length or the number of time steps in each sample
# fea_dim is the number of features per time step

# Sanity Check for dimensions and forward pass
if __name__ == "__main__":

    # Fake config
    config = {
        "input_size": 26,                # fea_dim
        "hidden_size": 32,
        "latent_dim": 8,
        "num_layers": 2,
        "bidirectional": True,
        "dropout_lstm_encoder": 0.1,
        "dropout_layer_encoder": 0.1,

        "window_size": 30,              # seq_len for reconstruction
        "dropout_lstm_decoder": 0.1,
        "dropout_layer_decoder": 0.1,

        "regression_dims": 16,
        "dropout_regressor": 0.1,
        "reconstruct": True,

        "d_model": 32,                  # must be divisible by num_heads
        "num_heads": 4,
        "dropout_latent": 0.1,
        "wavelet_kernel_size": (3, 5, 7),
    }

    # Instantiate encoder, decoder, and full model
    encoder = Encoder(config)
    decoder = Decoder(config)
    model = TSHAE(config, encoder, decoder)

    # Dummy input: batch of 5 sequences, length 30, 26 features
    B, seq_len, fea_dim = 5, 30, 26
    x = torch.randn(B, seq_len, fea_dim)

    # Forward pass
    y_hat, z, mean, log_var, x_hat = model(x)

    print(">>> Sanity check:")
    print("Input x shape:     ", x.shape)
    print("Latent z shape:    ", z.shape)
    print("Mean shape:        ", mean.shape)
    print("Log var shape:     ", log_var.shape)
    print("RUL pred y_hat:    ", y_hat.shape)
    print("Reconstruction shape x_hat:", x_hat.shape)