import torch
from torch import nn
import torchaudio as audio

class StutterNet(nn.Module):
  def __init__(self, n_mels=40, 
               dropout=0.0, use_batchnorm=False, scale=1):
    '''Implementation of StutterNet
    from Sheikh et al. StutterNet: 
    "Stuttering Detection Using 
    Time Delay Neural Network" 2021

    Args:
      n_mels (int, optional): number of mel filter banks
      n_classes (int, optional): number of classes in output layer
      use_dropout (bool, optional): whether or not to use dropout in the
        last two linear layers
      use_batchnorm (bool, optional): whether ot not to batchnorm in the
        TDNN layers
      scale (float ,optional): width scale factor
    '''
    super(StutterNet, self).__init__()

    self.n_mels = n_mels
    
    self.spec = audio.transforms.MelSpectrogram(n_mels=n_mels, sample_rate=16000,
                                               n_fft=512, pad=1, f_max=8000, win_length=400,
                                                f_min=0, power=2.0, hop_length=160, norm='slaney')
    self.db = audio.transforms.AmplitudeToDB()
    # self.mfcc = audio.transforms.MFCC(16000, 40)
    self.tdnn_1 = nn.Conv1d(n_mels, int(512*scale), 5, dilation=1)
    self.tdnn_2 = nn.Conv1d(int(512*scale), int(1536*scale), 5, dilation=2)
    self.tdnn_3 = nn.Conv1d(int(1536*scale), int(512*scale), 7, dilation=3)
    self.tdnn_4 = nn.Conv1d(int(512*scale), int(512*scale), 1)
    self.tdnn_5 = nn.Conv1d(int(512*scale), int(1500*scale), 1)
    self.fc_1 = nn.Linear(int(3000*scale), 512)
    self.relu = nn.ReLU()
    self.bn_1 = nn.BatchNorm1d(int(512*scale))
    self.bn_2 = nn.BatchNorm1d(int(1536*scale))
    self.bn_3 = nn.BatchNorm1d(int(512*scale))
    self.bn_4 = nn.BatchNorm1d(int(512*scale))
    self.bn_5 = nn.BatchNorm1d(int(1500*scale))
    
    nn.init.xavier_uniform_(self.fc_1.weight)
    self.dropout_1 = nn.Dropout(dropout)
    self.fc_2 = nn.Linear(512, 512)
    nn.init.xavier_uniform_(self.fc_1.weight)
    self.dropout_2 = nn.Dropout(dropout)

    self.binary_head = nn.Linear(512, 6)
    self.class_head = nn.Linear(512, 6)

    self.sig = nn.Sigmoid()

  def forward(self, x):
    '''forward method'''
    batch_size = x.shape[0]

    x = self.spec(x)
    x = self.db(x)
    # x = self.mfcc(x)
    x = self.tdnn_1(x)
    x = self.relu(x)
    x = self.bn_1(x)
    x = self.tdnn_2(x)
    x = self.relu(x)
    x = self.bn_2(x)
    x = self.tdnn_3(x)
    x = self.relu(x)
    x = self.bn_3(x)
    x = self.tdnn_4(x)
    x = self.relu(x)
    x = self.bn_4(x)
    x = self.tdnn_5(x)
    x = self.relu(x)
    x = self.bn_5(x)
        
    mean = torch.mean(x,-1)
    std = torch.std(x,-1)
    x = torch.cat((mean,std),1)
    x = self.fc_1(x)
    x = self.dropout_1(x)
    x = self.fc_2(x)
    x = self.dropout_2(x)

    binary = self.binary_head(x)
    # binary = self.sig(binary)

    classes = self.class_head(x)
    # classes = self.sig(classes)

    # return torch.cat((classes, binary), dim=-1)
    return torch.cat((binary, classes), dim=-1)

class ConvEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.0):
        '''Transformer encoder with convolution fcn head

        Args:
          embed (int): input embedding shape
          num_heads (int): number of heads in multi-head attention
          ff_dim (int): feed-forward dimensions
          dropout (float, optional): dropout rate
        '''
        super(ConvEncoder, self).__init__()
        
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.conv_1 = nn.Conv1d(embed_dim, ff_dim, 3, padding=1)
        # self.relu = nn.ReLU()
        self.relu = nn.LeakyReLU()
        self.conv_2 = nn.Conv1d(ff_dim, embed_dim, 3, padding=1)
        
    def forward(self, x):
        '''forward method'''
        inputs = x
        batch_size = x.shape[0]
        embed_dim = x.shape[-1]
        
        x, _ = self.mha(x, x, x)
        x = self.norm(x)
        x = self.dropout(x)
        x = x + inputs
        x = self.norm(x)
        x = x.reshape((batch_size, embed_dim, -1))
        x = self.conv_1(x)
        x = self.dropout(x)
        x = self.conv_2(x)

        return x

class SpeechTransformer(nn.Module):
    def __init__(self, n_mels=40, n_classes=12, num_blocks=4, num_units=2, 
              hidden_dim=128, num_heads=40, ff_dim=120, dropout=0.0, mlp_dropout=0.0):
        '''transformer with convolutional heads

        Args:
        n_mels (int, optional): number of mel filter banks
        n_classes (int, optional): number of classes
        num_blocks (int, optional): number of transformer blocks
        num_units (int, optional): number of linear units
        hidden_dim (int, optional): number of hidden units in linear layers
        num_heads (int): number of heads in multi-head attention
        ff_dim (int): feed-forward dimensions
        dropout (float, optional): dropout rate
        mlp_dropout (float, optional): mlp dropout rate
        '''
        super(SpeechTransformer, self).__init__()

        self.num_blocks = num_blocks
        self.num_units = num_units
        
        self.n_mels = n_mels
        
        self.spec = audio.transforms.MelSpectrogram(n_mels=n_mels, sample_rate=16000,
                                               n_fft=512, win_length=400, pad=1, f_max=8000, f_min=0,
                                               power=2.0, hop_length=160, norm='slaney')
        self.db = audio.transforms.AmplitudeToDB()

        self.transformers = nn.ModuleList([ConvEncoder(n_mels, num_heads, ff_dim, dropout=dropout)
                        for i_ in range(num_blocks)])
        self.first_mlp = nn.Linear(2*n_mels, hidden_dim)
        self.mlps = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_units-1)])
        self.dropout = nn.Dropout(mlp_dropout)
        self.relu = nn.ReLU()
        # self.out = nn.Linear(hidden_dim, 12)
        self.binary_head = nn.Linear(hidden_dim, 6)
        self.class_head = nn.Linear(hidden_dim, 6)

        self.sig = nn.Sigmoid()

    def forward(self, x):
        '''forward method'''
        batch_size = x.shape[0]
        
        x = self.spec(x)
        x = self.db(x)

        for transformer in self.transformers:
            x = x.reshape((batch_size, -1, self.n_mels))
            x = transformer(x)
        # x = torch.mean(x,-1)
        m, s = torch.mean(x, -1), torch.std(x, -1)
        x = torch.cat((m,s),1)
        x = self.first_mlp(x)
        for mlp in self.mlps:
            x = mlp(x)
            x = self.relu(x)
            x = self.dropout(x)
        # x = self.out(x)

        binary = self.binary_head(x)
        # binary = self.sig(binary)

        classes = self.class_head(x)
        # classes = self.sig(classes)

        return torch.cat((binary, classes), dim=-1)
        # return x