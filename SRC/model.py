import torch 
import torch.nn as nn
class MyConvBlock( nn.Module ):
    def __init__(self,in_ch,out_ch,dropout_p):
        kernel_size=3
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,kernel_size,stride=1,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.MaxPool2d(2,stride=2)
        )
    def forward(self,x):
        return self.model(x)
def create_model(N_CLASSES=43,IMG_CHS=3):
    flattened_img_size=128*4*4
    base_model= nn.Sequential(
    MyConvBlock(IMG_CHS,32,0), #(32,16,16)
    MyConvBlock(32,64,0.2),#(64,8,8)
    MyConvBlock(64,128,0),#(128,4,4)
    nn.Flatten(),
    nn.Linear(flattened_img_size,512),
    nn.Dropout(.3),
    nn.ReLU(),
    nn.Linear(512,N_CLASSES)
)
device=torch.device("cuda"if torch.cuda.is_available() else "cpu")
torch._dynamo.config.suppress_errors = True
model=torch.compile(base_model.to(device))
return model 