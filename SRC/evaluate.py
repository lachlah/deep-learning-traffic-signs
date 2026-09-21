from model import create_model
import torch
def validate():
    model=create_model()
    loss=0
    accuracy=0
    model.eval()
    with torch.no_grad():
        for x,y in test_loader:
            output=model(x)
            loss+=loss_function(output,y).item()
            accuracy += get_batch_accuracy(output,y,test_N)
        print('Valid-Loss:{:.4f} Accuracy {:.4f}'.format(loss,accuracy))