from src.gravesnet import GravesPredictionNet
from src.train import optimize, OptimizationSizes

if __name__ == '__main__':
    sizes = OptimizationSizes(
        epoch_size=1000,
        train_size=1,
        eval_size=16,
        mini_batch_size=1
    )
    model = GravesPredictionNet(nhidden=100)
    optimize(model, sizes, data_dir="/home/karita/Desktop/res_listed/")
