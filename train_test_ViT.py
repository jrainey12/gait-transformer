import argparse
import logging
from GEI.Gender_ViT_GEI import main as GEI
from key_frames.Gender_ViT_KF import main as KF
from single_frame.Gender_ViT_SF import main as SF


def main(input_dir, input_type, mode, model_name, lr, bs, epochs):
    """
    Wrapper for training and testing ViT model on gait representations.
    Args:
        input_dir - Directory containing training, testing and validation datasets.
        input_type - Type of gait representation, GEI, keyframe or single.
        mode - Mode of execution, train, test or resume.
        model_name - Name of model to train, test or resume.
        lr - learning rate to use for training.
        bs - batch size to use for training.
        epochs - number of epochs to train for.
    """
    
    if input_type == 'GEI':
    
        GEI(input_dir, lr, bs, epochs, mode, model_name)

    elif input_type == 'keyframe':
    
        KF(input_dir, lr, bs, epochs, mode, model_name)

    elif input_type == 'single':

        SF(input_dir, lr, bs, epochs, mode, model_name)



if __name__=='__main__':

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Train or test ViT with various gait representations.")

    parser.add_argument(
        'input_dir',
        help="Directory of the data.")
    parser.add_argument(
        'input_type',
        choices=['GEI','keyframe','single'],
        help="Type of input, GEI, keyframe or single.")

    parser.add_argument(
        'mode',
        choices=['train','test','resume'],
        help="Mode of operation, train, test or resume.")

    parser.add_argument(
        'model_name',
        help="Path to save new model or path to existing model for testing or resuming.")

    parser.add_argument(
        '-lr',
        default=0.0001,
        type=float,
        help="Learning rate for the model. Default: 0.0001.")

    parser.add_argument(
        '-bs',
        default=64,
        type=int,
        help="Batch size to use. Default: 64.")

    parser.add_argument(
        '--epochs',
        default=1,
        type=int,
        help="Number of epochs to train for. Default: 1.")


    args = parser.parse_args()

    main(args.input_dir, args.input_type, args.mode, args.model_name, args.lr, args.bs, args.epochs)
