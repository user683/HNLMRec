import argparse

parser = argparse.ArgumentParser(description="Run a SELFRec model with GPU and model selection.")
parser.add_argument('--model', type=str, help="Name of the model to run.", default='LightGCN')
parser.add_argument('--gpu', type=int, help="GPU ID to use (-1 for CPU).", default=0)
parser.add_argument('--input_size', type=int, help='Input of the embeddings size', default=4096)
parser.add_argument('--output_size', type=int, help='Input of the embeddings size', default=512)

args = parser.parse_args()
