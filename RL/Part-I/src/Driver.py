
# coding: utf-8

# In[ ]:


from Config import Config
from MDP import MDP
from EDECMDP import  EDECMDP
from Env import Env
import argparse
import sys

parser = argparse.ArgumentParser(description='Reinforce With Baseline')
parser.add_argument("mdp", help="MDP# to run MAC on", type=int)
parser.add_argument("filename", help="Filename for logging iterations")
parser.add_argument("--numIterations", help="Number of Iterations to train, default:15000", type=int)
parser.add_argument("--runWithSavedModel", help="Whether to run with saved model, default:False", action="store_true")
parser.add_argument("--BaslinelearningRate", help="Learning Rate for Basline, default:0.0001", type=float)
parser.add_argument("--PolicylearningRate", help="Learning Rate for Policy, default:0.0001", type=float)
parser.add_argument("--batch", help="Batch Size, default:30", type=int)
parser.add_argument("--numUnitsPerLayer", help="Number of units per layer, default:32", type=int)
parser.add_argument("--alpha", help="Success Probability for Inspect actions, default:0.7", type=float)
parser.add_argument("--beta", help="Success Probability for Move actions, default:0.7", type=float)
parser.add_argument("--VFLayerNorm", help="default:False", action="store_true")
parser.add_argument("--PolLayerNorm", help="default:False", action="store_true")
parser.add_argument("--drate", help="Dropout rate, default:0.0", type=float)

args = parser.parse_args()
c = Config(args.mdp)
if args.numIterations:
    c.numIterations = args.numIterations
if args.runWithSavedModel:
    c.runWithSavedModel = args.runWithSavedModel
if args.BaslinelearningRate:
    c.BaslinelearningRate = args.BaslinelearningRate
if args.PolicylearningRate:
    c.PolicylearningRate = args.PolicylearningRate
if args.batch:
    c.batch = args.batch
if args.numUnitsPerLayer:
    c.numUnitsPerLayer = args.numUnitsPerLayer
if args.alpha:
    c.alpha = args.alpha
if args.VFLayerNorm:
    c.VFLayerNorm = args.VFLayerNorm
if args.PolLayerNorm:
    c.PolLayerNorm = args.PolLayerNorm
if args.drate:
    c.drate = args.drate
if args.beta:
    c.beta = args.beta

e = EDECMDP(c)
e.initializeRL(filename=args.filename)

# if __name__ == "__main__":
#     filename = "3_MDP_3.1"
# #     experiment = 2
# #     agents = 2
# #     numberOfLines = 2
# #     nPrivatePerLine = 2
# #     nShared = 3
# #     minSharing = 2
# #     maxSharing = 2
# #     minT = 6
# #     maxT = 6
# #     minTaction = 2
# #     maxTaction = 2
# #     repetition = 32
# #     args = (agents,numberOfLines,nPrivatePerLine,nShared,minSharing,maxSharing,minT,maxT,minTaction,maxTaction,repetition)
#     c = Config(3)
#     e = EDECMDP(c)
#     e.initializeRL(filename)


