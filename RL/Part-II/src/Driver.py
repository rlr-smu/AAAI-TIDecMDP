import numpy as np
import os
import tensorflow as tf
from Config import Config
from SR import EDECMDP
import argparse
import sys

parser = argparse.ArgumentParser(description='Mean Actor Critic')
parser.add_argument("mdp", help="MDP# to run MAC on", type=int)
parser.add_argument("filename", help="Filename for logging iterations")
parser.add_argument("--featureS", help="Number of features for S, default:30", type=int)
parser.add_argument("--featureSAS", help="Number of features for SAS, default:30", type=int)
parser.add_argument("--numIterations", help="Number of Iterations to train, default:15000", type=int)
parser.add_argument("--runWithSavedModel", help="Whether to run with saved model, default:False", action="store_true")
parser.add_argument("--loadPolicyFile", help="Which file to load model from?")
parser.add_argument("--EDlearningRate", help="Learning Rate for Encoder/Decoder, default:0.001", type=float)
parser.add_argument("--MlearningRate", help="Learning Rate for M Values, default:0.0005", type=float)
parser.add_argument("--PolMDPlearningRate", help="Learning Rate for Policy MDP, default:0.0001", type=float)
parser.add_argument("--PolEVlearningRate", help="Learning Rate for Policy Event, default:0.0001", type=float)
parser.add_argument("--batch", help="Batch Size, default:30", type=int)
parser.add_argument("--numUnitsPerLayer", help="Number of units per layer, default:32", type=int)
parser.add_argument("--explore", help="Whether to explore or not, default:False", action="store_true")
parser.add_argument("--lastKIter", help="Result as last k Iterations, default:-100", type=int)
parser.add_argument("--alpha", help="Success Probability for Inspect actions, default:0.95", type=float)
parser.add_argument("--beta", help="Success Probability for Move actions, default:0.95", type=float)
parser.add_argument("--normalizeMSAW", help="default:False", action="store_true")
parser.add_argument("--normalizeCK", help="default:False", action="store_true")
parser.add_argument("--NNlayerNorm", help="default:False", action="store_true")
parser.add_argument("--PollayerNorm", help="default:False", action="store_true")
parser.add_argument("--ReplayerNorm", help="default:False", action="store_true")

args = parser.parse_args()
c = Config(args.mdp)
if args.featureS:
    c.featsS = args.featureS
if args.featureSAS:
    c.featsSAS = args.featureSAS
if args.numIterations:
    c.numIterations = args.numIterations
if args.runWithSavedModel:
    c.runWithSavedModel = args.runWithSavedModel
if args.EDlearningRate:
    c.EDlearningRate = args.EDlearningRate
if args.MlearningRate:
    c.MlearningRate = args.MlearningRate
if args.PolMDPlearningRate:
    c.PolMDPlearningRate = args.PolMDPlearningRate
if args.PolEVlearningRate:
    c.PolEVlearningRate = args.PolEVlearningRate    
if args.batch:
    c.batch = args.batch
if args.numUnitsPerLayer:
    c.numUnitsPerLayer = args.numUnitsPerLayer
if args.explore:
    c.annealRandomActProb = args.explore
if args.lastKIter:
    c.lastKIter = args.lastKIter
if args.alpha:
    c.alpha = args.alpha
if args.beta:
    c.beta = args.beta
if args.loadPolicyFile:
    c.loadPolicyFile = str(args.loadPolicyFile)
    c.loadPolicyFileFull = c.saveModelsDir + c.loadPolicyFile + ".ckpt"
if args.normalizeMSAW:
    c.normalizeMSAW = args.normalizeMSAW
if args.normalizeCK:
    c.normalizeCK = args.normalizeCK
if args.NNlayerNorm:
    c.NNlayerNorm = args.NNlayerNorm
if args.PollayerNorm:
    c.PollayerNorm = args.PollayerNorm
if args.ReplayerNorm:
    c.ReplayerNorm = args.ReplayerNorm

e = EDECMDP(c)
e.initializeRL(filename=args.filename)
