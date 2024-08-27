from MCTS import MCTS_main
from ConvNets import DeepNN
from StateCNT import Dotdict, HarshState
import numpy as np
import pickle
from collections import deque
import time
import random
import multiprocessing
import pdb

##====## sequence parameters
# args = Dotdict({
#         'N': 12, # length of input image to DNN, i.e., N_prime in the paper
#         'K': 5, # width of input image to DNN, i.e., K_prime in the paper
#         'lpos': 5, # number of positions filled in each time step
#         'M': 3, # feature planes
#         'Q': 2, # 2 for binary sequence
#         'alpha': 0.05, # exploration noise, i.e., \alpha in the paper
#         'simBudget': 900, # MCTS simulation budget
#         'eval_games': 50,
#         'updateNNcycle': 300, # parameter G
#         'zDNN': 2, # parameter z
#         'numfilters1': 256,
#         'numfilters2': 512,
#         'l2_const': 1e-3,
#         'batchSize': 64,
#         'numEpisode': 100001,
#         'isMultiCore': 1,
#         'recordState': 0,
#         })

########## 周期序列
args = Dotdict({
        'N': 32, # length of input image to DNN, i.e., N_prime in the paper
        'K': 1, # width of input image to DNN, i.e., K_prime in the paper
        'lpos': 4, # number of positions filled in each time step
        'M': 3, # feature planes
        'Q': 2, # 2 for binary sequence
        'alpha': 0.05, # exploration noise, i.e., \alpha in the paper
        'simBudget': 400, # MCTS simulation budget
        'eval_games': 50,
        'updateNNcycle': 100, # parameter G
        'zDNN': 3, # parameter z
        'numfilters1': 256,
        'numfilters2': 512,
        'l2_const': 1e-3,
        'batchSize': 64,
        'numEpisode': 8000,
        'isMultiCore': 1,
        'recordState': 1,
        })

# ########## 互补码
# args = Dotdict({
#         'N': 8, # length of input image to DNN, i.e., N_prime in the paper
#         'K': 4, # width of input image to DNN, i.e., K_prime in the paper
#         'lpos': 4, # number of positions filled in each time step
#         'M': 3, # feature planes
#         'Q': 2, # 2 for binary sequence
#         'alpha': 0.05, # exploration noise, i.e., \alpha in the paper
#         'simBudget': 400, # MCTS simulation budget
#         'eval_games': 50,
#         'updateNNcycle': 100, # parameter G
#         'zDNN': 3, # parameter z
#         'numfilters1': 256,
#         'numfilters2': 512,
#         'l2_const': 1e-3,
#         'batchSize': 64,
#         'numEpisode': 8000,
#         'isMultiCore': 1,
#         'recordState': 1,
#         })

overallSteps = args.N * args.K
stepSize = args.lpos
n_steps = int(overallSteps/stepSize)
memorySize = n_steps * args.updateNNcycle * args.zDNN

####### Complementary Code J = 2, M = 2, N = 8
if np.mod(args.N, 2) == 1:
	worstMetric = 499
	bestMetric = 0
else:
	worstMetric = 496
	bestMetric = 0

######## pulse compression radar
# worstMetric = 37
# bestMetric = 0
# with segmented induction [0,15], [5,25], [10,37]
# worstMetric = 0
# bestMetric = 15

######## reward definition - complementary code
def calc_reward(currentState):
    # 计算周期函数
    currentState = np.array(currentState).flatten()
    N = len(currentState)
    currentState = np.array(currentState)
    Side = np.correlate(np.tile(currentState, 2), currentState, 'full')[N:N * 2 - 1]
    corrSum = np.sqrt(np.sum(np.abs(Side) ** 3))


    # Code set
    # NN = args.N
    #
    # aFig = currentState.reshape([NN, 4])
    # #
    # codeA1 = aFig[:, 0]
    # codeA2 = aFig[:, 1]
    # codeB1 = aFig[:, 2]
    # codeB2 = aFig[:, 3]
    #
    #
    # # cyclic auto-corr
    # # 同一用户的
    # res1 = np.correlate(codeA1, codeA1, 'full')
    # # print(res1[NN:NN * 2 - 1])
    # res2 = np.correlate(codeA2, codeA2, 'full')
    # # print(res2[NN:NN * 2 - 1])
    # zeroVec1 = np.sum(np.abs(res1[NN:NN * 2 - 1] + res2[NN:NN * 2 - 1]) ** 2)
    # res1 = np.correlate(codeB1, codeB1, 'full')
    # res2 = np.correlate(codeB2, codeB2, 'full')
    # zeroVec2 = np.sum(np.abs(res1[NN:NN * 2 - 1] + res2[NN:NN * 2 - 1]) ** 2)
    # # cyclic cross-corr
    # res1 = np.correlate(codeA1, codeB1, 'full')
    # res2 = np.correlate(codeA2, codeB2, 'full')
    # zeroVec5 = np.sum(np.abs(res1[NN:NN * 2] + res2[NN:NN * 2]) ** 2)
    # corrSum = np.sum([zeroVec1, zeroVec2, zeroVec5])
    #
    # for tau in range(1, NN):
    #     # flipped auto corr
    #     temp = np.dot(codeA1[range(NN - tau)], codeA1[range(tau, NN)]) - np.dot(codeA1[range(NN - tau, NN)],
    #                                                                             codeA1[range(tau)]) + np.dot(
    #         codeA2[range(NN - tau)], codeA2[range(tau, NN)]) - np.dot(codeA2[range(NN - tau, NN)], codeA2[range(tau)])
    #
    #     corrSum += np.abs(temp) ** 2
    #     temp = np.dot(codeB1[range(NN - tau)], codeB1[range(tau, NN)]) - np.dot(codeB1[range(NN - tau, NN)],
    #                                                                             codeB1[range(tau)]) + np.dot(
    #         codeB2[range(NN - tau)], codeB2[range(tau, NN)]) - np.dot(codeB2[range(NN - tau, NN)], codeB2[range(tau)])
    #     corrSum += np.abs(temp) ** 2
    #
    #     # flipped cross corr
    #     temp = np.dot(codeA1[range(NN - tau)], codeB1[range(tau, NN)]) - np.dot(codeA1[range(NN - tau, NN)],
    #                                                                             codeB1[range(tau)]) + np.dot(
    #         codeA2[range(NN - tau)], codeB2[range(tau, NN)]) - np.dot(codeA2[range(NN - tau, NN)], codeB2[range(tau)])
    #     corrSum += np.abs(temp) ** 2

    # aFig = currentState.reshape([NN, 4])
    # #J=2， M= 2，N= 8
    # #互补序列为[{A1,A2}, {B1,B2}]
    # codeA1 = aFig[:,0]
    # codeA2 = aFig[:,1]
    # codeB1 = aFig[:,2]
    # codeB2 = aFig[:,3]
    #
    # # cyclic auto-corr
    # # User1 两(M)个序列各自的自相关函数
    # res1 = np.correlate(np.tile(codeA1, 2), codeA1, 'full')
    # res2 = np.correlate(np.tile(codeA2, 2), codeA2, 'full')
    # # user1 两(M)个自相关函数求和
    # zeroVec1 = np.sum(np.abs(res1[NN:NN*2-1] + res2[NN:NN*2-1]))
    # # User2 两(M)个序列各自的自相关函数
    # res1 = np.correlate(np.tile(codeB1, 2), codeB1, 'full')
    # res2 = np.correlate(np.tile(codeB2, 2), codeB2, 'full')
    # # user2 两(M)个自相关函数求和
    # zeroVec2 = np.sum(np.abs(res1[NN:NN*2-1] + res2[NN:NN*2-1]))
    # # cyclic cross-corr
    # # user1和user2各自第一个序列互相关函数
    # res1 = np.correlate(np.tile(codeA1, 2), codeB1, 'full')
    # # user1和user2各自第二个序列互相关函数
    # res2 = np.correlate(np.tile(codeA2, 2), codeB2, 'full')
    # # user1 和 user2的两(M)个互相关函数求和
    # zeroVec5 = np.sum(np.abs(res1[NN:NN*2] + res2[NN:NN*2]))
    # corrSum = np.sum([zeroVec1, zeroVec2, zeroVec5])
    #
    # for tau in range(1,NN):
    #     # flipped auto corr
    #     # User1 两(M)个序列各自的翻转自相关函数
    #     temp = np.dot(codeA1[range(NN-tau)], codeA1[range(tau,NN)]) - np.dot(codeA1[range(NN-tau,NN)], codeA1[range(tau)]) + np.dot(codeA2[range(NN-tau)], codeA2[range(tau,NN)]) - np.dot(codeA2[range(NN-tau,NN)], codeA2[range(tau)])
    #     corrSum += np.abs(temp)
    #     temp = np.dot(codeB1[range(NN-tau)], codeB1[range(tau,NN)]) - np.dot(codeB1[range(NN-tau,NN)], codeB1[range(tau)]) + np.dot(codeB2[range(NN-tau)], codeB2[range(tau,NN)]) - np.dot(codeB2[range(NN-tau,NN)], codeB2[range(tau)])
    #     # user1第1个序列翻转自相关函数
    #     corrSum += np.abs(temp)
    #     # flipped cross corr
    #     # user1和user2的翻转互相关函数
    #     temp = np.dot(codeA1[range(NN-tau)], codeB1[range(tau,NN)]) - np.dot(codeA1[range(NN-tau,NN)], codeB1[range(tau)]) + np.dot(codeA2[range(NN-tau)], codeB2[range(tau,NN)]) - np.dot(codeA2[range(NN-tau,NN)], codeB2[range(tau)])
    #     corrSum += np.abs(temp)

    # given corrSum -> reward
    if corrSum <= worstMetric:
        reward = (worstMetric + bestMetric - 2 * corrSum) / (worstMetric - bestMetric) # -1 to 1
    else:
        reward = -1
    return reward, corrSum

######## reward definition - pulse compression radar
# def calc_reward(seq):
#     # detect error
#     if 0 in seq:
#         print("Reward Error - inputs contain 0")
#         pdb.set_trace()
#     seq = seq[0][:-1]
#     lenSeq = len(seq)
#
#     matrixC = np.zeros([lenSeq,2*lenSeq-2])
#
#     shift = 1 - lenSeq
#     xx = 0
#     while shift <= lenSeq - 1:
#         if shift != 0:
#             vecS = f_shiftadd0(seq, shift)
#             matrixC[:,xx] = vecS
#             xx += 1
#         shift += 1
#     matrixR = np.dot(matrixC, matrixC.T)
#
#     try:
#         temp1 = np.dot(seq,np.linalg.inv(matrixR))
#     except:
#         print("Error when calculating inverse")
#         print("======================================================")
#         print("======================================================")
#         print("======================================================")
#         return -1, 0
#
#     MF = np.dot(temp1,seq.T)
#
#     # reward
#     if MF >= worstMetric:
#         reward = (2 * MF - bestMetric - worstMetric) / (bestMetric - worstMetric) # -1 to 1
#     else:
#         reward = -1
#
#     return reward, MF

def f_shiftadd0(seq, shift):
    if shift > 0:
        return np.concatenate((np.zeros([shift]),seq[:-shift]),axis=0)
    else:
        return np.concatenate((seq[-shift:],np.zeros([-shift])),axis=0)

######## DNN player
def DNN_play(n_games, evaluating_fn):
    corrArray = []
    for _ in range(n_games):
        currentMove = np.array([])
        for eachstep in range(n_steps):
            currentState = np.append(currentMove, np.zeros(overallSteps -len(currentMove)))
            Prior_sa, value = evaluating_fn(currentState.reshape([1,len(currentState)]),0)
            nextMove = np.random.choice(args.Q ** stepSize, 1, p = Prior_sa[0])[0]
            realNextMoves = np.array([(nextMove>>k)&1 for k in range(0,stepSize)])[::-1]
            currentMove = np.append(currentMove, 2*realNextMoves-1)

        reward, corr = calc_reward(currentMove.reshape([1,len(currentMove)]))

        corrArray.append(corr)

    print("DNN play = ", corrArray)
    print("mean = ", np.mean(corrArray))
    # return np.mean(corrArray), np.max(corrArray)
    return np.mean(corrArray), np.min(corrArray)

######## AlphaSeq player (Noiseless games)
def evaluate_DNN(n_games, tau, evaluating_fn):
    # play 50 games, calculate the mean corr
    corrArray = []
    for _ in range(n_games):
        currentMove, _ = MCTS_main(args, VisitedState, stepSize, n_steps, DNN.evaluate_node, calc_reward, selfPlay = 0)

        # sequence found
        reward, corr = calc_reward(currentMove.reshape([1,len(currentMove)]))

        # record every reward
        corrArray.append(corr)

        # print("seq found = ", currentMove)
        # print("reward = ", reward)
        # print("corr = ", corr)

    print("MCTS + DNN play = ", corrArray)
    print("mean = ", np.mean(corrArray))
    # return np.mean(corrArray), np.max(corrArray)
    return np.mean(corrArray), np.min(corrArray)

######## Update DNN
def updateDNN(memoryBuffer, lr):
	######## without replacement
    # np.random.shuffle(memoryBuffer)
    # numBatch = int(len(memoryBuffer)/args.batchSize)
    # for ii in range(numBatch):
    #     mini_batch = []
    #     for jj in range(args.batchSize):
    #         mini_batch.append(memoryBuffer[ii*args.batchSize+jj])
    #     DNN.update_DNN(mini_batch, lr)
	######## with replacement
    # train DNN
    np.random.shuffle(memoryBuffer)
    numBatch = int(len(memoryBuffer)/args.batchSize) * 6
    for ii in range(numBatch):
        mini_batch = random.sample(memoryBuffer, args.batchSize)
        DNN.update_DNN(mini_batch, lr)

def main():
    # initialize memory buffer
    memoryBuffer = deque(maxlen = memorySize)

    # load/save latest structure
    # DNN.loadParams('./240730_bestParams/bestParams7600/net_params.ckpt')
    DNN.saveParams('./240812_bestParams/net_params.ckpt')

    # performance of current DNN
    DNNplayer, DNNmin = DNN_play(n_games = 100, evaluating_fn = DNN.evaluate_node)
    meanCorr , MCTSmin= evaluate_DNN(n_games = args.eval_games, tau = 0, evaluating_fn = DNN.evaluate_node)
    
    if args.recordState == 1:
        f = open('./240812_bestParams/Record.txt', 'w')
        f.write(str(0)+" "+str(DNNplayer)+" "+str(meanCorr)+" "+str(DNNmin)+" "+str(MCTSmin) + " " + str(0) + " ")
        f.write(str(0)+" ") # overall visited states
        f.write(str(0)+" ") # visited states in the last G episodes
        f.write(str(0)+" ") # mean entropy in the last G episodes
        f.write(str(0)+" ") # cross entropy in the last G episodes
        f.write(str(0)+";\n") # number of states being evaluated in the latest G episodes
        f.close()

    global worstMetric
    worstMetric = meanCorr

    print("-------------------------------------")
    print("worstMetric ="+str(worstMetric))
    print("bestMetric = "+str(bestMetric))

    overall_startTime = time.time()

    episode = 0
    while episode < args.numEpisode:
        print("----------------------------------------------  Episode %s:"%(episode))
        epi_time_start = time.time()

        # ---------------------- Part I: game-play with MCTS to gain experiences ----------------------
        cummulativeMove, temp_store = MCTS_main(args, VisitedState, stepSize, n_steps, DNN.evaluate_node, calc_reward, selfPlay = 1)

        # in each episode, we find a sequence - calculate reward
        reward, correlation = calc_reward(cummulativeMove.reshape([1,len(cummulativeMove)]))

        # Store n_steps experience
        print("seq found = ", cummulativeMove)
        print("reward = ", reward)
        print("corr = ", correlation)

        for state in temp_store:
            state.append(reward)

        memoryBuffer.extend(temp_store)

        if args.recordState == 1:
            print("...... The overall visited state so far =", VisitedState.printCnt())

        # --------------------------- Part II: DNN update ----------------------------
        lr = 0.0001

        # train NN
        if episode > 0 and episode % args.updateNNcycle == 0:
            # train new params
            updateDNN(memoryBuffer, lr)
            print("Deep Neural Network Updated, now evaluate the new DNN ...")
            print("learning rate = ",lr)

            # measure the performance of updated DNN
            print("---------------------------------------------------")

            DNNplayer, DNNmin = DNN_play(n_games=100, evaluating_fn=DNN.evaluate_node)
            print("DNN玩家100次游戏平均测评函数 = ", DNNplayer)
            print("DNN玩家100次游戏最小测评函数 = ", DNNmin)
            updatedCorr, MCTSmin = evaluate_DNN(n_games=args.eval_games, tau=0, evaluating_fn=DNN.evaluate_node)
            print("DNN-MCTS50次游戏玩家平均测评函数 = ", updatedCorr)
            print("DNN-MCTS50次游戏玩家最小测评函数 = ", MCTSmin)

            # ================================================================ store
            if args.recordState == 1:
                f = open('./240812_bestParams/Record.txt', 'a')
                f.write(str(episode)+" "+str(DNNplayer)+" "+str(updatedCorr)+" "+str(DNNmin)+" "+str(MCTSmin)+ " " + str(0)+ " ")
                f.write(str(VisitedState.printCnt())+" ") # overall visited states
                f.write(str(VisitedState.printCnt1())+" ") # visited states in the last G episodes
                VisitedState.renew()
                entropy, crossentropy, numStates = DNN.output_entropy()
                f.write(str(entropy)+" ") # mean entropy in the last G episodes
                f.write(str(crossentropy)+" ") # cross entropy in the last G episodes
                f.write(str(numStates)+";\n") # number of states being evaluated in the latest G episodes
                DNN.refresh_entropy()
                f.close()

            if args.recordState == 1:
                filename = "./240812_bestParams/bestParams" + str(episode) + "/net_params.ckpt"
                DNN.saveParams(filename)
                filename1 = "./240812_bestParams/States" + str(episode) + ".txt"
                f = open(filename1, 'w')
                f.write(str(VisitedState.visitedState))
                f.close()

        episode += 1
        print(time.time()-epi_time_start)

    # store experience
    # pickle.dump(DNN.memoryBuffer, open('./240731_bestParams/latestMemory', 'wb'))
    # loadedMemory = pickle.load(open('./240731_bestParams/latestMemory', 'rb'))

    print("-------------------------------------")
    print("-------------------------------------")
    print("After evaluation, the mean reward we get is %s"%(meanCorr))

    # seconds consumed from beginning
    print(time.time()-overall_startTime)

    DNN.plot_cost()
    pdb.set_trace()


if __name__ == "__main__":
    DNN = DeepNN(args, stepSize)
    VisitedState = HarshState(overallSteps)
    ##### load state counts
    # f = open('States3700.txt','r')
    # aa = f.read()
    # bb = eval(aa)
    # f.close()
    # VisitedState.visitedState = bb
    main()  
